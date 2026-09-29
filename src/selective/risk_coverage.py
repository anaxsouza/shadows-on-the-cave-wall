"""Curva risco-cobertura, AURC e o teste de valor adicionado.

Este é código novo, não adaptação: a busca no repositório por `risk_coverage`,
`aurc`, `abstention`, `selective` e `conformal` antes da reorganização retornou
zero arquivos (docs/tese/reorg/corte.md, §5).

O QUE A CURVA É, OPERACIONALMENTE

Cada entidade predita recebe um escore de confiança. Ordenam-se as entidades da
mais confiante para a menos confiante e percorre-se essa ordem retendo prefixos:
retendo as k mais confiantes, a COBERTURA é k/n e o RISCO é a perda média
**entre as retidas**. Varrer k de 1 a n dá a curva. Curva mais baixa é melhor:
significa que, para a mesma fração de entidades entregues, o operador erra
menos.

A SUTILEZA QUE A NOTAÇÃO ESCONDE: O DENOMINADOR É CONDICIONAL

`risk = mean(losses[retidas])` divide pelo número de RETIDAS, não pelo total. É
por isso que a curva de um escore inútil é uma HORIZONTAL na taxa de erro base,
e não uma reta descendente: descartar entidades ao acaso remove erros e acertos
na mesma proporção, então a média entre as sobreviventes não se move. Quem
esperava que "abster-se de metade" reduzisse o risco pela metade está pensando
no risco não condicional, que é outra quantidade e não é a que interessa a quem
opera o extrator. Toda a força do teste do piso vem daí: bater a horizontal já
é dizer que o escore ordena erro.

EMPATES, E POR QUE ELES NÃO SÃO DETALHE

Se duas entidades têm o mesmo escore, a ordem entre elas é arbitrária, e um
prefixo que corte no meio de um grupo empatado produz um ponto de curva que
depende dessa arbitrariedade. A massa de atenção sobre o span é contínua e
empata pouco; a confiança do softmax, depois de agregada por entidade, empata
mais do que se imagina (entidades de um token com probabilidade saturada). Por
isso a curva é reportada apenas nas FRONTEIRAS dos grupos de empate: nesses
pontos o conjunto retido é o mesmo qualquer que seja a ordem interna.

EXEMPLO NUMÉRICO, QUE É ILUSTRAÇÃO E NÃO MEDIÇÃO

Seis entidades, três erradas: taxa de erro base 0,5. Três escores, os três
valores obtidos rodando este módulo:

- ordenação perfeita (as três corretas em cima): risco 0 até cobertura 0,5,
  depois 0,25, 0,4 e 0,5. AURC = 0,15.
- escore constante: um único ponto reportado, cobertura 1,0 e risco 0,5, porque
  as seis entidades formam um só grupo de empate. AURC = 0,50, que é a taxa de
  erro base — é o que "não ter instrumento" vale nesta escala.
- ordenação invertida (as três erradas em cima): AURC = 0,85, simétrico de 0,15
  em torno de 0,5.

Nenhum desses números vem de modelo: são a aritmética da definição, e estão
fixados como asserção em `tests/test_selective_risk_coverage.py`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

__all__ = [
    "RiskCoverageCurve",
    "risk_coverage_curve",
    "random_abstention_risk",
    "aurc",
    "delta_aurc_paired_bootstrap",
    "DeltaAURC",
]


@dataclass(frozen=True)
class RiskCoverageCurve:
    """Uma curva risco-cobertura já reduzida às fronteiras de empate.

    `coverage` e `risk` têm o mesmo comprimento, com cobertura crescente. `n` é o
    número de entidades e `base_risk` é o risco em cobertura total — a taxa de
    erro do extrator sem abstenção nenhuma, o ponto onde todas as curvas se
    encontram.
    """

    coverage: np.ndarray
    risk: np.ndarray
    n: int
    base_risk: float
    n_tied_groups: int

    @property
    def aurc(self) -> float:
        return aurc(self)

    def risk_at(self, target_coverage: float) -> float:
        """Risco na maior cobertura reportada que não excede `target_coverage`.

        Interpolar aqui seria inventar um ponto operacional que não existe: entre
        duas fronteiras de empate não há conjunto retido intermediário.
        """
        if not 0 < target_coverage <= 1:
            raise ValueError(f"cobertura alvo fora de (0, 1]: {target_coverage}")
        elegiveis = np.nonzero(self.coverage <= target_coverage + 1e-12)[0]
        if elegiveis.size == 0:
            raise ValueError(
                f"nenhuma cobertura reportada é <= {target_coverage}; "
                f"a menor é {self.coverage[0]:.4f}"
            )
        return float(self.risk[elegiveis[-1]])

    def coverage_at_risk(self, target_risk: float) -> float:
        """Maior cobertura cujo risco não excede `target_risk`.

        É esta a direção que a literatura canônica usa, e não a inversa:
        Geifman & El-Yaniv (2017) declaram um risco alvo r* e procuram a
        selecão que MAXIMIZA a cobertura sujeita a R(f,g) <= r*; a Tabela 1
        deles é indexada por r* e reporta a cobertura obtida. O controle
        conformal de risco declara alpha com o mesmo papel.

        Devolve 0.0 quando nem a menor cobertura reportada atinge o risco alvo —
        que é um resultado, não um erro: significa que nem abstendo-se ao máximo
        o critério é alcançável com este sinal. Levantar exceção aqui
        transformaria um resultado negativo legítimo em falha de execução.
        """
        if not 0 < target_risk <= 1:
            raise ValueError(f"risco alvo fora de (0, 1]: {target_risk}")
        elegiveis = np.nonzero(self.risk <= target_risk + 1e-12)[0]
        if elegiveis.size == 0:
            return 0.0
        return float(self.coverage[elegiveis].max())


def _validar(losses: Sequence[float], scores: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    perdas = np.asarray(losses, dtype=float)
    escores = np.asarray(scores, dtype=float)
    if perdas.ndim != 1 or escores.ndim != 1:
        raise ValueError("losses e scores têm de ser unidimensionais")
    if perdas.size != escores.size:
        raise ValueError(f"tamanhos diferentes: {perdas.size} perdas e {escores.size} escores")
    if perdas.size == 0:
        raise ValueError("nenhuma entidade: a curva não existe com n = 0")
    if not np.isfinite(perdas).all():
        raise ValueError("há perda não finita")
    if not np.isfinite(escores).all():
        raise ValueError(
            "há escore não finito — decida a convenção para o caso degenerado antes de rodar"
        )
    if (perdas < 0).any():
        raise ValueError("perda negativa")
    return perdas, escores


def risk_coverage_curve(losses: Sequence[float], scores: Sequence[float]) -> RiskCoverageCurve:
    """Constrói a curva. `scores` alto = mais confiante = retido primeiro.

    A perda é por entidade e não precisa ser 0/1, mas o controle conformal de
    risco em `conformal.py` supõe perda limitada e usa o limite superior
    declarado no pré-registro.
    """
    perdas, escores = _validar(losses, scores)
    ordem = np.argsort(-escores, kind="stable")
    perdas_ord = perdas[ordem]
    escores_ord = escores[ordem]

    k = np.arange(1, perdas.size + 1)
    risco_todos = np.cumsum(perdas_ord) / k

    # Fronteira de empate: k é fronteira quando o escore seguinte é estritamente
    # menor. O último k é sempre fronteira, e corresponde à cobertura total.
    fronteira = np.empty(perdas.size, dtype=bool)
    fronteira[:-1] = escores_ord[:-1] > escores_ord[1:]
    fronteira[-1] = True

    return RiskCoverageCurve(
        coverage=(k[fronteira] / perdas.size),
        risk=risco_todos[fronteira],
        n=int(perdas.size),
        base_risk=float(perdas.mean()),
        n_tied_groups=int(np.sum(~fronteira[:-1])),
    )


def random_abstention_risk(losses: Sequence[float]) -> float:
    """A linha de base do piso: a horizontal na taxa de erro base.

    Não há escore aqui, e é esse o ponto. Abstenção aleatória não altera o risco
    condicional esperado em nenhuma cobertura, então a "curva" do acaso é uma
    constante. Comparar contra ela é comparar contra não ter instrumento.
    """
    perdas = np.asarray(losses, dtype=float)
    if perdas.size == 0:
        raise ValueError("nenhuma entidade")
    return float(perdas.mean())


def aurc(curve: RiskCoverageCurve) -> float:
    """Área sob a curva risco-cobertura, por trapézios sobre a grade de cobertura.

    Duas convenções ficam declaradas porque mudam o número na terceira casa e não
    são universais na literatura:

    1. A integração é sobre a grade de cobertura efetivamente reportada (as
       fronteiras de empate), não sobre uma grade uniforme. Com escore contínuo
       as duas coincidem; com escore que empata, a grade uniforme exigiria
       inventar pontos operacionais inexistentes.
    2. Abaixo da primeira cobertura reportada a área é o retângulo do primeiro
       risco. Ignorar essa faixa é o que faz duas implementações de AURC
       discordarem em amostra pequena.

    A AURC de um escore constante é a taxa de erro base. Menor é melhor, e a
    comparação só faz sentido entre escores medidos nas MESMAS entidades.
    """
    cobertura, risco = curve.coverage, curve.risk
    if cobertura.size == 1:
        return float(risco[0])
    area = float(np.trapezoid(risco, cobertura))
    area += float(risco[0] * cobertura[0])  # faixa inicial, convenção 2
    return area / float(cobertura[-1])


@dataclass(frozen=True)
class DeltaAURC:
    """Diferença de AURC entre dois escores, com intervalo por reamostragem.

    `delta` é `aurc_b - aurc_a`: negativo significa que B é melhor, porque AURC
    menor é melhor. `contains_zero` é o veredito pré-registrado — o intervalo
    conter zero é a condição de refutação declarada no documento 04.
    """

    aurc_a: float
    aurc_b: float
    delta: float
    ci_low: float
    ci_high: float
    level: float
    n_resamples: int
    n_entities: int
    # Unidades INDEPENDENTES da reamostragem. Entidades da mesma sentença
    # compartilham a matriz de atenção, então o tamanho efetivo da amostra é o
    # número de sentenças e não o de entidades. Reportar os dois deixa visível o
    # quanto eles diferem — aqui, cerca de 2,7 entidades por sentença.
    n_groups: int
    contains_zero: bool
    n_degenerate_resamples: int = 0
    labels: tuple[str, str] = field(default=("a", "b"))

    def render(self) -> str:
        melhor = self.labels[1] if self.delta < 0 else self.labels[0]
        veredito = (
            "SIM — não há diferença detectável"
            if self.contains_zero
            else f"NÃO — {melhor} ordena melhor"
        )
        return (
            f"AURC {self.labels[0]} = {self.aurc_a:.4f} | AURC {self.labels[1]} = {self.aurc_b:.4f}\n"
            f"ΔAURC = {self.delta:+.4f}  IC {self.level:.0%} "
            f"[{self.ci_low:+.4f}, {self.ci_high:+.4f}]"
            f"  ({self.n_resamples} reamostragens de {self.n_groups} sentenças, "
            f"{self.n_entities} entidades)\n"
            f"Contém zero: {veredito}"
        )


def delta_aurc_paired_bootstrap(
    losses: Sequence[float],
    scores_a: Sequence[float],
    scores_b: Sequence[float],
    *,
    groups: Sequence[Any],
    n_resamples: int,
    level: float,
    seed: int = 42,
    labels: tuple[str, str] = ("a", "b"),
) -> DeltaAURC:
    """Intervalo para a diferença de AURC entre dois escores, por percentil.

    A reamostragem é PAREADA, e isso não é preciosismo: os dois escores são
    medidos nas mesmas unidades, então cada reamostragem sorteia UMA vez e
    recalcula as duas AURC no mesmo sorteio. Sortear independentemente para cada
    escore trataria como independentes duas quantidades que compartilham a
    amostra inteira, e produziria intervalo largo demais — errando na direção de
    parecer conservador, que é pior do que errar de forma visível.

    A reamostragem é por GRUPO, e `groups` não tem valor padrão de propósito.
    Entidades da mesma sentença compartilham a matriz de atenção: os números
    delas saíram da mesma medição, e não são observações independentes. Sortear
    entidades i.i.d. trataria 8.104 medições como 8.104 informações quando são
    ~3.000 — e o intervalo sairia ESTREITO demais, na direção de fazer uma
    diferença excluir o zero quando não devia. Erro que não aparece como erro,
    aparece como resultado bom.

    Para reproduzir a reamostragem por entidade — que é o que a `decl-01`
    declarou, e não o que a `decl-02` declara —, passe `groups=range(len(losses))`.

    `n_resamples` e `level` não têm valor padrão por decisão de desenho: são itens
    do pré-registro (documento 04, §3, itens 5 e 6) e quem chama tem de passá-los.
    Um padrão aqui seria um grau de liberdade escolhido depois de ver o resultado.

    Reamostragem degenerada — todas as entidades sorteadas com a mesma perda — é
    contada e descartada, não silenciada: com n pequeno ela acontece, e a contagem
    entra no relatório.
    """
    perdas, escores_a = _validar(losses, scores_a)
    _, escores_b = _validar(losses, scores_b)
    if n_resamples < 1:
        raise ValueError("n_resamples tem de ser >= 1")
    if not 0 < level < 1:
        raise ValueError(f"nível fora de (0, 1): {level}")

    a0 = risk_coverage_curve(perdas, escores_a).aurc
    b0 = risk_coverage_curve(perdas, escores_b).aurc

    grupos = np.asarray(list(groups))
    if grupos.size != perdas.size:
        raise ValueError(
            f"groups tem {grupos.size} entradas e losses tem {perdas.size}: "
            f"é um grupo por observação"
        )
    # Posições de cada grupo, calculadas UMA vez. Sortear grupos e concatenar as
    # posições deles é o bootstrap por conglomerado: quando uma sentença é
    # sorteada, todas as entidades dela vêm juntas.
    unicos, inverso = np.unique(grupos, return_inverse=True)
    posicoes = [np.flatnonzero(inverso == g) for g in range(unicos.size)]

    rng = np.random.default_rng(seed)
    n = perdas.size
    n_g = unicos.size
    deltas: list[float] = []
    degeneradas = 0
    for _ in range(n_resamples):
        sorteados = rng.integers(0, n_g, size=n_g)
        idx = np.concatenate([posicoes[g] for g in sorteados])
        p = perdas[idx]
        if np.all(p == p[0]):
            degeneradas += 1
            continue
        deltas.append(
            risk_coverage_curve(p, escores_b[idx]).aurc
            - risk_coverage_curve(p, escores_a[idx]).aurc
        )
    if not deltas:
        raise RuntimeError(
            f"todas as {n_resamples} reamostragens saíram degeneradas (n = {n}); "
            "a amostra é pequena demais para este intervalo"
        )

    alpha = 1.0 - level
    baixo, alto = np.quantile(deltas, [alpha / 2, 1 - alpha / 2])
    return DeltaAURC(
        aurc_a=a0,
        aurc_b=b0,
        delta=b0 - a0,
        ci_low=float(baixo),
        ci_high=float(alto),
        level=level,
        n_resamples=len(deltas),
        n_entities=int(n),
        n_groups=int(n_g),
        contains_zero=bool(baixo <= 0.0 <= alto),
        n_degenerate_resamples=degeneradas,
        labels=labels,
    )
