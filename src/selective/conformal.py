"""Controle conformal de risco: o ponto operacional único, com garantia.

Arcabouço: *conformal risk control* (arXiv 2208.02814), sobre predição conforme
(arXiv 2107.07511). É a garantia formal que sustenta a alternativa C4 e a razão
pela qual ela pontuou força 5 no critério de validação científica.

O QUE O PROCEDIMENTO ENTREGA

Dado um nível de risco tolerado `alpha`, ele escolhe um limiar de confiança a
partir de uma partição de CALIBRAÇÃO e devolve o limiar que entrega a MAIOR
cobertura cujo risco satisfaz a desigualdade do teorema. Em operação: "entregue
o máximo de entidades que puder, sem deixar o erro entre as entregues passar de
alpha".

A GARANTIA É MARGINAL, NÃO CONDICIONAL — E ISSO MUDA A FRASE QUE SE PODE ESCREVER

O teorema controla o valor ESPERADO da perda, onde a esperança é sobre o sorteio
da partição de calibração. Ele NÃO diz que nesta execução, com esta calibração,
o risco ficou abaixo de alpha; diz que o procedimento, repetido, tem risco médio
abaixo de alpha. A tese pode escrever "risco esperado controlado em alpha" e não
pode escrever "risco garantido abaixo de alpha neste conjunto". A distinção é a
mesma entre cobertura marginal e condicional em predição conforme, e a ressalva
está registrada no documento 03 na linha de C4.

A HIPÓTESE QUE TEM DE VALER: PERMUTABILIDADE

Calibração e teste têm de ser permutáveis. Numa única distribuição — mesmo
corpus, mesma anotação, partição aleatória — isso é satisfeito por construção.
Entre corpora (calibrar em CoNLL-2003 e operar em GENIA) NÃO é, e a garantia se
perde. Por isso o desenho calibra e opera dentro do mesmo corpus, e o contraste
entre corpora aparece na comparação de ganho, não na transferência de limiar.

MONOTONICIDADE, QUE É O DETALHE QUE PODE INVALIDAR A APLICAÇÃO

O teorema exige que o risco empírico seja monótono no parâmetro. Aqui o
parâmetro é o limiar: subir o limiar retém menos entidades e, se o escore
ordenar erro, o risco cai. "Se o escore ordenar erro" é justamente o que o teste
do piso investiga — logo a monotonicidade não pode ser assumida, tem de ser
verificada na própria calibração. Esta implementação verifica e reporta a
violação em vez de escondê-la: `monotone=False` com uma violação máxima grande
significa que o limiar devolvido não carrega a garantia do teorema, e a tese tem
de dizer isso em vez de citar o teorema.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

__all__ = ["CRCThreshold", "crc_threshold"]


@dataclass(frozen=True)
class CRCThreshold:
    """O ponto operacional escolhido na calibração, com o que ele custa e garante.

    `threshold` é o limiar de escore: retém-se a entidade cujo escore é >= ele.
    `bound` é o lado direito da desigualdade do teorema, `alpha - (B - alpha)/n`,
    e é ele — não `alpha` — que o risco empírico tem de respeitar. Quando não há
    limiar viável, `feasible` é False e `threshold` é NaN: nesse caso a resposta
    honesta é que nenhuma cobertura atinge o risco pedido nesta calibração.
    """

    threshold: float
    coverage: float
    empirical_risk: float
    alpha: float
    bound: float
    n_calibration: int
    loss_bound: float
    feasible: bool
    monotone: bool
    max_monotonicity_violation: float

    def render(self) -> str:
        # A nota de monotonicidade vale nos dois casos, e sai antes do desvio de
        # inviabilidade por uma razão: a hipótese do teorema falhar é diagnóstico
        # sobre o ESCORE, e continua sendo informação útil mesmo quando nenhum
        # limiar é viável. Reportá-la só no caminho viável a esconderia
        # exatamente quando o resultado está pior.
        nota = (
            ""
            if self.monotone
            else (
                f"\nATENÇÃO: risco empírico não monótono no limiar "
                f"(violação máxima {self.max_monotonicity_violation:.4f}). A hipótese de "
                f"monotonicidade do teorema não se verifica nesta calibração, então este "
                f"limiar não carrega a garantia formal."
            )
        )
        if not self.feasible:
            return (
                f"Nenhum limiar viável: com alpha = {self.alpha:.3f} e n = {self.n_calibration}, "
                f"o teorema exige risco empírico <= {self.bound:.4f}, e nenhuma cobertura o atinge.\n"
                f"Leitura: o extrator não chega a esse nível de risco nem abstendo-se ao máximo."
                f"{nota}"
            )
        return (
            f"Limiar: {self.threshold:.6f}  cobertura {self.coverage:.1%}  "
            f"risco empírico {self.empirical_risk:.4f}\n"
            f"alpha = {self.alpha:.3f}, limite do teorema = {self.bound:.4f} "
            f"(n = {self.n_calibration}, perda limitada por {self.loss_bound:g})"
            f"{nota}"
        )


def crc_threshold(
    losses: Sequence[float],
    scores: Sequence[float],
    alpha: float,
    loss_bound: float = 1.0,
) -> CRCThreshold:
    """Escolhe o limiar de maior cobertura cujo risco satisfaz o teorema.

    Implementa a desigualdade do *conformal risk control* na forma
    `R̂(λ) <= alpha - (B - alpha)/n`, com `B = loss_bound` (1 para perda 0/1) e
    `n` o tamanho da calibração. A correção `(B - alpha)/n` é o preço de estimar
    o risco na amostra finita, e é ela que torna o procedimento conservador em
    calibração pequena — com n = 50 e alpha = 0,1, o risco empírico exigido é
    0,082, não 0,1.

    Argumentos:
        losses: perda por entidade na partição de CALIBRAÇÃO. Não pode ser a
            partição de teste: usar a mesma partição para escolher o limiar e
            para reportar o risco destrói a garantia e é o erro mais fácil de
            cometer aqui.
        scores: escore de confiança das mesmas entidades, alto = mais confiante.
        alpha: risco tolerado. Item 6 do pré-registro.
        loss_bound: limite superior da perda. Perda 0/1 tem limite 1.

    A varredura é sobre os escores observados, do menor para o maior (cobertura
    total para cobertura mínima), e devolve o PRIMEIRO limiar viável — o de maior
    cobertura, que é o que interessa a quem opera.
    """
    perdas = np.asarray(losses, dtype=float)
    escores = np.asarray(scores, dtype=float)
    if perdas.size != escores.size:
        raise ValueError(f"tamanhos diferentes: {perdas.size} perdas e {escores.size} escores")
    if perdas.size == 0:
        raise ValueError("calibração vazia")
    if not (0 < alpha < loss_bound):
        raise ValueError(f"alpha tem de estar em (0, {loss_bound}): recebido {alpha}")
    if (perdas < 0).any() or (perdas > loss_bound).any():
        raise ValueError(f"há perda fora de [0, {loss_bound}]")

    n = perdas.size
    limite = alpha - (loss_bound - alpha) / n

    # Candidatos: cada escore observado, decrescente. Retendo score >= candidato,
    # a cobertura decresce à medida que o candidato sobe.
    candidatos = np.unique(escores)[::-1]
    riscos = np.empty(candidatos.size)
    coberturas = np.empty(candidatos.size)
    for i, tau in enumerate(candidatos):
        retidas = escores >= tau
        coberturas[i] = retidas.mean()
        riscos[i] = perdas[retidas].mean()

    # Monotonicidade, e a direção aqui é fácil de inverter: `candidatos` é
    # DEcrescente em escore, então percorrer i é AUMENTAR a cobertura (limiar
    # mais baixo retém mais). Um escore que ordena erro faz o risco SUBIR com a
    # cobertura — subida é o comportamento esperado, não a violação. A violação
    # é uma QUEDA: retiver mais entidades e errar menos entre as retidas
    # significa que o escore está pondo erro na frente de acerto em alguma
    # faixa, e é isso que rompe a hipótese do teorema.
    variacao = np.diff(riscos)  # índice crescente = cobertura crescente
    violacao = float(max(0.0, -float(np.min(variacao)) if variacao.size else 0.0))

    viaveis = np.nonzero(riscos <= limite)[0]
    if viaveis.size == 0:
        return CRCThreshold(
            threshold=float("nan"),
            coverage=float("nan"),
            empirical_risk=float(riscos[-1]) if riscos.size else float("nan"),
            alpha=alpha,
            bound=float(limite),
            n_calibration=int(n),
            loss_bound=loss_bound,
            feasible=False,
            monotone=violacao <= 1e-12,
            max_monotonicity_violation=violacao,
        )

    # Entre os viáveis, o de maior cobertura.
    i = int(viaveis[np.argmax(coberturas[viaveis])])
    return CRCThreshold(
        threshold=float(candidatos[i]),
        coverage=float(coberturas[i]),
        empirical_risk=float(riscos[i]),
        alpha=alpha,
        bound=float(limite),
        n_calibration=int(n),
        loss_bound=loss_bound,
        feasible=True,
        monotone=violacao <= 1e-12,
        max_monotonicity_violation=violacao,
    )
