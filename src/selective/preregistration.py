"""O pré-registro como estrutura de dados, e não como parágrafo de boa intenção.

Os seis itens do §3 do `docs/tese/04_c4_fundamentacao.md` — a regra de agregação
de confiança por entidade, a convenção de sumidouro e a faixa de camadas e
cabeças, a regra de combinação e a partição de calibração, os níveis de
cobertura e o ponto operacional único, o critério de valor adicionado e o número
de reamostragens, e o alpha do controle conformal — não têm valor padrão neste
módulo, e isso é o ponto.

POR QUE NÃO HÁ PADRÃO

Um padrão em código é uma escolha feita por quem escreveu o código, num momento
em que ele podia já ter visto um resultado. A objeção `R4` da banca é
exatamente sobre cortes numéricos sem sistemática, e a resposta não é escolher
cortes melhores: é tornar impossível escolher depois. Sem a seção `selective` no
`configs/config.yaml`, `load_preregistration` levanta erro e nada roda. O
documento de pré-registro e essa seção são a mesma declaração em dois formatos,
e `PREREGISTRO.md` é a versão que a banca lê.

A CONSEQUÊNCIA PRÁTICA, QUE É DESCONFORTÁVEL DE PROPÓSITO

Mudar qualquer um dos seis itens depois de ver a curva exige editar um arquivo
versionado, o que aparece no `git log` com data. Não impede a mudança — nada
impede — mas deixa rastro, que é tudo o que um pré-registro pode fazer.
"""

from __future__ import annotations

import hashlib
import json

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

__all__ = ["Preregistration", "load_preregistration", "PreregistrationError"]

POLITICAS_SUMIDOURO = ("keep", "drop_from_denominator", "drop_from_queries_and_denominator")
# O espaço de combinações do protocolo, restrito ao que tem precedente citável.
# `convex` é o de Zheng et al. (2026): um peso, ajustado em partição separada.
# `logistic` saiu: intercepto, dois coeficientes e uma ligação, e ninguém na
# linha citada usa.
COMBINACOES = ("convex", "rank_average", "product")

# Escores de COMPARAÇÃO: ordenações contra as quais o sinal elaborado tem de se
# provar. Três deles não usam modelo nenhum, e é isso que os torna exigentes —
# uma base que só pode enfraquecer a própria afirmação não é grau de liberdade,
# é um nulo mais difícil de bater. Adotados por serem prática estabelecida
# (Saghir 2026, arXiv 2605.00269: base trivial de comprimento, residualização
# contra comprimento, e relato explícito da correlação escore-comprimento).
ESCORES_COMPARACAO = (
    "span_size",            # k, o número de tokens do span
    "geometric_fraction",   # k/|K|, a massa ESPERADA sob permutabilidade das chaves
    "enrichment",           # massa / esperado: a desconfundição por razão
    "geometric_residual",   # massa menos o ajuste em k/|K|: a desconfundição por regressão
    "sentence_length",      # T, que é o que decide o sinal da massa (ver PROTOCOLO.md)

    # Os nove SINAIS das três famílias (`src/selective/signals.py`). O ESTATUTO
    # do nulo de cada um NÃO se repete aqui: ele está no registro de `signals.py`,
    # fixado em teste, e é de lá que o relatório o lê. Declarar duas vezes criaria
    # duas versões da mesma verdade, e elas divergiriam no primeiro sinal novo.
    #
    # Família de ATENÇÃO, nulo EXATO — sai do orçamento fixo, nada ajustado:
    "row_entropy",          # entropia das linhas do span; esperança log|K|
    "row_max",              # máximo das linhas do span; esperança 1/|K|
    "emitted_mass",         # massa que o span emite para fora de si
    # Família de ESTADOS OCULTOS, nulo EMPÍRICO — sem orçamento, logo regressão:
    "hidden_norm",
    "hidden_dist_centroide",
    "hidden_delta_camadas",
    # Família de LOGITS, nulo EMPÍRICO pela mesma razão:
    "logit_margin",
    "logit_entropy",
    "logit_max",            # é a própria confiança: verificação de identidade
)

# Variáveis de ESTRATIFICAÇÃO obrigatória no relato. `span_size` está aqui por
# medida: no GENIA a associação entre enriquecimento e erro tem sinal OPOSTO no
# agregado e dentro das faixas de tamanho (paradoxo de Simpson), então um efeito
# não estratificado não é interpretável.
ESTRATIFICACOES = ("span_size", "is_nested")

# Escores que podem aparecer numa comparação, além dos de comparação: o
# instrumento cru e a confiança do próprio modelo. Uma combinação declara-se com
# "+", e é ajustada na calibração pela regra declarada em `combination_rule`.
# Os nomes CAUSAIS e as SONDAS, acrescentados em 18/09/2026 para o braço decoder.
# Ficam aqui e o ESTATUTO do nulo de cada um continua só em `signals.py`, pela
# mesma razão que os nove originais: duas versões da mesma verdade divergiriam.
ESCORES_COMPARACAO = ESCORES_COMPARACAO + (
    "row_entropy_causal", "row_max_causal",
    "sonda_ocultos", "sonda_atencao_cabecas",
)

# `aggseq` é LINHA DE BASE e não sinal, e é por isso que entra aqui e NÃO no
# registro de `signals.py`: ele não é uma leitura do interior do modelo, é um
# estimador de confiança melhor que o nosso sobre a MESMA saída — agregação das
# probabilidades de sequência sobre o feixe de busca. A referência mede que ele
# reduz o erro de calibração esperado no NER de 0,075 para 0,030 contra a
# probabilidade do trecho, que é exatamente a nossa `model_confidence`.
#
# A consequência para o relato, declarada: com ele no lugar, a afirmação deixa
# de ser "os sinais internos não superam a confiança do modelo" e passa a ser
# "não superam nem a confiança nem o melhor estimador publicado" — e a segunda é
# muito mais difícil de atacar. O braço encoder não pode tê-lo: o GLiNER não
# gera, então não há feixe. A assimetria é declarada, não escondida.
ESCORES_BASE = ESCORES_COMPARACAO + ("span_mass", "model_confidence", "aggseq")

# Formas de veredito admitidas. Uma só, por enquanto, e nomeá-la impede que o
# critério vire prosa no momento de aplicar.
CRITERIOS = (
    "paired_delta_aurc_ci_excludes_zero",
    # Veredito na TAREFA: a carga de revisão exigida para atingir a mesma
    # qualidade entregue, comparada entre dois supervisores. Existe porque AURC
    # é média sobre coberturas e não é unidade em que alguém decide — "revisar
    # 12% a mais para a mesma precisão" é.
    "paired_review_load_ci_excludes_zero",
)

TIPOS_COMPARACAO = ("descriptive", "verdict", "task_verdict")

# Cada tipo de veredito admite UM critério, e o par é fixado aqui. Sem isso,
# declarar um veredito de tarefa com critério de AURC passaria pela validação e
# produziria número que não responde à pergunta declarada.
CRITERIO_DO_TIPO = {
    "verdict": "paired_delta_aurc_ci_excludes_zero",
    "task_verdict": "paired_review_load_ci_excludes_zero",
}

# As métricas da tarefa, todas calculadas ENTRE OS ENTREGUES. O supervisor só
# remove predições: ele nunca cria, então as entidades anotadas que o extrator
# não apontou são invisíveis para ele e o recall tem teto fixo. Declarar as
# quatro juntas impede relatar só a que subiu.
METRICAS_TAREFA = (
    "precision_delivered",   # 1 - erro médio entre os entregues
    "recall_delivered",      # entregues corretos / total de entidades anotadas
    "f1_delivered",          # harmônica das duas acima
    "review_load",           # fração das predições mandadas para revisão humana
)


@dataclass(frozen=True)
class Comparison:
    """Uma comparação declarada, com o que ela afirma e como se refuta.

    `descriptive` NÃO tem veredito, e isso é deliberado: caracterizar o nulo é
    parte do resultado sem ser teste. Transformar caracterização em veredito
    gastaria um teste pré-registrado para confirmar o que a validação já mostrou.
    """

    id: str
    kind: str
    question: str
    score: str | None = None
    against: str | None = None
    criterion: str | None = None

    def render(self) -> str:
        if self.kind == "descriptive":
            return f"  {self.id} (descritivo, sem veredito): {self.question}"
        return (f"  {self.id}: {self.question}\n"
                f"      {self.score}  contra  {self.against}\n"
                f"      refuta-se por: {self.criterion}")


class PreregistrationError(RuntimeError):
    """Falta declaração, ou a declaração é inválida. Em nenhum dos casos se roda."""


@dataclass(frozen=True)
class Preregistration:
    """Os seis itens, mais o que é preciso para reproduzir a execução.

    Cada campo corresponde a um item numerado do §3 do documento 04, e a ordem
    aqui é a ordem de lá.
    """

    # QUAL MODELO. Item de MEDIÇÃO a partir da versão 4 do conjunto: duas escalas
    # do mesmo modelo produzem tabelas diferentes, e até 15/09/2026 as duas
    # carregariam o mesmo measurement_hash — o guarda de procedência aceitaria
    # cruzar uma tabela do base com uma declaração do large sem reclamar.
    # Vazio nas versões 1 a 3, e o vazio é informação: aquelas declarações não
    # fixam modelo, e as tabelas medidas sob elas são identificadas pelo bloco
    # `modelo` do MEDIDA.json, não pelo hash.
    model: str | None
    sink_policy: str                            # item 2
    layers: tuple[int, ...]                     # item 2 -- CONFIRMATÓRIO
    heads: tuple[int, ...] | None               # item 2
    combination_rule: str                       # item 3
    calibration_fraction: float                 # item 3
    coverage_levels: tuple[float, ...]           # item 4
    # O ponto operacional NÃO é declarado: é derivado do risco alvo.
    # Geifman & El-Yaniv (2017) fixam r* e maximizam a cobertura sujeita a ele —
    # a Tabela 1 deles é indexada por r* e reporta a cobertura obtida. Declarar
    # cobertura seria inconsistente com o método da garantia, que declara risco.
    # RELATO: a grade de riscos alvo, forma da Tabela 1 de Geifman & El-Yaniv
    # (2017). Não custa grau de liberdade porque os critérios dos dois testes
    # são intervalos sobre AURC e não olham para r*.
    target_risk_grid: tuple[float, ...]
    added_value_ci_level: float                  # item 5
    n_bootstrap_resamples: int                   # item 5
    conformal_alpha: float                       # item 6
    loss_bound: float
    seed: int
    # Contra o que o sinal elaborado tem de se provar, e por qual variável o
    # relato é estratificado. Itens de ANÁLISE: não tocam na produção da tabela.
    comparison_scores: tuple[str, ...]
    stratify_by: tuple[str, ...]
    # A GRADE de metas de qualidade, e não uma meta única. Uma meta só, fixada no
    # escuro, pode cair onde nenhum supervisor chega — com erro base de 48% e 53%,
    # precisão 0,90 entre os entregues pode exigir abster-se de quase tudo, e a
    # declaração não decidiria nada. Com a grade, "inalcançável nesta meta" deixa
    # de ser execução perdida e passa a ser uma linha da tabela. Mesma forma da
    # grade de riscos, e pela mesma razão.
    #
    # E ela não custa grau de liberdade: o veredito é a comparação entre
    # supervisores em CADA meta, então não há meta a escolher depois.
    task_quality_grid: tuple[float, ...]
    task_metrics: tuple[str, ...]
    # As bordas das faixas de tamanho de span, DECLARADAS. Sem elas, `stratify_by`
    # diz que estratificar e não como, e o corte muda a leitura descritiva: no
    # GENIA a inversão de sinal vai de 3-de-4 estratos a 4-de-4 a 1-de-2 conforme
    # onde a borda cai. O veredito não depende delas — é sobre a amostra inteira —
    # mas a tabela descritiva depende, e borda escolhida depois é grau de liberdade.
    span_size_bins: tuple[tuple[int, int | None], ...]
    comparisons: tuple[Comparison, ...]
    hash_version: int
    # A qual medição esta análise se amarra, quando ela reusa a de outra
    # declaração. É DECLARADO e não inferido: o guarda de procedência não deve
    # adivinhar que duas declarações compartilham medição — quem afirma isso
    # assina. Formato (declaration_id, declaration_hash) da declaração de origem.
    inherits_measurement_from: tuple[str, str] | None
    declaration_id: str
    source: str
    # EXPLORATÓRIO, e a separação é o que impede o perfil de virar pescaria: ele
    # descreve onde o sinal mora ao longo da profundidade, mas não escolhe a
    # faixa nem decide veredito. Medir N camadas e depois anunciar a melhor daria
    # N chances ao acaso — o veredito sai de `layers`, e só dele. Fica por último
    # e com padrão porque é o único campo que NÃO é item pré-registrado.
    layer_profile: bool = False

    # Os itens que ENTRAM no hash, em ordem fixa. `source` e `declaration_id`
    # ficam de fora de propósito: mover o arquivo de lugar ou renomear a
    # declaração não muda o que foi declarado, e um hash que mudasse por isso
    # seria inútil como identidade do conteúdo.
    # O hash é PARTIDO EM DOIS, e a razão é científica e não de organização.
    #
    # Uma declaração nova pode mudar só a ANÁLISE — acrescentar um escore de
    # comparação, um critério — sem mudar nada de como a tabela foi produzida.
    # Se houvesse um hash só, `decl-02` invalidaria tabelas medidas sob `decl-01`
    # com parâmetros de medição IDÊNTICOS, e a alternativa seria remedir por
    # burocracia ou afrouxar o guarda. As duas são ruins.
    #
    # MEDIÇÃO: itens que entram na produção da tabela. Mudar qualquer um deles
    # produz números diferentes, e a tabela antiga deixa de servir.
    # RELATO, versão 5: a fração do VÃO entre a precisão de base e 1,0 que a meta
    # fecha. Substitui a grade absoluta como eixo de comparação entre modelos,
    # porque meta absoluta mede a qualidade do ARRANJO: 0,80 é trivial num
    # extrator que já entrega 0,8997 sem abster-se e inalcançável num que entrega
    # 0,4690. A fração é declarada; a meta absoluta é DERIVADA da precisão de base
    # da partição de CALIBRAÇÃO — nunca da de avaliação, que produz o veredito.
    task_gap_grid: tuple[float, ...] = ()

    _ITENS_MEDICAO_V1 = ("sink_policy", "layers", "heads")
    # Versão 4 acrescenta o MODELO à identidade da medição. É item de medição e
    # não de análise porque mudar de modelo muda os números DENTRO da tabela.
    _ITENS_MEDICAO_V4 = ("model",) + _ITENS_MEDICAO_V1
    # ANÁLISE: itens que só agem depois, sobre a tabela pronta.
    _ITENS_ANALISE_V1 = (
        "combination_rule", "calibration_fraction", "coverage_levels",
        "target_risk_grid", "added_value_ci_level", "n_bootstrap_resamples",
        "conformal_alpha", "loss_bound", "seed",
    )
    _ITENS_ANALISE_V2 = _ITENS_ANALISE_V1 + (
        "comparison_scores", "stratify_by", "inherits_measurement_from",
        "span_size_bins", "comparisons",
    )

    # O CONJUNTO DE ITENS É VERSIONADO, e a razão é a assinatura.
    #
    # Acrescentar um item ao hash muda o hash de TODAS as declarações, inclusive
    # as já assinadas. Um hash assinado que deixa de ser reproduzível não prova
    # mais nada: a defesa inteira do pré-registro é que o commit da assinatura
    # veio antes da medição, e isso só se verifica recalculando o hash.
    #
    # Então o conjunto vira versão. `decl-01`, assinada em 02/09/2026, declara
    # versão 1 e continua reproduzindo o hash que assinou para sempre. Versão 2
    # acrescenta os escores de comparação e a estratificação, e é o que qualquer
    # declaração nova usa.
    # Versão 3 acrescenta a grade de metas de qualidade e as métricas da tarefa.
    # `decl-02`, assinada em 10/09/2026, declara versão 2 e segue reproduzindo o
    # hash que assinou.
    _ITENS_ANALISE_V3 = _ITENS_ANALISE_V2 + ("task_quality_grid", "task_metrics")
    # Versão 5 acrescenta a grade RELATIVA à precisão de base. É item de
    # ANÁLISE e não de medição: ela muda a meta contra a qual a tabela é
    # julgada, e não um número dentro da tabela. Por isso uma declaração v5
    # pode herdar a medição de uma v4 — e é isso que permite refazer o relato
    # dos três modelos já medidos sem medir de novo.
    _ITENS_ANALISE_V5 = _ITENS_ANALISE_V3 + ("task_gap_grid",)

    _ITENS_POR_VERSAO = {
        1: _ITENS_MEDICAO_V1 + _ITENS_ANALISE_V1,
        2: _ITENS_MEDICAO_V1 + _ITENS_ANALISE_V2,
        3: _ITENS_MEDICAO_V1 + _ITENS_ANALISE_V3,
        4: _ITENS_MEDICAO_V4 + _ITENS_ANALISE_V3,
        5: _ITENS_MEDICAO_V4 + _ITENS_ANALISE_V5,
    }

    def _itens_medicao(self) -> tuple[str, ...]:
        """Os itens de medição DA VERSÃO desta declaração.

        Versionado pela mesma razão do conjunto inteiro: acrescentar o modelo ao
        hash de medição mudaria o hash das três declarações já assinadas, e
        assinatura que deixa de ser reproduzível não prova mais ordem nenhuma.
        """
        return self._ITENS_MEDICAO_V4 if self.hash_version >= 4 else self._ITENS_MEDICAO_V1

    @property
    def target_risk(self) -> float:
        """O risco alvo r*: a fração de entidades entregues que se admite errada.

        É o nível da GARANTIA, e o mesmo número que `conformal_alpha`: o controle
        conformal de risco admite um só. A grade `target_risk_grid` é de relato.

        Ter um nome próprio é o ponto:
        Geifman & El-Yaniv (2017) fixam r* e maximizam a cobertura sujeita a ele,
        e o controle conformal de risco declara α com o mesmo papel. Eram dois
        parâmetros declarados; a literatura mostra que é um.
        """
        return self.conformal_alpha

    @staticmethod
    def _serializavel(v):
        """Forma estável para o hash. Uma estrutura tem de virar dado, e não
        endereço de memória: `default=str` sobre uma dataclass geraria hash
        diferente a cada execução se o repr mudasse."""
        if isinstance(v, Comparison):
            return [v.id, v.kind, v.question, v.score, v.against, v.criterion]
        if isinstance(v, (list, tuple)):
            return [Preregistration._serializavel(x) for x in v]
        return v

    def _hash_de(self, itens: tuple[str, ...]) -> str:
        bruto = json.dumps({k: self._serializavel(getattr(self, k)) for k in itens},
                           sort_keys=True, ensure_ascii=True, default=list)
        return hashlib.sha256(bruto.encode("utf-8")).hexdigest()[:16]

    @property
    def measurement_hash(self) -> str:
        """Identidade dos parâmetros que PRODUZEM a tabela.

        É este que o guarda de procedência compara: uma tabela medida sob outra
        declaração é admissível se, e só se, os parâmetros de medição forem os
        mesmos. O `declaration_hash` continua gravado, para que a linhagem
        apareça no relatório mesmo quando a medição é compartilhada.
        """
        return self._hash_de(self._itens_medicao())

    @property
    def analysis_hash(self) -> str:
        """Identidade do que se faz DEPOIS da tabela, na versão declarada."""
        itens = self._ITENS_POR_VERSAO[self.hash_version]
        medicao = self._itens_medicao()
        return self._hash_de(tuple(i for i in itens if i not in medicao))

    @property
    def declaration_hash(self) -> str:
        """Identidade do CONTEÚDO declarado, para amarrar medida e teste.

        Gravado ao lado de cada medição. Sem isso, uma tabela medida sob uma
        declaração poderia ser testada sob outra — e o resultado sairia sem
        erro, parecendo válido, com metade dos parâmetros de cada uma.
        """
        return self._hash_de(self._ITENS_POR_VERSAO[self.hash_version])

    def render(self) -> str:
        cabecas = "todas" if self.heads is None else ", ".join(str(h) for h in self.heads)
        return (
            f"Declaração {self.declaration_id}  ·  hash {self.declaration_hash}\n"
            f"  carregada de {self.source}\n"
            f"  1. sumidouro: {self.sink_policy} | camadas: "
            f"{', '.join(str(c) for c in self.layers)} | cabeças: {cabecas}\n"
            f"  2. combinação: {self.combination_rule} | partição de calibração: "
            f"{self.calibration_fraction:.0%} das sentenças\n"
            f"  3. riscos alvo relatados: "
            f"{', '.join(f'{g:.2f}' for g in self.target_risk_grid)} | GARANTIA em "
            f"r* = {self.target_risk:.3f} "
            f"(perda limitada por {self.loss_bound:g}); o ponto operacional é a "
            f"COBERTURA OBTIDA a esse risco, não um número declarado\n"
            f"  4. coberturas reportadas na curva: "
            f"{', '.join(f'{c:.0%}' for c in self.coverage_levels)}\n"
            f"  5. valor adicionado: IC {self.added_value_ci_level:.0%} com "
            f"{self.n_bootstrap_resamples} reamostragens\n"
            f"  semente: {self.seed}\n"
            f"  perfil por camada (EXPLORATÓRIO, não decide veredito): "
            f"{'sim' if self.layer_profile else 'não'}"
        )


def _exigir(d: Mapping[str, Any], chave: str, onde: str) -> Any:
    if chave not in d:
        raise PreregistrationError(
            f"item de pré-registro ausente: '{chave}' em {onde}. "
            f"Declare-o em configs/config.yaml (seção 'selective') e em "
            f"docs/tese/PREREGISTRO.md antes de rodar."
        )
    return d[chave]


def _fracao(valor: Any, nome: str) -> float:
    """Fração estritamente interna a (0, 1): partição, nível de IC, alpha.

    Um alpha de 1 ou uma partição de calibração de 100% não são escolhas
    conservadoras, são erros de digitação.
    """
    v = float(valor)
    if not 0 < v < 1:
        raise PreregistrationError(f"{nome} tem de estar em (0, 1): recebido {v}")
    return v


def _cobertura(valor: Any, nome: str) -> float:
    """Cobertura vive em (0, 1], e o 1 é legítimo: é o ponto de cobertura total.

    Separado de `_fracao` porque a cobertura total é o ponto onde todas as
    curvas se encontram — a taxa de erro sem abstenção nenhuma — e reportá-la é
    o que dá referência às outras coberturas. Tratá-la como fração inválida foi
    um erro apanhado pelos testes de ponta a ponta do runner.
    """
    v = float(valor)
    if not 0 < v <= 1:
        raise PreregistrationError(f"{nome} tem de estar em (0, 1]: recebido {v}")
    return v


def load_preregistration(config_path: str | Path) -> Preregistration:
    """Lê a seção `selective` do config e valida os seis itens.

    Validar aqui, e não no ponto de uso, é deliberado: um item inválido tem de
    impedir a execução antes de qualquer passagem de modelo, e não depois — para
    que ninguém tenha visto número nenhum quando o erro aparece.
    """
    import yaml

    caminho = Path(config_path)
    if not caminho.is_file():
        raise PreregistrationError(f"config não encontrado: {caminho}")
    try:
        bruto = yaml.safe_load(caminho.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as e:
        # Erro de parsing vira PreregistrationError e não vaza: quem chama
        # trata uma exceção só, e apontar o arquivo errado (um .toml, por
        # exemplo) é o mesmo problema de declaração que uma seção ausente.
        raise PreregistrationError(f"{caminho} não é YAML válido: {e}") from e
    if not isinstance(bruto, dict):
        raise PreregistrationError(
            f"{caminho} não contém um mapeamento no topo; recebido {type(bruto).__name__}"
        )
    if "selective" not in bruto:
        raise PreregistrationError(
            f"{caminho} não tem a seção 'selective'. Os seis itens do pré-registro "
            f"(documento 04, §3) têm de estar declarados antes de rodar qualquer teste: "
            f"não há valor padrão para nenhum deles, por desenho."
        )
    s = bruto["selective"]
    onde = f"{caminho}:selective"

    sumidouro = str(_exigir(s, "sink_policy", onde))
    if sumidouro not in POLITICAS_SUMIDOURO:
        raise PreregistrationError(
            f"política de sumidouro desconhecida: {sumidouro!r}; use uma de {POLITICAS_SUMIDOURO}"
        )

    camadas = _exigir(s, "layers", onde)
    if not isinstance(camadas, Sequence) or isinstance(camadas, str) or not len(camadas):
        raise PreregistrationError("'layers' tem de ser uma lista não vazia de índices de camada")
    camadas_t = tuple(int(c) for c in camadas)

    cabecas = s.get("heads", None)
    if cabecas is None:
        cabecas_t = None
    else:
        if not isinstance(cabecas, Sequence) or isinstance(cabecas, str) or not len(cabecas):
            raise PreregistrationError("'heads' tem de ser lista não vazia, ou ausente para todas")
        cabecas_t = tuple(int(h) for h in cabecas)

    combinacao = str(_exigir(s, "combination_rule", onde))
    if combinacao not in COMBINACOES:
        raise PreregistrationError(f"combinação desconhecida: {combinacao!r}; use uma de {COMBINACOES}")

    coberturas = _exigir(s, "coverage_levels", onde)
    if not isinstance(coberturas, Sequence) or isinstance(coberturas, str) or not len(coberturas):
        raise PreregistrationError("'coverage_levels' tem de ser lista não vazia")
    coberturas_t = tuple(_cobertura(c, "cobertura") for c in coberturas)

    # O ponto operacional é derivado, e a declaração tem de dizer isso
    # explicitamente. Um número aqui seria a parametrização invertida: declarar
    # cobertura e ler risco, quando o método da garantia declara risco.
    ponto = str(_exigir(s, "operating_point", onde))
    if ponto != "derived_from_target_risk":
        raise PreregistrationError(
            f"operating_point tem de ser 'derived_from_target_risk', não {ponto!r}. "
            f"Geifman & El-Yaniv (2017) fixam o risco alvo e maximizam a cobertura "
            f"sujeita a ele; o controle conformal de risco tem a mesma forma. Declarar "
            f"uma cobertura fixa inverteria a parametrização do próprio método."
        )

    grade = _exigir(s, "target_risk_grid", onde)
    if not isinstance(grade, Sequence) or isinstance(grade, str) or not len(grade):
        raise PreregistrationError("'target_risk_grid' tem de ser lista não vazia de riscos")
    grade_t = tuple(sorted(_cobertura(g, "risco alvo") for g in grade))

    n_boot = int(_exigir(s, "n_bootstrap_resamples", onde))
    if n_boot < 100:
        raise PreregistrationError(
            f"n_bootstrap_resamples = {n_boot} é baixo demais para um intervalo por percentil"
        )

    alpha = _fracao(_exigir(s, "conformal_alpha", onde), "conformal_alpha")
    if alpha not in grade_t:
        raise PreregistrationError(
            f"conformal_alpha = {alpha} não está na grade de relato {grade_t}. O nível da "
            f"garantia tem de aparecer também na tabela de coberturas obtidas, senão o "
            f"número que sustenta a garantia é o único que não se lê no relato."
        )

    versao = int(s.get("hash_version", 1))
    if versao not in Preregistration._ITENS_POR_VERSAO:
        raise PreregistrationError(
            f"hash_version {versao} desconhecida; conhecidas: "
            f"{sorted(Preregistration._ITENS_POR_VERSAO)}")

    herdado = s.get("inherits_measurement_from")
    if herdado is None:
        herda = None
    else:
        if not isinstance(herdado, dict):
            raise PreregistrationError(
                "inherits_measurement_from tem de ser um mapa com declaration_id e declaration_hash")
        falta = [c for c in ("declaration_id", "declaration_hash") if not herdado.get(c)]
        if falta:
            raise PreregistrationError(
                f"inherits_measurement_from sem {', '.join(falta)}: herança sem hash não é "
                f"verificável, e herança não verificável é adivinhação")
        herda = (str(herdado["declaration_id"]), str(herdado["declaration_hash"]))

    if versao == 1:
        # Versão 1 é o conjunto congelado de decl-01: ela não conhece estes itens,
        # e declará-los nela seria declarar algo que o hash não cobre — um item
        # fora do hash não é declaração, é comentário.
        for proibido in ("comparison_scores", "stratify_by", "inherits_measurement_from",
                         "span_size_bins", "comparisons",
                         "task_quality_grid", "task_metrics"):
            if proibido in s:
                raise PreregistrationError(
                    f"{proibido} exige hash_version 2; na versão 1 ele não entraria no hash")
        comparacoes_t, estratos_t = (), ()
        faixas_t, comps_t = (), ()
        metas_t, metricas_t, vaos_t = (), (), ()
        if "model" in s:
            raise PreregistrationError(
                "model exige hash_version 4; na versão 1 ele não entraria no hash de medição")
        modelo = None
    else:
        comparacoes = _exigir(s, "comparison_scores", onde)
        if isinstance(comparacoes, str) or not comparacoes:
            raise PreregistrationError("comparison_scores tem de ser uma lista não vazia")
        comparacoes_t = tuple(str(c) for c in comparacoes)
        for c in comparacoes_t:
            if c not in ESCORES_COMPARACAO:
                raise PreregistrationError(
                    f"escore de comparação desconhecido: {c!r}; use algum de {ESCORES_COMPARACAO}")

        faixas = _exigir(s, "span_size_bins", onde)
        if isinstance(faixas, str) or not faixas:
            raise PreregistrationError("span_size_bins tem de ser uma lista de pares [min, max]")
        faixas_t = []
        anterior = 0
        for par in faixas:
            if not isinstance(par, (list, tuple)) or len(par) != 2:
                raise PreregistrationError(f"faixa mal formada: {par!r}; use [min, max] com max null no fim")
            lo, hi = par
            lo = int(lo)
            hi = None if hi is None else int(hi)
            if lo != anterior + 1:
                raise PreregistrationError(
                    f"faixa começando em {lo} deixa buraco depois de {anterior}: as faixas têm de "
                    f"cobrir os tamanhos sem lacuna, senão entidades somem do relato")
            if hi is not None and hi < lo:
                raise PreregistrationError(f"faixa invertida: [{lo}, {hi}]")
            faixas_t.append((lo, hi))
            anterior = hi if hi is not None else 10 ** 9
        if faixas_t[-1][1] is not None:
            raise PreregistrationError(
                "a última faixa tem de ser aberta (max null): span maior que a última borda "
                "ficaria fora do relato")
        faixas_t = tuple(faixas_t)

        if versao >= 4:
            modelo = str(_exigir(s, "model", onde))
            if not modelo.strip():
                raise PreregistrationError(
                    "model vazio: a identidade do modelo é o que distingue duas escalas, e "
                    "declaração sem ela deixa o guarda de procedência cruzar tabelas")
        else:
            if "model" in s:
                raise PreregistrationError(
                    f"model exige hash_version 4; na versão {versao} ele não entraria no hash "
                    f"de medição, e item fora do hash não é declaração")
            modelo = None

        if versao >= 3:
            # Nome PRÓPRIO e não `grade`: o parsing do risco alvo acima já usa
            # `grade`/`grade_t`, e reusá-los zerou a grade de riscos das duas
            # declarações assinadas — os testes de hash apanharam. Colisão de nome
            # de variável não aparece como erro, aparece como hash que muda.
            metas = _exigir(s, "task_quality_grid", onde)
            if isinstance(metas, str) or not metas:
                raise PreregistrationError(
                    "task_quality_grid tem de ser uma lista não vazia de metas de precisão")
            metas_t = tuple(_fracao(q, "meta de qualidade") for q in metas)
            if list(metas_t) != sorted(metas_t):
                raise PreregistrationError(
                    f"task_quality_grid fora de ordem: {metas_t}. Ordem crescente para a tabela "
                    f"não depender da ordem de digitação")
            metricas = _exigir(s, "task_metrics", onde)
            if isinstance(metricas, str) or not metricas:
                raise PreregistrationError("task_metrics tem de ser uma lista não vazia")
            metricas_t = tuple(str(m) for m in metricas)
            for m in metricas_t:
                if m not in METRICAS_TAREFA:
                    raise PreregistrationError(
                        f"métrica de tarefa desconhecida: {m!r}; use alguma de {METRICAS_TAREFA}")
            faltando = [m for m in METRICAS_TAREFA if m not in metricas_t]
            if faltando:
                raise PreregistrationError(
                    f"task_metrics omite {faltando}. As quatro são declaradas juntas: o "
                    f"supervisor só remove predições, então precisão sobe enquanto recall cai, "
                    f"e relatar só uma delas é relatar a que subiu")
        else:
            for proibido in ("task_quality_grid", "task_metrics"):
                if proibido in s:
                    raise PreregistrationError(
                        f"{proibido} exige hash_version 3; na versão {versao} ele não entraria "
                        f"no hash, e item fora do hash não é declaração")
            metas_t, metricas_t = (), ()

        if versao >= 5:
            vaos = _exigir(s, "task_gap_grid", onde)
            if isinstance(vaos, str) or not vaos:
                raise PreregistrationError(
                    "task_gap_grid tem de ser uma lista não vazia de frações do vão")
            vaos_t = tuple(_fracao(v, "fração do vão") for v in vaos)
            if list(vaos_t) != sorted(vaos_t):
                raise PreregistrationError(
                    f"task_gap_grid fora de ordem: {vaos_t}. Ordem crescente para a tabela não "
                    f"depender da ordem de digitação")
        else:
            if "task_gap_grid" in s:
                raise PreregistrationError(
                    f"task_gap_grid exige hash_version 5; na versão {versao} ele não entraria no "
                    f"hash, e item fora do hash não é declaração — a meta relativa poderia mudar "
                    f"sem nenhum hash mudar")
            vaos_t = ()

        comps = _exigir(s, "comparisons", onde)
        if isinstance(comps, str) or not comps:
            raise PreregistrationError("comparisons tem de ser uma lista não vazia")
        comps_t, vistos = [], set()
        for c in comps:
            if not isinstance(c, dict):
                raise PreregistrationError(f"comparação mal formada: {c!r}")
            ident = str(_exigir(c, "id", "comparisons"))
            if ident in vistos:
                raise PreregistrationError(f"comparação {ident} declarada duas vezes")
            vistos.add(ident)
            tipo = str(_exigir(c, "kind", f"comparisons[{ident}]"))
            if tipo not in TIPOS_COMPARACAO:
                raise PreregistrationError(
                    f"{ident}: tipo {tipo!r} desconhecido; use um de {TIPOS_COMPARACAO}")
            pergunta = str(_exigir(c, "question", f"comparisons[{ident}]"))
            if tipo == "descriptive":
                for proibido in ("score", "against", "criterion"):
                    if c.get(proibido):
                        raise PreregistrationError(
                            f"{ident} é descritiva e declara {proibido}: descritiva não tem "
                            f"veredito, e declarar um seria criar veredito sem dizer que criou")
                comps_t.append(Comparison(ident, tipo, pergunta))
                continue
            escore = str(_exigir(c, "score", f"comparisons[{ident}]"))
            contra = str(_exigir(c, "against", f"comparisons[{ident}]"))
            crit = str(_exigir(c, "criterion", f"comparisons[{ident}]"))
            esperado = CRITERIO_DO_TIPO[tipo]
            if crit != esperado:
                raise PreregistrationError(
                    f"{ident} é do tipo {tipo!r} e declara critério {crit!r}; o critério desse "
                    f"tipo é {esperado!r}. Um veredito de tarefa julgado por critério de AURC "
                    f"produziria número que não responde à pergunta declarada")
            if tipo == "task_verdict" and versao < 3:
                raise PreregistrationError(
                    f"{ident} é veredito de tarefa e exige hash_version 3: sem a grade de metas "
                    f"no hash, a meta poderia ser escolhida depois de ver a carga de revisão")
            for nome in (escore, contra):
                for parte in nome.split("+"):
                    if parte.strip() not in ESCORES_BASE:
                        raise PreregistrationError(
                            f"{ident}: escore {parte.strip()!r} desconhecido; use algum de "
                            f"{ESCORES_BASE} (combinação com '+')")
            if crit not in CRITERIOS:
                raise PreregistrationError(
                    f"{ident}: critério {crit!r} desconhecido; use um de {CRITERIOS}")
            comps_t.append(Comparison(ident, tipo, pergunta, escore, contra, crit))
        comps_t = tuple(comps_t)
        # Os DOIS tipos de veredito contam. O guarda existe para impedir
        # declaração que não se pode perder — não para exigir que o veredito seja
        # em AURC. Uma declaração de tarefa refuta em carga de revisão, que é
        # veredito igual, só em outra unidade.
        if not any(c.kind in CRITERIO_DO_TIPO for c in comps_t):
            raise PreregistrationError(
                f"nenhuma comparação com veredito (tipos que refutam: "
                f"{tuple(CRITERIO_DO_TIPO)}): uma declaração só descritiva não refuta nada")

        estratos = _exigir(s, "stratify_by", onde)
        if isinstance(estratos, str) or not estratos:
            raise PreregistrationError("stratify_by tem de ser uma lista não vazia")
        estratos_t = tuple(str(e) for e in estratos)
        for e in estratos_t:
            if e not in ESTRATIFICACOES:
                raise PreregistrationError(
                    f"estratificação desconhecida: {e!r}; use alguma de {ESTRATIFICACOES}")

    return Preregistration(
        model=modelo,
        sink_policy=sumidouro,
        layers=camadas_t,
        heads=cabecas_t,
        combination_rule=combinacao,
        calibration_fraction=_fracao(_exigir(s, "calibration_fraction", onde), "calibration_fraction"),
        coverage_levels=coberturas_t,
        added_value_ci_level=_fracao(_exigir(s, "added_value_ci_level", onde), "added_value_ci_level"),
        n_bootstrap_resamples=n_boot,
        target_risk_grid=grade_t,
        conformal_alpha=_fracao(_exigir(s, "conformal_alpha", onde), "conformal_alpha"),
        loss_bound=float(s.get("loss_bound", 1.0)),
        seed=int(s.get("seed", 42)),
        comparison_scores=comparacoes_t,
        stratify_by=estratos_t,
        task_quality_grid=metas_t,
        task_gap_grid=vaos_t,
        task_metrics=metricas_t,
        span_size_bins=faixas_t,
        comparisons=comps_t,
        hash_version=versao,
        inherits_measurement_from=herda,
        declaration_id=str(_exigir(s, "declaration_id", onde)),
        source=str(caminho),
        layer_profile=bool(s.get("layer_profile", False)),
    )
