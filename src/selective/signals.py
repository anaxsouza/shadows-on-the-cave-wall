"""As três famílias de sinal interno, e o ESTATUTO do nulo de cada uma.

POR QUE ESTE MÓDULO EXISTE

O resultado confirmatório é que a massa de atenção sobre o span não acrescenta
poder de decisão sobre a confiança que o próprio modelo reporta. A pergunta que
vem depois dele, e que a banca faria, é: *então onde mora a informação de
decisão, se não na atenção?*

Responder exige medir mais de uma família de sinal. E exige uma distinção que
este módulo existe para tornar impossível de esquecer: **as famílias não têm o
mesmo tipo de nulo**, e misturá-las sem dizer qual é qual é o erro que
transformaria um resultado limpo em confusão.

OS DOIS ESTATUTOS DE NULO

`NULO_EXATO` — a família de ATENÇÃO. Cada linha da matriz soma 1, por
construção. Desse orçamento fixo sai um valor esperado EXATO para qualquer
agregado dos pesos, derivável e verificável por permutação, sem ajustar nada ao
dado. A massa tem esperança `k/|K|`; a entropia da linha tem esperança
`log|K|`; o máximo tem esperança `1/|K|`. É a mesma álgebra três vezes, e é por
isso que a literatura observa dependência de comprimento em toda variante que
tenta: não há variante que escape de um orçamento fixo.

`NULO_EMPIRICO` — as famílias de ESTADOS OCULTOS e de LOGITS. Não há orçamento
fixo: a norma de um vetor de estado não é restrita a somar coisa alguma, e o
escore de um trecho não compete com os escores dos outros trechos por uma massa
constante. Sem orçamento não há esperança derivável, então a desconfundição é
por REGRESSÃO nas covariáveis geométricas — e isso é estatisticamente mais fraco
em dois sentidos que ficam declarados em cada linha de resultado: a forma da
relação é suposta (linear nas covariáveis) e os coeficientes são ajustados ao
mesmo dado.

A consequência prática, e a razão de o estatuto viajar com o número: um nulo
exato refutado é uma afirmação sobre o modelo; um nulo empírico refutado pode ser
uma afirmação sobre a forma da regressão. Um revisor tem direito a saber qual
dos dois está lendo, em cada linha.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np

from .geometry import GeometryError, n_allowed_keys

Estatuto = Literal["exato", "empirico"]
Familia = Literal["atencao", "estados_ocultos", "logits"]


class SignalError(ValueError):
    """Entrada que não permite calcular o sinal ou o nulo dele."""


@dataclass(frozen=True)
class SignalSpec:
    """Um sinal, com o estatuto do nulo dele à vista.

    `covariaveis` só tem sentido quando o estatuto é empírico: são as colunas em
    que a desconfundição regride. Para um sinal de nulo exato ela fica vazia, e
    isso é a asserção de que nada foi ajustado ao dado.
    """

    nome: str
    familia: Familia
    estatuto: Estatuto
    descricao: str
    covariaveis: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.estatuto == "exato" and self.covariaveis:
            raise SignalError(
                f"{self.nome}: nulo exato não regride em covariável nenhuma — "
                f"a esperança sai da álgebra, não do ajuste")
        if self.estatuto == "empirico" and not self.covariaveis:
            raise SignalError(
                f"{self.nome}: nulo empírico exige as covariáveis declaradas, senão "
                f"não há como dizer do que o sinal foi desconfundido")

    def render(self) -> str:
        extra = f" (regride em {', '.join(self.covariaveis)})" if self.covariaveis else ""
        return f"{self.nome} [{self.familia}, nulo {self.estatuto}]{extra}"


# =============================================================================
# FAMÍLIA 1 — atenção. Nulo EXATO, do orçamento fixo.
# =============================================================================

def expected_row_entropy(n_tokens: int, sink_policy: str) -> float:
    """Esperança da entropia de uma linha de atenção sob permutabilidade: log|K|.

    A linha soma 1 sobre |K| chaves permitidas. A distribuição de máxima entropia
    sob essa única restrição é a uniforme, e a entropia dela é `log|K|` exato. É
    a forma fechada do `Theta(log T)` que a literatura de confundimento por
    comprimento relata empiricamente.
    """
    K = n_allowed_keys(n_tokens, sink_policy=sink_policy)
    if K < 1:
        raise SignalError(f"nenhuma chave permitida com T={n_tokens} e política {sink_policy!r}")
    return float(np.log(K))


def expected_row_max(n_tokens: int, sink_policy: str) -> float:
    """Esperança do máximo de uma linha sob a uniforme: 1/|K|, exato.

    Vale a ressalva de que é a esperança do máximo DA UNIFORME e não a esperança
    do máximo de uma linha aleatória qualquer com soma 1 — esta segunda depende da
    distribuição sobre o simplex e não tem forma fechada única. Declarado assim
    porque é o nulo que a permutabilidade das chaves sustenta: sob permutação, o
    valor esperado de cada peso é 1/|K|.
    """
    K = n_allowed_keys(n_tokens, sink_policy=sink_policy)
    if K < 1:
        raise SignalError(f"nenhuma chave permitida com T={n_tokens} e política {sink_policy!r}")
    return 1.0 / float(K)


def row_entropy(pesos: np.ndarray) -> np.ndarray:
    """Entropia de cada linha, em nats. Linhas que não somam 1 são recusadas."""
    p = np.asarray(pesos, dtype=float)
    if p.ndim != 2:
        raise SignalError(f"esperava matriz [consulta, chave], recebi forma {p.shape}")
    soma = p.sum(axis=1)
    if not np.allclose(soma, 1.0, atol=1e-3):
        raise SignalError(
            f"linhas não somam 1 (min {soma.min():.4f}, máx {soma.max():.4f}): sem o "
            f"orçamento fixo o nulo exato não vale, e usá-lo seria afirmar o que não se tem")
    with np.errstate(divide="ignore", invalid="ignore"):
        termo = np.where(p > 0, p * np.log(p), 0.0)
    return -termo.sum(axis=1)


ATENCAO = (
    SignalSpec("span_mass", "atencao", "exato",
               "proporção da massa de atenção sobre os tokens do span; esperança k/|K|"),
    SignalSpec("row_entropy", "atencao", "exato",
               "entropia das linhas das consultas do span; esperança log|K|"),
    SignalSpec("row_max", "atencao", "exato",
               "máximo das linhas das consultas do span; esperança 1/|K| sob permutabilidade"),
    SignalSpec("emitted_mass", "atencao", "exato",
               "massa que o span EMITE para fora dele; complemento da massa recebida"),

    # OS NOMES CAUSAIS, acrescentados em 18/09/2026 para o braço decoder. Eles
    # são sinais PRÓPRIOS e não variantes dos de cima, e a razão é que o nulo
    # muda de forma: sob máscara causal a linha `i` só vê `i+1` chaves, então o
    # teto de cada linha depende da LARGURA DELA — `log(w)` e `1/w` — e a
    # esperança do conjunto é a média desses tetos sobre as linhas do trecho.
    # Reaproveitar `row_entropy` aqui faria o veredito ser lido contra `log|K|`,
    # que é o teto bidirecional, e a comparação sairia contra o nulo errado.
    #
    # Estatuto EXATO, e sem covariável: a esperança sai da contagem de chaves
    # permitidas, que é aritmética da máscara e não ajuste ao dado. É por isso
    # que a afirmação de atenção sobrevive à objeção de escala — a derivação não
    # contém largura, profundidade nem número de cabeças.
    SignalSpec("row_entropy_causal", "atencao", "exato",
               "entropia das linhas causais do span, cada uma sobre as chaves "
               "permitidas; esperança = média de log(i+1) sobre as linhas"),
    SignalSpec("row_max_causal", "atencao", "exato",
               "máximo das linhas causais do span; esperança = média de 1/(i+1) "
               "sobre as linhas, sob permutabilidade das chaves permitidas"),
)

# =============================================================================
# FAMÍLIA 2 — estados ocultos. Nulo EMPÍRICO: não há orçamento.
# =============================================================================

COVARIAVEIS = ("span_size", "sentence_length", "span_position")

ESTADOS_OCULTOS = (
    SignalSpec("hidden_norm", "estados_ocultos", "empirico",
               "norma L2 média dos estados ocultos dos tokens do span", COVARIAVEIS),
    SignalSpec("hidden_dist_centroide", "estados_ocultos", "empirico",
               "distância do estado médio do span ao centroide da camada", COVARIAVEIS),
    SignalSpec("hidden_delta_camadas", "estados_ocultos", "empirico",
               "norma da variação do estado do span entre a primeira e a última camada",
               COVARIAVEIS),
)

# =============================================================================
# SONDAS SUPERVISIONADAS — o compartimento EXPLORATÓRIO, acrescentado em
# 18/09/2026 por decisão de escopo do autor.
# =============================================================================
# POR QUE ELAS EXISTEM. Os sinais acima são resumos NÃO SUPERVISIONADOS: norma,
# distância, média sobre cabeças. A literatura que ENCONTRA sinal nestas duas
# famílias quase nunca usa isso — usa sonda treinada sobre o vetor de ativação,
# e features de atenção POR CABEÇA em vez da média. A distinção decide o que um
# nulo pode afirmar: uma norma falhar diz que aquele resumo escalar não decide;
# uma sonda falhar diz que a informação não é linearmente recuperável do vetor,
# que é afirmação sobre a REPRESENTAÇÃO e muito mais difícil de atacar.
#
# O PREÇO, e ele é declarado e não escondido: estatuto EMPÍRICO nas duas, também
# na de atenção. A força da família de atenção é o nulo EXATO — `k/|K|`,
# `log(i+1)`, `1/(i+1)` saem da álgebra do orçamento e não contêm escala. Uma
# combinação supervisionada de features de cabeça tem pesos AJUSTADOS, logo
# abandona esse estatuto. A sonda é estatisticamente mais poderosa e
# epistemicamente mais fraca, e por isso ela ACOMPANHA o escalar em vez de
# substituí-lo: os dois resultados vão para o relato.
#
# UMA comparação por família, e não 336. As features por cabeça entram pela
# agregação supervisionada, ajustada SÓ na partição de calibração. Tratar cada
# cabeça como comparação própria multiplicaria por 336 as chances de exclusão de
# zero por sorteio — o erro de contagem que a apuração de 16/09/2026 apanhou.
SONDAS = (
    SignalSpec("sonda_ocultos", "estados_ocultos", "empirico",
               "escore de sonda logística sobre o vetor de estados ocultos do span, "
               "por camada; ajustada na calibração, aplicada na avaliação",
               COVARIAVEIS),
    SignalSpec("sonda_atencao_cabecas", "atencao", "empirico",
               "escore de sonda logística sobre as features de atenção POR CABEÇA "
               "(massa, entropia e máximo de linha em cada camada e cabeça); "
               "ajustada na calibração. ABANDONA o nulo exato da família",
               COVARIAVEIS),
)

# =============================================================================
# FAMÍLIA 3 — logits. Nulo EMPÍRICO pela mesma razão.
# =============================================================================

LOGITS = (
    SignalSpec("logit_margin", "logits", "empirico",
               "diferença entre o maior e o segundo maior escore de rótulo do trecho",
               COVARIAVEIS),
    SignalSpec("logit_entropy", "logits", "empirico",
               "entropia da distribuição de escores sobre os rótulos, normalizada",
               COVARIAVEIS),
    SignalSpec("logit_max", "logits", "empirico",
               "maior escore de rótulo do trecho; é a confiança reportada, aqui como sinal",
               COVARIAVEIS),
)

# As sondas entram na família a que PERTENCEM — a de atenção na atenção, a de
# estados ocultos nos estados ocultos —, e não numa quarta família própria. A
# razão: elas medem a MESMA grandeza dos escalares, com outro instrumento, e
# separá-las em família nova faria a apuração por família comparar coisas que não
# são alternativas entre si. O que as distingue do escalar é o `estatuto` do nulo
# (empírico contra exato), que já viaja com cada sinal.
#
# Este agrupamento é conferido contra `TODOS` em
# `tests/test_selective_signals.py::test_o_registro_nao_tem_nome_repetido`, e foi
# esse teste que apanhou eu ter acrescentado as sondas a `TODOS` e esquecido
# aqui — 12 contra 14.
FAMILIAS: dict[Familia, tuple[SignalSpec, ...]] = {
    "atencao": ATENCAO + tuple(s for s in SONDAS if s.familia == "atencao"),
    "estados_ocultos": ESTADOS_OCULTOS + tuple(
        s for s in SONDAS if s.familia == "estados_ocultos"),
    "logits": LOGITS,
}
TODOS: tuple[SignalSpec, ...] = ATENCAO + ESTADOS_OCULTOS + LOGITS + SONDAS


def por_nome(nome: str) -> SignalSpec:
    for s in TODOS:
        if s.nome == nome:
            return s
    raise SignalError(f"sinal desconhecido: {nome!r}; conhecidos {[s.nome for s in TODOS]}")


def row_ceiling_entropy(width: int) -> float:
    """Teto de entropia de UMA linha causal, dada a largura dela, em nats.

    A distinção com `expected_row_entropy_causal` é a que eu errei ao escrever a
    medição, e por isso está nomeada aqui em vez de comentada no chamador:

    - ESTA função é o teto de UMA linha: `log(w)`, com `w` chaves permitidas.
    - A outra é a MÉDIA do teto sobre as `T` linhas de uma matriz causal inteira:
      `log(T!)/T`. Ela vale quando se agrega sobre todas as consultas.

    Quem agrega só sobre as linhas de um trecho precisa desta, aplicada à largura
    de cada linha. Usar a outra ali subestima o teto — medido: para as linhas 10,
    11 e 12 o teto correto é 2,4826 e a fórmula agregada dá 1,6638.
    """
    w = int(width)
    if w < 1:
        raise SignalError(f"largura tem de ser >= 1: {w}")
    return float(np.log(w))


def row_ceiling_max(width: int) -> float:
    """Máximo esperado de UMA linha causal sob uniforme: `1/w`.

    Mesma distinção de `row_ceiling_entropy`: a média sobre todas as linhas de
    uma matriz causal é `H_T/T`, que é outra quantidade.
    """
    w = int(width)
    if w < 1:
        raise SignalError(f"largura tem de ser >= 1: {w}")
    return 1.0 / w


def expected_row_entropy_causal(n_tokens: int) -> float:
    """Entropia esperada de linha sob MÁSCARA CAUSAL, em nats.

    POR QUE ESTA FUNÇÃO EXISTE, e por que reusar a bidirecional seria ERRADO

    `expected_row_entropy` vale quando toda consulta vê o mesmo conjunto de
    chaves: a entropia máxima é `log|K|`, igual para todas as linhas. Sob máscara
    causal a linha `i` vê `i+1` chaves, então o teto dela é `log(i+1)` — e a
    média sobre as `T` consultas é:

        E[entropia] = (1/T) Σ_{i=0}^{T-1} log(i+1) = log(T!)/T

    A diferença não é pequena nem conservadora. Medida: em T=60 o nulo causal é
    3,1438 contra 4,0775 do bidirecional (razão 0,77), e a razão CRESCE com T.
    Usar o bidirecional aqui declararia um teto que nenhuma linha pode atingir, e
    toda entropia observada pareceria baixa por razão aritmética.

    ESTATUTO: exato. Sai da contagem de chaves permitidas, sem ajustar nada ao
    dado — a mesma álgebra do caso bidirecional, com |K| variando por linha.
    """
    T = int(n_tokens)
    if T < 1:
        raise SignalError(f"n_tokens tem de ser >= 1: {T}")
    import math
    return float(math.lgamma(T + 1) / T)          # log(T!)/T, estável para T grande


def expected_row_max_causal(n_tokens: int) -> float:
    """Máximo esperado de linha sob MÁSCARA CAUSAL.

    Sob atenção uniforme, a linha `i` tem máximo `1/(i+1)`, então:

        E[máximo] = (1/T) Σ_{i=0}^{T-1} 1/(i+1) = H_T / T

    com `H_T` o T-ésimo número harmônico. Isto é MUITO maior que o `1/|K|`
    bidirecional: medido, 3,4x em T=20, 4,6x em T=60 e 5,3x em T=120. A razão é a
    primeira linha, que vê uma chave só e portanto tem máximo 1,0 — e nenhuma
    quantidade de comprimento dilui isso.

    ESTATUTO: exato.
    """
    T = int(n_tokens)
    if T < 1:
        raise SignalError(f"n_tokens tem de ser >= 1: {T}")
    return float(np.sum(1.0 / np.arange(1, T + 1)) / T)


def residuo_empirico(sinal: np.ndarray, covariaveis: np.ndarray) -> np.ndarray:
    """Desconfundição por regressão, para os sinais SEM orçamento fixo.

    Devolve o resíduo de mínimos quadrados do sinal nas covariáveis, com termo
    constante. Duas fraquezas ficam declaradas aqui e viajam com cada número que
    sai daqui: a forma da relação é SUPOSTA linear, e os coeficientes são
    ajustados no MESMO dado em que o resíduo é avaliado. Um nulo exato não tem
    nenhuma das duas.
    """
    y = np.asarray(sinal, dtype=float)
    X = np.asarray(covariaveis, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    if y.ndim != 1 or X.shape[0] != y.size:
        raise SignalError(f"formas incompatíveis: sinal {y.shape}, covariáveis {X.shape}")
    if y.size < X.shape[1] + 2:
        raise SignalError(
            f"amostra de {y.size} para {X.shape[1]} covariáveis: resíduo sem grau de liberdade")
    A = np.column_stack([np.ones(y.size), X])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    return y - A @ beta


__all__ = [
    "ATENCAO",
    "COVARIAVEIS",
    "ESTADOS_OCULTOS",
    "FAMILIAS",
    "LOGITS",
    "SignalError",
    "SignalSpec",
    "TODOS",
    "expected_row_entropy",
    "expected_row_max",
    "expected_row_entropy_causal",
    "row_ceiling_entropy",
    "row_ceiling_max",
    "expected_row_max_causal",
    "por_nome",
    "residuo_empirico",
    "row_entropy",
]
