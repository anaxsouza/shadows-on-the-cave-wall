"""A geometria da massa de atenção: o nulo analítico e o que sobra dele.

POR QUE ESTE MÓDULO EXISTE

A massa de atenção sobre um span não é uma medida solta: ela tem um valor
esperado exato sob um nulo, e esse valor depende só de dois números geométricos —
o tamanho do span e o da sentença. Medido na validação em 03/09/2026, a massa
declarada tem R² de 96% contra a fração k/T, e tudo o que ela contribui para a
decisão a fração contribui igual. Sem separar a geometria do resto, "massa de
atenção" mede o lugar que a entidade ocupa e não o que o modelo faz com ela.

O NULO, E POR QUE ELE É EXATO

Seja A a matriz de atenção média sobre camadas e cabeças, Q o conjunto de
consultas e K o de chaves permitidas pela política de sumidouro. A massa é

    massa(S) = Σ_{q∈Q} Σ_{j∈S} A[q,j]  /  Σ_{q∈Q} Σ_{j∈K} A[q,j]

Sob o nulo em que a identidade do span é PERMUTÁVEL com as demais chaves — isto
é, sorteando quais k das |K| chaves formam o span —, a linearidade da esperança
dá E[numerador] = (k/|K|)·denominador, e portanto

    E[massa(S)] = k / |K|

Não é aproximação nem assintótico: vale para qualquer matriz linha-estocástica,
qualquer modelo e qualquer entrada. Verificado por permutação em 03/09/2026 sobre
716 entidades dos dois corpora, 200 permutações cada: o desvio médio entre a
massa permutada e k/(T−1) foi +0,00013 e +0,00017, contra desvios OBSERVADOS de
−0,043 e −0,058, ou seja 327× e 335× maiores.

O QUE SE FAZ COM ISSO

`enrichment` é a razão observado/esperado. Vale 1,0 quando a entidade recebe
exatamente a parte que o nulo prevê, e é a forma desconfundida do instrumento:
não é uma correção cosmética, é a única leitura em que "a atenção se concentrou
nesta entidade" tem sentido, porque desconta o lugar ocupado. Medido: mediana
0,562 no GENIA e 0,528 no CoNLL-2003 — entidades recebem cerca de METADE da parte
prevista, de forma estável em dois corpora muito diferentes.

`geometric_residual` é a outra desconfundição possível, por regressão em vez de
razão. As duas existem porque erram em direções opostas: a razão sobrecorrige (a
correlação com k/T fica NEGATIVA, −0,43 e −0,51), a regressão subcorrige. Relatar
as duas é mais honesto que escolher a que der o número desejado.
"""
from __future__ import annotations

from typing import Literal, Sequence

import numpy as np

SinkPolicy = Literal["keep", "drop_from_denominator", "drop_from_queries_and_denominator"]

#: Chaves que cada política remove do denominador. `keep` não remove nenhuma.
_REMOVE_SUMIDOURO = {"drop_from_denominator", "drop_from_queries_and_denominator"}


class GeometryError(ValueError):
    """Entrada geometricamente impossível — não é caso de degradar em silêncio."""


def n_allowed_keys(n_tokens: int, sink_policy: SinkPolicy) -> int:
    """|K|: quantas chaves entram no denominador, dada a política de sumidouro."""
    if n_tokens < 2:
        raise GeometryError(f"sentença com {n_tokens} token(s): não há denominador")
    if sink_policy in _REMOVE_SUMIDOURO:
        return n_tokens - 1
    if sink_policy != "keep":
        raise GeometryError(f"política de sumidouro desconhecida: {sink_policy!r}")
    return n_tokens


def expected_mass(
    span_size: np.ndarray | Sequence[int] | int,
    n_tokens: np.ndarray | Sequence[int] | int,
    *,
    sink_policy: SinkPolicy = "drop_from_denominator",
) -> np.ndarray:
    """Massa esperada sob o nulo de permutabilidade: k / |K|.

    Exata, não aproximada. Ver a derivação no topo do módulo.
    """
    k = np.asarray(span_size, dtype=float)
    T = np.asarray(n_tokens, dtype=float)
    if np.any(k < 1):
        raise GeometryError("span de tamanho zero: o mapeamento caractere→token falhou")
    if np.any(T < 2):
        raise GeometryError("sentença com menos de 2 tokens")
    fora = 1.0 if sink_policy in _REMOVE_SUMIDOURO else 0.0
    if sink_policy not in _REMOVE_SUMIDOURO and sink_policy != "keep":
        raise GeometryError(f"política de sumidouro desconhecida: {sink_policy!r}")
    K = T - fora
    if np.any(k > K):
        raise GeometryError("span maior que o número de chaves permitidas")
    return k / K


def causal_mass(
    attention: np.ndarray,
    span_token_indices: Sequence[int],
    *,
    layers: Sequence[int] | None = None,
    heads: Sequence[int] | None = None,
    query_positions: Sequence[int] | None = None,
) -> float:
    """A massa observada sobre o trecho, com DENOMINADOR POR CONSULTA.

    POR QUE ESTA FUNÇÃO EXISTE, e por que `span_attention_mass` não serve aqui

    A função do encoder divide pela massa sobre um conjunto FIXO de chaves
    permitidas, o mesmo para toda consulta. Num decoder isso está errado por
    construção: a consulta na posição `i` tem `i+1` chaves permitidas, e as
    demais entradas da linha são zero porque a máscara as zerou — não porque o
    modelo não tenha olhado para elas. Usar um denominador fixo somaria zeros
    estruturais ao denominador e deprimiria a massa das consultas iniciais por
    razão puramente aritmética.

    Aqui cada linha é normalizada pelas chaves que ELA pode ver, e só então se
    tira a média sobre as consultas. É essa média que a derivação de
    `expected_mass_causal` prevê.

    `query_positions` escolhe DE ONDE se olha. Vazio significa todas as
    consultas, que é o análogo direto do caso bidirecional. Num decoder
    generativo o conjunto interessante é outro — as posições em que o modelo
    ESCREVE a menção, porque é ali que ele decide —, e passar as duas leituras
    transforma uma escolha em comparação medida.

    A política de sumidouro NÃO é argumento, e a omissão é deliberada: no caso
    causal a primeira posição é a única chave que TODA consulta vê, então
    descartá-la do denominador muda o nulo de forma que a derivação não cobre.
    Manter o sumidouro é o único regime em que `expected_mass_causal` vale.
    """
    a = np.asarray(attention, dtype=float)
    if a.ndim != 4:
        raise GeometryError(f"esperava [camadas, cabeças, consultas, chaves], recebi {a.shape}")
    L, H, Q, K = a.shape
    if Q != K:
        raise GeometryError(f"atenção causal tem de ser quadrada: {Q} consultas, {K} chaves")
    idx = np.asarray(sorted(set(int(i) for i in span_token_indices)), dtype=int)
    if idx.size == 0:
        raise GeometryError("trecho vazio")
    if idx.min() < 0 or idx.max() >= K:
        raise GeometryError(f"índices do trecho fora de [0, {K}): {idx.min()}..{idx.max()}")

    cam = list(range(L)) if layers is None else [int(c) for c in layers]
    cab = list(range(H)) if heads is None else [int(h) for h in heads]
    if max(cam) >= L or max(cab) >= H:
        raise GeometryError(
            f"faixa pede camada {max(cam)} e cabeça {max(cab)}, tensor tem {L}x{H}")
    media = a[np.ix_(cam, cab)].mean(axis=(0, 1))          # [Q, K]

    consultas = np.arange(Q) if query_positions is None else np.asarray(
        sorted(set(int(q) for q in query_positions)), dtype=int)
    if consultas.size == 0:
        raise GeometryError("nenhuma posição de consulta")
    if consultas.min() < 0 or consultas.max() >= Q:
        raise GeometryError(f"consultas fora de [0, {Q}): {consultas.min()}..{consultas.max()}")

    razoes = []
    for q in consultas:
        permitidas = np.arange(q + 1)                      # a máscara causal
        den = media[q, permitidas].sum()
        if den <= 0:
            continue
        no_span = idx[idx <= q]
        razoes.append(media[q, no_span].sum() / den if no_span.size else 0.0)
    if not razoes:
        raise GeometryError("todas as linhas somam zero: atenção degenerada")
    return float(np.mean(razoes))


def enrichment(
    mass: np.ndarray | Sequence[float] | float,
    span_size: np.ndarray | Sequence[int] | int,
    n_tokens: np.ndarray | Sequence[int] | int,
    *,
    sink_policy: SinkPolicy = "drop_from_denominator",
) -> np.ndarray:
    """Observado sobre esperado: 1,0 = a entidade recebe exatamente a sua parte.

    É a forma desconfundida do instrumento. Acima de 1 a atenção se concentra na
    entidade mais do que o lugar dela justifica; abaixo, menos.
    """
    m = np.asarray(mass, dtype=float)
    esp = expected_mass(span_size, n_tokens, sink_policy=sink_policy)
    if np.any(m < 0) or np.any(m > 1 + 1e-9):
        raise GeometryError("massa fora de [0, 1]: não é proporção")
    return m / esp


def expected_mass_causal(
    span_start: int | np.ndarray,
    span_size: int | np.ndarray,
    n_tokens: int | np.ndarray,
) -> float | np.ndarray:
    """Massa esperada sob permutabilidade das chaves COM MÁSCARA CAUSAL.

    POR QUE ESTA FUNÇÃO EXISTE

    `expected_mass` vale para atenção bidirecional, onde toda consulta vê todas
    as chaves. Num modelo autorregressivo a consulta na posição `i` só vê chaves
    até `i`, então o número de chaves permitidas DEPENDE DA CONSULTA — e o nulo
    deixa de ser `k/|K|`.

    Para um trecho contíguo em `[a, a+k-1]`, somando sobre as `T` consultas:

        E[massa] = (1/T) [ Σ_{i=a}^{a+k-1} (i-a+1)/(i+1) + k·(H_T − H_{a+k}) ]

    com `H_n` o n-ésimo número harmônico. Aproximadamente `(k/T)·ln(T/(a+k))`.

    A CONSEQUÊNCIA QUE IMPORTA

    O nulo passa a depender da POSIÇÃO. Trecho no começo da sentença é visto por
    todas as consultas posteriores; trecho no fim, por poucas. Medido com esta
    fórmula: em T=60, um trecho de 1 token em a=0 tem esperança 0,0780, e um de 4
    tokens em a=45 tem 0,0169 — a posição domina o tamanho, e na direção oposta.
    Em NER a posição correlaciona com estrutura sintática e com tipo de entidade,
    então a máscara causal ACRESCENTA uma dimensão ao confundidor em vez de
    remover.

    ESTATUTO: verificada numericamente por permutação das chaves permitidas de
    cada linha, sobre matriz causal linha-estocástica de densidade gama (não
    uniforme, de propósito): desvios abaixo de 2,7 erros-padrão da média em seis
    configurações de (T, a, k). Ver `tests/test_selective_geometry.py`.

    NÃO É USADA por nenhuma declaração vigente: o modelo medido
    (`urchade/gliner_base`) é bidirecional. Ela existe para a pergunta do modelo
    autorregressivo (§8.1 de docs/tese/FORMALIZACAO.md) e para deixar registrado
    que a régua generaliza — com outra fórmula, não com a mesma.
    """
    a = np.asarray(span_start, dtype=int)
    k = np.asarray(span_size, dtype=int)
    T = np.asarray(n_tokens, dtype=int)
    if np.any(k < 1) or np.any(T < 1):
        raise GeometryError("tamanho de trecho e de sentença têm de ser >= 1")
    if np.any(a < 0) or np.any(a + k > T):
        raise GeometryError(
            "trecho fora da sentença: exige 0 <= a e a + k <= T; um trecho que passa do fim "
            "não tem consulta que o veja por completo, e a fórmula não descreveria nada")

    a_f, k_f, T_f = np.atleast_1d(a), np.atleast_1d(k), np.atleast_1d(T)
    fora = np.empty(np.broadcast(a_f, k_f, T_f).shape, dtype=float)
    it = np.nditer([np.broadcast_to(x, fora.shape) for x in (a_f, k_f, T_f)],
                   flags=["multi_index"])
    H = np.concatenate([[0.0], np.cumsum(1.0 / np.arange(1, int(T_f.max()) + 1))])
    for ai, ki, Ti in it:
        ai, ki, Ti = int(ai), int(ki), int(Ti)
        # Parte de DENTRO do trecho: a consulta i em [a, a+k-1] vê só (i-a+1) dos
        # k tokens dele, porque os posteriores ainda não existem para ela.
        parcial = sum((i - ai + 1) / (i + 1) for i in range(ai, min(ai + ki, Ti)))
        # Parte DEPOIS do trecho: a consulta vê os k tokens, entre (i+1) chaves.
        cauda = ki * (H[Ti] - H[min(ai + ki, Ti)]) if ai + ki < Ti else 0.0
        fora[it.multi_index] = (parcial + cauda) / Ti
    # Escalar entra, escalar sai: a função é usada tanto por entidade quanto em
    # vetor, e devolver array de um elemento para entrada escalar obrigaria todo
    # chamador a desempacotar.
    todos_escalares = all(np.ndim(x) == 0 for x in (span_start, span_size, n_tokens))
    return float(fora.reshape(-1)[0]) if todos_escalares else fora.reshape(
        np.broadcast(a_f, k_f, T_f).shape)


def geometric_residual(
    mass: np.ndarray | Sequence[float],
    span_size: np.ndarray | Sequence[int],
    n_tokens: np.ndarray | Sequence[int],
    *,
    sink_policy: SinkPolicy = "drop_from_denominator",
) -> np.ndarray:
    """Resíduo da massa após remover a fração geométrica por mínimos quadrados.

    A desconfundição por REGRESSÃO, alternativa à razão. Ajusta
    massa ≈ a·(k/|K|) + b sobre a amostra inteira e devolve o que sobra. Exige
    amostra: não faz sentido para uma entidade sozinha, e a função recusa em vez
    de devolver zero.
    """
    m = np.asarray(mass, dtype=float)
    esp = expected_mass(span_size, n_tokens, sink_policy=sink_policy)
    if m.ndim != 1 or m.size < 3:
        raise GeometryError("resíduo geométrico exige uma amostra (>= 3 entidades)")
    if np.allclose(esp, esp[0]):
        raise GeometryError("fração geométrica constante na amostra: nada a remover")
    a, b = np.polyfit(esp, m, 1)
    return m - (a * esp + b)
