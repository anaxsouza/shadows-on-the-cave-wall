"""Janelas de controle (decl-15, C1) e taxonomia de erros (decl-14, C4).

Este módulo só tem numpy e biblioteca padrão, de propósito: ele viaja no pacote
de fonte dos jobs da SageMaker junto com `geometry.py` e `attention_mass.py`,
e `src.selective` resolve lá como pacote de espaço de nomes (sem `__init__`).

O QUE ESTE MÓDULO FAZ, E O QUE NÃO FAZ

Constrói as janelas e calcula o enriquecimento com as MESMAS funções da medição
original (`attention_mass.span_attention_mass`, `geometry.expected_mass`,
`geometry.causal_mass`, `geometry.expected_mass_causal`). Não calcula vereditos:
J1–J3 e E1–E3 são de outra etapa.

JANELAS (decl-15)

Para uma entidade prevista com k tokens, as janelas são TODAS as sequências de k
índices de token consecutivos, dentro dos tokens de conteúdo da frase, que não
tocam nenhum token de entidade anotada nem prevista da mesma frase. Sem sorteio.

TAXONOMIA (decl-14), por precedência, em caracteres:
1. acerto    — mesmas fronteiras e mesmo rótulo de uma anotada;
2. rotulo    — mesmas fronteiras de uma anotada, rótulo diferente;
3. fronteira — não vale (1) nem (2) e há sobreposição de >= 1 caractere;
4. sem_par   — nenhuma sobreposição.
(A decl-14 numera `rótulo` como 1 e chama de acerto o casamento estrito; a
checagem de acerto vem antes porque acerto não é erro.)
"""

from __future__ import annotations

from typing import Callable, Iterable, Sequence

import numpy as np

TIPOS = ("acerto", "rotulo", "fronteira", "sem_par")


def tipo_erro(inicio: int, fim: int, rotulo: str,
              ouro: Iterable[tuple[int, int, str]]) -> str:
    """Tipo da previsão `[inicio, fim)` com `rotulo`, contra o ouro `(ini, fim, rotulo)`."""
    ouro = list(ouro)
    if any(g0 == inicio and g1 == fim and gr == rotulo for g0, g1, gr in ouro):
        return "acerto"
    if any(g0 == inicio and g1 == fim for g0, g1, _ in ouro):
        return "rotulo"
    if any(g0 < fim and g1 > inicio for g0, g1, _ in ouro):
        return "fronteira"
    return "sem_par"


def janelas_livres(tokens_frase: Sequence[int], k: int,
                   ocupados: Iterable[int]) -> list[tuple[int, ...]]:
    """Todas as janelas de `k` tokens consecutivos livres de `ocupados`.

    `tokens_frase`: índices de token de conteúdo da frase (no GLiNER, fora de
    [CLS]/[SEP]; no Qwen, só a frase dentro do prompt). Consecutivos significa
    índices a, a+1, ..., a+k-1 TODOS em `tokens_frase`.
    """
    if k < 1:
        raise ValueError("k < 1")
    validos = set(int(i) for i in tokens_frase)
    ocup = set(int(i) for i in ocupados)
    saida = []
    for a in sorted(validos):
        janela = tuple(range(a, a + k))
        if all(j in validos and j not in ocup for j in janela):
            saida.append(janela)
    return saida


def media_encoder(atencao: np.ndarray, camadas: Sequence[int],
                  cabecas: Sequence[int] | None) -> np.ndarray:
    """A média [camadas, cabeças] que `span_attention_mass` faz internamente.

    Devolvida como [1, 1, T, T] para ser passada de volta a `span_attention_mass`
    com `layers=[0]`: a média de um único elemento é ele mesmo, então o número é
    o da medição original, a um custo T^2 por sentença em vez de L*H*T^2 por janela.
    """
    a = np.asarray(atencao, dtype=float)
    sel = a[list(camadas)]
    if cabecas is not None:
        sel = sel[:, list(cabecas)]
    return sel.mean(axis=(0, 1))[None, None]


def media_decoder(atencao: np.ndarray) -> np.ndarray:
    """Idem para `causal_mass` com `layers=None, heads=None` (todas)."""
    return np.asarray(atencao, dtype=float).mean(axis=(0, 1))[None, None]


def enriquecimento_encoder(media: np.ndarray, idx: Sequence[int], n_tokens: int,
                           sink_policy: str) -> float:
    """Massa recebida / k/|K| (Eq. 3), com as funções do repositório."""
    from .attention_mass import span_attention_mass
    from .geometry import expected_mass

    m = span_attention_mass(media, idx, sink_policy=sink_policy, layers=[0], heads=None).mass
    esp = float(expected_mass(len(set(idx)), n_tokens, sink_policy=sink_policy))
    return m / esp


def enriquecimento_decoder(media: np.ndarray, idx: Sequence[int], n_tokens: int) -> float:
    """Massa causal recebida / esperado causal (Eq. 4), com a posição da janela."""
    from .geometry import causal_mass, expected_mass_causal

    idx = sorted(set(int(i) for i in idx))
    m = causal_mass(media, idx)
    esp = float(expected_mass_causal(idx[0], len(idx), n_tokens))
    return m / esp


def janelas_da_entidade(
    idx: Sequence[int], tokens_frase: Sequence[int], ocupados: Iterable[int],
    enr_fn: Callable[[Sequence[int]], float],
) -> tuple[int, float]:
    """`(n janelas, média do enriquecimento das janelas)`; média NaN se n == 0."""
    k = len(set(idx))
    js = janelas_livres(tokens_frase, k, ocupados)
    if not js:
        return 0, float("nan")
    return len(js), float(np.mean([enr_fn(j) for j in js]))
