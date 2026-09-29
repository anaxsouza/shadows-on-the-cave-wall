"""A massa de atenção sobre o span: o instrumento da tese.

Esta é a primeira das duas peças de código que a alternativa C4 exigia. A parte
difícil — mapear span de caractere para índice de token — já existia em
`src/shared/analysis/xai_utils.py` e continua lá. O que não existia é isto: a
PROPORÇÃO da massa de atenção que cai sobre os tokens do span.

POR QUE PROPORÇÃO, E NÃO MÉDIA

O código anterior devolvia médias de submatrizes de atenção. Média sobre uma
linha de matriz de atenção é uma quantidade sem conteúdo: a linha soma 1 por
construção (é uma distribuição softmax sobre as chaves), então a média de uma
linha inteira é exatamente 1/T para qualquer modelo, qualquer texto e qualquer
entidade — mede o comprimento da sentença, não a atenção. A proporção da massa
sobre o span, ao contrário, é razão adimensional em [0, 1] e é comparável entre
entidades de tamanhos diferentes na mesma sentença.

POR QUE ISSO RESPONDE À OBJEÇÃO SOBRE O INSTRUMENTO

A entropia da distribuição de atenção não é comparável entre modelos porque
depende do vocabulário e do comprimento da sequência (é o achado registrado em
`docs/tese/01_estado_atual.md`). A massa sobre o span é uma razão calculada
DENTRO de um único modelo e de uma única sentença, e o desenho da tese nunca
compara o valor bruto entre modelos — compara o poder de ordenação dentro de
cada um. Trocar o instrumento foi a resposta à objeção, em vez de defender o
anterior.

O SUMIDOURO, QUE É A DECISÃO QUE MAIS MEXE NO NÚMERO

Em transformadores treinados, o primeiro token ([CLS], ou o BOS) recebe uma
fração desproporcional da atenção de todas as consultas, sem carregar conteúdo
proporcional — é o *attention sink*. Se ele fica no denominador, ele domina, e a
massa sobre o span de conteúdo fica esmagada num intervalo estreito perto de
zero, o que destrói poder de ordenação por compressão de escala. Se sai, a
proporção passa a ser "entre os tokens de conteúdo, quanto vai para o span",
que é a pergunta que interessa. As duas escolhas são defensáveis e dão números
diferentes; o que não é defensável é escolher depois de ver a curva. Por isso
`sink_policy` é argumento obrigatório e item do pré-registro.

EXEMPLO NUMÉRICO, QUE É ILUSTRAÇÃO E NÃO MEDIÇÃO

Sentença de 5 tokens, o primeiro sendo o sumidouro, span nos tokens 2-3. Suponha
que cada consulta distribua 0,60 para o sumidouro, 0,30 para o span e 0,10 para
o resto. Com o sumidouro no denominador a massa do span é 0,30; excluindo-o do
denominador é 0,30 / 0,40 = 0,75. O mesmo modelo, a mesma entidade, e um número
2,5 vezes maior — é a ordem de grandeza do efeito da convenção, e a razão de ela
ser declarada antes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np

__all__ = ["SinkPolicy", "SpanAttentionMass", "span_attention_mass"]

SinkPolicy = Literal["keep", "drop_from_denominator", "drop_from_queries_and_denominator"]


@dataclass(frozen=True)
class SpanAttentionMass:
    """A massa e o que foi preciso decidir para obtê-la.

    `mass` é a proporção em [0, 1]. Os demais campos existem para que o número
    nunca circule sem a convenção que o produziu: duas execuções com políticas de
    sumidouro diferentes produzem massas incomparáveis, e o relatório precisa
    poder dizer qual foi usada.
    """

    mass: float
    n_span_tokens: int
    n_query_tokens: int
    n_key_tokens_in_denominator: int
    sink_policy: str
    layers: tuple[int, ...]
    heads: tuple[int, ...] | None


def span_attention_mass(
    attention: np.ndarray,
    span_token_indices: Sequence[int],
    *,
    sink_policy: SinkPolicy,
    layers: Sequence[int] | None = None,
    heads: Sequence[int] | None = None,
    sink_index: int = 0,
    special_token_indices: Sequence[int] = (),
) -> SpanAttentionMass:
    """Proporção da massa de atenção que cai sobre os tokens do span.

    Argumentos:
        attention: tensor de atenção com forma [camadas, cabeças, consultas,
            chaves]. Já sem a dimensão de lote — quem chama seleciona a sentença.
        span_token_indices: índices de token do span, na mesma indexação das
            chaves. Vem do mapeamento caractere -> token de
            `AttentionAnalyzer._map_entity_to_tokens`.
        sink_policy: "keep" mantém o sumidouro no denominador;
            "drop_from_denominator" o remove das chaves;
            "drop_from_queries_and_denominator" também o remove das consultas,
            de modo que a média não inclui a linha do próprio sumidouro.
        layers: camadas a agregar. `None` usa todas — o que raramente é o que se
            quer, e é por isso que a faixa é item do pré-registro: as camadas
            iniciais carregam posição e as finais carregam tarefa.
        heads: cabeças a agregar. `None` usa todas.
        sink_index: índice do token sumidouro. Zero para [CLS] e para BOS.
        special_token_indices: outros tokens especiais a remover do denominador
            junto com o sumidouro, quando a política os remove ([SEP], padding).

    A média sobre camadas e cabeças é feita ANTES da razão, não depois: a razão
    de médias é a quantidade que o desenho declara, e média de razões daria peso
    igual a cabeças que participam pouco da distribuição.
    """
    a = np.asarray(attention, dtype=float)
    if a.ndim != 4:
        raise ValueError(f"esperado [camadas, cabeças, consultas, chaves]; recebido forma {a.shape}")
    n_camadas, n_cabecas, n_consultas, n_chaves = a.shape
    if n_consultas != n_chaves:
        raise ValueError(f"matriz não quadrada: {n_consultas} consultas e {n_chaves} chaves")
    if not np.isfinite(a).all():
        raise ValueError("há peso de atenção não finito")

    idx_span = np.unique(np.asarray(list(span_token_indices), dtype=int))
    if idx_span.size == 0:
        raise ValueError("span vazio: o mapeamento caractere -> token não encontrou nenhum token")
    if idx_span.min() < 0 or idx_span.max() >= n_chaves:
        raise ValueError(
            f"índice de span fora da sequência: {idx_span.min()}..{idx_span.max()} "
            f"para {n_chaves} tokens"
        )

    camadas = tuple(range(n_camadas)) if layers is None else tuple(int(i) for i in layers)
    cabecas = None if heads is None else tuple(int(i) for i in heads)
    for i in camadas:
        if not 0 <= i < n_camadas:
            raise ValueError(f"camada {i} fora de [0, {n_camadas})")
    if cabecas is not None:
        for i in cabecas:
            if not 0 <= i < n_cabecas:
                raise ValueError(f"cabeça {i} fora de [0, {n_cabecas})")

    sel = a[list(camadas)]
    if cabecas is not None:
        sel = sel[:, list(cabecas)]
    media = sel.mean(axis=(0, 1))  # [consultas, chaves]

    removidos = set(int(i) for i in special_token_indices)
    if sink_policy in ("drop_from_denominator", "drop_from_queries_and_denominator"):
        removidos.add(int(sink_index))
    elif sink_policy != "keep":
        raise ValueError(f"política de sumidouro desconhecida: {sink_policy!r}")

    if removidos & set(int(i) for i in idx_span):
        raise ValueError(
            "o span inclui um token que a política de sumidouro remove; "
            "isso indica erro no mapeamento caractere -> token, não uma escolha de convenção"
        )

    chaves = np.array([i for i in range(n_chaves) if i not in removidos], dtype=int)
    if chaves.size == 0:
        raise ValueError("a política removeu todas as chaves")

    if sink_policy == "drop_from_queries_and_denominator":
        consultas = np.array([i for i in range(n_consultas) if i not in removidos], dtype=int)
        if consultas.size == 0:
            raise ValueError("a política removeu todas as consultas")
    else:
        consultas = np.arange(n_consultas)

    bloco = media[np.ix_(consultas, chaves)]
    denominador = float(bloco.sum())
    if denominador <= 0:
        raise ValueError("massa total nula no denominador")
    numerador = float(media[np.ix_(consultas, idx_span)].sum())

    return SpanAttentionMass(
        mass=numerador / denominador,
        n_span_tokens=int(idx_span.size),
        n_query_tokens=int(consultas.size),
        n_key_tokens_in_denominator=int(chaves.size),
        sink_policy=sink_policy,
        layers=camadas,
        heads=cabecas,
    )
