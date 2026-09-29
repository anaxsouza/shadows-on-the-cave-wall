"""O nulo geométrico e a razão de enriquecimento.

O teste central deste arquivo não é de encanamento: é a verificação NUMÉRICA da
derivação. Se a massa esperada sob permutabilidade é k/|K|, então permutar quais
chaves formam o span tem de reproduzir esse número dentro do erro de Monte Carlo.
Um teste que só conferisse a fórmula contra ela mesma não verificaria nada.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.selective.attention_mass import span_attention_mass
from src.selective.geometry import (
    expected_mass_causal,
    GeometryError,
    enrichment,
    expected_mass,
    geometric_residual,
    n_allowed_keys,
)


def _uniforme(L=2, H=2, T=12) -> np.ndarray:
    """Atenção uniforme: cada consulta reparte igualmente entre todas as chaves."""
    return np.full((L, H, T, T), 1.0 / T, dtype=float)


class TestONulo:
    def test_a_permutacao_reproduz_k_sobre_K(self):
        """A verificação da derivação, e é ela que dá sentido ao módulo.

        Matriz ALEATÓRIA linha-estocástica, não uniforme: se o nulo dependesse da
        forma da distribuição, este teste falharia.
        """
        rng = np.random.default_rng(11)
        T, k, n_perm = 30, 4, 400
        bruta = rng.gamma(shape=0.7, size=(1, 1, T, T))
        a = bruta / bruta.sum(axis=-1, keepdims=True)

        esp = float(expected_mass(k, T, sink_policy="drop_from_denominator"))
        amostras = [
            span_attention_mass(a, rng.choice(np.arange(1, T), size=k, replace=False),
                                sink_policy="drop_from_denominator").mass
            for _ in range(n_perm)
        ]
        media, erro = float(np.mean(amostras)), float(np.std(amostras) / np.sqrt(n_perm))
        assert abs(media - esp) < 4 * erro, (
            f"média permutada {media:.5f} contra k/|K| = {esp:.5f}; "
            f"erro de Monte Carlo {erro:.5f}. A derivação está errada ou a implementação divergiu."
        )

    def test_o_nulo_nao_depende_do_conteudo_da_matriz(self):
        """k/|K| não olha para a matriz: é o mesmo para qualquer modelo e entrada."""
        assert expected_mass(3, 41) == expected_mass(3, 41)
        assert float(expected_mass(3, 41)) == pytest.approx(3 / 40)
        assert float(expected_mass(3, 41, sink_policy="keep")) == pytest.approx(3 / 41)

    def test_chaves_permitidas_por_politica(self):
        assert n_allowed_keys(20, "keep") == 20
        assert n_allowed_keys(20, "drop_from_denominator") == 19
        assert n_allowed_keys(20, "drop_from_queries_and_denominator") == 19


class TestEnriquecimento:
    def test_vale_um_sob_atencao_uniforme(self):
        """A propriedade que define a escala: sob atenção uniforme a entidade
        recebe exatamente a sua parte, e o instrumento tem de marcar 1,0."""
        T, span = 12, [3, 4, 5]
        m = span_attention_mass(_uniforme(T=T), span, sink_policy="drop_from_denominator").mass
        assert float(enrichment(m, len(span), T)) == pytest.approx(1.0, abs=1e-9)

    def test_invariante_a_T_quando_a_fracao_e_a_mesma(self):
        """Sob uniforme, dobrar span e sentença juntos não muda o enriquecimento —
        é isso que 'desconfundido da geometria' significa operacionalmente."""
        vals = []
        for T, k in ((21, 2), (41, 4), (81, 8)):
            span = list(range(1, 1 + k))
            m = span_attention_mass(_uniforme(T=T), span, sink_policy="drop_from_denominator").mass
            vals.append(float(enrichment(m, k, T)))
        assert np.allclose(vals, 1.0, atol=1e-9), vals

    def test_abaixo_de_um_quando_a_atencao_foge_do_span(self):
        T, span = 12, [3, 4]
        a = _uniforme(T=T).copy()
        a[..., span] *= 0.2
        a = a / a.sum(axis=-1, keepdims=True)
        m = span_attention_mass(a, span, sink_policy="drop_from_denominator").mass
        assert float(enrichment(m, len(span), T)) < 1.0

    def test_recusa_massa_que_nao_e_proporcao(self):
        with pytest.raises(GeometryError, match="proporção"):
            enrichment(1.4, 2, 10)


class TestRecusas:
    def test_span_vazio(self):
        with pytest.raises(GeometryError, match="tamanho zero"):
            expected_mass(0, 10)

    def test_span_maior_que_as_chaves(self):
        with pytest.raises(GeometryError, match="maior que o número de chaves"):
            expected_mass(10, 10, sink_policy="drop_from_denominator")

    def test_politica_desconhecida(self):
        with pytest.raises(GeometryError, match="política de sumidouro"):
            expected_mass(2, 10, sink_policy="ignorar")

    def test_residuo_exige_amostra(self):
        with pytest.raises(GeometryError, match="amostra"):
            geometric_residual([0.1, 0.2], [1, 2], [10, 10])

    def test_residuo_recusa_fracao_constante(self):
        with pytest.raises(GeometryError, match="constante"):
            geometric_residual([0.1, 0.2, 0.3], [2, 2, 2], [10, 10, 10])


def test_o_residuo_remove_a_tendencia_e_nao_a_media():
    """O resíduo tira a parte linear na fração e deixa o resto — inclusive o sinal
    que não é geométrico. Testado com sinal PLANTADO ortogonal à fração."""
    rng = np.random.default_rng(3)
    k = rng.integers(1, 6, size=200)
    T = rng.integers(20, 60, size=200)
    frac = k / (T - 1)
    plantado = rng.normal(0, 0.004, size=200)
    m = np.clip(0.5 * frac + 0.01 + plantado, 1e-6, 1.0)
    r = geometric_residual(m, k, T)
    assert abs(np.corrcoef(r, frac)[0, 1]) < 0.05, "o resíduo ainda carrega a fração"
    assert np.corrcoef(r, plantado)[0, 1] > 0.9, "o resíduo perdeu o sinal plantado"


# =============================================================================
# O nulo COM MÁSCARA CAUSAL — a régua generaliza, com outra fórmula
# =============================================================================
# Este arquivo já verifica numericamente o nulo bidirecional. Aqui se faz o mesmo
# para o caso autorregressivo, e a verificação é o que separa "derivei" de
# "conferi": a fórmula tem um fator harmônico que é fácil de escrever errado.


def _matriz_causal(T: int, rng) -> np.ndarray:
    """Linha-estocástica sobre as chaves PERMITIDAS, densidade gama.

    Gama e não uniforme de propósito: se o nulo dependesse da forma da
    distribuição, este teste falharia — e é justamente o que se quer descartar.
    """
    A = np.zeros((T, T))
    for i in range(T):
        w = rng.gamma(2.0, 1.0, size=i + 1)
        A[i, : i + 1] = w / w.sum()
    return A


def _massa_causal(A: np.ndarray, inicio: int, k: int, T: int) -> float:
    num = den = 0.0
    for i in range(T):
        num += A[i, [j for j in range(inicio, inicio + k) if j <= i]].sum()
        den += A[i, : i + 1].sum()
    return num / den


@pytest.mark.parametrize("T,a,k", [(40, 3, 2), (60, 0, 1), (60, 45, 4), (25, 10, 3)])
def test_permutar_as_chaves_permitidas_reproduz_o_nulo_causal(T, a, k):
    """A verificação central: permutação bate a derivação dentro do erro de MC.

    Permuta-se os pesos entre as chaves PERMITIDAS de cada linha, com o trecho
    fixo — é o análogo exato da verificação bidirecional. Tolerância em erros-
    padrão da média, não em valor absoluto: o desvio aceitável depende do número
    de permutações, e fixar um absoluto esconderia isso.
    """
    rng = np.random.default_rng(11)
    A0 = _matriz_causal(T, rng)
    amostras = []
    for _ in range(1200):
        A = np.zeros_like(A0)
        for i in range(T):
            A[i, : i + 1] = A0[i, rng.permutation(i + 1)]
        amostras.append(_massa_causal(A, a, k, T))
    obs = float(np.mean(amostras))
    erro_padrao = float(np.std(amostras)) / np.sqrt(len(amostras))
    der = expected_mass_causal(a, k, T)
    assert abs(obs - der) < 4.0 * erro_padrao, (
        f"T={T} a={a} k={k}: permutado {obs:.5f} contra derivado {der:.5f}, "
        f"{abs(obs - der) / erro_padrao:.1f} erros-padrão"
    )


def test_no_decoder_a_POSICAO_domina_o_tamanho(self=None):
    """O achado que a fórmula traz, fixado em teste.

    Um trecho de 1 token no COMEÇO tem massa esperada MAIOR que um de 4 tokens
    perto do fim. No caso bidirecional é o contrário, porque lá só o tamanho
    conta. É o que torna a máscara causal um confundidor pior, e não melhor.
    """
    comeco = expected_mass_causal(0, 1, 60)
    fim = expected_mass_causal(45, 4, 60)
    assert comeco > fim, f"começo {comeco:.4f} não excede fim {fim:.4f}"
    assert comeco / fim > 4.0
    # no caso bidirecional, o de 4 tokens recebe QUATRO vezes o de 1
    assert expected_mass(4, 60) / expected_mass(1, 60) == pytest.approx(4.0)


def test_trecho_fora_da_sentenca_e_recusado():
    with pytest.raises(GeometryError, match="fora da sentença"):
        expected_mass_causal(58, 4, 60)


# =============================================================================
# A MEDIÇÃO causal: `causal_mass` bate a derivação `expected_mass_causal`?
# =============================================================================
# A derivação já estava verificada por permutação sobre matriz sintética. O que
# faltava era o lado da MEDIÇÃO: a função que calcula a massa observada num
# tensor causal real tem de convergir para a fórmula quando a atenção é
# indiferente a QUAIS tokens formam o trecho. É isso que se fixa aqui, e é o
# par que torna o braço decoder mensurável.


def _causal_estocastica(L, H, T, rng, forma=0.5):
    """Tensor causal linha-estocástico, densidade gama (NÃO uniforme)."""
    a = rng.gamma(shape=forma, scale=1.0, size=(L, H, T, T)) + 1e-12
    mascara = np.tril(np.ones((T, T)))
    a = a * mascara
    return a / a.sum(axis=-1, keepdims=True)


def test_causal_mass_converge_para_a_DERIVACAO_sob_permutabilidade():
    """O par derivação-medição, verificado.

    Sorteando QUAIS posições formam o trecho contíguo e medindo com
    `causal_mass`, a média tem de bater `expected_mass_causal(a, k, T)` dentro
    do erro de Monte Carlo. Se a normalização por consulta estivesse errada — um
    denominador fixo em vez de um por linha —, a medida sairia deprimida nas
    consultas iniciais e a média não bateria.
    """
    from src.selective.geometry import causal_mass, expected_mass_causal

    rng = np.random.default_rng(11)
    for T, k in ((40, 1), (40, 4), (80, 2)):
        A = _causal_estocastica(2, 2, T, rng)
        inicios = rng.integers(0, T - k, size=140)
        medidas, esperadas = [], []
        for a0 in inicios:
            medidas.append(causal_mass(A, list(range(int(a0), int(a0) + k))))
            esperadas.append(float(expected_mass_causal(int(a0), k, T)))
        m, e = np.mean(medidas), np.mean(esperadas)
        ep = np.std(medidas) / np.sqrt(len(medidas))
        assert abs(m - e) < 3.0 * ep + 1e-4, (
            f"T={T} k={k}: medido {m:.5f}, derivado {e:.5f}, erro-padrão {ep:.5f}")


def test_a_POSICAO_domina_o_tamanho_na_medida_tambem():
    """A consequência que torna o decoder pior que o encoder.

    No bidirecional, 4 tokens recebem exatamente 4x o de 1 token. No causal, um
    trecho de 1 token no começo pode receber MAIS que um de 4 no fim. A medida
    tem de reproduzir isso, senão ela não é causal.
    """
    from src.selective.geometry import causal_mass

    rng = np.random.default_rng(5)
    T = 60
    A = _causal_estocastica(2, 2, T, rng)
    inicio_1 = np.mean([causal_mass(A, [0]) for _ in range(1)])
    fim_4 = np.mean([causal_mass(A, list(range(45, 49))) for _ in range(1)])
    assert inicio_1 > fim_4, (
        f"1 token em a=0 deu {inicio_1:.4f} e 4 tokens em a=45 deram {fim_4:.4f}; "
        f"a posição tem de dominar o tamanho")


def test_denominador_e_POR_CONSULTA_e_nao_fixo():
    """A prova direta: cada linha somada sobre as chaves que ELA vê dá 1.

    Com denominador por consulta, um trecho que cobre TODAS as chaves visíveis
    de uma linha recebe massa 1 naquela linha. Com denominador fixo receberia
    menos, porque os zeros estruturais entrariam embaixo.
    """
    from src.selective.geometry import causal_mass

    rng = np.random.default_rng(7)
    T = 12
    A = _causal_estocastica(1, 1, T, rng)
    # a consulta 0 só vê a chave 0; o trecho [0] cobre tudo que ela vê
    assert causal_mass(A, [0], query_positions=[0]) == pytest.approx(1.0)
    # a consulta 3 vê as chaves 0..3; o trecho [0,1,2,3] cobre tudo
    assert causal_mass(A, [0, 1, 2, 3], query_positions=[3]) == pytest.approx(1.0)


def test_query_positions_seleciona_DE_ONDE_se_olha():
    from src.selective.geometry import causal_mass

    rng = np.random.default_rng(9)
    A = _causal_estocastica(2, 2, 30, rng)
    todas = causal_mass(A, [5, 6])
    so_finais = causal_mass(A, [5, 6], query_positions=list(range(20, 30)))
    assert todas != so_finais, "a leitura tem de depender de onde se olha"


def test_causal_mass_recusa_entrada_invalida():
    from src.selective.geometry import GeometryError, causal_mass

    rng = np.random.default_rng(1)
    A = _causal_estocastica(2, 2, 10, rng)
    with pytest.raises(GeometryError, match="quadrada|camadas"):
        causal_mass(A[:, :, :5, :], [0])
    with pytest.raises(GeometryError, match="trecho vazio"):
        causal_mass(A, [])
    with pytest.raises(GeometryError, match="fora de"):
        causal_mass(A, [99])
    with pytest.raises(GeometryError, match="consultas fora de"):
        causal_mass(A, [0], query_positions=[99])
