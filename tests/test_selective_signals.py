"""As três famílias, e a VERIFICAÇÃO NUMÉRICA dos nulos exatos.

O teste central deste arquivo não é de encanamento. Os nulos da família de
atenção são afirmações algébricas — entropia esperada `log|K|`, máximo esperado
`1/|K|` — e uma afirmação algébrica se verifica gerando matrizes e conferindo,
não relendo a derivação. É o que se faz aqui, sobre densidade NÃO uniforme, para
que o nulo não possa depender da forma da distribuição.

O segundo grupo de testes fixa a separação de estatuto: um sinal de nulo exato
não pode carregar covariável, e um de nulo empírico não pode ficar sem ela. Isso
existe para que acrescentar um sinal novo no futuro não possa esquecer de dizer
qual dos dois é.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.selective.signals import (
    ATENCAO,
    COVARIAVEIS,
    ESTADOS_OCULTOS,
    FAMILIAS,
    LOGITS,
    SignalError,
    SignalSpec,
    TODOS,
    expected_row_entropy,
    expected_row_max,
    por_nome,
    residuo_empirico,
    row_entropy,
)


def _linhas_estocasticas(n: int, T: int, rng, forma: float = 0.5) -> np.ndarray:
    """Matriz linha-estocástica de densidade gama, deliberadamente NÃO uniforme."""
    a = rng.gamma(shape=forma, scale=1.0, size=(n, T)) + 1e-12
    return a / a.sum(axis=1, keepdims=True)


# =============================================================================
# Os nulos EXATOS, verificados numericamente
# =============================================================================

def test_entropia_esperada_e_log_K_sob_a_uniforme():
    """Sob a uniforme a entropia da linha é log|K| exatamente, não aproximadamente."""
    for T in (5, 20, 97):
        K = T - 1                                   # drop_from_denominator
        u = np.full((30, K), 1.0 / K)
        medida = row_entropy(u)
        esperada = expected_row_entropy(T, "drop_from_denominator")
        assert np.allclose(medida, esperada, atol=1e-12), T
        assert esperada == pytest.approx(np.log(K))


def test_a_entropia_de_linha_aleatoria_fica_ABAIXO_do_nulo():
    """Toda distribuição não uniforme tem entropia menor que log|K|.

    O nulo é um TETO, e isso importa para a leitura: entropia alta não é sinal de
    dúvida do modelo, é sinal de que a mesa é grande.
    """
    rng = np.random.default_rng(0)
    for T in (12, 40, 80):
        A = _linhas_estocasticas(200, T - 1, rng)
        teto = expected_row_entropy(T, "drop_from_denominator")
        h = row_entropy(A)
        assert np.all(h <= teto + 1e-9)
        assert h.mean() < teto, "densidade gama deveria ficar estritamente abaixo do teto"


def test_o_nulo_da_entropia_cresce_como_log_do_comprimento():
    """A dependência Theta(log T) em forma fechada, e é ela que confunde o sinal."""
    Ts = np.array([10, 20, 40, 80, 160])
    nulos = np.array([expected_row_entropy(int(T), "drop_from_denominator") for T in Ts])
    # Dobrar T acrescenta log 2 ao nulo, a menos do -1 da política de sumidouro.
    difs = np.diff(nulos)
    assert np.allclose(difs, np.log(2), atol=0.06), difs


def test_maximo_esperado_e_um_sobre_K():
    for T in (4, 33, 128):
        assert expected_row_max(T, "drop_from_denominator") == pytest.approx(1.0 / (T - 1))
    assert expected_row_max(10, "keep") == pytest.approx(1.0 / 10)


def test_linha_que_nao_soma_um_e_recusada():
    """Sem orçamento fixo o nulo exato não vale, e usá-lo seria afirmar o que não se tem."""
    A = np.full((3, 5), 0.1)                       # soma 0,5
    with pytest.raises(SignalError, match="não somam 1"):
        row_entropy(A)


def test_matriz_de_dimensao_errada_e_recusada():
    with pytest.raises(SignalError, match="matriz"):
        row_entropy(np.ones(5) / 5)


def test_sentenca_sem_chave_permitida_e_recusada_a_montante():
    """A recusa vem de `geometry.n_allowed_keys`, e não deste módulo.

    Duplicar o guarda aqui criaria dois lugares para manter em sincronia, e é o
    contador de chaves permitidas que sabe o que cada política de sumidouro faz.
    O teste fixa QUE a recusa acontece e DE ONDE ela vem.
    """
    from src.selective.geometry import GeometryError

    with pytest.raises(GeometryError, match="não há denominador"):
        expected_row_entropy(1, "drop_from_denominator")
    with pytest.raises(GeometryError, match="não há denominador"):
        expected_row_max(1, "drop_from_denominator")


# =============================================================================
# A separação de ESTATUTO, que é o que o módulo existe para proteger
# =============================================================================

def test_toda_a_familia_de_atencao_tem_nulo_exato_e_zero_covariavel():
    for s in ATENCAO:
        assert s.estatuto == "exato", s.nome
        assert s.covariaveis == (), f"{s.nome} ajusta algo ao dado"


def test_as_outras_duas_familias_tem_nulo_empirico_e_covariaveis_declaradas():
    for s in ESTADOS_OCULTOS + LOGITS:
        assert s.estatuto == "empirico", s.nome
        assert s.covariaveis == COVARIAVEIS, s.nome


def test_nulo_exato_com_covariavel_e_recusado_na_construcao():
    with pytest.raises(SignalError, match="não regride"):
        SignalSpec("x", "atencao", "exato", "d", ("span_size",))


def test_nulo_empirico_sem_covariavel_e_recusado_na_construcao():
    with pytest.raises(SignalError, match="exige as covariáveis"):
        SignalSpec("x", "logits", "empirico", "d")


def test_o_registro_nao_tem_nome_repetido():
    nomes = [s.nome for s in TODOS]
    assert len(nomes) == len(set(nomes))
    assert sum(len(v) for v in FAMILIAS.values()) == len(TODOS)


def test_por_nome_encontra_e_recusa_desconhecido():
    assert por_nome("span_mass").familia == "atencao"
    with pytest.raises(SignalError, match="desconhecido"):
        por_nome("massa_inventada")


def test_render_mostra_o_estatuto():
    assert "nulo exato" in por_nome("span_mass").render()
    r = por_nome("hidden_norm").render()
    assert "nulo empirico" in r or "nulo empírico" in r
    assert "span_size" in r


# =============================================================================
# O resíduo empírico
# =============================================================================

def test_residuo_empirico_remove_a_tendencia_e_preserva_o_sinal_plantado():
    rng = np.random.default_rng(3)
    k = rng.integers(1, 6, size=300).astype(float)
    T = rng.integers(20, 90, size=300).astype(float)
    pos = rng.integers(0, 40, size=300).astype(float)
    plantado = rng.normal(0, 1.0, size=300)
    y = 3.0 * k - 0.4 * T + 0.2 * pos + 5.0 + plantado
    r = residuo_empirico(y, np.column_stack([k, T, pos]))
    for nome, cov in (("k", k), ("T", T), ("pos", pos)):
        assert abs(np.corrcoef(r, cov)[0, 1]) < 0.05, nome
    assert np.corrcoef(r, plantado)[0, 1] > 0.95


def test_residuo_recusa_amostra_sem_grau_de_liberdade():
    with pytest.raises(SignalError, match="grau de liberdade"):
        residuo_empirico(np.arange(4.0), np.arange(4.0)[:, None] @ np.ones((1, 3)))


def test_residuo_recusa_formas_incompativeis():
    with pytest.raises(SignalError, match="incompatíveis"):
        residuo_empirico(np.arange(10.0), np.arange(7.0)[:, None])


# =============================================================================
# Os nulos EXATOS da família de atenção sob MÁSCARA CAUSAL
# =============================================================================
# Reusar os nulos bidirecionais no decoder não seria conservador — seria errado,
# e o erro tem direção: o teto da entropia ficaria alto demais (toda entropia
# observada pareceria baixa) e o do máximo, baixo demais (todo máximo pareceria
# alto). Estes testes fixam as duas fórmulas e a MAGNITUDE da diferença, para
# que ninguém troque uma pela outra por descuido.


def _causal_uniforme(T, rng=None):
    """Linha `i` uniforme sobre as `i+1` chaves permitidas. É o nulo, literal."""
    a = np.zeros((T, T))
    for i in range(T):
        a[i, : i + 1] = 1.0 / (i + 1)
    return a


@pytest.mark.parametrize("T", [1, 2, 5, 20, 60, 120])
def test_a_entropia_causal_bate_a_forma_fechada(T):
    from src.selective.signals import expected_row_entropy_causal, row_entropy

    A = _causal_uniforme(T)
    # entropia de cada linha, sobre as chaves permitidas dela
    # `row_entropy` recebe MATRIZ [consulta, chave] e devolve uma entropia por
    # linha. Cada linha causal tem largura diferente, então cada uma entra
    # como matriz de uma linha só, com a largura dela.
    obs = np.mean([float(row_entropy(A[i : i + 1, : i + 1])[0]) for i in range(T)])
    assert obs == pytest.approx(expected_row_entropy_causal(T), abs=1e-9), (
        f"T={T}: observado {obs:.6f}, fórmula {expected_row_entropy_causal(T):.6f}")


@pytest.mark.parametrize("T", [1, 2, 5, 20, 60, 120])
def test_o_maximo_causal_bate_a_forma_fechada(T):
    from src.selective.signals import expected_row_max_causal

    A = _causal_uniforme(T)
    obs = np.mean([A[i, : i + 1].max() for i in range(T)])
    assert obs == pytest.approx(expected_row_max_causal(T), abs=1e-9)


def test_a_diferenca_entre_causal_e_bidirecional_e_GRANDE_e_cresce_com_T():
    """A prova de que a troca não é detalhe.

    Se a diferença fosse pequena, reusar o nulo bidirecional seria aproximação
    defensável. Ela não é: o máximo esperado causal é vários múltiplos do
    bidirecional, e a razão cresce com o comprimento.
    """
    from src.selective.signals import (
        expected_row_entropy,
        expected_row_entropy_causal,
        expected_row_max,
        expected_row_max_causal,
    )

    razoes_max = []
    for T in (20, 60, 120):
        ec = expected_row_entropy_causal(T)
        eb = expected_row_entropy(T, "drop_from_denominator")
        mc = expected_row_max_causal(T)
        mb = expected_row_max(T, "drop_from_denominator")
        # a entropia causal é MENOR: menos chaves, teto mais baixo
        assert ec < eb, f"T={T}: entropia causal {ec:.4f} deveria ser < bidir {eb:.4f}"
        # o máximo causal é MAIOR: a primeira linha vê uma chave só
        assert mc > mb, f"T={T}: máximo causal {mc:.5f} deveria ser > bidir {mb:.5f}"
        razoes_max.append(mc / mb)
        assert mc / mb > 3.0, f"T={T}: razão do máximo {mc / mb:.1f}x, esperava > 3"
    assert razoes_max == sorted(razoes_max), (
        f"a razão do máximo tem de CRESCER com T: {[round(r, 1) for r in razoes_max]}")


def test_a_entropia_causal_e_um_TETO_como_a_bidirecional():
    """Atenção concentrada dá entropia ABAIXO do nulo, nunca acima."""
    from src.selective.signals import expected_row_entropy_causal, row_entropy

    rng = np.random.default_rng(3)
    T = 40
    a = rng.gamma(shape=0.3, scale=1.0, size=(T, T)) + 1e-12
    a = a * np.tril(np.ones((T, T)))
    a = a / a.sum(axis=-1, keepdims=True)
    obs = np.mean([float(row_entropy(a[i : i + 1, : i + 1])[0]) for i in range(T)])
    assert obs < expected_row_entropy_causal(T), (
        "entropia concentrada tem de ficar abaixo do teto uniforme")


def test_os_nulos_causais_recusam_comprimento_invalido():
    from src.selective.signals import (
        SignalError,
        expected_row_entropy_causal,
        expected_row_max_causal,
    )

    for f in (expected_row_entropy_causal, expected_row_max_causal):
        with pytest.raises(SignalError, match=">= 1"):
            f(0)


# =============================================================================
# O teto POR LINHA contra a média sobre TODAS as linhas
# =============================================================================
# Esta distinção foi um defeito real na primeira versão da medição do decoder,
# apanhado por um teste com o nulo literal: usei a média sobre as T linhas onde
# cabia o teto da linha, e o teto saiu subestimado — 1,6638 contra os 2,4826
# corretos, para as linhas 10 a 12. Os testes abaixo fixam as duas quantidades e
# a relação entre elas, porque elas têm nomes parecidos e significados
# diferentes.


def test_o_teto_por_linha_e_log_da_largura():
    from src.selective.signals import row_ceiling_entropy, row_ceiling_max

    assert row_ceiling_entropy(1) == 0.0, "uma chave: nenhuma incerteza possível"
    assert row_ceiling_max(1) == 1.0, "uma chave: toda a massa nela"
    assert row_ceiling_entropy(11) == pytest.approx(np.log(11))
    assert row_ceiling_max(11) == pytest.approx(1 / 11)


def test_a_media_sobre_TODAS_as_linhas_e_a_media_dos_tetos_por_linha():
    """A relação entre as duas, que é o que torna a confusão possível."""
    from src.selective.signals import (
        expected_row_entropy_causal,
        expected_row_max_causal,
        row_ceiling_entropy,
        row_ceiling_max,
    )

    for T in (5, 20, 60):
        media_ent = np.mean([row_ceiling_entropy(w) for w in range(1, T + 1)])
        media_max = np.mean([row_ceiling_max(w) for w in range(1, T + 1)])
        assert media_ent == pytest.approx(expected_row_entropy_causal(T), abs=1e-9)
        assert media_max == pytest.approx(expected_row_max_causal(T), abs=1e-9)


def test_as_duas_DIVERGEM_para_um_subconjunto_de_linhas():
    """O caso que causou o defeito, com os três números que o distinguem.

    Para as linhas 10, 11 e 12 existem TRÊS quantidades parecidas:

    - CORRETO: média dos tetos por linha, `mean(log(11), log(12), log(13))`.
    - O BUG: média da fórmula AGREGADA avaliada em cada largura,
      `mean(log(11!)/11, log(12!)/12, log(13!)/13)`. Foi isto que a primeira
      versão da medição fez, e subestima o teto em ~33%.
    - A fórmula agregada em `T` inteiro, `log(30!)/30`, que é a média sobre TODAS
      as trinta linhas e não se aplica a um subconjunto.

    Os números vêm da execução que apanhou o defeito, não de estimativa.
    """
    from src.selective.signals import expected_row_entropy_causal, row_ceiling_entropy

    idx = [10, 11, 12]
    correto = float(np.mean([row_ceiling_entropy(q + 1) for q in idx]))
    o_bug = float(np.mean([expected_row_entropy_causal(q + 1) for q in idx]))
    agregado_em_T = expected_row_entropy_causal(30)

    assert correto == pytest.approx(2.4826, abs=1e-3)
    assert o_bug == pytest.approx(1.6638, abs=1e-3)
    assert agregado_em_T == pytest.approx(2.4886, abs=1e-3)
    assert correto > o_bug, "o bug subestimava o teto"
    assert abs(correto - o_bug) / correto > 0.3, (
        "a diferença é de um terço, não arredondamento")


def test_os_tetos_por_linha_recusam_largura_invalida():
    from src.selective.signals import SignalError, row_ceiling_entropy, row_ceiling_max

    for f in (row_ceiling_entropy, row_ceiling_max):
        with pytest.raises(SignalError, match=">= 1"):
            f(0)
