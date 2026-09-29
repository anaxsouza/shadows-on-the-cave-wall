"""Testes da curva risco-cobertura, da AURC e do teste de valor adicionado.

Todos os números aqui são aritmética da definição, não medição de modelo: seis
entidades com três erradas, escores escolhidos à mão. É o que permite que este
arquivo rode em milissegundos e falhe por motivo interpretável.

Estes testes existem sobretudo para fixar as CONVENÇÕES, que é onde duas
implementações honestas de AURC divergem: o denominador condicional do risco, o
tratamento de empates e a faixa inicial da integração.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.selective.risk_coverage import (
    aurc,
    delta_aurc_paired_bootstrap,
    random_abstention_risk,
    risk_coverage_curve,
)

PERDAS = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]  # três acertos, três erros
PERFEITO = [1.0, 0.9, 0.8, 0.3, 0.2, 0.1]
INVERTIDO = [0.1, 0.2, 0.3, 0.8, 0.9, 1.0]
CONSTANTE = [0.5] * 6


def test_curva_com_ordenacao_perfeita_e_a_aritmetica_da_definicao():
    c = risk_coverage_curve(PERDAS, PERFEITO)
    assert np.allclose(c.coverage, [1 / 6, 2 / 6, 3 / 6, 4 / 6, 5 / 6, 1.0])
    assert np.allclose(c.risk, [0.0, 0.0, 0.0, 0.25, 0.4, 0.5])
    assert c.base_risk == 0.5
    assert c.n_tied_groups == 0
    assert c.aurc == pytest.approx(0.15)


def test_escore_constante_colapsa_em_um_ponto_e_sua_aurc_e_a_taxa_base():
    """Empate total: um só grupo, logo uma só fronteira, logo um só ponto.

    Este é o teste que fixa a convenção de empate. Sem ela, a curva de um escore
    constante teria seis pontos cujos riscos dependeriam da ordem de chegada das
    entidades — e a AURC mudaria ao reordenar o arquivo de entrada.
    """
    c = risk_coverage_curve(PERDAS, CONSTANTE)
    assert c.coverage.tolist() == [1.0]
    assert c.risk.tolist() == [0.5]
    assert c.n_tied_groups == 5
    assert c.aurc == pytest.approx(random_abstention_risk(PERDAS))


def test_ordenacao_invertida_e_simetrica_da_perfeita_em_torno_da_taxa_base():
    perfeita = risk_coverage_curve(PERDAS, PERFEITO).aurc
    invertida = risk_coverage_curve(PERDAS, INVERTIDO).aurc
    assert invertida == pytest.approx(0.85)
    assert perfeita + invertida == pytest.approx(1.0)


def test_abstencao_aleatoria_nao_depende_de_escore_nenhum():
    """A linha de base do piso é uma horizontal, e é isso que a torna uma prova.

    Descartar entidades ao acaso remove erros e acertos na mesma proporção, então
    o risco condicional esperado não se move. Quem espera que abster-se de metade
    reduza o risco pela metade está pensando no risco não condicional.
    """
    assert random_abstention_risk(PERDAS) == 0.5
    assert random_abstention_risk([0.0, 1.0]) == 0.5
    assert random_abstention_risk([1.0, 1.0, 1.0]) == 1.0


def test_risco_em_cobertura_alvo_nao_interpola():
    c = risk_coverage_curve(PERDAS, PERFEITO)
    # 0,6 não é fronteira; a resposta é a maior fronteira que não a excede (0,5).
    assert c.risk_at(0.6) == pytest.approx(0.0)
    assert c.risk_at(0.5) == pytest.approx(0.0)
    assert c.risk_at(1.0) == pytest.approx(0.5)
    with pytest.raises(ValueError, match="nenhuma cobertura reportada"):
        risk_coverage_curve(PERDAS, CONSTANTE).risk_at(0.5)


def test_aurc_ignora_transformacao_monotona_do_escore():
    """AURC depende só da ORDEM, e a consequência é uma frase que não se pode escrever.

    Calibrar um escore por qualquer função monótona crescente não muda a AURC nem
    o conjunto retido em cada cobertura. Logo o teste do piso não fica "mais
    justo" por causa da calibração: a comparação é justa porque os dois escores
    são medidos nas mesmas entidades. A calibração serve para dar sentido de
    risco ao limiar, e é aí que ela é indispensável.
    """
    original = risk_coverage_curve(PERDAS, PERFEITO).aurc
    esticado = risk_coverage_curve(PERDAS, [np.exp(3 * s) for s in PERFEITO]).aurc
    deslocado = risk_coverage_curve(PERDAS, [0.1 * s + 7 for s in PERFEITO]).aurc
    assert original == pytest.approx(esticado) == pytest.approx(deslocado)


def test_delta_aurc_pareado_detecta_o_melhor_e_exige_os_itens_do_preregistro():
    d = delta_aurc_paired_bootstrap(
        PERDAS, CONSTANTE, PERFEITO,
        # Uma entidade por grupo reproduz a reamostragem por entidade, que é o
        # que a decl-01 declarou. A decl-02 declara por sentença.
        groups=range(len(PERDAS)),
        n_resamples=400, level=0.95, seed=7,
        labels=("softmax", "combinado"),
    )
    assert d.aurc_a == pytest.approx(0.5)
    assert d.aurc_b == pytest.approx(0.15)
    assert d.delta < 0  # negativo = o segundo é melhor
    assert not d.contains_zero
    assert "combinado ordena melhor" in d.render()
    with pytest.raises(TypeError):
        delta_aurc_paired_bootstrap(PERDAS, CONSTANTE, PERFEITO)  # type: ignore[call-arg]


def test_delta_aurc_de_um_escore_contra_si_mesmo_contem_zero():
    d = delta_aurc_paired_bootstrap(
        PERDAS, PERFEITO, PERFEITO, groups=range(len(PERDAS)),
        n_resamples=400, level=0.95, seed=7
    )
    assert d.delta == pytest.approx(0.0)
    assert d.contains_zero


def test_reamostrar_por_grupo_alarga_o_intervalo():
    """O defeito que a reamostragem por sentença corrige, medido.

    Entidades da mesma sentença compartilham a matriz de atenção. Tratá-las como
    independentes produz intervalo ESTREITO demais — na direção de fazer uma
    diferença excluir o zero quando não devia.

    Aqui o sinal é construído com correlação DENTRO do grupo: cada grupo tem
    perda e escore homogêneos, então há 8 informações e não 40.
    """
    rng = np.random.default_rng(5)
    grupos, perdas, escore_a, escore_b = [], [], [], []
    for g in range(8):
        base = rng.normal()
        alvo = float(g % 2)
        for _ in range(5):
            grupos.append(g)
            perdas.append(alvo)
            escore_a.append(base + rng.normal(0, 0.01))
            escore_b.append(-base + rng.normal(0, 0.01))

    por_entidade = delta_aurc_paired_bootstrap(
        perdas, escore_a, escore_b, groups=range(len(perdas)),
        n_resamples=600, level=0.95, seed=3)
    por_sentenca = delta_aurc_paired_bootstrap(
        perdas, escore_a, escore_b, groups=grupos,
        n_resamples=600, level=0.95, seed=3)

    largura = lambda d: d.ci_high - d.ci_low
    assert por_sentenca.n_groups == 8 and por_entidade.n_groups == 40
    assert largura(por_sentenca) > largura(por_entidade), (
        f"por sentença {largura(por_sentenca):.4f} não é mais largo que por "
        f"entidade {largura(por_entidade):.4f}: o conglomerado não está agindo"
    )
    assert por_sentenca.delta == pytest.approx(por_entidade.delta), (
        "a unidade de reamostragem não pode mudar a estimativa pontual"
    )


def test_entrada_invalida_falha_alto():
    with pytest.raises(ValueError, match="tamanhos diferentes"):
        risk_coverage_curve([0.0, 1.0], [0.5])
    with pytest.raises(ValueError, match="n = 0"):
        risk_coverage_curve([], [])
    with pytest.raises(ValueError, match="não finita"):
        risk_coverage_curve([0.0, float("nan")], [1.0, 0.0])
    with pytest.raises(ValueError, match="escore não finito"):
        risk_coverage_curve([0.0, 1.0], [1.0, float("inf")])
    with pytest.raises(ValueError, match="negativa"):
        risk_coverage_curve([-1.0, 1.0], [1.0, 0.0])


def test_aurc_de_curva_com_um_ponto_e_o_proprio_risco():
    c = risk_coverage_curve([1.0], [0.3])
    assert aurc(c) == pytest.approx(1.0)
