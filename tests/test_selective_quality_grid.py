"""A grade relativa, e a propriedade que ela existe para ter.

O teste central deste arquivo não é de encanamento: é a INVARIÂNCIA. A grade
absoluta falhou porque a mesma meta era trivial num extrator bom e inalcançável
num ruim. A grade relativa só serve se a mesma fração do vão exigir abstenção
nos dois — e é isso que se fixa aqui, com as precisões de base REAIS medidas nos
três modelos do projeto.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.selective.quality_grid import (
    QualityGridError,
    absolute_targets,
    base_precision,
    gap_closed,
)

# Precisões de base medidas na partição de avaliação, dos três modelos (T1, linha
# "(sem supervisor)"). Entram como dado e não como exemplo inventado: o defeito
# que a grade conserta foi observado nestes números.
P_BASE = {"gliner-base/genia": 0.5191, "gliner-base/conll": 0.4690,
          "ajustado/genia": 0.7582, "ajustado/conll": 0.8997}


def _perda_com_precisao(p: float, n: int = 10_000) -> np.ndarray:
    """Vetor de perda 0/1 cuja média de acertos é p, por construção."""
    l = np.ones(n)
    l[: int(round(p * n))] = 0.0
    return l


def test_precisao_de_base_e_a_fracao_de_acertos():
    assert base_precision(np.array([0.0, 0.0, 1.0, 1.0])) == 0.5
    assert base_precision(np.zeros(7)) == 1.0


def test_perda_que_nao_e_zero_um_e_recusada():
    with pytest.raises(QualityGridError, match="0/1"):
        base_precision(np.array([0.0, 0.5, 1.0]))


def test_particao_vazia_e_recusada():
    with pytest.raises(QualityGridError, match="vazia"):
        base_precision(np.array([]))


def test_a_meta_absoluta_e_a_formula_do_vao():
    (t,) = absolute_targets([0.5], _perda_com_precisao(0.8))
    assert t.base_precision == pytest.approx(0.8)
    assert t.absolute == pytest.approx(0.9)          # 0,8 + 0,5 * 0,2
    (t2,) = absolute_targets([0.25], _perda_com_precisao(0.6))
    assert t2.absolute == pytest.approx(0.7)         # 0,6 + 0,25 * 0,4


def test_INVARIANCIA_a_mesma_fracao_exige_abstencao_em_todo_extrator():
    """A propriedade pela qual a grade existe.

    Para CADA um dos quatro pares modelo/corpus medidos, e para cada fração
    declarada, a meta derivada tem de ficar ESTRITAMENTE acima da precisão de
    base — isto é, tem de exigir abster-se de algo. A grade absoluta falhava
    exatamente aqui: 0,70 e 0,80 ficavam abaixo de 0,8997 e portanto eram
    atingidas com carga zero.
    """
    fracoes = (0.25, 0.50, 0.75, 0.90)
    for nome, p in P_BASE.items():
        for t in absolute_targets(fracoes, _perda_com_precisao(p)):
            assert t.absolute > t.base_precision, f"{nome} fração {t.gap_fraction}"
            assert t.absolute < 1.0, f"{nome} fração {t.gap_fraction} exige perfeição"


def test_a_grade_ABSOLUTA_de_fato_falhava_no_extrator_bom():
    """O contraste que justifica a troca, fixado como teste e não como prosa.

    Com precisão de base 0,8997, duas das quatro metas absolutas antigas são
    atingidas sem abstenção nenhuma. É o defeito medido.
    """
    p = P_BASE["ajustado/conll"]
    antigas = (0.70, 0.80, 0.90, 0.95)
    triviais = [q for q in antigas if q <= p]
    assert triviais == [0.70, 0.80], "o defeito que motivou a grade relativa mudou de forma"


def test_fracao_fora_do_intervalo_e_recusada():
    for ruim in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(QualityGridError, match="fração do vão"):
            absolute_targets([ruim], _perda_com_precisao(0.8))


def test_extrator_perfeito_nao_define_vao():
    with pytest.raises(QualityGridError, match="fora de"):
        absolute_targets([0.5], np.zeros(10))


def test_vao_fechado_e_a_leitura_inversa():
    assert gap_closed(0.9, 0.8) == pytest.approx(0.5)
    assert gap_closed(0.8, 0.8) == pytest.approx(0.0)
    # Ponto de operação que PIORA a precisão dá vão negativo, e isso é
    # informativo: significa que o supervisor entregou pior que não supervisionar.
    assert gap_closed(0.7, 0.8) == pytest.approx(-0.5)


def test_a_meta_derivada_e_monotona_na_fracao():
    ts = absolute_targets([0.25, 0.5, 0.75, 0.9], _perda_com_precisao(0.6))
    assert list(t.absolute for t in ts) == sorted(t.absolute for t in ts)


def test_render_mostra_a_conta():
    (t,) = absolute_targets([0.5], _perda_com_precisao(0.8))
    texto = t.render()
    assert "50%" in texto and "0.8" in texto and "0.9" in texto
