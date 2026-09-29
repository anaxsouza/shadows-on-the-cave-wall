"""A varredura deriva a forma do tensor, e não a supõe.

O defeito que estes testes impedem de voltar: as faixas de camada e o conjunto
de cabeças estavam fixos em `range(12)`, o que só valia para o `gliner_base`. No
`gliner_large`, de 24 camadas e 16 cabeças, a varredura percorreria metade da
rede e indexaria cabeças inexistentes — e sairia como número plausível, porque
varrer menos camadas ainda produz uma distribuição de ΔAURC com cara de
resultado.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "varredura", RAIZ / "tools" / "perimetro1_varredura.py")
varredura = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(varredura)


@pytest.mark.parametrize("n_camadas,n_cabecas", [(12, 12), (24, 16), (6, 8)])
def test_a_varredura_cobre_TODAS_as_camadas_e_cabecas_do_modelo(n_camadas, n_cabecas):
    vistas_c, vistas_h = set(), set()
    for _d, _s, _nc, cs, _nh, hs in varredura.leituras(n_camadas, n_cabecas):
        vistas_c |= set(cs)
        vistas_h |= set(hs)
    assert vistas_c == set(range(n_camadas)), f"camadas não cobertas: {set(range(n_camadas)) - vistas_c}"
    assert vistas_h == set(range(n_cabecas)), f"cabeças não cobertas: {set(range(n_cabecas)) - vistas_h}"


@pytest.mark.parametrize("n_camadas,n_cabecas", [(12, 12), (24, 16)])
def test_a_varredura_nao_indexa_camada_nem_cabeca_INEXISTENTE(n_camadas, n_cabecas):
    for _d, _s, _nc, cs, _nh, hs in varredura.leituras(n_camadas, n_cabecas):
        assert max(cs) < n_camadas, f"camada {max(cs)} não existe em modelo de {n_camadas}"
        assert max(hs) < n_cabecas, f"cabeça {max(hs)} não existe em modelo de {n_cabecas}"


def test_as_faixas_sao_TERCOS_e_nao_indices_absolutos():
    """Comparabilidade entre escalas: 'o primeiro terço' é comparável, 'as quatro
    primeiras camadas' não. Mesma razão da grade de qualidade relativa."""
    F12, F24 = varredura.faixas(12), varredura.faixas(24)
    assert F12["inicio"] == (0, 1, 2, 3)
    assert F24["inicio"] == tuple(range(8)), "o primeiro terço de 24 tem 8 camadas"
    for F, n in ((F12, 12), (F24, 24)):
        assert F["todas"] == tuple(range(n))
        # as três faixas particionam a rede, sem buraco nem sobreposição
        juntas = sorted(F["inicio"] + F["meio"] + F["fim"])
        assert juntas == list(range(n)), f"faixas não particionam {n} camadas: {juntas}"


def test_o_numero_de_leituras_cresce_com_a_forma():
    n12 = sum(1 for _ in varredura.leituras(12, 12))
    n24 = sum(1 for _ in varredura.leituras(24, 16))
    assert n24 > n12, "modelo mais profundo tem de gerar mais leituras"
    # 4 direções x 3 sumidouros x (1 + L + 3) faixas x (1 + H) cabeças
    assert n12 == 4 * 3 * (1 + 12 + 3) * (1 + 12)
    assert n24 == 4 * 3 * (1 + 24 + 3) * (1 + 16)
