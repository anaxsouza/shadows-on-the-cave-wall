"""Testes da taxonomia de erros (decl-14) e das janelas de controle (decl-15),
com casos pequenos feitos à mão."""
import math

import numpy as np
import pytest

from src.selective.attention_mass import span_attention_mass
from src.selective.janelas import (
    enriquecimento_decoder, enriquecimento_encoder, janelas_da_entidade,
    janelas_livres, media_decoder, media_encoder, tipo_erro)

OURO = [(0, 5, "A"), (10, 20, "B"), (10, 14, "A")]


# ---- taxonomia -----------------------------------------------------------

@pytest.mark.parametrize("ini,fim,rot,esperado", [
    (0, 5, "A", "acerto"),
    (0, 5, "B", "rotulo"),            # mesmas fronteiras, rótulo diferente
    (10, 20, "B", "acerto"),
    (10, 14, "A", "acerto"),          # aninhada anotada também é acerto
    (10, 14, "B", "rotulo"),
    (0, 4, "A", "fronteira"),         # dentro de uma anotada, fronteira diferente
    (3, 8, "A", "fronteira"),         # sobrepõe 2 caracteres
    (4, 6, "A", "fronteira"),         # sobrepõe exatamente 1 caractere
    (5, 8, "A", "sem_par"),           # encosta (fim == início) mas não sobrepõe
    (6, 9, "A", "sem_par"),
    (25, 30, "A", "sem_par"),
])
def test_taxonomia(ini, fim, rot, esperado):
    assert tipo_erro(ini, fim, rot, OURO) == esperado


def test_taxonomia_precedencia_rotulo_sobre_fronteira():
    # (10, 14) casa fronteiras com a anotada (10, 14, "A") E sobrepõe (10, 20):
    # com rótulo errado vale `rotulo`, não `fronteira`.
    assert tipo_erro(10, 14, "X", OURO) == "rotulo"


def test_taxonomia_acerto_vence_se_ha_duas_anotadas_com_as_mesmas_fronteiras():
    ouro = [(0, 5, "A"), (0, 5, "B")]
    assert tipo_erro(0, 5, "B", ouro) == "acerto"
    assert tipo_erro(0, 5, "C", ouro) == "rotulo"


def test_taxonomia_sem_ouro():
    assert tipo_erro(0, 5, "A", []) == "sem_par"


# ---- janelas -------------------------------------------------------------

def test_janelas_k2_com_entidade_no_meio():
    frase = range(1, 9)               # tokens 1..8 (0 é o [CLS])
    assert janelas_livres(frase, 2, {3, 4}) == [(1, 2), (5, 6), (6, 7), (7, 8)]


def test_janelas_k1_sao_os_tokens_livres():
    assert janelas_livres(range(1, 5), 1, {2}) == [(1,), (3,), (4,)]


def test_janelas_nao_atravessam_o_fim_da_frase():
    assert janelas_livres(range(1, 4), 3, set()) == [(1, 2, 3)]
    assert janelas_livres(range(1, 4), 4, set()) == []


def test_janelas_exigem_tokens_consecutivos_da_frase():
    # o 3 não é token de conteúdo (ex.: fora do trecho da frase): nenhuma janela o cruza
    assert janelas_livres([1, 2, 4, 5], 2, set()) == [(1, 2), (4, 5)]


def test_janelas_ocupadas_por_qualquer_token():
    assert janelas_livres(range(1, 7), 3, {2}) == [(3, 4, 5), (4, 5, 6)]
    assert janelas_livres(range(1, 7), 3, {3, 4}) == []


def test_janelas_sem_possibilidade_conta_zero_e_media_nan():
    n, m = janelas_da_entidade([2, 3, 4], range(1, 6), {2, 3, 4}, lambda w: 1.0)
    assert n == 0 and math.isnan(m)


def test_janelas_da_entidade_media_do_enriquecimento():
    n, m = janelas_da_entidade([1], range(1, 5), {1}, lambda w: float(w[0]))
    assert n == 3 and m == pytest.approx((2 + 3 + 4) / 3)


# ---- enriquecimento: mesma conta da medição original ---------------------

def _atencao_aleatoria(L=3, H=2, T=9, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.gamma(0.7, size=(L, H, T, T))
    return (a / a.sum(-1, keepdims=True)).astype(np.float16)


def test_encoder_media_previa_reproduz_span_attention_mass():
    a = _atencao_aleatoria()
    camadas, cabecas = [0, 1, 2], None
    media = media_encoder(a, camadas, cabecas)
    for idx in ([3], [2, 3], [4, 5, 6]):
        orig = span_attention_mass(a, idx, sink_policy="drop_from_denominator",
                                   layers=camadas, heads=cabecas).mass
        via = span_attention_mass(media, idx, sink_policy="drop_from_denominator",
                                  layers=[0], heads=None).mass
        assert via == pytest.approx(orig, abs=1e-12)
        esp = len(idx) / (a.shape[-1] - 1)
        assert enriquecimento_encoder(media, idx, a.shape[-1], "drop_from_denominator") \
            == pytest.approx(orig / esp, abs=1e-12)


def test_encoder_atencao_uniforme_da_enriquecimento_um():
    T = 8
    a = np.full((2, 2, T, T), 1.0 / T)
    media = media_encoder(a, [0, 1], None)
    assert enriquecimento_encoder(media, [2, 3], T, "drop_from_denominator") == pytest.approx(1.0)


def test_decoder_atencao_causal_uniforme_da_enriquecimento_um_em_qualquer_posicao():
    T = 10
    a = np.zeros((2, 3, T, T))
    for q in range(T):
        a[:, :, q, : q + 1] = 1.0 / (q + 1)
    media = media_decoder(a)
    for idx in ([0], [4], [3, 4, 5], [7, 8, 9]):
        assert enriquecimento_decoder(media, idx, T) == pytest.approx(1.0)


def test_decoder_media_previa_reproduz_causal_mass():
    from src.selective.geometry import causal_mass
    T = 7
    rng = np.random.default_rng(1)
    a = rng.gamma(0.8, size=(2, 2, T, T)) * np.tril(np.ones((T, T)))
    a = a / a.sum(-1, keepdims=True)
    for idx in ([2], [3, 4]):
        assert causal_mass(media_decoder(a), idx) == pytest.approx(causal_mass(a, idx), abs=1e-12)
