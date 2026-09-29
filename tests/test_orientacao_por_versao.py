"""A regra de orientação do escore é da versão 5 do protocolo, e só dela.

Defeito achado em 24/09/2026 ao regerar as tabelas publicadas: a regra entrou com
decl-07/08 (hash_version 5), mas o executor a aplicava a TODA declaração. Sob o
código daquele dia, decl-02, 03, 04 e 06 não reproduziam as próprias tabelas
publicadas. Uma regra nova de protocolo não pode reescrever o veredito de uma
declaração assinada antes dela.
"""
import dataclasses
import numpy as np
from src.selective.comparisons import _combinar
from src.selective.preregistration import load_preregistration

# escore que PIORA com o acerto: a regra v5 o vira; as versões anteriores, não.
cal = {"s": np.array([1.0, 2.0, 3.0, 4.0]), "loss": np.array([0.0, 0.0, 1.0, 1.0])}
aval = {"s": np.array([10.0, 20.0]), "loss": np.array([0.0, 1.0])}


def _decl(versao):
    p = load_preregistration("configs/decl-09-sinais-large.yaml")
    return dataclasses.replace(p, hash_version=versao)


def test_versoes_anteriores_a_5_usam_o_escore_como_medido():
    for v in (2, 3, 4):
        assert np.array_equal(_combinar("s", cal, aval, _decl(v), []), aval["s"])


def test_versao_5_orienta_pela_calibracao():
    assert np.array_equal(_combinar("s", cal, aval, _decl(5), []), -aval["s"])
