"""Testes do controle conformal de risco e da massa de atenção sobre o span.

Sem modelo e sem GPU: matrizes de atenção construídas à mão, com linhas que
somam 1 como as de verdade. É o que permite testar o instrumento da tese em
milissegundos.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.selective.attention_mass import span_attention_mass
from src.selective.conformal import crc_threshold


# ---------------------------------------------------------------- conformal ---

def test_limiar_escolhe_a_maior_cobertura_que_respeita_o_limite_do_teorema():
    # 20 entidades; as 10 mais confiantes estão certas, as 10 menos estão erradas.
    perdas = [0.0] * 10 + [1.0] * 10
    escores = list(np.linspace(1.0, 0.05, 20))
    r = crc_threshold(perdas, escores, alpha=0.1)
    assert r.feasible
    assert r.empirical_risk <= r.bound
    # o limite é mais exigente que alpha, e a correção é (B - alpha)/n
    assert r.bound == pytest.approx(0.1 - 0.9 / 20)
    assert r.coverage == pytest.approx(0.5)  # todas as certas, nenhuma errada


def test_a_correcao_de_amostra_finita_e_o_preco_declarado():
    """Com n pequeno o teorema exige risco empírico bem abaixo de alpha.

    Este teste existe para que ninguém leia o alpha do pré-registro como o risco
    exigido na calibração. Com n = 50 e alpha = 0,1, o exigido é 0,082.
    """
    perdas = [0.0] * 45 + [1.0] * 5
    escores = list(np.linspace(1.0, 0.0, 50))
    r = crc_threshold(perdas, escores, alpha=0.1)
    assert r.bound == pytest.approx(0.082)
    assert r.n_calibration == 50


def test_quando_nenhuma_cobertura_atinge_o_risco_a_resposta_e_nao_ha_limiar():
    """Extrator que erra sempre: não existe limiar viável, e inventar um seria pior.

    O modo de falha que este teste proíbe é devolver o limiar de menor risco
    disponível como se ele cumprisse o pedido.
    """
    r = crc_threshold([1.0] * 10, list(np.linspace(1.0, 0.1, 10)), alpha=0.1)
    assert not r.feasible
    assert np.isnan(r.threshold)
    assert "Nenhum limiar viável" in r.render()


def test_o_caso_ideal_e_monotono_e_o_anticorrelacionado_nao():
    """Fixa a DIREÇÃO do diagnóstico, que é onde ele já esteve invertido.

    Percorrer limiares do mais alto para o mais baixo é aumentar a cobertura. Um
    escore que ordena erro faz o risco SUBIR com a cobertura: subida é o
    comportamento esperado. A violação é uma QUEDA — reter mais entidades e
    errar menos entre as retidas só acontece se o escore estiver pondo erro na
    frente de acerto em alguma faixa.

    Sem este teste, um diagnóstico invertido passa: ele acusaria o caso ideal e
    absolveria o pior caso, que é a falha mais perigosa possível num aviso.
    """
    escores = [1.0, 0.9, 0.8, 0.3, 0.2, 0.1]
    ideal = crc_threshold([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], escores, alpha=0.4)
    assert ideal.monotone
    assert ideal.max_monotonicity_violation == 0.0

    anticorrelacionado = crc_threshold([1.0, 1.0, 1.0, 0.0, 0.0, 0.0], escores, alpha=0.4)
    assert not anticorrelacionado.monotone
    assert anticorrelacionado.max_monotonicity_violation == pytest.approx(0.25)


def test_nao_monotonicidade_e_reportada_e_nao_escondida():
    """A hipótese do teorema tem de ser verificada, porque aqui ela pode falhar.

    O teorema exige risco monótono no limiar. Se o escore ordena erro AO
    CONTRÁRIO — o que é justamente a possibilidade que o teste do piso
    investiga — a monotonicidade se rompe, e citar o teorema passa a ser
    incorreto. O código reporta em vez de assumir.
    """
    perdas = [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    escores = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]
    r = crc_threshold(perdas, escores, alpha=0.4)
    assert not r.monotone
    assert r.max_monotonicity_violation > 0
    assert "não monótono" in r.render()


def test_alpha_e_perda_fora_de_faixa_falham_antes_de_qualquer_conta():
    with pytest.raises(ValueError, match="alpha"):
        crc_threshold([0.0, 1.0], [1.0, 0.0], alpha=0.0)
    with pytest.raises(ValueError, match="alpha"):
        crc_threshold([0.0, 1.0], [1.0, 0.0], alpha=1.0)
    with pytest.raises(ValueError, match="perda fora"):
        crc_threshold([0.0, 2.0], [1.0, 0.0], alpha=0.1)
    with pytest.raises(ValueError, match="vazia"):
        crc_threshold([], [], alpha=0.1)


# ---------------------------------------------------- massa sobre o span ---

def _atencao(distribuicao: list[float], n_camadas: int = 2, n_cabecas: int = 3) -> np.ndarray:
    """Tensor [camadas, cabeças, consultas, chaves] onde toda consulta distribui igual."""
    linha = np.asarray(distribuicao, dtype=float)
    assert linha.sum() == pytest.approx(1.0), "linha de atenção tem de somar 1"
    seq = linha.size
    return np.broadcast_to(linha, (n_camadas, n_cabecas, seq, seq)).copy()


def test_a_media_de_uma_linha_de_atencao_e_1_sobre_T_e_por_isso_nao_mede_nada():
    """Fixa em teste o motivo de o instrumento antigo ter sido trocado.

    `given` era a média das linhas do span. Como cada linha de atenção soma 1 por
    construção, essa média é 1/T para qualquer entrada — mede o comprimento da
    sentença. Aqui: três sequências com distribuições completamente diferentes e
    o mesmo valor.
    """
    for dist in ([0.6, 0.1, 0.2, 0.1], [0.25] * 4, [0.97, 0.01, 0.01, 0.01]):
        a = _atencao(dist)
        media_da_linha = a.mean(axis=(0, 1))[0, :].mean()
        assert media_da_linha == pytest.approx(1 / 4)


def test_massa_do_span_e_proporcao_e_a_politica_de_sumidouro_muda_o_numero():
    """O exemplo declarado no docstring do módulo, verificado.

    Sequência de 5 tokens, sumidouro em 0 com 0,60, span em 2-3 com 0,30 no
    total, resto 0,10. Mantendo o sumidouro no denominador a massa é 0,30;
    excluindo-o, 0,75. Mesmo modelo, mesma entidade, número 2,5 vezes maior.
    """
    a = _atencao([0.60, 0.05, 0.15, 0.15, 0.05])
    com = span_attention_mass(a, [2, 3], sink_policy="keep")
    sem = span_attention_mass(a, [2, 3], sink_policy="drop_from_denominator")
    assert com.mass == pytest.approx(0.30)
    assert sem.mass == pytest.approx(0.75)
    assert sem.mass / com.mass == pytest.approx(2.5)
    assert com.n_span_tokens == sem.n_span_tokens == 2
    assert sem.n_key_tokens_in_denominator == 4


def test_massa_e_comparavel_entre_spans_de_tamanhos_diferentes_na_mesma_sentenca():
    a = _atencao([0.2, 0.2, 0.2, 0.2, 0.2])
    um = span_attention_mass(a, [1], sink_policy="keep")
    dois = span_attention_mass(a, [1, 2], sink_policy="keep")
    assert um.mass == pytest.approx(0.2)
    assert dois.mass == pytest.approx(0.4)


def test_selecao_de_camadas_e_cabecas_e_respeitada():
    seq = 4
    a = np.zeros((3, 2, seq, seq))
    a[0] = 1.0 / seq                      # camada 0: uniforme
    a[1, :, :, 1] = 1.0                   # camada 1: tudo no token 1
    a[2, 0, :, 2] = 1.0                   # camada 2, cabeça 0: tudo no token 2
    a[2, 1] = 1.0 / seq
    assert span_attention_mass(a, [1], sink_policy="keep", layers=[1]).mass == pytest.approx(1.0)
    assert span_attention_mass(a, [1], sink_policy="keep", layers=[0]).mass == pytest.approx(0.25)
    assert span_attention_mass(
        a, [2], sink_policy="keep", layers=[2], heads=[0]
    ).mass == pytest.approx(1.0)


def test_span_que_cai_no_sumidouro_e_erro_de_mapeamento_nao_convencao():
    a = _atencao([0.6, 0.2, 0.2])
    with pytest.raises(ValueError, match="erro no mapeamento"):
        span_attention_mass(a, [0], sink_policy="drop_from_denominator")


def test_forma_e_indices_invalidos_falham_com_a_razao():
    a = _atencao([0.5, 0.3, 0.2])
    with pytest.raises(ValueError, match="camadas, cabeças"):
        span_attention_mass(a[0], [1], sink_policy="keep")
    with pytest.raises(ValueError, match="fora da sequência"):
        span_attention_mass(a, [7], sink_policy="keep")
    with pytest.raises(ValueError, match="span vazio"):
        span_attention_mass(a, [], sink_policy="keep")
    with pytest.raises(ValueError, match="política de sumidouro desconhecida"):
        span_attention_mass(a, [1], sink_policy="qualquer_coisa")  # type: ignore[arg-type]
