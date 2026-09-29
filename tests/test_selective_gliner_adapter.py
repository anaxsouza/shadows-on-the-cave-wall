"""O adaptador do GLiNER: as funções puras e as recusas, sem baixar pesos.

As quatro funções que decidem a medida — quais tokens formam o span, o que conta
como acerto, o que conta como aninhado, e como o texto e o ouro se alinham — são
puras e testadas aqui com offsets escritos à mão. Os caminhos internos do pacote
foram verificados rodando contra `urchade/gliner_base` (gliner 0.2.28) e estão
no docstring do módulo; o que este arquivo garante é que a mudança de um deles
falhe com mensagem que diz qual.
"""

from __future__ import annotations

import pytest

from src.selective.gliner_adapter import (
    AdapterError,
    GLiNERAdapter,
    GoldEntity,
    _casa_ouro,
    _e_aninhada,
    _texto_e_ouro,
    _tokens_do_span,
)
from src.selective.preregistration import load_preregistration

# Offsets reais do tokenizador do gliner_base para
# "Barack Obama was born in Honolulu, Hawaii.", conferidos na sondagem:
# ['[CLS]', '▁Barack', '▁Obama', '▁was', '▁born', '▁in', '▁Honolulu', ',', ...]
OFFSETS = [
    [0, 0], [0, 6], [6, 12], [12, 16], [16, 21], [21, 24], [24, 33], [33, 34],
]


def _cfg(tmp_path, layers="[6, 7]"):
    p = tmp_path / "config.yaml"
    p.write_text(
        f"""
selective:
  declaration_id: decl-teste
  sink_policy: drop_from_denominator
  layers: {layers}
  combination_rule: convex
  calibration_fraction: 0.3
  coverage_levels: [0.5, 0.8, 1.0]
  operating_point: derived_from_target_risk
  added_value_ci_level: 0.95
  n_bootstrap_resamples: 200
  target_risk_grid: [0.1]
  conformal_alpha: 0.1
""",
        encoding="utf-8",
    )
    return load_preregistration(p)


# --------------------------------------------------------------------------
# Quais tokens formam o span
# --------------------------------------------------------------------------


def test_o_span_pega_os_tokens_que_o_cobrem():
    """'Barack Obama' é [0,12) e cai nos tokens 1 e 2, não no [CLS]."""
    assert _tokens_do_span(OFFSETS, 0, 12) == [1, 2]


def test_token_especial_fica_fora_do_span():
    """[CLS] tem offset (0,0) e sobreporia qualquer span que comece em 0.

    Incluí-lo no numerador inflaria a massa com exatamente o token que a
    convenção de sumidouro existe para tratar — o efeito seria dobrado e
    invisível.
    """
    assert 0 not in _tokens_do_span(OFFSETS, 0, 12)
    assert _tokens_do_span([[0, 0], [0, 0], [0, 4]], 0, 4) == [2]


def test_span_de_um_token_e_span_fora_do_texto():
    assert _tokens_do_span(OFFSETS, 24, 33) == [6]
    assert _tokens_do_span(OFFSETS, 900, 950) == []


def test_a_sobreposicao_e_estrita_nas_bordas():
    """Um token que termina exatamente onde o span começa não pertence a ele."""
    assert _tokens_do_span([[0, 5], [5, 10]], 5, 10) == [1]


# --------------------------------------------------------------------------
# O que conta como acerto
# --------------------------------------------------------------------------


def test_acerto_exige_fronteira_E_rotulo():
    ouro = [GoldEntity(0, 12, "PER"), GoldEntity(24, 33, "LOC")]
    assert _casa_ouro(0, 12, "PER", ouro)
    assert not _casa_ouro(0, 12, "LOC", ouro), "rótulo errado não é acerto"
    assert not _casa_ouro(0, 6, "PER", ouro), "fronteira parcial não é acerto"
    assert not _casa_ouro(0, 13, "PER", ouro), "fronteira maior não é acerto"


def test_casamento_parcial_nao_conta_como_acerto():
    """Trecho com fronteira errada é entrega errada.

    Chamar sobreposição parcial de acerto infla o desempenho justamente na
    direção do resultado desejado, e a unidade da curva é o que se entrega.
    """
    ouro = [GoldEntity(0, 12, "PER")]
    assert not _casa_ouro(7, 12, "PER", ouro)


# --------------------------------------------------------------------------
# O que conta como aninhado
# --------------------------------------------------------------------------


def test_aninhado_e_estar_estritamente_dentro_de_outra():
    ouro = [GoldEntity(0, 30, "ORG"), GoldEntity(10, 20, "PER")]
    assert _e_aninhada(10, 20, ouro)
    assert not _e_aninhada(0, 30, ouro), "a externa não é aninhada em si mesma"


def test_span_identico_a_outro_nao_e_aninhado():
    """Sem a exigência de ser ESTRITAMENTE menor, toda entidade seria aninhada."""
    ouro = [GoldEntity(5, 15, "ORG"), GoldEntity(5, 15, "PER")]
    assert not _e_aninhada(5, 15, ouro)


def test_o_aninhamento_vem_da_anotacao_e_nao_da_predicao():
    """Se dependesse do previsto, o estrato mudaria com a qualidade do modelo.

    Aqui um trecho previsto que não existe no ouro ainda é marcado aninhado por
    estar dentro de uma entidade ANOTADA — o estrato é do corpus, não do modelo,
    e é isso que torna as duas execuções comparáveis.
    """
    ouro = [GoldEntity(0, 40, "ORG")]
    assert _e_aninhada(12, 18, ouro)


# --------------------------------------------------------------------------
# Texto e ouro alinhados na MESMA string
# --------------------------------------------------------------------------


def test_o_ouro_e_lido_e_o_aninhamento_marcado():
    texto, ouro = _texto_e_ouro(
        {
            "text": "A B C D",
            "entities": [
                {"start": 0, "end": 7, "label": "ORG"},
                {"start": 2, "end": 3, "label": "PER"},
            ],
        }
    )
    assert texto == "A B C D"
    assert [g.is_nested for g in ouro] == [False, True]


def test_sem_texto_o_texto_vem_dos_tokens():
    texto, _ = _texto_e_ouro({"tokens": ["Barack", "Obama", "was"], "entities": []})
    assert texto == "Barack Obama was"


def test_exemplo_sem_texto_e_sem_tokens_e_recusado():
    with pytest.raises(AdapterError, match="nada a medir"):
        _texto_e_ouro({"entities": []})


def test_entidade_de_ouro_incompleta_e_recusada():
    with pytest.raises(AdapterError, match="sem start/end/label"):
        _texto_e_ouro({"text": "x", "entities": [{"start": 0, "end": 1}]})


# --------------------------------------------------------------------------
# As recusas na construção, que são o valor deste módulo
# --------------------------------------------------------------------------


class _TransformerFalso:
    class config:  # noqa: N801
        num_hidden_layers = 12
        num_attention_heads = 12


class _ModeloFalso:
    """Espelha os caminhos internos verificados em gliner 0.2.28."""

    def __init__(self, com_tokenizer: bool = True, com_transformer: bool = True):
        class _Bert:
            model = _TransformerFalso()

        class _TokenRep:
            bert_layer = _Bert()

        class _Model:
            token_rep_layer = _TokenRep()

        class _DP:
            transformer_tokenizer = object()

        if com_transformer:
            self.model = _Model()
        if com_tokenizer:
            self.data_processor = _DP()


def test_faixa_pre_registrada_fora_do_modelo_e_recusada(tmp_path):
    """A recusa central: 12 camadas não contêm a faixa 8-15.

    Truncar por conveniência mudaria em silêncio quais camadas produziram a
    medida — o número sairia, pareceria certo, e seria de outras camadas.
    """
    with pytest.raises(AdapterError, match="não existe neste modelo"):
        GLiNERAdapter(
            _ModeloFalso(),
            prereg=_cfg(tmp_path, layers="[8, 15]"),
            labels=["person"],
            label_to_corpus={"person": "PER"},
        )


def test_faixa_valida_e_aceita(tmp_path):
    a = GLiNERAdapter(
        _ModeloFalso(),
        prereg=_cfg(tmp_path, layers="[6, 7]"),
        labels=["person"],
        label_to_corpus={"person": "PER"},
    )
    assert (a.n_layers, a.n_heads) == (12, 12)


def test_caminho_interno_que_muda_falha_dizendo_qual(tmp_path):
    with pytest.raises(AdapterError, match="data_processor.transformer_tokenizer"):
        GLiNERAdapter(
            _ModeloFalso(com_tokenizer=False),
            prereg=_cfg(tmp_path),
            labels=["person"],
            label_to_corpus={"person": "PER"},
        )
    with pytest.raises(AdapterError, match="token_rep_layer"):
        GLiNERAdapter(
            _ModeloFalso(com_transformer=False),
            prereg=_cfg(tmp_path),
            labels=["person"],
            label_to_corpus={"person": "PER"},
        )


def test_mapa_de_rotulos_incoerente_e_recusado(tmp_path):
    """Rótulo no mapa que não é dado ao modelo nunca casaria com predição."""
    with pytest.raises(AdapterError, match="não são dadas ao modelo"):
        GLiNERAdapter(
            _ModeloFalso(),
            prereg=_cfg(tmp_path),
            labels=["person"],
            label_to_corpus={"organization": "ORG"},
        )
