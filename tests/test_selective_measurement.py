"""A passagem de medição, sem modelo, sem rede e sem GPU.

O adaptador `predict` existe para isto: o modelo fica atrás dele, e o teste
passa uma função determinística. Não é abstração por gosto — é a diferença
entre um teste de milissegundos e um que baixa 1,3 GB de pesos.

O que se verifica aqui é o encanamento e as recusas, não a massa de atenção
(essa é `test_selective_conformal.py`) nem a curva (`test_selective_risk_coverage.py`).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from src.selective.measurement import (
    COLUNAS,
    MeasurementError,
    estimate_disk,
    measure,
)
from src.selective.preregistration import load_preregistration

N_LAYERS, N_HEADS = 4, 2

CONFIG = """
selective:
  declaration_id: decl-teste
  sink_policy: drop_from_denominator
  layers: [1, 2]
  combination_rule: convex
  calibration_fraction: 0.3
  coverage_levels: [0.5, 0.8, 1.0]
  operating_point: derived_from_target_risk
  added_value_ci_level: 0.95
  n_bootstrap_resamples: 200
  target_risk_grid: [0.1]
  conformal_alpha: 0.1
"""


def _atencao(T: int, semente: int) -> tuple:
    """Uma tupla por camada de tensores [1, cabeças, T, T] com linhas somando 1."""
    rng = np.random.default_rng(semente)
    saida = []
    for _ in range(N_LAYERS):
        m = rng.random((1, N_HEADS, T, T)) + 0.05
        m /= m.sum(axis=-1, keepdims=True)
        saida.append(m)
    return tuple(saida)


def fabrica_predict(n_ent: int = 2, T: int = 6, sem_atencao: bool = False, lote: int = 1):
    """Adaptador determinístico: metade das entidades erradas, uma aninhada."""

    def predict(sent, tokens_only: bool = False):
        i = int(str(sent).lstrip("s") or 0)
        if tokens_only:
            return {"tokens": ["t"] * T, "n_layers": N_LAYERS, "n_heads": N_HEADS}
        attn = () if sem_atencao else _atencao(T, semente=i)
        if lote != 1 and not sem_atencao:
            attn = tuple(np.repeat(a, lote, axis=0) for a in attn)
        preditas = [
            {
                "token_indices": [1 + k],
                "confidence": 0.9 - 0.1 * k,
                "correct": (i + k) % 2 == 0,
                "is_nested": k == 1,
            }
            for k in range(n_ent)
        ]
        return {"tokens": ["t"] * T, "attentions": attn, "predicted": preditas, "n_gold": n_ent + 1}

    return predict


@pytest.fixture()
def prereg(tmp_path: Path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(CONFIG, encoding="utf-8")
    return load_preregistration(cfg)


# --------------------------------------------------------------------------
# A estimativa de disco
# --------------------------------------------------------------------------


def test_a_estimativa_usa_a_media_dos_quadrados_e_nao_o_quadrado_da_media():
    """O custo é quadrático no comprimento, e confundir os dois subestima.

    Duas sentenças de 10 e 30 tokens custam 10² + 30² = 1000, não 2 x 20² = 800.
    Subestimar disco é o erro que interrompe uma execução de horas no meio, e é
    por isso que este teste existe com números escolhidos para separar as duas
    contas.
    """
    e = estimate_disk([10, 30], n_layers=1, n_heads=1, max_disk_gb=1.0)
    assert e.projected_bytes == (10 * 10 + 30 * 30) * 2
    assert e.projected_bytes != 2 * 20 * 20 * 2
    assert e.mean_tokens == 20.0 and e.max_tokens == 30


def test_a_estimativa_escala_com_camadas_e_cabecas():
    um = estimate_disk([8] * 3, n_layers=1, n_heads=1, max_disk_gb=1.0)
    muitos = estimate_disk([8] * 3, n_layers=24, n_heads=16, max_disk_gb=1.0)
    assert muitos.projected_bytes == um.projected_bytes * 24 * 16


def test_estimativa_recusa_entrada_impossivel():
    with pytest.raises(MeasurementError, match="nada a estimar"):
        estimate_disk([], 1, 1, 1.0)
    with pytest.raises(MeasurementError, match="não positivo"):
        estimate_disk([4, 0], 1, 1, 1.0)


# --------------------------------------------------------------------------
# A recusa por orçamento, que é o ponto do pré-voo
# --------------------------------------------------------------------------


def test_orcamento_estourado_recusa_antes_de_escrever_qualquer_byte(prereg, tmp_path: Path):
    saida = tmp_path / "out"
    with pytest.raises(MeasurementError, match="NÃO CABE"):
        measure(
            sentences=[f"s{i}" for i in range(50)],
            predict=fabrica_predict(),
            prereg=prereg,
            model_id="m-teste",
            output_dir=saida,
            max_disk_gb=1e-9,
        )
    assert not (saida / "entities.csv").exists(), "escreveu tabela apesar de recusar"
    assert not (saida / "attention").exists(), "criou diretório de atenção apesar de recusar"


# --------------------------------------------------------------------------
# A tabela emitida
# --------------------------------------------------------------------------


def test_a_tabela_tem_o_contrato_que_o_runner_consome(prereg, tmp_path: Path):
    rel = measure(
        sentences=[f"s{i}" for i in range(8)],
        predict=fabrica_predict(n_ent=2),
        prereg=prereg,
            model_id="m-teste",
        output_dir=tmp_path / "out",
        shard_size=3,
    )
    linhas = list(csv.DictReader((tmp_path / "out" / "entities.csv").open(encoding="utf-8")))
    assert tuple(linhas[0].keys()) == COLUNAS, "a ordem das colunas é contrato"
    # token_indices existe para a análise de perímetro: sem ela o tensor guardado
    # não permite recompor a massa sob outra leitura, e guardar tudo perde o
    # propósito. Formato "3|4|5".
    idx = [int(v) for v in linhas[0]["token_indices"].split("|")]
    assert idx and all(i >= 0 for i in idx)
    assert len(linhas) == 16 == rel.n_predicted
    assert {l["loss"] for l in linhas} == {"0", "1"}
    assert all(0.0 <= float(l["span_mass"]) <= 1.0 for l in linhas)
    assert {l["is_nested"] for l in linhas} == {"0", "1"}


def test_a_taxa_de_erro_base_e_medida_e_nao_afirmada(prereg, tmp_path: Path):
    """É a horizontal do acaso da curva risco-cobertura; errá-la move o veredito."""
    rel = measure(
        sentences=[f"s{i}" for i in range(10)],
        predict=fabrica_predict(n_ent=2),
        prereg=prereg,
            model_id="m-teste",
        output_dir=tmp_path / "out",
    )
    linhas = list(csv.DictReader((tmp_path / "out" / "entities.csv").open(encoding="utf-8")))
    erradas = sum(int(l["loss"]) for l in linhas)
    assert rel.error_rate == pytest.approx(erradas / len(linhas))


def test_o_recall_nao_entra_na_curva_mas_e_reportado(prereg, tmp_path: Path):
    """Falso negativo não é candidato a abstenção: não se abstém do que não se produziu."""
    rel = measure(
        sentences=[f"s{i}" for i in range(4)],
        predict=fabrica_predict(n_ent=2),
        prereg=prereg,
            model_id="m-teste",
        output_dir=tmp_path / "out",
    )
    assert rel.n_gold == 12 and rel.n_predicted == 8
    assert "não entra na curva" in rel.render()


# --------------------------------------------------------------------------
# A atenção persistida e a procedência
# --------------------------------------------------------------------------


def test_a_atencao_e_persistida_completa_em_float16(prereg, tmp_path: Path):
    rel = measure(
        sentences=[f"s{i}" for i in range(7)],
        predict=fabrica_predict(T=6),
        prereg=prereg,
            model_id="m-teste",
        output_dir=tmp_path / "out",
        shard_size=3,
    )
    fatias = sorted((tmp_path / "out" / "attention").glob("shard_*.npz"))
    assert len(fatias) == rel.n_shards == 3, "7 sentenças em fatias de 3 dão 3 arquivos"
    with np.load(fatias[0]) as z:
        chaves = list(z.keys())
        a = z[chaves[0]]
    assert len(chaves) == 3
    assert a.shape == (N_LAYERS, N_HEADS, 6, 6), "todas as camadas e cabeças, decisão de 02/09"
    assert a.dtype == np.float16


def test_a_procedencia_grava_a_faixa_que_produziu_a_medida(prereg, tmp_path: Path):
    """Tabela medida com uma faixa e citada como outra é erro invisível no CSV."""
    measure(
        sentences=[f"s{i}" for i in range(3)],
        predict=fabrica_predict(),
        prereg=prereg,
            model_id="m-teste",
        output_dir=tmp_path / "out",
    )
    m = json.loads((tmp_path / "out" / "MEDIDA.json").read_text(encoding="utf-8"))
    assert m["preregistro"]["layers"] == [1, 2]
    assert m["preregistro"]["sink_policy"] == "drop_from_denominator"
    assert m["atencao"] == {
        "camadas": N_LAYERS,
        "cabecas": N_HEADS,
        "dtype": "float16",
        "fatias": m["atencao"]["fatias"],
        "bytes": m["atencao"]["bytes"],
    }
    assert m["colunas"] == list(COLUNAS)


def test_a_faixa_de_camadas_vem_do_preregistro_e_nao_de_argumento():
    """A fricção que o tensor completo remove volta por aqui.

    `measure` não aceita `layers`: quem quiser outra faixa edita o
    config.yaml, que é mudança versionada e visível. Guardar todas as camadas
    permite reescolher depois de ver o resultado, e a faixa é o item 2 do
    pré-registro — este teste é o que impede o atalho.
    """
    import inspect

    params = inspect.signature(measure).parameters
    for proibido in ("layers", "heads", "sink_policy"):
        assert proibido not in params, (
            f"measure() aceita {proibido!r} como argumento, o que permite medir com faixa "
            f"diferente da declarada sem tocar no pré-registro"
        )
    assert "prereg" in params


# --------------------------------------------------------------------------
# As recusas
# --------------------------------------------------------------------------


def test_modelo_sem_atencao_falha_alto(prereg, tmp_path: Path):
    with pytest.raises(MeasurementError, match="output_attentions=True"):
        measure(
            sentences=["s0"],
            predict=fabrica_predict(sem_atencao=True),
            prereg=prereg,
            model_id="m-teste",
            output_dir=tmp_path / "out",
        )


def test_lote_maior_que_um_e_recusado(prereg, tmp_path: Path):
    """Preenchimento colocaria massa sobre token de preenchimento.

    Isso contaminaria o denominador da proporção — e contaminar o denominador
    custa o resultado, enquanto perder o lote custa tempo.
    """
    with pytest.raises(MeasurementError, match="lote"):
        measure(
            sentences=["s0"],
            predict=fabrica_predict(lote=4),
            prereg=prereg,
            model_id="m-teste",
            output_dir=tmp_path / "out",
        )


def test_nenhuma_entidade_predita_e_recusa_e_nao_tabela_vazia(prereg, tmp_path: Path):
    with pytest.raises(MeasurementError, match="nenhuma entidade predita"):
        measure(
            sentences=["s0", "s1"],
            predict=fabrica_predict(n_ent=0),
            prereg=prereg,
            model_id="m-teste",
            output_dir=tmp_path / "out",
        )


def test_entidade_sem_token_mapeado_e_ressalva_e_nao_silencio(prereg, tmp_path: Path):
    def predict(sent, tokens_only: bool = False):
        base = fabrica_predict(n_ent=1)(sent, tokens_only=tokens_only)
        if tokens_only:
            return base
        base["predicted"] = [
            {"token_indices": [], "confidence": 0.9, "correct": True, "is_nested": False},
            {"token_indices": [2], "confidence": 0.8, "correct": False, "is_nested": False},
        ]
        return base

    rel = measure(
        sentences=["s0", "s1"],
        predict=predict,
        prereg=prereg,
            model_id="m-teste",
        output_dir=tmp_path / "out",
    )
    assert rel.n_predicted == 2, "só as mapeadas entram"
    assert any("sem token mapeado" in n for n in rel.notes)
    assert "Ressalvas desta execução" in rel.render()


def test_model_id_e_obrigatorio_e_nao_pode_ganhar_valor_padrao(prereg, tmp_path):
    """Sem identidade de modelo, duas escalas produzem tabelas indistinguíveis.

    Até 15/09/2026 o MEDIDA.json não registrava o modelo: o único traço era o
    nome do diretório. Com um modelo só isso não machucava; com dois, uma tabela
    do large carregaria o mesmo measurement_hash de uma do base e o guarda de
    procedência aceitaria cruzá-las. Um valor padrão aqui reabriria o buraco em
    silêncio, então a ausência do argumento tem de ser erro.
    """
    import inspect

    from src.selective.measurement import measure

    par = inspect.signature(measure).parameters["model_id"]
    assert par.default is inspect.Parameter.empty, (
        "model_id ganhou valor padrão: medição sem identidade de modelo volta a ser possível"
    )
    assert par.kind is inspect.Parameter.KEYWORD_ONLY


def test_o_registro_grava_o_modelo_que_produziu_a_tabela(prereg, tmp_path):
    """O bloco `modelo` do MEDIDA.json, com o que o modelo TEM de fato."""
    import json

    saida = tmp_path / "out"
    measure(sentences=["s0", "s1", "s2"], predict=fabrica_predict(n_ent=1), prereg=prereg,
            output_dir=saida, model_id="fulano/modelo-x")
    reg = json.loads((saida / "MEDIDA.json").read_text(encoding="utf-8"))
    assert reg["modelo"]["id"] == "fulano/modelo-x"
    assert reg["modelo"]["camadas"] == reg["atencao"]["camadas"]
    assert reg["modelo"]["cabecas"] == reg["atencao"]["cabecas"]


# =============================================================================
# Os SINAIS das três famílias, medidos na passagem
# =============================================================================
# Estes testes existem porque `_sinais_do_span` calcula dezoito colunas numa
# função só, e um erro de índice ali sai como número plausível — não como erro.
# O que se fixa aqui são as propriedades que um número plausível errado violaria.


def _predict_com_sinais(n_ent=2, T=12, n_rot=5, seed=0, L=2, H=2):
    """Adaptador sintético que devolve atenção, estados ocultos e escores.

    Determinístico e com PLANTAS conhecidas: a atenção é uniforme nas linhas, o
    que fixa a entropia em log|K| exato e o máximo em 1/|K|; os estados ocultos
    têm norma crescente com o índice do token, o que dá ordem conhecida.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    d = 8

    def predict(sent, tokens_only: bool = False):
        if tokens_only:
            return {"tokens": ["t"] * T, "n_layers": L, "n_heads": H}
        uni = np.full((T, T), 1.0 / T, dtype=np.float32)
        attn = tuple(np.broadcast_to(uni, (1, H, T, T)).copy() for _ in range(L))
        # L+1 tensores, como o transformers devolve
        oc = []
        for c in range(L + 1):
            base = np.arange(1, T + 1, dtype=np.float32)[:, None] * np.ones((1, d), np.float32)
            oc.append((base * (c + 1))[None, :, :])
        preditas = []
        for k in range(n_ent):
            preditas.append({
                "token_indices": [2 + k, 3 + k],
                "confidence": 0.9 - 0.1 * k,
                "correct": k == 0,
                "is_nested": False,
                "label_scores": sorted(rng.random(n_rot).tolist(), reverse=True),
            })
        return {"tokens": ["t"] * T, "attentions": attn, "hidden_states": tuple(oc),
                "predicted": preditas, "n_gold": n_ent}

    return predict


def _medir_com_sinais(tmp_path, prereg, **kw):
    # A forma do tensor vem da DECLARAÇÃO e não de constante no teste: a faixa de
    # camadas declarada é o que a medição vai indexar, e um sintético menor que
    # ela falha por índice fora de faixa em vez de testar o que se quer.
    kw.setdefault("L", max(prereg.layers) + 1)
    kw.setdefault("H", (max(prereg.heads) + 1) if prereg.heads else 2)
    return measure(sentences=[f"s{i}" for i in range(6)],
                   predict=_predict_com_sinais(**kw), prereg=prereg,
                   model_id=prereg.model, output_dir=tmp_path / "out")


def test_as_dezoito_colunas_saem_no_csv_na_ordem_declarada(prereg, tmp_path):
    _medir_com_sinais(tmp_path, prereg)
    cab = (tmp_path / "out" / "entities.csv").read_text(encoding="utf-8").splitlines()[0]
    assert cab.strip().split(",") == list(COLUNAS)


def test_entropia_da_linha_bate_o_nulo_EXATO_sob_atencao_uniforme(prereg, tmp_path):
    """Atenção uniforme tem de dar entropia log|K| e máximo 1/|K|, exatos.

    É a verificação de que a conta na passagem de medição é a MESMA que a
    derivação de `signals.py`. Se o denominador da política de sumidouro
    estivesse errado aqui, o número sairia plausível e errado.
    """
    import csv as _csv
    import numpy as np
    from src.selective.signals import expected_row_entropy, expected_row_max

    _medir_com_sinais(tmp_path, prereg, T=12)
    linhas = list(_csv.DictReader((tmp_path / "out" / "entities.csv").open(encoding="utf-8")))
    esp_h = expected_row_entropy(12, prereg.sink_policy)
    esp_m = expected_row_max(12, prereg.sink_policy)
    for l in linhas:
        assert float(l["row_entropy"]) == pytest.approx(esp_h, abs=1e-4), l["row_entropy"]
        assert float(l["row_max"]) == pytest.approx(esp_m, abs=1e-4), l["row_max"]


def test_as_covariaveis_geometricas_sao_o_que_dizem(prereg, tmp_path):
    import csv as _csv

    _medir_com_sinais(tmp_path, prereg, T=12, n_ent=2)
    linhas = list(_csv.DictReader((tmp_path / "out" / "entities.csv").open(encoding="utf-8")))
    for l in linhas:
        toks = [int(v) for v in l["token_indices"].split("|")]
        assert int(l["span_size"]) == len(toks)
        assert int(l["sentence_length"]) == 12
        assert int(l["span_position"]) == min(toks), "posição tem de ser o PRIMEIRO token"


def test_margem_e_entropia_dos_logits_sao_coerentes(prereg, tmp_path):
    import csv as _csv

    _medir_com_sinais(tmp_path, prereg, n_rot=5)
    linhas = list(_csv.DictReader((tmp_path / "out" / "entities.csv").open(encoding="utf-8")))
    for l in linhas:
        assert float(l["logit_margin"]) >= 0.0, "margem é o maior menos o segundo, nunca negativa"
        assert 0.0 <= float(l["logit_entropy"]) <= 1.0, "entropia normalizada vive em [0, 1]"
        assert float(l["logit_max"]) >= float(l["logit_margin"])


def test_sem_estados_ocultos_a_familia_sai_NaN_com_RESSALVA(prereg, tmp_path):
    """Adaptador antigo não deve derrubar a medição, mas deve DECLARAR a falta."""
    import csv as _csv
    import math

    base = _predict_com_sinais(L=max(prereg.layers) + 1,
                               H=(max(prereg.heads) + 1) if prereg.heads else 2)

    def predict(sent, tokens_only: bool = False):
        r = base(sent, tokens_only=tokens_only)
        if not tokens_only:
            r.pop("hidden_states", None)
        return r

    rel = measure(sentences=["s0", "s1"], predict=predict, prereg=prereg,
                  model_id=prereg.model, output_dir=tmp_path / "out")
    linhas = list(_csv.DictReader((tmp_path / "out" / "entities.csv").open(encoding="utf-8")))
    for l in linhas:
        assert math.isnan(float(l["hidden_norm"]))
    assert any("SEM ESTADOS OCULTOS" in n.upper() for n in rel.notes), \
        "a falta tem de ser declarada, não silenciosa"


def test_a_ressalva_dos_estados_ocultos_aparece_UMA_vez(prereg, tmp_path):
    """Uma ressalva por sentença encheria o relatório e esconderia as outras."""
    base = _predict_com_sinais(L=max(prereg.layers) + 1,
                               H=(max(prereg.heads) + 1) if prereg.heads else 2)

    def predict(sent, tokens_only: bool = False):
        r = base(sent, tokens_only=tokens_only)
        if not tokens_only:
            r.pop("hidden_states", None)
        return r

    rel = measure(sentences=[f"s{i}" for i in range(8)], predict=predict, prereg=prereg,
                  model_id=prereg.model, output_dir=tmp_path / "out")
    assert sum("SEM ESTADOS OCULTOS" in n.upper() for n in rel.notes) == 1


def test_norma_dos_estados_ocultos_cresce_com_o_indice_do_token(prereg, tmp_path):
    """O sintético planta norma crescente no índice; a medida tem de refletir.

    Se a função indexasse a dimensão errada do tensor, a norma sairia igual para
    todas as entidades — plausível e errada.
    """
    import csv as _csv

    _medir_com_sinais(tmp_path, prereg, n_ent=3, T=14)
    linhas = list(_csv.DictReader((tmp_path / "out" / "entities.csv").open(encoding="utf-8")))
    por_pos = {}
    for l in linhas:
        por_pos[int(l["span_position"])] = float(l["hidden_norm"])
    posicoes = sorted(por_pos)
    valores = [por_pos[p] for p in posicoes]
    assert valores == sorted(valores), f"norma não cresce com a posição: {list(zip(posicoes, valores))}"
    assert len(set(valores)) > 1, "todas as normas iguais: a indexação do tensor está errada"
