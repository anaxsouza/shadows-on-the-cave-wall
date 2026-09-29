"""As comparações declaradas, sobre tabela sintética.

Sintética e não medida: aqui se verifica que o módulo executa o que a declaração
manda — e SÓ o que ela manda —, não que a atenção funcione. A tabela é construída
com sinal PLANTADO num escore e ruído no outro; se o veredito não separar os dois,
o defeito é do módulo.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from src.selective.comparisons import (
    ROTULOS_DE_RELATO,
    ComparisonsError,
    run_declared_comparisons,
    sentence_lengths,
)
from src.selective.preregistration import load_preregistration

CONFIG = """
selective:
  declaration_id: 'd-teste'
  hash_version: 2
  sink_policy: 'drop_from_denominator'
  layers: [0, 1]
  heads: null
  combination_rule: 'convex'
  calibration_fraction: 0.3
  coverage_levels: [0.5, 1.0]
  operating_point: 'derived_from_target_risk'
  added_value_ci_level: 0.95
  n_bootstrap_resamples: 200
  target_risk_grid: [0.2]
  conformal_alpha: 0.2
  loss_bound: 1.0
  seed: 11
  comparison_scores: [span_size, geometric_fraction, enrichment, geometric_residual, sentence_length]
  stratify_by: [span_size, is_nested]
  span_size_bins: [[1, 1], [2, null]]
  comparisons:
    - id: 'C1'
      kind: 'descriptive'
      question: 'q1'
    - id: 'C2'
      kind: 'verdict'
      question: 'q2'
      score: 'span_mass'
      against: 'geometric_fraction'
      criterion: 'paired_delta_aurc_ci_excludes_zero'
    - id: 'C3'
      kind: 'verdict'
      question: 'q3'
      score: 'model_confidence+enrichment'
      against: 'model_confidence'
      criterion: 'paired_delta_aurc_ci_excludes_zero'
"""


def _montar(tmp_path: Path, n_sent=120, com_indices=True, sentenca_sem_matriz=False) -> Path:
    """Tabela + fatias de atenção. `span_mass` carrega o sinal; a confiança não."""
    rng = np.random.default_rng(3)
    d = tmp_path / "tabela"
    (d / "attention").mkdir(parents=True)
    linhas, matrizes = [], {}
    for i in range(n_sent):
        sid = f"s{i}"
        T = int(rng.integers(12, 40))
        matrizes[sid] = np.zeros((1, 1, T, T), dtype=np.float16)
        for j in range(int(rng.integers(1, 4))):
            k = int(rng.integers(1, 5))
            erro = int(rng.random() < 0.5)
            # SINAL PLANTADO, na direção da convenção: escore ALTO significa
            # ENTREGAR, então um escore útil é BAIXO quando a entidade está
            # errada. A parte geométrica fica deliberadamente presente, para o
            # instrumento ter de vencê-la e não apenas herdá-la.
            massa = float(np.clip(0.5 * k / (T - 1) - 0.05 * erro + rng.normal(0, 0.004), 1e-4, 1))
            linhas.append({
                "sentence_id": sid, "loss": erro,
                "model_confidence": round(float(rng.random()), 5),
                # 0/1 e não True/False: é o que a passagem de medição grava, e
                # aceitar as duas formas esconderia tabela de outra procedência.
                "span_mass": round(massa, 6), "is_nested": int(j > 0),
                "token_indices": "|".join(str(x) for x in range(1, 1 + k)),
            })
    if sentenca_sem_matriz:
        linhas.append({**linhas[0], "sentence_id": "fantasma"})
    campos = list(linhas[0])
    if not com_indices:
        campos.remove("token_indices")
    with (d / "entities.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=campos, extrasaction="ignore")
        w.writeheader(); w.writerows(linhas)
    np.savez_compressed(d / "attention" / "shard_000.npz", **matrizes)
    return d


@pytest.fixture
def prereg(tmp_path):
    cfg = tmp_path / "d.yaml"
    cfg.write_text(CONFIG, encoding="utf-8")
    return load_preregistration(str(cfg))


def test_le_o_comprimento_pelo_cabecalho_sem_descompactar(tmp_path):
    d = _montar(tmp_path, n_sent=20)
    L = sentence_lengths(d / "attention")
    assert len(L) == 20
    assert all(12 <= v < 40 for v in L.values())


def test_executa_exatamente_as_comparacoes_declaradas(tmp_path, prereg):
    rel = run_declared_comparisons(_montar(tmp_path), prereg, "sintetico")
    declaradas = {c.id for c in prereg.comparisons}
    executadas = {l["comparacao"] for l in rel.linhas} - set(ROTULOS_DE_RELATO)
    assert executadas == declaradas, (
        f"declaradas {declaradas}, executadas {executadas} — o módulo não pode "
        f"inventar comparação nem deixar de fazer uma declarada"
    )


def test_a_descritiva_nao_produz_veredito(tmp_path, prereg):
    rel = run_declared_comparisons(_montar(tmp_path), prereg, "sintetico")
    c1 = [l for l in rel.linhas if l["comparacao"] == "C1"]
    assert c1, "C1 não produziu linha nenhuma"
    assert all("veredito" not in l for l in c1)
    assert {l["metrica"] for l in c1} >= {
        "R2_massa_vs_fracao_geometrica", "mediana_observado_sobre_esperado"}


def test_o_veredito_usa_reamostragem_por_sentenca(tmp_path, prereg):
    rel = run_declared_comparisons(_montar(tmp_path), prereg, "sintetico")
    todos = [l for l in rel.linhas
             if l["comparacao"] == "C2" and l["estrato"] == "todos" and "delta" in l]
    assert len(todos) == 1
    l = todos[0]
    assert l["n_sentencas"] < l["n_entidades"], (
        "sentenças e entidades iguais: a reamostragem não está por conglomerado"
    )


def test_detecta_o_sinal_plantado_sobre_a_geometria(tmp_path, prereg):
    """A massa tem sinal ALÉM da geometria por construção, e C2 tem de vê-lo."""
    rel = run_declared_comparisons(_montar(tmp_path, n_sent=200), prereg, "sintetico")
    l = [x for x in rel.linhas
         if x["comparacao"] == "C2" and x["estrato"] == "todos" and "delta" in x][0]
    assert l["delta"] < 0, f"ΔAURC {l['delta']} não é negativo: o sinal plantado não foi visto"
    assert l["veredito"] == "ACRESCENTA"


def test_estratifica_pelas_faixas_declaradas_e_por_aninhamento(tmp_path, prereg):
    rel = run_declared_comparisons(_montar(tmp_path, n_sent=200), prereg, "sintetico")
    estratos = {l["estrato"] for l in rel.linhas if l["comparacao"] == "C2"}
    assert {"k=1", "k>=2"} <= estratos, f"faixas declaradas ausentes: {estratos}"
    assert {"aninhado", "plano"} <= estratos


def test_relata_a_correlacao_de_cada_escore_com_o_comprimento(tmp_path, prereg):
    """O controle que Saghir (2026) recomenda: correlação relatada, não suposta."""
    rel = run_declared_comparisons(_montar(tmp_path), prereg, "sintetico")
    controles = {l["metrica"] for l in rel.linhas if l["comparacao"] == "controle"}
    for nome in prereg.comparison_scores + ("span_mass", "model_confidence"):
        assert f"corr_{nome}_vs_comprimento" in controles


def test_tabela_sem_token_indices_e_recusada(tmp_path, prereg):
    d = _montar(tmp_path, com_indices=False)
    with pytest.raises(ComparisonsError, match="token_indices"):
        run_declared_comparisons(d, prereg, "sintetico")


def test_sentenca_sem_matriz_de_atencao_e_recusada(tmp_path, prereg):
    """Tabela e fatias de execuções diferentes: erro que sairia como número."""
    d = _montar(tmp_path, sentenca_sem_matriz=True)
    with pytest.raises(ComparisonsError, match="não têm matriz"):
        run_declared_comparisons(d, prereg, "sintetico")


def test_a_combinacao_e_ajustada_na_calibracao_e_o_peso_e_reportado(tmp_path, prereg):
    rel = run_declared_comparisons(_montar(tmp_path), prereg, "sintetico")
    assert any("ajustado na calibração" in n for n in rel.notas)
    assert any("COLAPSANDO" in n for n in rel.notas), (
        "a ressalva do colapso tem de viajar com o peso: é o que distingue "
        "'descartou o sinal' de 'empatou'"
    )


# =============================================================================
# Roteamento por tipo: cada comparação é julgada por UM runner, e pelo seu
# critério
# =============================================================================
# A decl-04 é a primeira declaração a trazer os dois tipos no mesmo arquivo, e
# foi ela que expôs o defeito: comparisons.py tratava tudo que não era
# `descriptive` como veredito de ΔAURC, então imprimia veredito de AURC para uma
# comparação cujo critério declarado é carga de revisão. Quantidade não declarada
# com etiqueta de veredito é o pior defeito possível aqui.

def test_cada_comparacao_declarada_e_julgada_por_exatamente_um_runner():
    import yaml
    from pathlib import Path

    from src.selective.preregistration import load_preregistration

    raiz = Path(__file__).resolve().parents[1]
    for fonte in sorted((raiz / "configs").glob("decl-*.yaml")):
        p = load_preregistration(str(fonte))
        for c in p.comparisons:
            geometria = c.kind in ("descriptive", "verdict")
            tarefa = c.kind == "task_verdict"
            assert geometria != tarefa, (
                f"{fonte.name}: {c.id} tem kind={c.kind!r}, que nenhum runner reivindica "
                f"ou que os dois reivindicam"
            )
            if tarefa:
                assert "review_load" in (c.criterion or ""), (
                    f"{fonte.name}: {c.id} é task_verdict mas o critério "
                    f"{c.criterion!r} não é de carga de revisão"
                )
            elif c.kind == "verdict":
                assert "aurc" in (c.criterion or "").lower(), (
                    f"{fonte.name}: {c.id} é verdict mas o critério {c.criterion!r} "
                    f"não é de AURC"
                )


def test_o_runner_de_geometria_omite_as_comparacoes_de_tarefa(tmp_path):
    """A omissão é DECLARADA numa ressalva, não silenciosa."""
    from pathlib import Path

    from src.selective.comparisons import run_declared_comparisons
    from src.selective.preregistration import load_preregistration

    raiz = Path(__file__).resolve().parents[1]
    fonte = raiz / "configs" / "decl-04-escala.yaml"
    tabela = raiz / "results" / "gliner-large" / "genia" / "test"
    if not (tabela / "entities.csv").is_file():
        import pytest

        pytest.skip("tabela do gliner-large não medida neste checkout")

    p = load_preregistration(str(fonte))
    rel = run_declared_comparisons(tabela, p, "genia")
    julgadas = {l["comparacao"] for l in rel.linhas}
    tarefa = {c.id for c in p.comparisons if c.kind == "task_verdict"}
    assert not (julgadas & tarefa), f"o runner de geometria julgou {julgadas & tarefa}"
    for cid in tarefa:
        assert any(cid in n and "não é julgada aqui" in n for n in rel.notas), (
            f"{cid} foi omitida sem ressalva declarada"
        )


# =============================================================================
# ORIENTAÇÃO do escore: o invariante que substitui testar as duas
# =============================================================================
# A curva entrega por escore decrescente. Um sinal em que ALTO significa
# "revise isto" sai pior que o acaso por razão trivial. Testar as duas
# orientações e reportar a melhor dobraria as chances ao acaso — então a
# orientação é decidida pela correlação com acerto na CALIBRAÇÃO, nunca na
# avaliação. Estes testes fixam que é isso que acontece.


def _particoes_com_sinal(n=400, seed=1):
    """Duas partições com dois sinais plantados em orientações OPOSTAS."""
    import numpy as np

    rng = np.random.default_rng(seed)
    loss = (rng.random(n) < 0.4).astype(float)
    acerto = 1.0 - loss
    bom = acerto + rng.normal(0, 0.3, n)        # alto = acerto  -> +1
    ruim = -acerto + rng.normal(0, 0.3, n)      # alto = erro    -> -1
    t = {"loss": loss, "bom": bom, "ruim": ruim,
         "model_confidence": acerto + rng.normal(0, 0.2, n),
         "logit_max": acerto + rng.normal(0, 0.2, n),
         "constante": np.full(n, 0.7)}
    return t, t


def test_orientacao_positiva_quando_alto_significa_acerto():
    from src.selective.comparisons import _orientar

    cal, aval = _particoes_com_sinal()
    notas = []
    x = _orientar("bom", cal, aval, notas)
    assert np.allclose(x, aval["bom"]), "sinal já na orientação certa não deve ser invertido"
    assert any("orientação +1" in n for n in notas)


def test_orientacao_INVERTIDA_quando_alto_significa_erro():
    from src.selective.comparisons import _orientar

    cal, aval = _particoes_com_sinal()
    notas = []
    x = _orientar("ruim", cal, aval, notas)
    assert np.allclose(x, -aval["ruim"]), "sinal anticorrelacionado tem de ser invertido"
    assert any("orientação -1" in n for n in notas)


def test_a_orientacao_e_DECIDIDA_na_calibracao_e_nao_na_avaliacao():
    """A prova de que a avaliação não entra: mudá-la não muda a orientação.

    Se a orientação viesse da avaliação, inverter o sinal lá mudaria a decisão —
    e seria escolher a orientação olhando o dado que produz o veredito.
    """
    import numpy as np
    from src.selective.comparisons import _orientar

    cal, _ = _particoes_com_sinal()
    aval_invertida = dict(cal)
    aval_invertida["bom"] = -cal["bom"]
    notas = []
    x = _orientar("bom", cal, aval_invertida, notas)
    # orientação +1 decidida na calibração, aplicada à avaliação como ela está
    assert np.allclose(x, aval_invertida["bom"])
    assert any("orientação +1" in n for n in notas)


def test_confianca_e_logit_max_tem_orientacao_FIXA_por_definicao():
    """Eles são a mesma quantidade e são confiança por definição.

    Fixá-los é o que faz a verificação de identidade valer: se a orientação de
    `logit_max` viesse do dado, ele poderia sair invertido em relação à confiança
    e o delta deixaria de ser zero por razão que não é defeito de encanamento.
    """
    import numpy as np
    from src.selective.comparisons import ORIENTACAO_FIXA, _orientar

    assert ORIENTACAO_FIXA == {"model_confidence": 1.0, "logit_max": 1.0}
    cal, aval = _particoes_com_sinal()
    notas = []
    for nome in ("model_confidence", "logit_max"):
        assert np.allclose(_orientar(nome, cal, aval, notas), aval[nome])
    assert not notas, "orientação fixa não deve gerar ressalva de decisão"


def test_escore_constante_gera_RESSALVA_e_nao_erro():
    import numpy as np
    from src.selective.comparisons import _orientar

    cal, aval = _particoes_com_sinal()
    notas = []
    x = _orientar("constante", cal, aval, notas)
    assert np.allclose(x, aval["constante"])
    assert any("orientação indeterminada" in n for n in notas)


def test_sinal_ausente_da_tabela_da_erro_NOMEADO():
    from src.selective.comparisons import ComparisonsError, _combinar

    cal, aval = _particoes_com_sinal()
    with pytest.raises(ComparisonsError, match="não está na tabela"):
        _combinar("hidden_norm", cal, aval, None, [])
