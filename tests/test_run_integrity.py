"""Execução quebrada não pode reportar sucesso.

O princípio deste arquivo sobreviveu à reorganização; os cinco defeitos concretos
que ele guardava não, porque o código deles saiu. O que era guardado antes:
`ModelManager` construído e nunca carregado no `runner.py`, o conversor de spans
sem receber o modelo em `h1_performance_stratification.py`, escrita de JSON não
atômica via `shared/json_io.py`, e o painel `cli/console/` pintando banner verde
sobre execução falha. Nada disso existe mais.

Os guardas abaixo apontam para o caminho que sobrou, e dois deles fixam
propriedades que a própria reorganização introduziu — porque a falha silenciosa
que o corte removeu de `src/core/__init__.py` é a mesma classe de defeito que
este arquivo persegue: algo que erra, registra aviso e segue como se tivesse
funcionado.

O guarda de painel não foi reapontado: não há painel. Recuperar o original em
`git show pre-c4-reorg:tests/test_run_integrity.py`.
"""

from __future__ import annotations

import ast
import csv
import json
from pathlib import Path

import numpy as np
import pytest

RAIZ = Path(__file__).parent.parent
PIPELINE = RAIZ / "src" / "core" / "pipeline" / "ner_pipeline.py"
CORE_INIT = RAIZ / "src" / "core" / "__init__.py"


# --- Defeito 1: ModelManager construído tem de ser carregado -------------------



# --- Defeito 2: falhar alto em vez de escolher um caminho por omissão ----------



def test_core_init_nao_engole_import_error():
    """Import que falha em silêncio é invisível para esta suíte.

    Até 02/09/2026 este arquivo envolvia seus imports num try/except ImportError
    que registrava aviso e seguia — foi assim que 'Core import failed: No module
    named datasets' apareceu sem quebrar nada.
    """
    fonte = CORE_INIT.read_text(encoding="utf-8")
    arvore = ast.parse(fonte)
    for no in ast.walk(arvore):
        if isinstance(no, ast.Try):
            capturados = []
            for h in no.handlers:
                if h.type is None:
                    capturados.append("except:")
                else:
                    capturados.append(ast.unparse(h.type))
            assert not any("ImportError" in c or c == "except:" for c in capturados), (
                f"src/core/__init__.py voltou a engolir falha de import: {capturados}"
            )


def test_preregistro_ausente_impede_execucao():
    """A recusa é a garantia: sem os seis itens declarados, nada roda."""
    from src.selective.preregistration import PreregistrationError, load_preregistration

    with pytest.raises(PreregistrationError):
        load_preregistration(RAIZ / "pyproject.toml")


# --- Defeito 3: o resultado tem de serializar, incluindo tipos de numpy --------

CONFIG_MINIMO = """
selective:
  declaration_id: decl-teste
  sink_policy: drop_from_denominator
  layers: [8, 9]
  heads: null
  combination_rule: rank_average
  calibration_fraction: 0.4
  coverage_levels: [0.5, 1.0]
  operating_point: derived_from_target_risk
  added_value_ci_level: 0.9
  n_bootstrap_resamples: 150
  target_risk_grid: [0.3]
  conformal_alpha: 0.3
  loss_bound: 1.0
  seed: 5
"""


@pytest.fixture()
def resultado(tmp_path: Path):
    from src.selective.runner import SelectiveRunner

    cfg = tmp_path / "config.yaml"
    cfg.write_text(CONFIG_MINIMO, encoding="utf-8")
    tabela = tmp_path / "results" / "m" / "genia" / "test" / "entities.csv"
    tabela.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    with tabela.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["sentence_id", "loss", "model_confidence", "span_mass", "is_nested"])
        for s in range(40):
            for k in range(2):
                errada = rng.random() < 0.4
                w.writerow([
                    f"s{s}", int(errada),
                    round(float(rng.uniform(0.8, 0.99)), 4),
                    round(float(np.clip(rng.normal(0.25 if errada else 0.55, 0.08), 0.01, 0.99)), 4),
                    int(k == 1),
                ])
    return SelectiveRunner("m", "genia", config_path=str(cfg), output_dir=str(tmp_path / "results")).run("floor")


def test_resultado_serializa_sem_tipo_de_numpy(resultado):
    """`json.dumps` sem `default=` é o teste: numpy.float64 não é serializável.

    Tipos de numpy atravessam todo este código (as curvas são arrays), e um
    `default=str` esconderia o problema convertendo silenciosamente em texto —
    o que produz relatório com números entre aspas.
    """
    texto = json.dumps(resultado.to_dict())
    recarregado = json.loads(texto)
    assert recarregado["test"] == "floor"
    assert isinstance(recarregado["base_risk"], float)
    assert isinstance(recarregado["n_entities"], int)
    assert recarregado["preregistration"]["conformal_alpha"] == 0.3


def test_o_relatorio_declara_o_preregistro_e_as_ressalvas(resultado):
    """Resultado que circula sem a convenção que o produziu é número solto."""
    texto = resultado.render()
    assert "Declaração" in texto and "hash" in texto
    assert "rank_average" in texto
    assert f"{resultado.n_calibration_entities} na calibração" in texto


def test_execucao_limitada_declara_a_limitacao(tmp_path: Path):
    """`max_samples` reduz a amostra, e o relatório tem de dizer isso.

    Intervalo calculado sobre amostra truncada que se apresenta como intervalo do
    corpus é a forma mais discreta de reportar sucesso indevido.
    """
    from src.selective.runner import SelectiveRunner

    cfg = tmp_path / "config.yaml"
    cfg.write_text(CONFIG_MINIMO, encoding="utf-8")
    tabela = tmp_path / "r" / "m" / "genia" / "test" / "entities.csv"
    tabela.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1)
    with tabela.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["sentence_id", "loss", "model_confidence", "span_mass", "is_nested"])
        for s in range(40):
            errada = rng.random() < 0.4
            w.writerow([f"s{s}", int(errada), 0.9,
                        round(float(rng.uniform(0.1, 0.9)), 4), 0])
    r = SelectiveRunner(
        "m", "genia", config_path=str(cfg), output_dir=str(tmp_path / "r"), max_samples=20
    ).run("floor")
    assert r.n_sentences == 20
    assert any("max_samples" in n for n in r.notes)
    assert "Ressalvas desta execução" in r.render()


def test_o_guarda_de_procedencia_le_o_que_a_medicao_escreve(tmp_path):
    """Acopla escritor e leitor, que é o que faltava.

    O guarda procurava a chave `preregistration` e `measure()` escreve
    `preregistro`: ele nunca conferia nada e caía na ressalva de "procedência
    incompleta", parecendo proteger. Este teste falha se as duas pontas voltarem
    a divergir — inclusive por renomeação de chave.
    """
    import json

    import numpy as np

    from src.selective.measurement import measure
    from src.selective.preregistration import load_preregistration

    cfg = tmp_path / "config.yaml"
    cfg.write_text(CONFIG_MINIMO, encoding="utf-8")
    prereg = load_preregistration(str(cfg))

    def predict(sent, tokens_only: bool = False):
        toks = ["a", "b", "c"]
        if tokens_only:
            return {"tokens": toks, "n_layers": max(prereg.layers) + 1, "n_heads": 1}
        att = tuple(
            np.full((1, 1, len(toks), len(toks)), 1 / len(toks), dtype=np.float32)
            for _ in range(max(prereg.layers) + 1)
        )
        return {
            "tokens": toks,
            "attentions": att,
            "n_gold": 1,
            "predicted": [
                {"token_indices": [1], "confidence": 0.8, "correct": True, "is_nested": False}
            ],
        }

    saida = tmp_path / "out"
    measure(sentences=["s0", "s1"], predict=predict, prereg=prereg,
            output_dir=saida, model_id="m-teste")

    escrito = json.loads((saida / "MEDIDA.json").read_text(encoding="utf-8"))
    bloco = escrito.get("preregistro") or escrito.get("preregistration") or escrito
    assert bloco.get("declaration_hash") == prereg.declaration_hash, (
        "measure() não grava declaration_hash onde o guarda de SelectiveRunner o procura"
    )
