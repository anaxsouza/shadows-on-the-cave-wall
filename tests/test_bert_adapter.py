"""Testes do adaptador do classificador por token (decl-22). Sem rede e sem pesos."""
import json
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from src.selective.bert_adapter import (  # noqa: E402
    bio_plano, confianca_geometrica, decodifica_conlleval, f1_estrito, ler_corpus,
    normaliza_rotulo, rotulos_bio, spans_de_caracteres)


def test_conlleval_i_sem_b_abre_trecho():
    assert decodifica_conlleval(["I-PER", "I-PER", "O", "I-LOC"]) == [(0, 1, "PER"), (3, 3, "LOC")]


def test_conlleval_troca_de_tipo_e_b_consecutivos():
    assert decodifica_conlleval(["B-A", "I-B", "B-B", "B-B", "I-B"]) == [
        (0, 0, "A"), (1, 1, "B"), (2, 2, "B"), (3, 4, "B")]


def test_conlleval_vazio_e_so_o():
    assert decodifica_conlleval([]) == [] and decodifica_conlleval(["O", "O"]) == []


def test_bio_plano_sobrescreve_na_ordem():
    # externo [0,2] depois interno [1,1]: o interno sobrescreve
    assert bio_plano(4, [(0, 2, "X"), (1, 1, "Y")]) == ["B-X", "B-Y", "I-X", "O"]
    assert bio_plano(3, [(1, 1, "Y"), (0, 2, "X")]) == ["B-X", "I-X", "I-X"]


def test_rotulos_bio_ordem_estavel():
    r = rotulos_bio("bc5cdr")
    assert r == ["O", "B-Chemical", "I-Chemical", "B-Disease", "I-Disease"]
    assert len(rotulos_bio("genia")) == 11 and len(rotulos_bio("conll2003")) == 9


def test_normaliza_rotulo():
    assert normaliza_rotulo("genia", "cell type") == "CELL_TYPE"
    assert normaliza_rotulo("genia", "DNA") == "DNA"
    assert normaliza_rotulo("conll2003", "person") == "PER"
    assert normaliza_rotulo("conll2003", "MISC") == "MISC"
    with pytest.raises(KeyError):
        normaliza_rotulo("genia", "banana")


def test_confianca_media_geometrica():
    assert confianca_geometrica([0.5, 0.5]) == pytest.approx(0.5)
    assert confianca_geometrica([1.0, 0.25]) == pytest.approx(0.5)
    assert confianca_geometrica([0.9]) == pytest.approx(0.9)


def test_f1_estrito():
    r = f1_estrito([[(0, 1, "A"), (3, 3, "B")], []], [[(0, 1, "A"), (3, 4, "B")], [(0, 0, "A")]])
    assert (r["tp"], r["n_previstos"], r["n_ouro"]) == (1, 2, 3)
    assert r["precisao"] == pytest.approx(0.5) and r["recall"] == pytest.approx(1 / 3)
    assert f1_estrito([[]], [[]])["f1"] == 0.0


def test_spans_de_caracteres_como_trechos_de_ouro():
    toks = ["Nadim", "Ladki", "went"]
    assert spans_de_caracteres(toks, [[0, 11, "PER"]]) == [(0, 1, "PER")]
    assert spans_de_caracteres(toks, [[12, 16, "X"]]) == [(2, 2, "X")]


def test_ler_corpus_jsonl_e_bc5cdr(tmp_path):
    (tmp_path / "genia_train.jsonl").write_text(json.dumps(
        {"tokenized_text": ["a", "b", "c", "d"], "ner": [[0, 2, "protein"], [1, 1, "cell type"]]}) + "\n")
    ex = ler_corpus("genia", "train", tmp_path)[0]
    assert ex["ouro"] == [(0, 2, "PROTEIN"), (1, 1, "CELL_TYPE")]       # aninhado preservado
    assert ex["tags_planas"] == ["B-PROTEIN", "B-CELL_TYPE", "I-PROTEIN", "O"]
    (tmp_path / "genia_test.jsonl").write_text(json.dumps(
        {"tokenized_text": ["xx", "yy"], "ner_char": [[3, 5, "RNA"]]}) + "\n")
    assert ler_corpus("genia", "test", tmp_path)[0]["ouro"] == [(1, 1, "RNA")]
    b = tmp_path / "bc"
    b.mkdir()
    (b / "label.json").write_text(json.dumps({"O": 0, "B-Chemical": 1, "B-Disease": 2, "I-Disease": 3, "I-Chemical": 4}))
    (b / "valid.json").write_text(json.dumps({"tags": [1, 0, 2, 3], "tokens": ["w", "x", "y", "z"]}) + "\n")
    ex = ler_corpus("bc5cdr", "validation", b)[0]
    assert ex["ouro"] == [(0, 0, "Chemical"), (2, 3, "Disease")]
