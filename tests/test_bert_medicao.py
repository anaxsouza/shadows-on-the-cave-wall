"""Testes da medição do BERT (decl-22): offsets, nulo, aninhamento, procedência.

Usa um BERT minúsculo construído na hora (vocabulário próprio), sem rede.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "tools"))

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

import medir_bert as mb  # noqa: E402
from src.selective.attention_mass import span_attention_mass  # noqa: E402
from src.selective.geometry import expected_mass  # noqa: E402


@pytest.fixture(scope="module")
def mini(tmp_path_factory):
    d = tmp_path_factory.mktemp("minibert")
    palavras = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "a", "b", "c", "un", "##known", "##ing",
                "Hello", "world", ",", "."]
    (d / "vocab.txt").write_text("\n".join(palavras) + "\n")
    tok = transformers.BertTokenizerFast(str(d / "vocab.txt"), do_lower_case=False)
    cfg = transformers.BertConfig(vocab_size=len(palavras), hidden_size=16, num_hidden_layers=12, attn_implementation="eager",
                                  num_attention_heads=2, intermediate_size=32, num_labels=3,
                                  id2label={0: "O", 1: "B-X", 2: "I-X"}, label2id={"O": 0, "B-X": 1, "I-X": 2})
    torch.manual_seed(0)
    m = transformers.BertForTokenClassification(cfg)
    return m.eval(), tok


def test_subtokens_pelos_offsets_igual_a_word_ids(mini):
    m, tok = mini
    toks = ["Hello", "unknowning", "a", ",", "world"]
    r = mb.medir_sentenca(m, tok, toks, {0: "O", 1: "B-X", 2: "I-X"})
    enc = tok(toks, is_split_into_words=True)
    for w, t in enumerate(toks):
        ci = r["inicios"][w]
        por_offset = mb.indices_de_subtokens(r["offsets"], ci, ci + len(t))
        por_word_id = [j for j, x in enumerate(enc.word_ids()) if x == w]
        assert por_offset == por_word_id
    assert r["T"] == len(enc["input_ids"]) and r["att"].shape == (12, 2, r["T"], r["T"])
    assert 0 not in mb.indices_de_subtokens(r["offsets"], 0, 10**6)      # [CLS] nunca é do trecho


def test_massa_do_trecho_com_cls_fora_do_denominador(mini):
    m, tok = mini
    toks = ["Hello", "world", "a", "b", "c"]
    r = mb.medir_sentenca(m, tok, toks, {0: "O", 1: "B-X", 2: "I-X"})
    idx = mb.indices_de_subtokens(r["offsets"], 0, len("Hello world"))
    massa = span_attention_mass(r["att"], idx, sink_policy="drop_from_denominator").mass
    assert 0.0 < massa < 1.0
    # a soma sobre todos os sub-tokens exceto [CLS] é 1: nulo exato do trecho inteiro
    todos = list(range(1, r["T"]))
    assert span_attention_mass(r["att"], todos, sink_policy="drop_from_denominator").mass == pytest.approx(1.0)
    assert expected_mass(len(idx), r["T"], sink_policy="drop_from_denominator") == pytest.approx(len(idx) / (r["T"] - 1))


def test_aninhado_estrito():
    assert mb.aninhado((1, 2), {(0, 3), (1, 2)}) is True
    assert mb.aninhado((1, 2), {(1, 2)}) is False
    assert mb.aninhado((0, 3), {(0, 3), (1, 2)}) is False


def test_medir_grava_tabela_no_formato_do_encoder(mini, tmp_path, monkeypatch):
    m, tok = mini
    d = tmp_path / "modelo"
    m.save_pretrained(d, safe_serialization=True)
    tok.save_pretrained(d)
    dados = tmp_path / "dados"
    dados.mkdir()
    (dados / "genia_test.jsonl").write_text(
        '{"tokenized_text": ["Hello", "world", "a", "b"], "ner_char": [[0, 11, "DNA"]]}\n' * 3)
    monkeypatch.setenv("SENTINEL_DADOS", str(dados))
    # o modelo minúsculo tem 3 rótulos; força a decodificação a achar trecho X
    import csv
    monkeypatch.setattr(mb, "decodifica_conlleval", lambda tags: [(0, 1, "DNA")])
    monkeypatch.setattr(mb, "ler_corpus", lambda c, s, d_: [
        {"tokens": ["Hello", "world", "a", "b"], "ouro": [(0, 1, "DNA")], "tags_planas": []}] * 3)
    saida = tmp_path / "s"
    out = mb.medir("genia", str(d), saida)
    linhas = list(csv.DictReader((saida / "entities.csv").open()))
    assert len(linhas) == 3 and {l["loss"] for l in linhas} == {"0"}
    for c in ("sentence_id", "loss", "model_confidence", "span_mass", "is_nested", "token_indices"):
        assert c in linhas[0]
    assert all(0 < float(l["model_confidence"]) <= 1 for l in linhas)
    assert out["modelo"]["id"] == "bert-base-cased-ft-genia"
    assert out["preregistro"]["declaration_id"] == "decl-23-bert-genia"
    assert out["preregistro"]["measurement_hash"] == "c5068b677fbfdf82"
    assert (saida / "sentence_lengths.csv").is_file()
