"""Exporta o BC5CDR para os jsonl dos jobs (treino/validação por palavra, teste por caractere).

Mesmos dois formatos de `sentinel/dados/*.jsonl` dos outros corpora:
  - `bc5cdr_train.jsonl` e `bc5cdr_validation.jsonl`: `{"tokenized_text", "ner"}`, com `ner` =
    [início_palavra, fim_palavra INCLUSIVO, descrição do rótulo] — a descrição ('chemical',
    'disease') vem de `tools/rotulos.py`, a fonte única.
  - `bc5cdr_test.jsonl`: `{"tokenized_text", "ner_char"}`, com posição de CARACTERE e o rótulo
    do carregador ('Chemical', 'Disease'), como `tools/exportar_corpus_teste.py`.
Os três vêm do MESMO carregador (mesma ordem, mesmas entidades). Contêm texto do corpus: ficam em
`dados_decoder/*.jsonl` (gitignored) e não são publicados. O manifesto (sem texto) é
`dados_decoder/MANIFESTO_bc5cdr.json`, SEPARADO do MANIFESTO_test.json existente.
"""
import hashlib
import json
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(RAIZ / "tools"))
from rotulos import descricao  # noqa: E402
from src.core.loaders.biomedical.bc5cdr import BC5CDRLoader, SENTENCAS_ESPERADAS  # noqa: E402

D = RAIZ / "dados_decoder"


def _palavras(tokens, ini, fim):
    """Índices de palavra [i0, i1] (inclusivo) de um trecho em caracteres [ini, fim)."""
    pos, i0, i1 = 0, None, None
    for i, t in enumerate(tokens):
        a, b = pos, pos + len(t)
        if i0 is None and a <= ini < b:
            i0 = i
        if a < fim <= b:
            i1 = i
        pos = b + 1
    assert i0 is not None and i1 is not None and i1 >= i0, (tokens, ini, fim)
    return i0, i1


def main() -> int:
    D.mkdir(exist_ok=True)
    man = {}
    for split, nome in (("train", "train"), ("validation", "validation"), ("test", "test")):
        ex = BC5CDRLoader().load_split(split)
        assert len(ex) == SENTENCAS_ESPERADAS[split], (split, len(ex))
        alvo = D / f"bc5cdr_{nome}.jsonl"
        with alvo.open("w", encoding="utf-8") as fh:
            for e in ex:
                toks = list(e.tokens)
                if split == "test":
                    reg = {"tokenized_text": toks,
                           "ner_char": [[int(s.start), int(s.end), str(s.label)] for s in e.entities]}
                else:
                    ner = []
                    for s in e.entities:
                        i0, i1 = _palavras(toks, int(s.start), int(s.end))
                        ner.append([i0, i1, descricao("bc5cdr", str(s.label))])
                    reg = {"tokenized_text": toks, "ner": ner}
                fh.write(json.dumps(reg, ensure_ascii=False) + "\n")
        man[alvo.name] = {"sha256": hashlib.sha256(alvo.read_bytes()).hexdigest(),
                          "n_sentencas": len(ex),
                          "n_entidades": sum(len(e.entities) for e in ex),
                          "campo_entidades": "ner_char (CARACTERE)" if split == "test"
                          else "ner (INDICE DE PALAVRA, fim inclusivo)"}
        print(alvo.name, man[alvo.name])
    (D / "MANIFESTO_bc5cdr.json").write_text(
        json.dumps({"origem": "BC5CDRLoader, dados_bc5cdr com SHA-256 da decl-16", "arquivos": man},
                   ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
