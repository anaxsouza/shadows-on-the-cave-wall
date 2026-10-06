"""Gold-mention statistics of the three test partitions (no corpus text written).

Reads dados_decoder/<corpus>_test.jsonl (character offsets over the space-joined tokens)
and writes docs/tese/resultados/revisao/corpus_mencoes.csv. Nesting: a mention is nested
when it contains, or is contained in, another gold mention of the same sentence.
"""
import csv, json, pathlib
import numpy as np

RAIZ = pathlib.Path(__file__).resolve().parents[1]
SAIDA = RAIZ / "docs/tese/resultados/revisao/corpus_mencoes.csv"

def main():
    linhas = []
    for nome, arq in (("genia", "genia_test"), ("conll2003", "conll2003_test"), ("bc5cdr", "bc5cdr_test")):
        comp, aninh, n_sent = [], 0, 0
        for linha in open(RAIZ / "dados_decoder" / f"{arq}.jsonl"):
            d = json.loads(linha); n_sent += 1
            txt = " ".join(d["tokenized_text"])
            sp = [(a, b) for a, b, _ in d["ner_char"]]
            for a, b, _ in d["ner_char"]:
                comp.append(len(txt[a:b].split()))
                if any((x, y) != (a, b) and ((x <= a and b <= y) or (a <= x and y <= b)) for x, y in sp):
                    aninh += 1
        c = np.array(comp)
        linhas.append(dict(corpus=nome, sentences=n_sent, mentions=len(c), mean_words=round(float(c.mean()), 2),
                           pct_multiword=round(float((c > 1).mean() * 100), 1),
                           pct_three_plus=round(float((c >= 3).mean() * 100), 1),
                           pct_nested=round(aninh / len(c) * 100, 1)))
    with open(SAIDA, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(linhas[0])); w.writeheader(); w.writerows(linhas)
    print(SAIDA)

if __name__ == "__main__":
    main()
