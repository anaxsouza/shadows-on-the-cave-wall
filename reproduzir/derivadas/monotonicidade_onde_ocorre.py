# Recuperado da linhagem do artefato que produziu a peça publicada; só os CAMINHOS
# foram trocados para o repositório. Rode por reproduzir/derivadas.py, que o executa
# em saida/derivadas/ e compara o que ele escreve com o publicado.
from pathlib import Path as _P
RAIZ = _P(__file__).resolve().parents[2]
import csv
import sys
import zipfile
import numpy as np
from pathlib import Path
from numpy.lib import format as npy

REPO = RAIZ
sys.path.insert(0, str(REPO))


def carregar(corpus):
    d = REPO / f"dados_medidos/gliner-base/{corpus}/validation"
    r = list(csv.DictReader((d / "entities.csv").open(encoding="utf-8")))
    T = {x["sentence_id"]: int(x["n_tokens"]) for x in
         csv.DictReader((d / "sentence_lengths.csv").open(encoding="utf-8"))}
    k = np.array([len(x["token_indices"].split("|")) for x in r], float)
    Ts = np.array([T[x["sentence_id"]] for x in r], float)
    massa = np.array([float(x["span_mass"]) for x in r])
    return dict(perda=np.array([float(x["loss"]) for x in r]),
                model_confidence=np.array([float(x["model_confidence"]) for x in r]),
                span_mass=massa, enrichment=massa / (k / (Ts - 1)))


def onde_a_violacao(perdas, escores):
    cand = np.unique(escores)[::-1]
    riscos, cobs, ns = [], [], []
    for tau in cand:
        ret = escores >= tau
        riscos.append(perdas[ret].mean()); cobs.append(ret.mean()); ns.append(int(ret.sum()))
    riscos, cobs, ns = np.array(riscos), np.array(cobs), np.array(ns)
    var = np.diff(riscos)
    i = int(np.argmin(var))
    for piso in (30, 100):
        m = np.array(ns[1:]) >= piso
        v_r = float(max(0.0, -var[m].min())) if m.any() else float("nan")
        yield piso, v_r
    yield "loc", (float(max(0.0, -var.min())), cobs[i], ns[i], ns[i + 1])


linhas = []
for corpus in ("genia", "conll2003"):
    g = carregar(corpus)
    for esc in ("model_confidence", "span_mass", "enrichment"):
        saida = list(onde_a_violacao(g["perda"], g[esc]))
        (_, v30), (_, v100), (_, (vt, cob, n_i, _)) = saida
        linhas.append(dict(corpus=corpus, escore=esc,
                           violacao_total=round(vt, 6),
                           cobertura_no_ponto=round(float(cob), 6),
                           n_entregue_no_ponto=int(n_i),
                           violacao_n_min_30=round(v30, 6),
                           violacao_n_min_100=round(v100, 6),
                           razao_total_sobre_n100=round(vt / v100, 1) if v100 else None))

alvo = Path("monotonicidade_onde_ocorre.csv")
with alvo.open("w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=list(linhas[0])); w.writeheader(); w.writerows(linhas)