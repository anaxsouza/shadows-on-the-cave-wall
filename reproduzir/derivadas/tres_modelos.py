# Recuperado da linhagem do artefato que produziu a peça publicada; só os CAMINHOS
# foram trocados para o repositório. Rode por reproduzir/derivadas.py, que o executa
# em saida/derivadas/ e compara o que ele escreve com o publicado.
from pathlib import Path as _P
RAIZ = _P(__file__).resolve().parents[2]
import csv
import json
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# Load data
def load_csv(path):
    return list(csv.DictReader(open(path, encoding="utf-8-sig")))

FONTES = {
    "gliner-base":  (str(RAIZ / "docs/tese/resultados" / "confirmatorio_decl02.csv"), str(RAIZ / "docs/tese/resultados" / "confirmatorio_decl03_tarefa.csv")),
    "gliner-large": (str(RAIZ / "docs/tese/resultados" / "confirmatorio_decl04_geometria.csv"), str(RAIZ / "docs/tese/resultados" / "confirmatorio_decl04_tarefa.csv")),
    "ajustado":     (str(RAIZ / "docs/tese/resultados" / "confirmatorio_ajustado_geometria.csv"), str(RAIZ / "docs/tese/resultados" / "confirmatorio_ajustado_tarefa.csv")),
}

geo = {}
tar = {}
for m, (fg, ft) in FONTES.items():
    geo[m] = load_csv(fg)
    tar[m] = load_csv(ft)

def g1(m, corpus, comp, metrica=None, estrato="todos"):
    for r in geo[m]:
        if r["corpus"] == corpus and r["comparacao"] == comp and r.get("estrato", "todos") == estrato:
            if metrica is None or r.get("metrica") == metrica:
                return r

def lt(m, corpus, comp, meta):
    for r in tar[m]:
        if r["corpus"] == corpus and r["comparacao"] == comp and r["meta"] == meta:
            return r

# Erro base per model/corpus (from MEDIDA.json data, hardcoded from trace)
erro = {
    ("gliner-base", "genia"): 0.4818,
    ("gliner-base", "conll2003"): 0.5310,
    ("gliner-large", "genia"): 0.4224,
    ("gliner-large", "conll2003"): 0.4667,
    ("ajustado", "genia"): 0.2458,
    ("ajustado", "conll2003"): 0.1028,
}

MOD = ["gliner-base", "gliner-large", "ajustado"]
COR = {"genia": "#C1272D", "conll2003": "#1F5FA9"}
NOME = {"genia": "GENIA", "conll2003": "CoNLL-2003"}
META_GREY = "0.45"

# Figure style setup
plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
})

fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(7.4, 2.9))
x = np.arange(3)
ROT2 = ["base", "large", "ajustado"]

# --- a: the extractor improved ---
for c_ in ("genia", "conll2003"):
    y = [erro[(m, c_)] for m in MOD]
    axA.plot(x, y, "o-", color=COR[c_], lw=1.6, ms=5, label=NOME[c_])
    axA.annotate(f"{y[-1]:.3f}".replace(".", ","), xy=(2, y[-1]), xytext=(3, -9),
                 textcoords="offset points", fontsize=7, color=COR[c_])
axA.set_ylim(0, 0.58)
axA.set_ylabel("taxa de erro base")
axA.set_title("O extrator ficou competente", loc="left")
axA.legend(frameon=False, loc="lower left", fontsize=7)
axA.set_xlabel("extrator  →  de prateleira para ajustado ao corpus", fontsize=7)

# --- b: geometry does not move ---
for c_ in ("genia", "conll2003"):
    y = [float(g1(m, c_, "C1", "R2_massa_vs_fracao_geometrica")["valor"]) for m in MOD]
    axB.plot(x, y, "o-", color=COR[c_], lw=1.6, ms=5)
    axB.annotate(NOME[c_], xy=(2, y[-1]), xytext=(-4, 9 if c_ == "genia" else -13),
                 textcoords="offset points", fontsize=7, color=COR[c_], ha="right")
axB.set_ylim(0.80, 1.0)
axB.set_ylabel("$R^2$ da massa vs. fração geométrica")
axB.set_title("A massa segue sendo geometria", loc="left")
axB.set_xlabel(" ", fontsize=7)

# --- c: the positive signal disappeared ---
for i, c_ in enumerate(("genia", "conll2003")):
    d = np.array([float(g1(m, c_, "C2")["delta"]) for m in MOD])
    lo = np.array([float(g1(m, c_, "C2")["ci_low"]) for m in MOD])
    hi = np.array([float(g1(m, c_, "C2")["ci_high"]) for m in MOD])
    dx = (i - 0.5) * 0.16
    axC.errorbar(x + dx, d, yerr=[d - lo, hi - d], fmt="o", color=COR[c_],
                 ms=5, lw=1.4, capsize=2.5)
axC.axhline(0, color=META_GREY, lw=0.9, ls="--", zorder=0)
axC.set_ylabel("ΔAURC da massa sobre a geometria", labelpad=1)
axC.set_title("O único sinal positivo desapareceu", loc="left")
axC.set_xlabel(" ", fontsize=7)
axC.annotate("IC exclui zero:\na massa acrescenta", xy=(-0.08, -0.0039), xytext=(-0.42, -0.0062),
             fontsize=7, color=COR["genia"], ha="left", va="top",
             arrowprops=dict(arrowstyle="-", lw=0.7, color=COR["genia"]))
axC.annotate("IC contém zero", xy=(1.92, -0.0014), xytext=(1.05, -0.0052),
             fontsize=7, color=COR["genia"], ha="left",
             arrowprops=dict(arrowstyle="-", lw=0.7, color=COR["genia"]))
axC.set_ylim(-0.0092, 0.0050)

for ax in (axA, axB, axC):
    ax.set_xticks(x)
    ax.set_xticklabels(ROT2, fontsize=7)
    ax.margins(x=0.12)

# Panel letters
for ax, letra in zip((axA, axB, axC), "abc"):
    ax.text(-0.13, 1.04, letra, transform=ax.transAxes, fontsize=9,
            fontweight="bold", va="top", ha="left")

fig.tight_layout(w_pad=1.9)
fig.savefig("tres_modelos.png", dpi=300, bbox_inches="tight")