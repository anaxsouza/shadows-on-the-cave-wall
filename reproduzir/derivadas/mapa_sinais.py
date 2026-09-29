# Recuperado da linhagem do artefato que produziu a peça publicada; só os CAMINHOS
# foram trocados para o repositório. Rode por reproduzir/derivadas.py, que o executa
# em saida/derivadas/ e compara o que ele escreve com o publicado.
from pathlib import Path as _P
RAIZ = _P(__file__).resolve().parents[2]
import csv
import json
import subprocess
import os
import statistics as st
from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

REPO = RAIZ
RES = REPO / "docs/tese/resultados"

# skill:figure-style kernel.py (auto-injected on skill load)
# Apply figure style
def apply_figure_style(sizes=(9, 8, 7)):
    plt.rcParams.update({
        'font.size': sizes[1],
        'axes.titlesize': sizes[0],
        'axes.labelsize': sizes[1],
        'xtick.labelsize': sizes[2],
        'ytick.labelsize': sizes[2],
        'legend.fontsize': sizes[2],
        'figure.dpi': 150,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })

META_GREY = "0.55"

def panel_letter(ax, letter):
    ax.text(-0.13, 1.02, letter, transform=ax.transAxes,
            fontsize=10, fontweight='bold', va='bottom', ha='left')

apply_figure_style(sizes=(9, 8, 7))

ta = list(csv.DictReader((RES / "confirmatorio_sinais_tarefa.csv").open(encoding="utf-8-sig")))
notas = (RES / "confirmatorio_sinais_notas.txt").read_text(encoding="utf-8").splitlines()

SIG = {"S1":"row_entropy","S2":"row_max","S3":"emitted_mass","S4":"hidden_norm",
       "S5":"hidden_dist_centroide","S6":"hidden_delta_camadas",
       "S7":"logit_margin","S8":"logit_entropy","S9":"logit_max"}

def num(v):
    try: return float(v)
    except (TypeError, ValueError): return None

cel = [x for x in ta if x["comparacao"] in SIG and x.get("grade")=="vao"]
com_ic = [x for x in cel if num(x["ci_low"]) is not None and num(x["ci_high"]) is not None]
fav  = [x for x in com_ic if num(x["ci_high"]) < 0]
desf = [x for x in com_ic if num(x["ci_low"]) > 0]
nd   = [x for x in com_ic if num(x["ci_low"]) <= 0 <= num(x["ci_high"])]
inal = [x for x in cel if num(x["ci_low"]) is None]

esperado = 0.025 * len(com_ic)

ORDEM = [("row_entropy","atenção"),("row_max","atenção"),("emitted_mass","atenção"),
         ("hidden_norm","ocultos"),("hidden_dist_centroide","ocultos"),("hidden_delta_camadas","ocultos"),
         ("logit_margin","logits"),("logit_entropy","logits"),("logit_max","logits")]
INV = {v:k for k,v in SIG.items()}
COR = {"atenção":"#1F5FA9","ocultos":"#4C8C3F","logits":"#C1272D"}
CURTO = {"row_entropy":"entropia da linha","row_max":"máximo da linha","emitted_mass":"massa emitida",
         "hidden_norm":"norma do estado","hidden_dist_centroide":"dist. ao centroide",
         "hidden_delta_camadas":"Δ entre camadas","logit_margin":"margem de rótulo",
         "logit_entropy":"entropia de rótulo","logit_max":"maior escore"}

fig, (axA, axB) = plt.subplots(1, 2, figsize=(7.3, 3.5),
                               gridspec_kw={"width_ratios": [1.35, 1]})

# --- a: todos os deltas por sinal -------------------------------------------
for i, (nome, fam) in enumerate(ORDEM):
    y = len(ORDEM) - 1 - i
    linhas = [x for x in ta if x["comparacao"] == INV[nome] and x.get("grade") == "vao"]
    ds  = [num(x["delta"]) for x in linhas if num(x["delta"]) is not None]
    los = [num(x["ci_low"]) for x in linhas if num(x["ci_low"]) is not None]
    his = [num(x["ci_high"]) for x in linhas if num(x["ci_high"]) is not None]
    n_inal = sum(num(x["ci_low"]) is None for x in linhas)
    for d, lo, hi in zip(ds, los, his):
        favor = hi < 0
        axB_c = COR[fam]
        axA.plot([lo, hi], [y, y], color=axB_c, lw=1.0, alpha=0.5, solid_capstyle="butt", zorder=2)
        axA.plot([d], [y], marker="o", ms=4.2, mfc=("white" if not favor else axB_c),
                 mec=axB_c, mew=1.1, zorder=3)
    if n_inal:
        axA.text(0.80, y, f"{n_inal} inalc.", fontsize=7, color=COR[fam], va="center", ha="left")
axA.axvline(0, color="0.25", lw=0.9, zorder=1)
axA.set_yticks(range(len(ORDEM)))
axA.set_yticklabels([CURTO[n] for n, _ in ORDEM][::-1])
for t_, (n_, f_) in zip(axA.get_yticklabels()[::-1], ORDEM):
    t_.set_color(COR[f_])
axA.set_xlabel("Δ carga de revisão contra a confiança do modelo\n"
               "← sinal exige MENOS revisão   |   exige MAIS →")
axA.set_xlim(-0.15, 0.95)
axA.set_title("Nenhuma família bate a confiança do próprio modelo")
panel_letter(axA, "a")

# --- b: favoráveis contra o que o acaso produz ------------------------------
cats = ["favoráveis\n(IC < 0)", "desfavoráveis\n(IC > 0)"]
obs = [len(fav), len(desf)]
axB.bar([0, 1], obs, color=["#4C8C3F", "0.55"], width=0.58, zorder=2)
axB.axhline(esperado, color="#C1272D", lw=1.2, ls="--", zorder=3)
axB.text(1.42, esperado, f"acaso a 95%\n≈ {esperado:.1f}", color="#C1272D",
         fontsize=7, va="center", ha="right")
for x_, v in zip([0, 1], obs):
    axB.text(x_, v + 0.7, str(v), ha="center", fontsize=8)
axB.set_xticks([0, 1]); axB.set_xticklabels(cats)
axB.set_ylabel("células declaradas (de 49 com intervalo)")
axB.set_ylim(0, 34)
axB.set_title("Os 2 favoráveis cabem no acaso;\nos 30 desfavoráveis não")
panel_letter(axB, "b")

fig.tight_layout(w_pad=1.8)
fig.savefig("mapa_sinais.png", dpi=300, bbox_inches="tight")

axA.set_title("Nenhuma família bate a confiança", pad=10)
axB.set_title("Os 2 favoráveis cabem no acaso;\nos 30 desfavoráveis não", pad=10)
axB.set_ylabel("células declaradas", labelpad=2)
for t_ in list(axB.texts):
    if "acaso a 95%" in t_.get_text():
        t_.remove()
axB.annotate("o que o acaso a 95%\nproduziria (≈1,2)", xy=(0.30, esperado), xytext=(0.30, 11),
             color="#C1272D", fontsize=7, ha="center", va="bottom",
             arrowprops=dict(arrowstyle="-", lw=0.8, color="#C1272D"))
axA.set_xlim(-0.16, 1.02)
fig.tight_layout(w_pad=2.4)
fig.savefig("mapa_sinais.png", dpi=300, bbox_inches="tight")