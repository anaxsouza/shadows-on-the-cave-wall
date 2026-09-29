# Recuperado da linhagem do artefato que produziu a peça publicada; só os CAMINHOS
# foram trocados para o repositório. Rode por reproduzir/derivadas.py, que o executa
# em saida/derivadas/ e compara o que ele escreve com o publicado.
from pathlib import Path as _P
RAIZ = _P(__file__).resolve().parents[2]
import csv
import numpy as np
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt

REPO = RAIZ

# Global state
import matplotlib as mpl

def apply_figure_style(*, frame="open", font=None, sizes=(8, 7, 6), grid=False):
    import matplotlib as mpl
    if frame not in ("open", "boxed", "none"):
        raise ValueError(f"frame must be 'open'|'boxed'|'none', got {frame!r}")

    try:
        import os, sys, glob, matplotlib.font_manager as fm
        fdir = os.path.join(os.environ.get("CONDA_PREFIX") or sys.prefix, "fonts")
        if os.path.isdir(fdir):
            known = {f.fname for f in fm.fontManager.ttflist}
            for f in glob.glob(os.path.join(fdir, "*.ttf")):
                if f not in known:
                    fm.fontManager.addfont(f)
    except Exception:
        pass
    base, secondary, tick = sizes
    boxed = (frame == "boxed")
    rc = {
        "font.family": "sans-serif",
        "font.size": base,
        "axes.labelsize": base,
        "axes.titlesize": base,
        "legend.fontsize": secondary,
        "xtick.labelsize": tick,
        "ytick.labelsize": tick,
        "axes.linewidth": 0.6,
        "xtick.direction": "out", "ytick.direction": "out",
        "xtick.major.size": 3, "ytick.major.size": 3,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "axes.spines.top": boxed, "axes.spines.right": boxed,
        "axes.spines.left": frame != "none", "axes.spines.bottom": frame != "none",
        "axes.grid": bool(grid),
        "legend.frameon": False,
        "figure.dpi": 200,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.titleweight": "normal",
        "axes.titlelocation": "left",
        "axes.labelweight": "normal",
        "lines.linewidth": 1.2,
        "patch.linewidth": 0.6,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    }
    if font:
        rc["font.sans-serif"] = [font, "DejaVu Sans"]
    mpl.rcParams.update(rc)

apply_figure_style(sizes=(9, 8, 7))

import sys
sys.path.insert(0, str(REPO))

from src.selective.risk_coverage import risk_coverage_curve, aurc

# Load entities.csv for each corpus/split
guardar = {}
for corpus in ("genia", "conll2003"):
    d = REPO / "dados_medidos/gliner-base" / corpus / "validation"
    rows = list(csv.DictReader((d / "entities.csv").open(encoding="utf-8")))
    perda = np.array([int(r["loss"]) for r in rows])
    conf  = np.array([float(r["model_confidence"]) for r in rows])
    massa = np.array([float(r["span_mass"]) for r in rows])
    ntok  = np.array([len(r["token_indices"].split("|")) for r in rows], dtype=float)
    guardar[corpus] = dict(perda=perda, conf=conf, massa=massa, ntok=ntok,
                           sent=np.array([r["sentence_id"] for r in rows]))

# Load perimetro1 varredura results
L = list(csv.DictReader((REPO / "docs/tese/resultados/perimetro1_leituras.csv").open(encoding="utf-8")))
for r in L:
    for k in ("aurc_leitura","delta_vs_acaso","peso_na_leitura","aurc_combinado","delta_vs_confianca","correlacao_com_tamanho"):
        r[k] = float(r[k])

COR = {"acaso": "#8C8C8C", "massa": "#C1272D", "comp": "#E08214", "conf": "#1F5FA9"}
NOME = {"genia": "GENIA (biomédico, aninhado)", "conll2003": "CoNLL-2003 (notícias, plano)"}

fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.3))
for ax, corpus in zip(axes[:2], ("genia", "conll2003")):
    g = guardar[corpus]; perda = g["perda"]
    base = perda.mean()
    ax.axhline(base, color=COR["acaso"], lw=1.2, ls="--")
    for chave, x, rot in (("comp", g["ntok"], "comprimento do span"),
                          ("massa", g["massa"], "massa de atenção"),
                          ("conf", g["conf"], "confiança do modelo")):
        c = risk_coverage_curve(perda, x)
        ax.plot(c.coverage, c.risk, color=COR[chave], lw=1.7,
                label=f"{rot} ({aurc(c):.3f})")
    ax.set_xlabel("cobertura"); ax.set_xlim(0, 1); ax.set_ylim(0, 0.78)
    ax.set_title(NOME[corpus])
    ax.text(0.03, base + 0.012, f"abstenção aleatória ({base:.3f})", fontsize=6.4,
            color=COR["acaso"], va="bottom")
    ax.legend(frameon=False, fontsize=6.6, loc="lower right")
axes[0].set_ylabel("risco entre as entidades entregues")
axes[0].text(0.02, 0.02, "abaixo = melhor", transform=axes[0].transAxes, fontsize=6.4, color="#444444")

ax = axes[2]
for corpus, cor in (("genia", "#C1272D"), ("conll2003", "#E08214")):
    dv = np.array([r["delta_vs_confianca"] for r in L if r["corpus"] == corpus])
    ax.hist(dv, bins=48, histtype="step", lw=1.5, color=cor, label=NOME[corpus].split(" (")[0])
    decl = [r for r in L if r["corpus"] == corpus and r["direcao"]=="recebida"
            and r["sumidouro"]=="drop_from_denominator" and r["camadas"]=="todas"
            and r["cabecas"]=="todas"][0]["delta_vs_confianca"]
    ax.axvline(decl, color=cor, lw=1.2, ls=":")
ax.axvline(0, color="#333333", lw=1.0)
ax.set_xlabel("ΔAURC da combinação vs confiança sozinha")
ax.set_ylabel("leituras de atenção")
ax.set_title("2.496 leituras por corpus")
ax.text(0.03, 0.96, "pontilhado = leitura declarada\nà esquerda de 0 = ajudaria",
        transform=ax.transAxes, va="top", fontsize=6.4, color="#444444")
ax.legend(frameon=False, fontsize=6.6, loc="center right")

fig.tight_layout()
fig.savefig("perimetro1.png", dpi=300, bbox_inches="tight")