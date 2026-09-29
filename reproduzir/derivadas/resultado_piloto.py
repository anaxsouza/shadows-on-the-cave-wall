# Recuperado da linhagem do artefato que produziu a peça publicada; só os CAMINHOS
# foram trocados para o repositório. Rode por reproduzir/derivadas.py, que o executa
# em saida/derivadas/ e compara o que ele escreve com o publicado.
from pathlib import Path as _P
RAIZ = _P(__file__).resolve().parents[2]
import sys, csv, numpy as np
from pathlib import Path

sys.path.insert(0, str(RAIZ))
from src.selective.risk_coverage import risk_coverage_curve, aurc

REPO = RAIZ

def tabela(corpus):
    d = REPO / "dados_medidos/gliner-base" / corpus / "test"
    ent = list(csv.DictReader((d / "entities.csv").open(encoding="utf-8")))
    perfil = list(csv.DictReader((d / "layer_profile.csv").open(encoding="utf-8")))
    return ent, perfil

resumo = {}
for corpus in ("genia", "conll2003"):
    ent, perfil = tabela(corpus)
    perda = np.array([float(e["loss"]) for e in ent])
    conf  = np.array([float(e["model_confidence"]) for e in ent])
    massa = np.array([float(e["span_mass"]) for e in ent])

    chave = {}
    for r in perfil:
        chave.setdefault(int(r["layer"]), []).append(float(r["span_mass"]))
    por_camada = {}
    for c, vals in sorted(chave.items()):
        v = np.array(vals)
        if v.size == perda.size:
            por_camada[c] = aurc(risk_coverage_curve(perda, v))

    resumo[corpus] = {
        "n": len(ent),
        "erro_base": float(perda.mean()),
        "aurc_conf": aurc(risk_coverage_curve(perda, conf)),
        "aurc_massa": aurc(risk_coverage_curve(perda, massa)),
        "por_camada": por_camada,
        "curvas": (risk_coverage_curve(perda, conf), risk_coverage_curve(perda, massa), perda.mean()),
    }
    p = por_camada
    melhor = min(p, key=p.get) if p else None
    print(f"{corpus}: n={len(ent)} erro_base={perda.mean():.4f} | AURC conf={resumo[corpus]['aurc_conf']:.4f} "
          f"massa(agregada)={resumo[corpus]['aurc_massa']:.4f}")
    if p:
        print("   AURC por camada:", " ".join(f"{c}:{v:.3f}" for c, v in sorted(p.items())))
        print(f"   melhor camada isolada: {melhor} (AURC {p[melhor]:.4f}) | acaso = {perda.mean():.4f}")

META_GREY = "#888888"


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


import matplotlib as mpl, matplotlib.pyplot as plt
apply_figure_style(sizes=(9, 8, 7))

fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.1))
COR = {"acaso": "#8C8C8C", "massa": "#C1272D", "conf": "#1F5FA9"}
NOME = {"genia": "GENIA (aninhado)", "conll2003": "CoNLL-2003 (plano)"}

for ax, corpus in zip(axes[:2], ("genia", "conll2003")):
    c_conf, c_massa, base = resumo[corpus]["curvas"]
    ax.axhline(base, color=COR["acaso"], lw=1.2, ls="--",
               label=f"abstenção aleatória (AURC {base:.3f})")
    ax.plot(c_massa.coverage, c_massa.risk, color=COR["massa"], lw=1.6,
            label=f"massa de atenção ({resumo[corpus]['aurc_massa']:.3f})")
    ax.plot(c_conf.coverage, c_conf.risk, color=COR["conf"], lw=1.6,
            label=f"confiança do modelo ({resumo[corpus]['aurc_conf']:.3f})")
    ax.set_xlabel("cobertura"); ax.set_xlim(0, 1); ax.set_ylim(0, 0.75)
    ax.set_title(NOME[corpus], fontsize=9)
    ax.legend(frameon=False, fontsize=6.6, loc="lower right")
axes[0].set_ylabel("risco entre as entidades entregues")

ax = axes[2]
for corpus, mk in (("genia", "o"), ("conll2003", "s")):
    p = resumo[corpus]["por_camada"]
    cs = sorted(p)
    ax.plot(cs, [p[c] for c in cs], marker=mk, ms=3.4, lw=1.3,
            color=COR["massa"] if corpus == "genia" else "#E08214",
            label=NOME[corpus])
    ax.axhline(resumo[corpus]["erro_base"], lw=1.0, ls="--",
               color=COR["massa"] if corpus == "genia" else "#E08214", alpha=0.6)
ax.set_xlabel("camada do encoder"); ax.set_ylabel("AURC da massa isolada")
ax.set_title("perfil por camada (exploratório)", fontsize=9)
ax.set_xticks(range(0, 12, 2))
ax.legend(frameon=False, fontsize=6.6, loc="center right")
ax.text(0.02, 0.965, "tracejado = acaso\nabaixo é melhor", transform=ax.transAxes,
        va="top", ha="left", fontsize=6.4, color="#444444")

fig.tight_layout()
fig.savefig("resultado_piloto.png", dpi=300, bbox_inches="tight")

r = fig.canvas.get_renderer()
tx = [(t, t.get_window_extent(r)) for t in fig.findobj(mpl.text.Text)
      if t.get_text().strip() and t.get_visible()]
ov = [(a.get_text()[:24], b.get_text()[:24]) for i, (a, ba) in enumerate(tx)
      for b, bb in tx[i+1:] if ba.overlaps(bb)]
print("sobreposições:", ov[:6], "| total", len(ov))