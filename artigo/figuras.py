"""The manuscript figures, drawn from the same tables as numeros.py.

    python figuras.py      # writes img/fig_mechanism.pdf|png and img/fig_tallies.pdf|png
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
import numeros as N  # noqa: E402  (executes: recomputes every value from the tables)

OUT = AQUI / "img"
OUT.mkdir(exist_ok=True)
BLUE, ORANGE, GREY = "#2f5d8c", "#c4622d", "#8c8c8c"
FAV, NUL, ADV = "#1b7f7a", "#d9d9d9", "#c4622d"


def estilo():
    mpl.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
                         "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "savefig.dpi": 300})


def fig_mecanismo():
    """AURC of attention mass vs AURC of the pure geometric fraction, both minus chance."""
    fig, ax = plt.subplots(figsize=(3.4, 3.1))
    MARK = {"gliner-base": "o", "gliner-large": "s", "gliner_base-ft-genia": "D",
            "gliner_base-ft-conll2003": "D"}
    enc = N.T.copy()
    enc["x"] = enc.geo - enc.chance; enc["y"] = enc.mass - enc.chance
    dec = N.ed.copy()
    dec["x"] = dec.aurc_geometric_fraction - dec.aurc_acaso
    dec["y"] = dec.aurc_span_mass - dec.aurc_acaso
    lim = 1.12 * max(np.abs(np.r_[enc.x, enc.y, dec.x, dec.y]).max(), 0.01)
    ax.axhline(0, color=GREY, lw=0.6, zorder=1); ax.axvline(0, color=GREY, lw=0.6, zorder=1)
    ax.plot([-lim, lim], [-lim, lim], color=GREY, lw=0.8, ls="--", zorder=1)
    for _, r in enc.iterrows():
        c = BLUE if r.corpus == "genia" else ORANGE
        ax.scatter(r.x, r.y, marker=MARK[r.modelo], s=34, color=c, edgecolor="white", lw=0.6, zorder=3)
    for pt, r in dec.iterrows():
        c = BLUE if "genia" in pt else ORANGE
        ax.scatter(r.x, r.y, marker="^" if "05b" in pt else "v", s=36, facecolor="white",
                   edgecolor=c, lw=1.1, zorder=3)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal")
    ax.set_xlabel("AURC of geometric fraction $k/|K|$ − chance")
    ax.set_ylabel("AURC of attention mass − chance")
    ax.set_title("Attention mass tracks the geometric fraction, sign included", loc="left")
    ax.text(0.05 * lim, 0.95 * lim, "both worse\nthan chance", fontsize=7, color=GREY, va="top", ha="left")
    ax.text(-0.95 * lim, -0.12 * lim, "both better\nthan chance", fontsize=7, color=GREY, va="top", ha="left")
    ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(4)); ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(4, prune="lower"))
    from matplotlib.lines import Line2D
    h = [Line2D([], [], ls="", marker="o", color=BLUE, label="GENIA"),
         Line2D([], [], ls="", marker="o", color=ORANGE, label="CoNLL-2003"),
         Line2D([], [], ls="", marker="o", color="k", mfc="k", label="GLiNER base"),
         Line2D([], [], ls="", marker="s", color="k", label="GLiNER large"),
         Line2D([], [], ls="", marker="D", color="k", label="GLiNER fine-tuned"),
         Line2D([], [], ls="", marker="^", mfc="white", color="k", label="Qwen2.5-0.5B (causal)"),
         Line2D([], [], ls="", marker="v", mfc="white", color="k", label="Qwen2.5-1.5B (causal)")]
    ax.legend(handles=h, frameon=False, loc="center left", bbox_to_anchor=(1.01, 0.5), handletextpad=0.3)
    ax.margins(0.04)
    return fig, dict(enc=enc[["modelo", "corpus", "x", "y"]], dec=dec[["x", "y"]])


def fig_contagens():
    """Share of favourable / null / adverse intervals per comparison family, with chance."""
    geo, tar = N.geo[N.geo.tipo == "verdict"], N.tar
    t9 = pd.read_csv(N.R / "confirmatorio_sinais_tarefa.csv"); t9 = t9[t9.contra == "model_confidence"]
    t78 = pd.read_csv(N.R / "confirmatorio_sinais_ajustado_tarefa.csv"); t78 = t78[t78.contra == "model_confidence"]
    todos = N.todos
    ATT_DEC = ["span_mass", "row_entropy_causal", "row_max_causal"]
    G = [("GLiNER — AURC", [
            ("attention mass vs. geometric fraction", geo[geo.comparacao == "C2"]),
            ("confidence + deconfounded attention vs. confidence", geo[geo.comparacao == "C3"])]),
         ("GLiNER — review load vs. confidence", [
            ("attention mass", tar[tar.comparacao == "T2"]),
            ("deconfounded attention (enrichment)", tar[tar.comparacao == "T3"]),
            ("confidence + deconfounded attention", tar[tar.comparacao == "T4"]),
            ("row entropy, row maximum, emitted mass", pd.concat([t9[t9.familia == "atencao"], t78[t78.familia == "atencao"]]))]),
         ("Qwen2.5 0.5B/1.5B — AURC, pooled stratum", [
            ("attention signals vs. confidence", todos[todos.escore.isin(ATT_DEC) & (todos.contra == "model_confidence")]),
            ("attention signals vs. AggSeq", todos[todos.escore.isin(ATT_DEC) & (todos.contra == "aggseq")])])]
    linhas, ys, cab, pos = [], [], [], 0
    for grupo, itens in G:
        cab.append((grupo, pos)); pos += 1
        for rot, d in itens:
            linhas.append((grupo, rot, N.contar(d))); ys.append(pos); pos += 1
    n = len(linhas)
    fig, ax = plt.subplots(figsize=(6.8, 0.24 * pos + 0.9))
    y = pos - 1 - np.array(ys)
    for yi, (_, rot, c) in zip(y, linhas):
        esq = 0.0
        for k, cor in (("fav", FAV), ("nul", NUL), ("adv", ADV)):
            w = c[k] / c["n"]; ax.barh(yi, w, left=esq, color=cor, height=0.66, edgecolor="white", lw=0.5); esq += w
        ax.text(1.01, yi, f"{c['fav']}/{c['n']} fav · {c['adv']}/{c['n']} adv"
                + (f" · {c['deg']} degen." if c["deg"] else ""), va="center", fontsize=7)
    ax.axvline(0.025, color="k", lw=0.8, ls=":")
    ax.set_yticks(y); ax.set_yticklabels([r for _, r, _ in linhas])
    for g, pg in cab:
        ax.text(-0.005, pos - 1 - pg, g, transform=ax.get_yaxis_transform(), ha="right", va="center",
                fontsize=8, weight="bold")
    ax.set_xlim(0, 1); ax.set_ylim(-0.6, pos - 0.4)
    ax.set_xlabel("share of non-degenerate 95% intervals")
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    ax.legend(handles=[Patch(color=FAV, label="favourable (CI < 0)"), Patch(color=NUL, label="contains 0"),
                       Patch(color=ADV, label="adverse (CI > 0)"),
                       Line2D([], [], color="k", lw=0.8, ls=":", label="chance rate of a favourable (2.5%)")],
              frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.45, -0.12))
    ax.set_title("Outcome of every declared interval, by comparison", loc="left")
    return fig, linhas


def fig_reducao():
    """Share of the risk of random abstention each signal removes, per extractor and corpus."""
    RR = N.RR
    fam = N.FAMILIA_RR
    def melhor(sigs):
        x = RR[RR.sinal.isin(sigs)]
        return x.loc[x.groupby(["extrator", "corpus"]).rr.idxmax()]
    LIN = [("model confidence", RR[RR.sinal == "model_confidence"]),
           ("span length $k$", RR[RR.sinal == "span_size"]),
           ("sentence length $T$", RR[RR.sinal == "sentence_length"]),
           ("geometric fraction $k/|K|$", RR[RR.sinal == "geometric_fraction"]),
           ("attention mass received", RR[RR.sinal == "span_mass"]),
           ("attention row entropy", RR[RR.sinal.isin(["row_entropy", "row_entropy_causal"])]),
           ("attention row maximum", RR[RR.sinal.isin(["row_max", "row_max_causal"])]),
           ("attention mass emitted", RR[RR.sinal == "emitted_mass"])]
    MARK = {"gliner-base": ("o", True), "gliner-large": ("s", True), "gliner_base-ft-genia": ("D", True),
            "gliner_base-ft-conll2003": ("D", True), "qwen05b-ft-genia": ("^", False), "qwen05b-ft-conll2003": ("^", False),
            "qwen15b-ft-genia": ("v", False), "qwen15b-ft-conll2003": ("v", False)}
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    n = len(LIN); y0 = np.arange(n)[::-1]
    off = {"genia": 0.16, "conll2003": -0.16}
    for yi, (rot, d) in zip(y0, LIN):
        ax.axhspan(yi - 0.45, yi + 0.45, color="#f4f4f4" if yi % 2 else "white", zorder=0, lw=0)
        for _, r in d.iterrows():
            m, cheio = MARK[r.extrator]; c = BLUE if r.corpus == "genia" else ORANGE
            dy = {"gliner-base": -0.06, "gliner-large": 0.0, "gliner_base-ft-genia": 0.06,
                  "gliner_base-ft-conll2003": 0.06}.get(r.extrator, 0.0 if "05b" in r.extrator else 0.05)
            ax.scatter(100 * r.rr, yi + off[r.corpus] + dy, marker=m, s=30, zorder=3, lw=1.0,
                       facecolor=c if cheio else "white", edgecolor=c if not cheio else "white")
    ax.axvline(0, color="k", lw=0.8)
    ax.set_yticks(y0); ax.set_yticklabels([r for r, _ in LIN])
    ax.set_xlabel("risk removed relative to random abstention (%)   [1 − AURC / AURC$_{chance}$]")
    ax.set_title("Confidence removes risk; attention does no better than the lengths it reflects", loc="left")
    ax.text(1, n - 0.35, "better than chance →", fontsize=7, color=GREY, va="center")
    ax.text(-1, n - 0.35, "← worse", fontsize=7, color=GREY, va="center", ha="right")
    ax.set_ylim(-0.6, n - 0.1)
    from matplotlib.lines import Line2D
    h = [Line2D([], [], ls="", marker="o", color=BLUE, label="GENIA"),
         Line2D([], [], ls="", marker="o", color=ORANGE, label="CoNLL-2003"),
         Line2D([], [], ls="", marker="o", color="k", label="GLiNER base"),
         Line2D([], [], ls="", marker="s", color="k", label="large"),
         Line2D([], [], ls="", marker="D", color="k", label="fine-tuned"),
         Line2D([], [], ls="", marker="^", mfc="white", color="k", label="Qwen2.5-0.5B"),
         Line2D([], [], ls="", marker="v", mfc="white", color="k", label="1.5B")]
    ax.legend(handles=h, frameon=False, ncol=7, loc="upper center", bbox_to_anchor=(0.38, -0.16),
              handletextpad=0.2, columnspacing=0.8)
    ax.margins(x=0.04)
    return fig


def fig_leituras():
    """The 2,496 readings per corpus: change in AURC of confidence when each is combined with it."""
    P = pd.read_csv(N.R / "perimetro1_leituras.csv")
    DIR = {"recebida": ("received", BLUE), "auto_foco": ("self-focus", ORANGE),
           "simetrica": ("their mean", "#6a9f58"), "recebida_por_token": ("received per token", GREY)}
    fig, axs = plt.subplots(1, 2, figsize=(6.8, 2.6), sharey=True)
    fig.subplots_adjust(wspace=0.12)
    for ax, (corp, nome) in zip(axs, (("genia", "GENIA"), ("conll2003", "CoNLL-2003"))):
        d = P[P.corpus == corp]
        lo, hi = P.delta_vs_confianca.quantile([0.002, 0.998])
        bins = np.linspace(lo, hi, 41)
        base = np.zeros(len(bins) - 1)
        for k_, (rot, cor) in DIR.items():
            h, _ = np.histogram(d[d.direcao == k_].delta_vs_confianca.clip(lo, hi), bins=bins)
            ax.bar(bins[:-1], h, width=np.diff(bins), bottom=base, align="edge", color=cor, lw=0, label=rot)
            base += h
        ax.axvline(0, color="k", lw=0.8)
        ax.set_title(f"{nome}: weight 0 in {100 * (d.peso_na_leitura == 0).mean():.0f}% of readings", loc="left")
        ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(4))
    fig.supxlabel("change in AURC of confidence when combined with the reading (negative = better)", fontsize=8, y=-0.04)
    axs[0].set_ylabel("readings")
    axs[0].legend(frameon=False, title="direction", loc="upper left", fontsize=6.5, title_fontsize=7)
    return fig


def fig_nulo():
    """Illustration of the two references for a 60-token sentence (formula only, no data)."""
    from src.selective.geometry import expected_mass_causal
    T = 60
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    for k_, cor in ((1, BLUE), (4, ORANGE)):
        a = np.arange(0, T - k_ + 1)
        ax.plot(a, [float(expected_mass_causal(int(x), k_, T)) for x in a], color=cor, lw=1.4, label=f"causal, $k={k_}$")
        ax.axhline(k_ / T, color=cor, lw=1.0, ls="--", label=f"bidirectional, $k={k_}$")
    ax.set_xlabel("position of the span's first token")
    ax.set_ylabel("expected attention mass")
    ax.legend(frameon=False, fontsize=6.5)
    ax.set_title("Reference for a span in a 60-token sentence", loc="left")
    return fig


def verificar(fig):
    r = fig.canvas.get_renderer()
    tx = [(t.get_text()[:30], t.get_window_extent(r)) for t in fig.findobj(mpl.text.Text)
          if t.get_text().strip() and t.get_visible()]
    return [(a[0], b[0]) for i, a in enumerate(tx) for b in tx[i + 1:] if a[1].overlaps(b[1])]


if __name__ == "__main__":
    estilo()
    f1, d1 = fig_mecanismo()
    for ext in ("pdf", "png"):
        f1.savefig(OUT / f"fig_mechanism.{ext}", bbox_inches="tight")
    sys.path.insert(0, str(N.R.parents[2]))
    for nome_, fn_ in (("fig_readings", fig_leituras), ("fig_null", fig_nulo)):
        f_ = fn_()
        for ext in ("pdf", "png"):
            f_.savefig(OUT / f"{nome_}.{ext}", bbox_inches="tight")
        print(f"overlaps {nome_}:", verificar(f_))
    f3 = fig_reducao()
    for ext in ("pdf", "png"):
        f3.savefig(OUT / f"fig_risk.{ext}", bbox_inches="tight")
    print("overlaps fig risk:", verificar(f3))
    f2, d2 = fig_contagens()
    for ext in ("pdf", "png"):
        f2.savefig(OUT / f"fig_tallies.{ext}", bbox_inches="tight")
    print("overlaps fig1:", verificar(f1)); print("overlaps fig2:", verificar(f2))
    print(d1["enc"].round(4).to_string(index=False)); print(d1["dec"].round(4).to_string())
