# Recuperado da linhagem do artefato que produziu a peça publicada; só os CAMINHOS
# foram trocados para o repositório. Rode por reproduzir/derivadas.py, que o executa
# em saida/derivadas/ e compara o que ele escreve com o publicado.
from pathlib import Path as _P
RAIZ = _P(__file__).resolve().parents[2]
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

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


apply_figure_style(sizes=(9, 8, 7))

EST = {
    "lit":    ("#8C8C8C", "#F2F2F2", "debate estabelecido — não é nó seu"),
    "resp":   ("#1F5FA9", "#E3ECF7", "RESPONDIDO"),
    "pronto": ("#E08214", "#FCEBD8", "SUBSTRATO PRONTO"),
    "inconc": ("#7B3294", "#EFE3F4", "INCONCLUSIVO"),
    "aberto": ("#8C8C8C", "#FFFFFF", "ABERTO"),
}


def seta(ax, a, b, texto="", cor="#555555", est="-|>", rad=0.0, dx=0, lado="c", fs=6.2):
    p = FancyArrowPatch(a, b, arrowstyle=est, mutation_scale=11, lw=1.15, color=cor,
                        connectionstyle=f"arc3,rad={rad}", zorder=1)
    ax.add_patch(p)
    if texto:
        mx, my = (a[0]+b[0])/2 + dx, (a[1]+b[1])/2
        ax.text(mx, my, texto, ha={"c":"center","l":"right","r":"left"}[lado], va="center",
                fontsize=fs, color=cor, style="italic",
                bbox=dict(fc="white", ec="none", pad=1.2), zorder=3)


def altura(t, s):
    return 4.6 + 2.3 * (t.count("\n") + 1) + 2.0 * (s.count("\n") + 1)


def caixa(ax, x, ytopo, w, t, s, est):
    ec, fc, rot = EST[est]
    h = altura(t, s)
    ax.add_patch(FancyBboxPatch((x - w/2, ytopo - h), w, h,
                                boxstyle="round,pad=0.5,rounding_size=1.0",
                                lw=1.5, ec=ec, fc=fc, zorder=2))
    ax.text(x, ytopo - 1.9, rot, ha="center", va="top", fontsize=6.0,
            color=ec, weight="bold", zorder=3)
    ax.text(x, ytopo - 4.3, t, ha="center", va="top", fontsize=8.3, weight="bold",
            zorder=3, linespacing=1.32)
    ax.text(x, ytopo - h + 1.9, s, ha="center", va="bottom", fontsize=6.3,
            color="#333333", zorder=3, linespacing=1.42)
    return ytopo - h


CENTRAL = [
 ("Q0", "lit", "Q0 · Pesos de atenção carregam informação decisória?",
  "Jain & Wallace 2019 · Wiegreffe & Pinter 2019 · Serrano & Smith 2019\nresposta da literatura: CONTESTADO"),
 ("Q1", "resp", "Q1 · A massa de atenção ordena erro melhor que o acaso?",
  "GENIA: SIM  ΔAURC −0,042 [−0,058; −0,027]\nCoNLL-2003: NÃO, SINAL INVERTIDO  +0,032 [+0,019; +0,044]"),
 ("Q2", "resp", "Q2 · Ela acrescenta sobre a confiança do modelo?",
  "NÃO nos dois: −0,001 [−0,004; +0,002] e +0,001 [−0,000; +0,003]\npeso convexo colapsou: 0,97/0,03 e 0,99/0,01"),
 ("Q4", "aberto", "Q4 · Se não é a atenção, onde mora a informação?",
  "estados ocultos, sondagem · Kadavath et al. 2022\nnenhum desses trabalhos é sobre NER"),
 ("Q6", "inconc", "Q6 · A abstenção dá garantia utilizável?",
  "monotonicidade violada nas 4 execuções\nnenhum risco alvo alcançável no GENIA"),
]
ROTULO_ARESTA = {
 "Q0": "torna a pergunta DECIDÍVEL:  'explica?' → 'decide?'",
 "Q1": "pré-condição prática",
 "Q2": None,
 "Q4": "RAMIFICA porque a resposta foi NÃO",
 "Q6": "só faz sentido se ALGUM sinal acrescentar",
}

fig, ax = plt.subplots(figsize=(9.8, 11.2))
ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")
W, XC, GAP = 46, 50, 5.4

ax.add_patch(FancyBboxPatch((XC - W/2, 92.5), W, 6.4, boxstyle="round,pad=0.6,rounding_size=1.2",
                            lw=1.7, ec="#C1272D", fc="#FBE9E9", zorder=2))
ax.text(XC, 97.6, "PERGUNTA CENTRAL", ha="center", va="top", fontsize=7.0,
        color="#C1272D", weight="bold")
ax.text(XC, 95.2, "A atenção adiciona poder de decisão ao modelo?", ha="center", va="center",
        fontsize=11, weight="bold")

y = 92.5
pos = {}
for k, est, t, s in CENTRAL:
    rot = ROTULO_ARESTA[k]
    cor = "#C1272D" if k == "Q4" else "#555555"
    ytopo = y - GAP
    ax.add_patch(FancyArrowPatch((XC, y - 0.4), (XC, ytopo + 0.4), arrowstyle="-|>",
                                 mutation_scale=12, lw=1.2, color=cor, zorder=1))
    if rot:
        ax.text(XC + 1.4, y - GAP/2, rot, ha="left", va="center", fontsize=6.4,
                color=cor, style="italic", bbox=dict(fc="white", ec="none", pad=1.2), zorder=3)
    ybase = caixa(ax, XC, ytopo, W, t, s, est)
    pos[k] = (ytopo, ybase)
    y = ybase

LAT = [("Q3", "pronto", 13.5, "#E08214",
        "Q3 · O veredito depende de\nQUAL leitura de atenção?",
        "> 15 mil leituras recalculáveis\nsem rodar o modelo de novo\nvalidação já medida"),
       ("Q5", "aberto", 86.5, "#8C8C8C",
        "Q5 · Sobrevive a um\nextrator competente?",
        "hoje erra 48% e 53%\ne sobreprediz\nJagannatha & Yu 2020")]
topo_q1, _ = pos["Q1"]; _, base_q2 = pos["Q2"]
meio = (topo_q1 + base_q2) / 2
for k, est, x, cor, t, s in LAT:
    h = altura(t, s)
    yb = caixa(ax, x, meio + h/2, 25, t, s, est)
    lado = 1 if x < XC else -1
    borda = x + lado * 12.5
    for alvo in (topo_q1 - 4.0, base_q2 + 4.0):
        ax.add_patch(FancyArrowPatch((borda, meio), (XC - lado * W/2, alvo), arrowstyle="-|>",
                                     mutation_scale=11, lw=1.15, color=cor,
                                     connectionstyle=f"arc3,rad={0.14 if alvo > meio else -0.14}",
                                     zorder=1))
    ax.text(x, yb - 2.4, "condição de VALIDADE de Q1 e Q2,\nnão consequência delas",
            ha="center", va="top", fontsize=6.3, color=cor, style="italic", linespacing=1.4)

ax.add_patch(FancyBboxPatch((4, 1.2), 92, 5.6, boxstyle="round,pad=0.5,rounding_size=0.9",
                            lw=1.2, ec="#C1272D", fc="#FFFFFF", ls="--", zorder=2))
ax.text(50, 5.5, "TESTE PARA SABER SE É NÓ  ·  se a resposta for \u201cnão\u201d, os nós abaixo ainda fazem sentido?",
        ha="center", va="top", fontsize=7.6, weight="bold", color="#C1272D", zorder=3)
ax.text(50, 2.9, "Se fizerem, não é nó — é item de lista, e o desenho paralelo que a banca rejeitou volta com nome novo.",
        ha="center", va="bottom", fontsize=6.6, color="#333333", zorder=3)

fig.savefig("arvore_perguntas.png", dpi=300, bbox_inches="tight")