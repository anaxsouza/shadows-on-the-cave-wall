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

AZ, VM, LR, CZ, VD, RX = "#1F5FA9", "#C1272D", "#E08214", "#8C8C8C", "#2E7D32", "#7B3294"


def cx(ax, x, ytopo, w, titulo, corpo, ec, fc, etiqueta=None, ft=8.2, fb=6.4):
    nt, nb = titulo.count("\n")+1, corpo.count("\n")+1
    h = 3.6 + nt*ft*0.325 + nb*fb*0.345 + (2.4 if etiqueta else 0)
    ax.add_patch(FancyBboxPatch((x-w/2, ytopo-h), w, h, boxstyle="round,pad=0.45,rounding_size=0.9",
                                lw=1.5, ec=ec, fc=fc, zorder=2))
    if etiqueta:
        ax.text(x, ytopo-1.6, etiqueta, ha="center", va="top", fontsize=5.9, color=ec,
                weight="bold", zorder=3)
        yt = ytopo-3.9
    else:
        yt = ytopo-2.0
    ax.text(x, yt, titulo, ha="center", va="top", fontsize=ft, weight="bold", zorder=3, linespacing=1.3)
    ax.text(x, ytopo-h+1.5, corpo, ha="center", va="bottom", fontsize=fb, color="#333333",
            zorder=3, linespacing=1.42)
    return ytopo - h


def flecha(ax, a, b, texto="", cor="#555555", dx=1.2):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=12, lw=1.25, color=cor, zorder=1))
    if texto:
        ax.text((a[0]+b[0])/2+dx, (a[1]+b[1])/2, texto, ha="left", va="center", fontsize=6.3,
                color=cor, style="italic", bbox=dict(fc="white", ec="none", pad=1.0), zorder=3)


fig, ax = plt.subplots(figsize=(10.4, 13.2))
ax.set_xlim(0, 100); ax.set_ylim(0, 118); ax.axis("off")
XC, W, G = 37, 56, 3.6

ax.add_patch(FancyBboxPatch((XC-W/2, 112.6), W, 4.4, boxstyle="round,pad=0.5,rounding_size=1.0",
                            lw=1.8, ec=VM, fc="#FBE9E9", zorder=2))
ax.text(XC, 114.8, "A massa de atenção sobre o span acrescenta poder de decisão\nsobre a confiança do PRÓPRIO modelo?",
        ha="center", va="center", fontsize=9.8, weight="bold", linespacing=1.3)

y = cx(ax, XC, 110.0, W, "ETAPA 1 · MEDIR, E SÓ UMA VEZ",
       "gliner-base prevê E fornece a atenção — os mesmos pesos\n"
       "GENIA (aninhado) e CoNLL-2003 (plano)\n"
       "validação = onde se explora   ·   teste = CONGELADO\n"
       "saída: uma linha por entidade PREDITA, com o tensor de atenção",
       AZ, "#E9F0F8", etiqueta="FEITO · 4 medições, 27.304 entidades")
flecha(ax, (XC, y-0.2), (XC, y-G), "a ficha de cada palpite")
y = cx(ax, XC, y-G, W, "errou? (0/1) · confiança do modelo · massa de atenção · aninhada?\ntokens do span  →  k = tamanho,  T = tokens da sentença",
       "", AZ, "#F4F8FC", etiqueta="entities.csv", ft=6.9)
flecha(ax, (XC, y-0.2), (XC, y-G), "quatro ordenações candidatas")

yt2 = y-G; h2 = 14.4
ax.add_patch(FancyBboxPatch((XC-W/2, yt2-h2), W, h2, boxstyle="round,pad=0.45,rounding_size=0.9",
                            lw=1.5, ec=VD, fc="#EDF5ED", zorder=2))
ax.text(XC, yt2-1.6, "ETAPA 2 · OS QUATRO ESCORES", ha="center", va="top", fontsize=5.9,
        color=VD, weight="bold", zorder=3)
for i, (t, s, c) in enumerate([("confiança\ndo modelo", "o número que\no extrator declara", AZ),
                               ("massa de\natenção", "fatia do olhar da\nsentença sobre o span", VM),
                               ("k", "tamanho do span\nSEM MODELO", LR),
                               ("k / T", "fração do lugar\nocupado · SEM MODELO", LR)]):
    xi = XC - W/2 + 3.2 + (W-6.4)*(i+0.5)/4
    ax.add_patch(FancyBboxPatch((xi-6.0, yt2-h2+1.5), 12.0, 9.6,
                                boxstyle="round,pad=0.3,rounding_size=0.6", lw=1.2, ec=c, fc="white", zorder=3))
    ax.text(xi, yt2-h2+10.5, t, ha="center", va="top", fontsize=7.4, weight="bold", color=c,
            zorder=4, linespacing=1.25)
    ax.text(xi, yt2-h2+2.3, s, ha="center", va="bottom", fontsize=5.7, color="#444444",
            zorder=4, linespacing=1.35)
y = yt2 - h2
flecha(ax, (XC, y-0.2), (XC, y-G), "cada escore ORDENA os palpites")

y = cx(ax, XC, y-G, W, "ETAPA 3 · ORDENAR, ENTREGAR, MEDIR O ERRO",
       "ordena do mais confiável ao menos · entrega os X% do topo\n"
       "mede o erro ENTRE OS ENTREGUES · varre X de 100% a ~0\n"
       "a área sob a curva (AURC) é um número só · MENOR É MELHOR\n"
       "sorteio = linha horizontal na taxa de erro base",
       RX, "#F3EAF7", etiqueta="curva risco-cobertura")
flecha(ax, (XC, y-0.2), (XC, y-G), "três comparações, e só três")

ytc = y-G; h3 = 20.2
ax.add_patch(FancyBboxPatch((XC-W/2, ytc-h3), W, h3, boxstyle="round,pad=0.45,rounding_size=0.9",
                            lw=1.5, ec=VM, fc="#FDF2F2", zorder=2))
ax.text(XC, ytc-1.6, "ETAPA 4 · AS TRÊS COMPARAÇÕES", ha="center", va="top", fontsize=5.9,
        color=VM, weight="bold", zorder=3)
for i, (rot, o, perg, est, c) in enumerate(
    [("C1 · PISO", "massa  vs  sorteio", "a atenção ordena melhor que o acaso?",
      "GENIA sim · CoNLL NÃO, invertido", VD),
     ("C2 · A TESE", "conf+massa  vs  conf", "acrescenta ao que o modelo já dá?",
      "refutado nos dois: IC contém zero", VD),
     ("C3 · O CONTROLE", "massa  vs  k , k/T", "o que a atenção dá, contar tokens já dava?",
      "k/T dá o MESMO: −0,0068 vs −0,0067", LR)]):
    yb = ytc - 4.4 - i*5.2
    ax.add_patch(FancyBboxPatch((XC-W/2+2.4, yb-4.4), W-4.8, 4.4,
                                boxstyle="round,pad=0.25,rounding_size=0.5", lw=1.1, ec=c, fc="white", zorder=3))
    ax.text(XC-W/2+4.0, yb-1.5, rot, ha="left", va="center", fontsize=6.6, weight="bold", color=c, zorder=4)
    ax.text(XC-W/2+16.0, yb-1.5, o, ha="left", va="center", fontsize=6.7, family="monospace", zorder=4)
    ax.text(XC-W/2+4.0, yb-3.2, perg, ha="left", va="center", fontsize=6.0, color="#333333", style="italic", zorder=4)
    ax.text(XC+W/2-4.0, yb-3.2, est, ha="right", va="center", fontsize=6.0, color=c, weight="bold", zorder=4)
y = ytc - h3
flecha(ax, (XC, y-0.2), (XC, y-G), "critério declarado ANTES de medir")

y = cx(ax, XC, y-G, W, "O CRITÉRIO DE REFUTAÇÃO",
       "diferença de AURC pareada · IC 95% com 2.000 reamostragens\n"
       "reamostra SENTENÇAS, não entidades — as da mesma sentença\n"
       "compartilham a matriz de atenção\n"
       "IC CONTÉM ZERO → não acrescenta. Sem exceção depois de ver.",
       "#333333", "#F5F5F5", etiqueta="pré-registro assinado em commit ANTERIOR à medição")

XL, WL = 84, 26
yl = cx(ax, XL, 108.0, WL, "O QUE A LITERATURA\nJÁ EXIGE",
        "Saghir (2026) · arXiv 2605.00269\n\n"
        "sinais de atenção em LLM são\nESTRUTURALMENTE confundidos\n"
        "por comprimento (|r| ≥ 0,61) e\ncaem a 0,491–0,527 (acaso 0,5)\nquando se controla\n\n"
        "base trivial de contar tokens,\nsem modelo: 0,874 e 0,919 —\nigual aos métodos elaborados\n\n"
        "TEORIA: a atenção opera sobre\num simplex que depende do\n"
        "comprimento, então QUALQUER\nagregado herda Θ(log T)\n\n"
        "→ prevê o que medimos: nenhuma\ndas 2.496 leituras escapa",
        CZ, "#F2F2F2", etiqueta="NÃO é nossa descoberta", ft=7.4, fb=5.9)
yl = cx(ax, XL, yl-3.0, WL, "ONDE ESTÁ O NOSSO",
        "neles T é o comprimento da\nENTRADA. Aqui a sentença é fixa\ne varia a fração k/T que o span\nocupa dentro dela\n\n"
        "medido: massa = 0,52·(k/T)\ncom R² = 96%\n\n"
        "e o que eles não relatam: em\ntexto plano a massa é PIOR que\no acaso, não igual",
        VM, "#FDF2F2", etiqueta="A CONTRIBUIÇÃO", ft=7.4, fb=5.9)
yl = cx(ax, XL, yl-3.0, WL, "O QUE FALTA",
        "decl-02 declarando k e k/T como\nbases de comparação, assinada\nantes de medir no TESTE\n\n"
        "hoje C3 é EXPLORATÓRIO:\nvalidação, sem intervalo",
        LR, "#FCEBD8", etiqueta="PENDENTE · decisão sua", ft=7.4, fb=5.9)
ax.text(XL, yl-2.6, "azul = vem do modelo\nlaranja = SEM modelo\nverde = já respondido",
        ha="center", va="top", fontsize=5.9, color="#555555", linespacing=1.5)

ax.set_ylim(-10, 118)
fig.savefig("protocolo.png", dpi=300, bbox_inches="tight")