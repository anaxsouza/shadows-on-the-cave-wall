"""Every number the manuscript cites, computed from the published tables.

    python numeros.py [<attention-selective-ner>/docs/tese/resultados]

Writes `numeros.tex` (one \\newcommand per value) and `numeros.json` (the same
values, for audit). The manuscript never types a result: it calls a macro, so a
number in the PDF can only come from a table that `reproduzir/analise.py` regenerates.

Counting convention, used everywhere: an interval is FAVOURABLE when the whole 95% CI
is below zero (lower AURC / lower review load than the baseline), ADVERSE when it is
entirely above zero, DEGENERATE when both bounds are exactly zero (the combination
collapsed onto the baseline, or neither score reaches the target), NULL otherwise.
Under the null of no difference each non-degenerate interval lands favourable with
probability 0.025. The intervals within a declaration share entities (strata are
nested), so n x 0.025 is a calibration of scale, not a test.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

AQUI = Path(__file__).resolve().parent
# Inside the public repository this file lives in artigo/, next to docs/; in the
# development tree it lives in writing/papers/<slug>/, beside the public repository.
_DENTRO = AQUI.parent / "docs" / "tese" / "resultados"
R = Path(sys.argv[1] if len(sys.argv) > 1 else os.environ.get(
    "ASNER_RESULTADOS", _DENTRO if _DENTRO.exists() else
    AQUI.parents[3] / "attention-selective-ner" / "docs" / "tese" / "resultados"))
M: dict[str, str] = {}


def put(nome: str, valor, fmt: str = "{}") -> None:
    assert nome.isalpha(), nome
    M[nome] = fmt.format(valor)


def f4(x: float) -> str:
    return f"{x:.4f}".replace("-", "\\text{-}") if False else f"{x:.4f}"


def ler(f: str, modelo: str | None = None) -> pd.DataFrame:
    d = pd.read_csv(R / f)
    if "modelo" not in d.columns:
        d.insert(0, "modelo", modelo)
    return d


def contar(d: pd.DataFrame) -> dict:
    ic = d[d.ci_low.notna() & d.ci_high.notna()]
    deg = (ic.ci_low == 0) & (ic.ci_high == 0)
    fav, adv = ic.ci_high < 0, ic.ci_low > 0
    n = int((~deg).sum())
    return dict(n=n, fav=int(fav.sum()), adv=int(adv.sum()), deg=int(deg.sum()),
                nul=n - int(fav.sum()) - int(adv.sum()), acaso=round(n * 0.025, 1))


def put_contagem(prefixo: str, c: dict) -> None:
    for k, v in c.items():
        put(prefixo + {"n": "N", "fav": "Fav", "adv": "Adv", "deg": "Deg", "nul": "Nul",
                       "acaso": "Chance"}[k], v)


ROMANO = {"gliner-base": "Base", "gliner-large": "Large",
          "gliner_base-ft-genia": "Tuned", "gliner_base-ft-conll2003": "Tuned"}
CORPUS = {"genia": "Genia", "conll2003": "Conll"}

# ---------------------------------------------------------------- encoder, C1-C3
geo = pd.concat([ler("confirmatorio_decl02.csv", "gliner-base"),
                 ler("confirmatorio_decl04_geometria.csv"),
                 ler("confirmatorio_ajustado_geometria.csv")], ignore_index=True)
esc = geo[geo.comparacao == "escores"].pivot_table(
    index=["modelo", "corpus"], columns="metrica", values="valor", aggfunc="first")
c1 = geo[(geo.comparacao == "C1")].groupby(["modelo", "corpus", "metrica"]).valor.first().unstack()
tabela_enc = []
for (mod, corp), row in esc.iterrows():
    p = ROMANO[mod] + CORPUS[corp]
    for k, col in (("Chance", "aurc_acaso"), ("Mass", "aurc_span_mass"), ("Geo", "aurc_geometric_fraction"),
                   ("Size", "aurc_span_size"), ("Sent", "aurc_sentence_length"), ("Enr", "aurc_enrichment"), ("Conf", "aurc_model_confidence")):
        put(f"enc{p}{k}", row[col], "{:.3f}")
    r2 = c1.loc[(mod, corp), "R2_massa_vs_fracao_geometrica"]
    oe = c1.loc[(mod, corp), "mediana_observado_sobre_esperado"]
    put(f"enc{p}Rsq", r2, "{:.3f}"); put(f"enc{p}ObsExp", oe, "{:.2f}")
    tabela_enc.append(dict(modelo=mod, corpus=corp, chance=row.aurc_acaso, mass=row.aurc_span_mass,
                           geo=row.aurc_geometric_fraction, size=row.aurc_span_size,
                           conf=row.aurc_model_confidence, r2=r2))
T = pd.DataFrame(tabela_enc)
put("encRsqMin", T.r2.min(), "{:.2f}"); put("encRsqMax", T.r2.max(), "{:.2f}")
put("encMassGeoMaxGap", (T.mass - T.geo).abs().max(), "{:.3f}")
put("encMassWorseThanChance", int((T.mass > T.chance).sum()))
LEN = esc.reset_index()
put("encConllSentBetter", int(((LEN.corpus == "conll2003") & (LEN.aurc_sentence_length < LEN.aurc_acaso)).sum()))
put("encGeniaSizeBetter", int(((LEN.corpus == "genia") & (LEN.aurc_span_size < LEN.aurc_acaso)).sum()))
oe_all = c1["mediana_observado_sobre_esperado"]
put("encObsExpMin", oe_all.min(), "{:.2f}"); put("encObsExpMax", oe_all.max(), "{:.2f}")

v = geo[geo.tipo == "verdict"]
put_contagem("encCtwo", contar(v[v.comparacao == "C2"]))
put_contagem("encCthree", contar(v[v.comparacao == "C3"]))
for mod in ("gliner-base", "gliner-large"):
    put_contagem(f"encCtwo{ROMANO[mod]}", contar(v[(v.comparacao == "C2") & (v.modelo == mod)]))
    put_contagem(f"encCthree{ROMANO[mod]}", contar(v[(v.comparacao == "C3") & (v.modelo == mod)]))
vt = v[v.modelo.str.contains("-ft-")]
put_contagem("encCtwoTuned", contar(vt[vt.comparacao == "C2"]))
put_contagem("encCthreeTuned", contar(vt[vt.comparacao == "C3"]))
# the pooled-stratum C2 for GENIA across the three extractors (the one that "disappears")
for mod in ("gliner-base", "gliner-large", "gliner_base-ft-genia"):
    r = v[(v.modelo == mod) & (v.corpus == "genia") & (v.comparacao == "C2") & (v.estrato == "todos")].iloc[0]
    p = ROMANO[mod]
    put(f"ctwoGenia{p}D", r.delta, "{:+.4f}"); put(f"ctwoGenia{p}Lo", r.ci_low, "{:+.4f}")
    put(f"ctwoGenia{p}Hi", r.ci_high, "{:+.4f}")

# ---------------------------------------------------------------- encoder, task side
tar = pd.concat([ler("confirmatorio_decl03_tarefa.csv", "gliner-base"), ler("confirmatorio_decl04_tarefa.csv"),
                 ler("confirmatorio_ajustado_tarefa.csv")], ignore_index=True)
tar = tar[tar.contra == "model_confidence"]
for comp, nome in (("T2", "Ttwo"), ("T3", "Tthree"), ("T4", "Tfour")):
    put_contagem(f"enc{nome}", contar(tar[tar.comparacao == comp]))

# ---------------------------------------------------------------- encoder, three families
def familias(prefixo, fg, ft):
    t = pd.read_csv(R / ft); t = t[t.contra == "model_confidence"]
    for fam, nome in (("atencao", "Att"), ("estados_ocultos", "Hid"), ("logits", "Logit"), ("geometrico", "Geo")):
        put_contagem(f"{prefixo}{nome}", contar(t[t.familia == fam]))
    put_contagem(f"{prefixo}All", contar(t))
    fav = t[(t.ci_high < 0)]
    return sorted(set(fav.supervisor)), sorted(set(fav.corpus))
put("famLargeFavSignals", ", ".join(s.replace("_", "\\_") for s in familias("famLarge", "confirmatorio_sinais_geometria.csv", "confirmatorio_sinais_tarefa.csv")[0]))
put("famTunedFavSignals", ", ".join(s.replace("_", "\\_") for s in familias("famTuned", "confirmatorio_sinais_ajustado_geometria.csv", "confirmatorio_sinais_ajustado_tarefa.csv")[0]))

# ---------------------------------------------------------------- fine-tuning, pilot, perimeter
for c, nome in (("genia", "Genia"), ("conll2003", "Conll")):
    a = json.loads((R / f"ajuste_{c}.json").read_text())
    put(f"ftFone{nome}Before", a["f1_val_antes"], "{:.3f}"); put(f"ftFone{nome}After", a["f1_val"], "{:.3f}")
pil = pd.read_csv(R / "resultado_piloto.csv")
for _, r in pil.iterrows():
    if r.teste == "piso":
        p = "Genia" if r.corpus == "GENIA" else "Conll"
        put(f"pilotFloor{p}D", float(r.delta), "{:+.4f}")
        put(f"pilotFloor{p}Lo", float(r.ic_baixo), "{:+.4f}"); put(f"pilotFloor{p}Hi", float(r.ic_alto), "{:+.4f}")
p1 = pd.read_csv(R / "perimetro1_leituras.csv")
put("periReadings", int(p1.groupby("corpus").size().iloc[0]))
for c, nome in (("genia", "Genia"), ("conll2003", "Conll")):
    g = p1[p1.corpus == c]
    put(f"peri{nome}ZeroWeight", 100 * (g.peso_na_leitura == 0).mean(), "{:.0f}")
    put(f"peri{nome}MaxWeight", g.peso_na_leitura.max(), "{:.2f}")
    put(f"peri{nome}BestDelta", g.delta_vs_confianca.min(), "{:+.4f}")
    put(f"peri{nome}SizeCorr", g.correlacao_com_tamanho.median(), "{:.2f}")

# ---------------------------------------------------------------- decoder
da = pd.read_csv(R / "decoder" / "decoder_aurc.csv")
dc = pd.read_csv(R / "decoder" / "decoder_carga.csv")
PT = {"qwen05b-ft-genia": "SmallGenia", "qwen05b-ft-conll2003": "SmallConll",
      "qwen15b-ft-genia": "LargeGenia", "qwen15b-ft-conll2003": "LargeConll"}
desc = da[da.tipo == "descriptive"]
_r2 = desc[(desc.metrica == "R2_massa_vs_fracao_geometrica") & (desc.comparacao == "D1")
           & (desc.estrato == "todos")].drop_duplicates(["ponto", "valor"])
assert _r2.ponto.is_unique, "R2 do decoder ambiguo"
r2d = _r2.set_index("ponto").valor
put("decRsqMin", r2d.min(), "{:.2f}"); put("decRsqMax", r2d.max(), "{:.2f}")
_ed = desc[desc.metrica.str.startswith("aurc_") & (desc.comparacao == "escores") & (desc.particao == "avaliação")
           & (desc.estrato == "todos")].drop_duplicates(["ponto", "metrica", "valor"])
assert not _ed.duplicated(["ponto", "metrica"]).any(), "AURC do decoder ambigua"
ed = _ed.pivot(index="ponto", columns="metrica", values="valor")
for pt, row in ed.iterrows():
    for k, col in (("Chance", "aurc_acaso"), ("Mass", "aurc_span_mass"), ("Geo", "aurc_geometric_fraction"),
                   ("Conf", "aurc_model_confidence"), ("Probe", "aurc_sonda_ocultos"), ("Size", "aurc_span_size")):
        put(f"dec{PT[pt]}{k}", row[col], "{:.3f}")
put("decMassWorseThanChance", int((ed.aurc_span_mass > ed.aurc_acaso).sum()))
# The decoder verdicts (C2) orient each score on the calibration part, as their declarations
# prescribe (NOTAS.json: orientation -1 for span_mass and geometric_fraction at all four points);
# the descriptive rows above keep the fixed direction (+1) used for the encoders. Both are
# reported, side by side, so the reader sees what the direction does to the decoder numbers.
_cal = da[(da.tipo == "verdict") & (da.origem != "adenda-01") & (da.comparacao == "C2")
          & (da.estrato == "todos")].drop_duplicates(["ponto"]).set_index("ponto")
assert set(_cal.index) == set(PT), _cal.index
assert (_cal.escore == "span_mass").all() and (_cal.contra == "geometric_fraction").all()
edc = ed.join(_cal[["aurc_escore", "aurc_contra", "delta", "ci_low", "ci_high"]])
for pt, row in edc.iterrows():
    put(f"dec{PT[pt]}MassCal", row.aurc_escore, "{:.3f}")
    put(f"dec{PT[pt]}GeoCal", row.aurc_contra, "{:.3f}")
put("decCalGapMin", edc.delta.min(), "{:.3f}"); put("decCalGapMax", edc.delta.max(), "{:.3f}")
put("decCalGapAdv", int((edc.ci_low > 0).sum()))
put("decRawGapMin", (edc.aurc_span_mass - edc.aurc_geometric_fraction).min(), "{:.3f}")
put("decRawGapMax", (edc.aurc_span_mass - edc.aurc_geometric_fraction).max(), "{:.3f}")
_rrm = 1 - edc.aurc_escore / edc.aurc_acaso; _rrg = 1 - edc.aurc_contra / edc.aurc_acaso
put("rrDecMassCalMin", 100 * _rrm.min(), "{:.0f}"); put("rrDecMassCalMax", 100 * _rrm.max(), "{:.0f}")
put("rrDecGeoCalMin", 100 * _rrg.min(), "{:.0f}"); put("rrDecGeoCalMax", 100 * _rrg.max(), "{:.0f}")
put("decCalBothBeatChance", int(((edc.aurc_escore < edc.aurc_acaso) & (edc.aurc_contra < edc.aurc_acaso)).sum()))
UNSUP = ["span_mass", "row_entropy_causal", "row_max_causal", "hidden_norm", "hidden_dist_centroide", "hidden_delta_camadas"]
vd = da[(da.tipo == "verdict") & (da.origem != "adenda-01")]
todos = vd[vd.estrato == "todos"]
u = todos[todos.escore.isin(UNSUP) & todos.contra.isin(["model_confidence", "aggseq"])]
put_contagem("decUnsupPooled", contar(u))
put_contagem("decUnsupAll", contar(vd[vd.escore.isin(UNSUP) & vd.contra.isin(["model_confidence", "aggseq"])]))
put_contagem("decCtwo", contar(todos[todos.comparacao == "C2"]))
put_contagem("decCthree", contar(todos[todos.comparacao == "C3"]))
ct = dc[dc.tipo.astype(str).str.contains("verdict") & dc.supervisor.isin(UNSUP) & dc.contra.isin(["model_confidence", "aggseq"])]
put_contagem("decUnsupLoad", contar(ct))
for sonda, nome in (("sonda_ocultos", "Hid"), ("sonda_atencao_cabecas", "Att")):
    for contra, cn in (("model_confidence", "Conf"), ("aggseq", "Agg")):
        put_contagem(f"decProbe{nome}{cn}", contar(todos[(todos.escore == sonda) & (todos.contra == contra)]))
ad1 = da[(da.origem == "adenda-01") & (da.tipo == "verdict")]
put_contagem("decAdOne", contar(ad1))
notas = json.loads((R / "decoder" / "NOTAS.json").read_text())
put("decHallucMin", min(n["inventadas_descartadas"] for n in notas.values()))
put("decHallucMax", max(n["inventadas_descartadas"] for n in notas.values()))

# test-set size: from the manifest of the exported test files (NOTAS counts only
# sentences with at least one anchored prediction, which is smaller)
man = json.loads((R.parents[2] / "dados_decoder" / "MANIFESTO_test.json").read_text())["arquivos"]
put("nSentGenia", f"{man['genia_test.jsonl']['n_sentencas']:,}")
put("nSentConll", f"{man['conll2003_test.jsonl']['n_sentencas']:,}")
# declared analysis parameters, read from the signed configuration
import re
Y = (R.parents[2] / "configs" / "decl-09-sinais-large.yaml").read_text(encoding="utf-8")
def yv(chave):
    return re.search(rf"^\s*{chave}:\s*([^#\n]+)", Y, re.M).group(1).strip()
put("declCalFrac", 100 * float(yv("calibration_fraction")), "{:.0f}")
put("declResamples", f"{int(yv('n_bootstrap_resamples')):,}")
put("declSeed", yv("seed"))
put("declQualityGrid", yv("task_quality_grid").strip("[]").replace(", ", ", "))
put("nDeclarations", len(list((R.parents[2] / "docs" / "tese" / "declaracoes").glob("decl-*.md"))))

# ---------------------------------------------------------------- risk reduction
# rr = 1 - AURC(signal)/AURC(random abstention): the share of the risk of random
# abstention that ranking by the signal removes. Positive is better than chance.
def _escores(d):
    return d[(d.tipo == "descriptive") & (d.comparacao == "escores") & (d.particao == "avaliação")
             & (d.estrato == "todos")]
linhas_rr = []
for (mod, corp), row in esc.iterrows():
    for sig in ("model_confidence", "span_mass", "geometric_fraction", "enrichment", "span_size", "sentence_length"):
        linhas_rr.append(dict(extrator=mod, corpus=corp, sinal=sig, rr=1 - row[f"aurc_{sig}"] / row.aurc_acaso))
g9 = pd.read_csv(R / "confirmatorio_sinais_geometria.csv")
g9["modelo"] = "gliner-large"; g9["corpus"] = np.where(g9.index < len(g9) / 2, "genia", "conll2003")
g78 = pd.read_csv(R / "confirmatorio_sinais_ajustado_geometria.csv")
g78["corpus"] = np.where(g78.modelo.str.contains("genia"), "genia", "conll2003")
g78["modelo"] = g78.modelo.str.replace("__sinais", "", regex=False)
for d in (g9, g78):
    e = _escores(d)
    for (mod, corp), g in e.groupby(["modelo", "corpus"]):
        v = dict(zip(g.metrica, g.valor)); ch = v["aurc_acaso"]
        assert abs(ch - esc.loc[(mod, corp), "aurc_acaso"]) < 1e-9, (mod, corp)   # same entities as decl-04/05/06
        for k_, val in v.items():
            if k_.startswith("aurc_") and k_ != "aurc_acaso":
                linhas_rr.append(dict(extrator=mod, corpus=corp, sinal=k_[5:], rr=1 - val / ch))
for pt, row in ed.iterrows():
    corp = "genia" if "genia" in pt else "conll2003"
    for k_ in row.index:
        if k_ != "aurc_acaso" and pd.notna(row[k_]):
            linhas_rr.append(dict(extrator=pt, corpus=corp, sinal=k_[5:], rr=1 - row[k_] / row.aurc_acaso))
RR = pd.DataFrame(linhas_rr).drop_duplicates(["extrator", "corpus", "sinal"])
RR.to_csv(AQUI / "reducao_de_risco.csv", index=False)
FAMILIA_RR = {"attention": ["span_mass", "row_entropy", "row_max", "emitted_mass", "row_entropy_causal", "row_max_causal"],
              "hidden": ["hidden_norm", "hidden_dist_centroide", "hidden_delta_camadas"],
              "output": ["logit_margin", "logit_entropy", "logit_max"]}
def pct(x): return f"{100 * x:.0f}"
enc = RR[RR.extrator.str.startswith("gliner")]; dec = RR[RR.extrator.str.startswith("qwen")]
def faixa(d, sinal, nome):
    x = d[d.sinal == sinal].rr
    put(nome + "Min", pct(x.min())); put(nome + "Max", pct(x.max()))
faixa(enc, "model_confidence", "rrEncConf"); faixa(dec, "model_confidence", "rrDecConf")
faixa(enc, "span_mass", "rrEncMass"); faixa(dec, "span_mass", "rrDecMass")
faixa(dec, "sonda_ocultos", "rrDecProbe")
for corp, nome in (("genia", "Genia"), ("conll2003", "Conll")):
    faixa(enc[enc.corpus == corp], "span_size", f"rrSize{nome}")
    faixa(enc[enc.corpus == corp], "sentence_length", f"rrSent{nome}")
    faixa(enc[enc.corpus == corp], "geometric_fraction", f"rrGeo{nome}")
faixa(dec[dec.corpus == "conll2003"], "row_entropy_causal", "rrDecEntConll")
faixa(dec[dec.corpus == "conll2003"], "model_confidence", "rrDecConfConll")
for fam, sigs in FAMILIA_RR.items():
    x = RR[RR.sinal.isin(sigs)]
    best = x.groupby(["extrator", "corpus"]).rr.max()
    put(f"rrBest{fam.capitalize()}Max", pct(best.max())); put(f"rrBest{fam.capitalize()}Min", pct(best.min()))
    xe = x[x.extrator.str.startswith("gliner")]
    if len(xe):
        be = xe.groupby(["extrator", "corpus"]).rr.max()
        put(f"rrEncBest{fam.capitalize()}Max", pct(be.max()))
hid = RR[RR.sinal.isin(FAMILIA_RR["hidden"])]
put("rrHiddenBelowChance", int((hid.groupby(["extrator", "corpus"]).rr.max() < 0).sum()))
put("rrHiddenCells", int(hid.groupby(["extrator", "corpus"]).ngroups))
put("rrGeniaMassBase", pct(RR[(RR.extrator == "gliner-base") & (RR.corpus == "genia") & (RR.sinal == "span_mass")].rr.iloc[0]))
put("rrGeniaConfBase", pct(RR[(RR.extrator == "gliner-base") & (RR.corpus == "genia") & (RR.sinal == "model_confidence")].rr.iloc[0]))
lm = RR[(RR.sinal == "logit_margin")]
for (mod, corp), g in lm.groupby(["extrator", "corpus"]):
    pass
put("rrLogitConllMin", pct(lm[lm.corpus == "conll2003"].rr.min())); put("rrLogitConllMax", pct(lm[lm.corpus == "conll2003"].rr.max()))
cf = RR[(RR.sinal == "model_confidence") & RR.extrator.isin(lm.extrator.unique()) & (RR.corpus == "conll2003")]
put("rrConfConllMin", pct(cf.rr.min())); put("rrConfConllMax", pct(cf.rr.max()))
# C3 magnitude: largest |change| in AURC when deconfounded attention is added to confidence
c3 = geo[(geo.comparacao == "C3") & (geo.tipo == "verdict") & (geo.estrato == "todos")]
put("cthreeMaxAbs", c3.delta.abs().max(), "{:.3f}")
put("cthreeZeroPairs", int(((c3.ci_low == 0) & (c3.ci_high == 0)).sum())); put("cthreePairs", len(c3))
put("cthreeMaxRel", pct((c3.delta.abs() / c3.merge(esc.reset_index()[["modelo", "corpus", "aurc_model_confidence"]],
    on=["modelo", "corpus"]).aurc_model_confidence.values).max()))
# review load at q = 0.90 for the tuned GENIA extractor
tq = tar[(tar.modelo == "gliner_base-ft-genia") & (tar.meta.astype(str) == "0.90") & (tar.comparacao == "T2")].iloc[0]
put("loadTunedGeniaConf", pct(float(tq.review_load_contra))); put("loadTunedGeniaMass", pct(float(tq.review_load)))

# ---------------------------------------------------------------- paper 1: attention only
ATT_DEC = ["span_mass", "row_entropy_causal", "row_max_causal"]
put_contagem("decAttPooled", contar(todos[todos.escore.isin(ATT_DEC) & todos.contra.isin(["model_confidence", "aggseq"])]))
put_contagem("decAttPooledConf", contar(todos[todos.escore.isin(ATT_DEC) & (todos.contra == "model_confidence")]))
put_contagem("decAttPooledAgg", contar(todos[todos.escore.isin(ATT_DEC) & (todos.contra == "aggseq")]))
put_contagem("decAttLoad", contar(dc[dc.tipo.astype(str).str.contains("verdict") & dc.supervisor.isin(ATT_DEC)
                                     & dc.contra.isin(["model_confidence", "aggseq"])]))
for sig, nome in (("row_entropy", "Ent"), ("row_max", "Max"), ("emitted_mass", "Emit")):
    faixa(enc, sig, f"rrEnc{nome}")
for sig, nome in (("row_entropy_causal", "Ent"), ("row_max_causal", "Max")):
    faixa(dec, sig, f"rrDec{nome}")
att_all = RR[RR.sinal.isin(["span_mass", "row_entropy", "row_max", "emitted_mass", "row_entropy_causal", "row_max_causal"])]
conf = RR[RR.sinal == "model_confidence"].set_index(["extrator", "corpus"]).rr
ai = att_all.set_index(["extrator", "corpus"])
put("rrAttBelowConfMin", pct((conf.reindex(ai.index).values - ai.rr.values).min()))
put("nAttCells", att_all.groupby(["extrator", "corpus"]).ngroups)

# ---------------------------------------------------------------- strata: C2 and C3 by span length
EST = [("k=1", "$k=1$"), ("k=2", "$k=2$"), ("k=3-4", "$k=3$--$4$"), ("k>=5", "$k\\ge5$"),
       ("aninhado", "nested"), ("plano", "flat"), ("todos", "all")]
PAR = [("gliner-base", "genia"), ("gliner-large", "genia"), ("gliner_base-ft-genia", "genia"),
       ("gliner-base", "conll2003"), ("gliner-large", "conll2003"), ("gliner_base-ft-conll2003", "conll2003")]
vv = geo[(geo.tipo == "verdict") & geo.comparacao.isin(["C2", "C3"])]
def cel(r):
    if r is None or pd.isna(r.delta):
        return "---"
    d_ = round(float(r.delta), 3)
    x = "0" if r.delta == 0 else ("0.000" if d_ == 0 else f"{d_:+.3f}".replace("-", "$-$").replace("+", "$+$"))
    if r.ci_high < 0: return r"\textbf{" + x + r"}$^{\downarrow}$"
    if r.ci_low > 0: return x + r"$^{\uparrow}$"
    return x
linhas = [r"\begin{tabular}{l" + "c" * 6 + "}", r"\toprule",
          r" & \multicolumn{3}{c}{GENIA} & \multicolumn{3}{c}{CoNLL-2003} \\",
          r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
          r"stratum & base & large & fine-tuned & base & large & fine-tuned \\", r"\midrule"]
for comp, titulo in (("C2", r"\multicolumn{7}{l}{\emph{(a) attention mass minus geometric fraction}} \\"),
                     ("C3", r"\multicolumn{7}{l}{\emph{(b) confidence $+$ deconfounded attention, minus confidence}} \\")):
    linhas.append(titulo)
    for e, rot in EST:
        cs = []
        for mod, corp in PAR:
            q = vv[(vv.comparacao == comp) & (vv.estrato == e) & (vv.modelo == mod) & (vv.corpus == corp)]
            cs.append(cel(q.iloc[0] if len(q) else None))
        linhas.append(rot + " & " + " & ".join(cs) + r" \\")
    if comp == "C2": linhas.append(r"\midrule")
linhas += [r"\bottomrule", r"\end{tabular}"]
(AQUI / "tab_estratos.tex").write_text("% GENERATED by numeros.py. Do not edit.\n" + "\n".join(linhas) + "\n", encoding="utf-8")
sub = vv[vv.estrato != "todos"]
for comp, nome in (("C2", "Ctwo"), ("C3", "Cthree")):
    x = sub[(sub.comparacao == comp) & sub.delta.notna()]
    put(f"strata{nome}N", len(x))
    put(f"strata{nome}Fav", int((x.ci_high < 0).sum()))
    put(f"strata{nome}Adv", int((x.ci_low > 0).sum()))
    put(f"strata{nome}MaxAbs", x.delta.abs().max(), "{:.3f}")
    # chance rate over NON-degenerate intervals only: a degenerate interval (both bounds zero)
    # cannot land favourable, so counting it inflates the expected number (Appendix convention).
    nd = x[~((x.ci_low == 0) & (x.ci_high == 0))]
    put(f"strata{nome}NonDeg", len(nd))
    put(f"strata{nome}Chance", 0.025 * len(nd), "{:.2f}")
    xa = vv[(vv.comparacao == comp) & vv.delta.notna()]
    put(f"strata{nome}NAll", len(xa))
c3f = sub[(sub.comparacao == "C3") & (sub.ci_high < 0)]
assert set(c3f.estrato) == {"k=1"} and set(c3f.corpus) == {"conll2003"}, c3f[["modelo", "corpus", "estrato"]]
put("cthreeKoneConllBaseD", float(c3f[c3f.modelo == "gliner-base"].delta.iloc[0]), "{:.4f}")
put("cthreeKoneConllLargeD", float(c3f[c3f.modelo == "gliner-large"].delta.iloc[0]), "{:.4f}")
put("cthreeKoneConllTunedD", float(sub[(sub.comparacao == "C3") & (sub.estrato == "k=1") & (sub.modelo == "gliner_base-ft-conll2003")].delta.iloc[0]), "{:.4f}")
# readings scan: medians
P1 = pd.read_csv(R / "perimetro1_leituras.csv")
for corp, nome in (("genia", "Genia"), ("conll2003", "Conll")):
    d = P1[P1.corpus == corp]
    put(f"peri{nome}MedDelta", d.delta_vs_confianca.median(), "{:.4f}")
    put(f"peri{nome}QuartDelta", d.delta_vs_confianca.quantile(0.25), "{:.4f}")
    put(f"peri{nome}QuartDeltaAbs", abs(d.delta_vs_confianca.quantile(0.25)), "{:.4f}")
    put(f"peri{nome}MedWeight", d.peso_na_leitura.median(), "{:.2f}")

# ---------------------------------------------------------------- decoder strata
PTD = [("qwen05b-ft-genia", "0.5B"), ("qwen15b-ft-genia", "1.5B"),
       ("qwen05b-ft-conll2003", "0.5B"), ("qwen15b-ft-conll2003", "1.5B")]
vd2 = da[(da.tipo == "verdict") & (da.origem != "adenda-01") & da.comparacao.isin(["C2", "C3"])]
lin = [r"\begin{tabular}{l" + "c" * 4 + "}", r"\toprule",
       r" & \multicolumn{2}{c}{GENIA} & \multicolumn{2}{c}{CoNLL-2003} \\",
       r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
       r"stratum & 0.5B & 1.5B & 0.5B & 1.5B \\", r"\midrule"]
for comp, titulo in (("C2", r"\multicolumn{5}{l}{\emph{(a) attention mass minus causal geometric fraction}} \\"),
                     ("C3", r"\multicolumn{5}{l}{\emph{(b) confidence $+$ deconfounded attention, minus confidence}} \\")):
    lin.append(titulo)
    for e, rot in EST:
        cs = []
        for pt, _ in PTD:
            q = vd2[(vd2.comparacao == comp) & (vd2.estrato == e) & (vd2.ponto == pt)]
            cs.append(cel(q.iloc[0] if len(q) else None))
        lin.append(rot + " & " + " & ".join(cs) + r" \\")
    if comp == "C2": lin.append(r"\midrule")
lin += [r"\bottomrule", r"\end{tabular}"]
(AQUI / "tab_estratos_decoder.tex").write_text("% GENERATED by numeros.py. Do not edit.\n" + "\n".join(lin) + "\n", encoding="utf-8")
subd = vd2[(vd2.estrato != "todos") & vd2.delta.notna()]
for comp, nome in (("C2", "Ctwo"), ("C3", "Cthree")):
    x = subd[subd.comparacao == comp]
    put(f"decStrata{nome}N", len(x)); put(f"decStrata{nome}Fav", int((x.ci_high < 0).sum()))
    nd = x[~((x.ci_low == 0) & (x.ci_high == 0))]
    put(f"decStrata{nome}Adv", int((x.ci_low > 0).sum())); put(f"decStrata{nome}Chance", 0.025 * len(nd), "{:.2f}")
    put(f"decStrata{nome}NonDeg", len(nd))
    put(f"decStrata{nome}MaxAbs", x.delta.abs().max(), "{:.3f}")
    put(f"decStrata{nome}ZeroW", int(((x.ci_low == 0) & (x.ci_high == 0)).sum()))

# ---------------------------------------------------------------- revision tests (decl-14 to decl-26)
exec((AQUI / "numeros_revisao.py").read_text(encoding="utf-8"))

(AQUI / "numeros.json").write_text(json.dumps(M, indent=1, ensure_ascii=False), encoding="utf-8")
with (AQUI / "numeros.tex").open("w", encoding="utf-8") as fh:
    fh.write("% GENERATED by numeros.py from the published tables. Do not edit.\n")
    for k, val in M.items():
        fh.write(f"\\newcommand{{\\{k}}}{{{val}}}\n")
print(f"{len(M)} values -> numeros.tex | source {R}")
