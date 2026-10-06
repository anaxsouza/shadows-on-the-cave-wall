"""Numbers of the revision tests (decl-14 to decl-26), executed by numeros.py.

Runs inside numeros.py (exec), so `R`, `put`, `pd`, `np`, `json` and `AQUI` are the ones defined there.
Every macro comes from a published table; nothing is typed.

    C3  confirmatorio_bert_geometria.csv + saida/bert/*/MEDIDA.json (F1 on test)
    C2  confirmatorio_bc5cdr_*_geometria.csv, decoder/decoder_aurc_bc5cdr.csv, confirmatorio_bc5cdr_previsao_insumos.csv
    C4  revisao/c4_tipos_de_erro.csv
    C1  revisao/c1_janelas.csv
"""
REP = R.parents[2]                     # .../attention-selective-ner
TOL_P2 = float(M["encMassGeoMaxGap"])  # the tolerance P2 inherits (decl-16)
assert abs(TOL_P2 - 0.004) < 1e-12, TOL_P2

# ------------------------------------------------------------------ C3: BERT
bg = pd.read_csv(R / "confirmatorio_bert_geometria.csv")
BCORP = {"bert-base-cased-ft-genia": ("genia", "Genia"), "bert-base-cased-ft-conll2003": ("conll2003", "Conll"),
         "bert-base-cased-ft-bc5cdr": ("bc5cdr", "Bc")}
_bert_r2, _bert_c2, _bert_c3z = [], [], 0
for mod, (corp, nome) in BCORP.items():
    x = bg[bg.modelo == mod]
    esc = x[(x.comparacao == "escores") & (x.particao == "avaliação") & (x.estrato == "todos")] \
        .drop_duplicates("metrica").set_index("metrica").valor
    for k, col in (("Chance", "aurc_acaso"), ("Mass", "aurc_span_mass"), ("Geo", "aurc_geometric_fraction"),
                   ("Conf", "aurc_model_confidence"), ("Size", "aurc_span_size")):
        put(f"bert{nome}{k}", float(esc[col]), "{:.3f}")
    put(f"rrBert{nome}Mass", 100 * (1 - esc["aurc_span_mass"] / esc["aurc_acaso"]), "{:.0f}")
    put(f"rrBert{nome}Geo", 100 * (1 - esc["aurc_geometric_fraction"] / esc["aurc_acaso"]), "{:.0f}")
    put(f"rrBert{nome}Conf", 100 * (1 - esc["aurc_model_confidence"] / esc["aurc_acaso"]), "{:.0f}")
    r2 = float(x[(x.comparacao == "C1") & (x.metrica == "R2_massa_vs_fracao_geometrica") & (x.estrato == "todos")].valor.iloc[0])
    put(f"bert{nome}Rsq", r2, "{:.2f}"); _bert_r2.append(r2)
    c2 = x[(x.comparacao == "C2") & (x.tipo == "verdict") & (x.estrato == "todos")].iloc[0]
    put(f"bert{nome}CtwoD", c2.delta, "{:+.3f}"); put(f"bert{nome}CtwoLo", c2.ci_low, "{:.3f}")
    put(f"bert{nome}CtwoHi", c2.ci_high, "{:.3f}"); _bert_c2.append(c2)
    c3 = x[(x.comparacao == "C3") & (x.tipo == "verdict") & (x.estrato == "todos")].iloc[0]
    _bert_c3z += int(c3.ci_low == 0 and c3.ci_high == 0)
    med = json.loads((REP / "saida" / "bert" / mod / corp / "test" / "MEDIDA.json").read_text())
    f1 = med["f1_estrito_teste"]; f1 = f1["f1"] if isinstance(f1, dict) else f1
    put(f"bert{nome}Fone", float(f1), "{:.2f}")
    put(f"bert{nome}Err", float(med["taxa_de_erro_base"]), "{:.2f}")
put("bertRsqMin", min(_bert_r2), "{:.2f}"); put("bertRsqMax", max(_bert_r2), "{:.2f}")
put("bertCtwoAdv", sum(int(c.ci_low > 0) for c in _bert_c2)); put("bertCthreeZeroW", _bert_c3z)
put("bertCtwoMin", min(c.delta for c in _bert_c2), "{:.3f}"); put("bertCtwoMax", max(c.delta for c in _bert_c2), "{:.3f}")

# ------------------------------------------------------------------ C2: BC5CDR and the prediction
ins = pd.read_csv(R / "confirmatorio_bc5cdr_previsao_insumos.csv").set_index("modelo")
bc = pd.read_csv(R / "decoder" / "decoder_aurc_bc5cdr.csv")
ENC_BC = {"gliner-base": ("decl-26-bc5cdr-gliner-base", "Base"), "gliner-large": ("decl-18-bc5cdr-gliner-large", "Large"),
          "gliner_base-ft-bc5cdr": ("decl-19-bc5cdr-ajustado", "Tuned")}
DEC_BC = {"qwen05b-ft-bc5cdr": "Small", "qwen15b-ft-bc5cdr": "Big"}
prev = []   # (name, P1, P2 or None)
for mod, (did, nome) in ENC_BC.items():
    g = pd.read_csv(R / f"confirmatorio_bc5cdr_{did}_geometria.csv")
    esc = g[(g.comparacao == "escores") & (g.particao == "avaliação") & (g.estrato == "todos")] \
        .drop_duplicates("metrica").set_index("metrica").valor
    ch, ma, ge, co = (float(esc[c]) for c in ("aurc_acaso", "aurc_span_mass", "aurc_geometric_fraction", "aurc_model_confidence"))
    assert abs(ch - ins.loc[mod, "aurc_acaso"]) < 6e-5 and abs(ma - ins.loc[mod, "aurc_massa"]) < 6e-5, mod
    for k, v in (("Chance", ch), ("Mass", ma), ("Geo", ge), ("Conf", co)):
        put(f"bc{nome}{k}", v, "{:.3f}")
    put(f"rrBc{nome}Mass", 100 * (1 - ma / ch), "{:.1f}"); put(f"rrBc{nome}Geo", 100 * (1 - ge / ch), "{:.1f}")
    put(f"rrBc{nome}Conf", 100 * (1 - co / ch), "{:.0f}")
    c2 = g[(g.comparacao == "C2") & (g.tipo == "verdict") & (g.estrato == "todos")].iloc[0]
    c3 = g[(g.comparacao == "C3") & (g.tipo == "verdict") & (g.estrato == "todos")].iloc[0]
    put(f"bc{nome}Gap", abs(ma - ge), "{:.4f}")
    put(f"bc{nome}CtwoD", c2.delta, "{:+.4f}"); put(f"bc{nome}CtwoLo", c2.ci_low, "{:.4f}"); put(f"bc{nome}CtwoHi", c2.ci_high, "{:.4f}")
    put(f"bc{nome}CthreeD", c3.delta, "{:+.4f}"); put(f"bc{nome}CthreeLo", c3.ci_low, "{:.4f}"); put(f"bc{nome}CthreeHi", c3.ci_high, "{:.4f}")
    p1 = np.sign(1 - ma / ch) == np.sign(1 - ge / ch); p2 = abs(ma - ge) <= TOL_P2
    prev.append((mod, bool(p1), bool(p2), c2, c3))
for pt, nome in DEC_BC.items():
    e = bc[(bc.ponto == pt) & (bc.origem != "adenda-01") & (bc.comparacao == "escores") & (bc.particao == "avaliação")
           & (bc.estrato == "todos")].drop_duplicates("metrica").set_index("metrica").valor
    ch, ma, ge, co = (float(e[c]) for c in ("aurc_acaso", "aurc_span_mass", "aurc_geometric_fraction", "aurc_model_confidence"))
    for k, v in (("Chance", ch), ("Mass", ma), ("Geo", ge), ("Conf", co)):
        put(f"bc{nome}{k}", v, "{:.3f}")
    put(f"rrBc{nome}Mass", 100 * (1 - ma / ch), "{:.1f}"); put(f"rrBc{nome}Geo", 100 * (1 - ge / ch), "{:.1f}")
    put(f"rrBc{nome}Conf", 100 * (1 - co / ch), "{:.0f}")
    v = bc[(bc.ponto == pt) & (bc.origem != "adenda-01") & (bc.tipo == "verdict") & (bc.estrato == "todos")]
    c2 = v[v.comparacao == "C2"].iloc[0]; c3 = v[v.comparacao == "C3"].iloc[0]
    put(f"bc{nome}CtwoD", c2.delta, "{:+.4f}"); put(f"bc{nome}CtwoLo", c2.ci_low, "{:.4f}"); put(f"bc{nome}CtwoHi", c2.ci_high, "{:.4f}")
    p1 = np.sign(1 - ma / ch) == np.sign(1 - ge / ch)
    prev.append((pt, bool(p1), None, c2, c3))
# BERT on BC5CDR
x = bg[bg.modelo == "bert-base-cased-ft-bc5cdr"]
ma, ge, ch = float(M["bertBcMass"]), float(M["bertBcGeo"]), float(M["bertBcChance"])
esc = x[(x.comparacao == "escores") & (x.particao == "avaliação") & (x.estrato == "todos")].drop_duplicates("metrica").set_index("metrica").valor
ma, ge, ch = float(esc["aurc_span_mass"]), float(esc["aurc_geometric_fraction"]), float(esc["aurc_acaso"])
put("bcBertGap", abs(ma - ge), "{:.4f}")
prev.append(("bert-base-cased-ft-bc5cdr", bool(np.sign(1 - ma / ch) == np.sign(1 - ge / ch)), bool(abs(ma - ge) <= TOL_P2),
             x[(x.comparacao == "C2") & (x.tipo == "verdict") & (x.estrato == "todos")].iloc[0],
             x[(x.comparacao == "C3") & (x.tipo == "verdict") & (x.estrato == "todos")].iloc[0]))
assert len(prev) == 6, prev
conf = [m for m, p1, p2, _, _ in prev if p1 and (p2 is None or p2)]
put("predN", len(prev)); put("predPoneOk", sum(p1 for _, p1, _, _, _ in prev)); put("predConfirmed", len(conf))
put("predPtwoFail", sum(1 for _, _, p2, _, _ in prev if p2 is False))
put("bcCtwoAdv", sum(int(c2.ci_low > 0) for *_, c2, _ in prev)); put("bcCtwoFav", sum(int(c2.ci_high < 0) for *_, c2, _ in prev))
put("bcCthreeFav", sum(int(c3.ci_high < 0) for *_, c3 in prev))
assert M["predConfirmed"] == "3" and M["predPoneOk"] == "5", (M["predConfirmed"], M["predPoneOk"])
for f, nome in (("ajuste_bc5cdr.json", "Tuned"), ("ajuste_decoder_qwen05b_bc5cdr.json", "Small"),
                ("ajuste_decoder_qwen15b_bc5cdr.json", "Big")):
    put(f"bc{nome}Fone", float(json.loads((R / f).read_text())["f1_val"]), "{:.2f}")
_cm = pd.read_csv(R / "revisao" / "corpus_mencoes.csv").set_index("corpus")
put("bcNtest", f"{int(_cm.loc['bc5cdr', 'sentences']):,}")
for _c, _n in (("genia", "Genia"), ("conll2003", "Conll"), ("bc5cdr", "Bc")):
    put(f"corp{_n}Ments", f"{int(_cm.loc[_c, 'mentions']):,}")
    put(f"corp{_n}MeanW", float(_cm.loc[_c, "mean_words"]), "{:.1f}")
    put(f"corp{_n}Multi", float(_cm.loc[_c, "pct_multiword"]), "{:.0f}")
    put(f"corp{_n}Nested", float(_cm.loc[_c, "pct_nested"]), "{:.0f}")

# ------------------------------------------------------------------ C4: error types
c4 = pd.read_csv(R / "revisao" / "c4_tipos_de_erro.csv")
TIP = {"rotulo": "Label", "fronteira": "Bound", "sem_par": "Spur"}
cnt = c4.drop_duplicates(["ponto", "corpus", "tipo"]).groupby("tipo").n_erros.sum()
for t, nome in TIP.items():
    put(f"errN{nome}", f"{int(cnt[t]):,}")
    for teste, tn in (("E1", "Eone"), ("E2", "Etwo")):
        v = c4[(c4.teste == teste) & (c4.tipo == t)]
        assert len(v) == 10, (teste, t, len(v))
        put(f"err{tn}{nome}Fav", int((v.ci_high < 0).sum())); put(f"err{tn}{nome}Adv", int((v.ci_low > 0).sum()))
        put(f"err{tn}{nome}Deg", int(((v.ci_low == 0) & (v.ci_high == 0)).sum()))
v1 = c4[c4.teste == "E1"]; v2 = c4[c4.teste == "E2"]
put("errEoneFav", int((v1.ci_high < 0).sum())); put("errEoneAdv", int((v1.ci_low > 0).sum()))
put("errEtwoFav", int((v2.ci_high < 0).sum())); put("errEtwoAdv", int((v2.ci_low > 0).sum()))
put("errEoneChance", 0.025 * int((~((v1.ci_low == 0) & (v1.ci_high == 0))).sum()), "{:.2f}")
lab = v1[(v1.tipo == "rotulo") & (v1.ci_high < 0)]
assert set(lab.ponto.str.startswith("gliner")) == {True}
put("errLabelMin", lab.delta.max(), "{:.3f}"); put("errLabelMax", lab.delta.min(), "{:.3f}")

# ------------------------------------------------------------------ C1: control windows
c1 = pd.read_csv(R / "revisao" / "c1_janelas.csv")
enc, dec = c1[c1.ponto.str.startswith("gliner")], c1[c1.ponto.str.startswith("qwen")]
assert (enc.J1_veredito == "atenua").all() and (dec.J1_veredito == "destaca").all()
put("winEncN", len(enc)); put("winDecN", len(dec))
put("winEncPctMin", 100 * (1 - np.exp(enc.J1_mediana_log_r.max())), "{:.0f}")
put("winEncPctMax", 100 * (1 - np.exp(enc.J1_mediana_log_r.min())), "{:.0f}")
put("winDecPctMin", 100 * (np.exp(dec.J1_mediana_log_r.min()) - 1), "{:.0f}")
put("winDecPctMax", 100 * (np.exp(dec.J1_mediana_log_r.max()) - 1), "{:.0f}")
dc = dec[dec.corpus == "conll2003"]
assert (dc.J3_ci_high < 0).all()
put("winDecConllJthreeMin", dc.J3_dif_erros_menos_acertos.max(), "{:.2f}")
put("winDecConllJthreeMax", dc.J3_dif_erros_menos_acertos.min(), "{:.2f}")
ec = enc[(enc.corpus == "conll2003") & (enc.J3_ci_low > 0)]
put("winEncConllJthreeN", len(ec))
put("winNoWindowMax", int(c1.n_sem_janela.max()))

# ------------------------------------------------------------------ generated tables
def _f(x, nd=3):
    return f"{x:.{nd}f}".replace("-", "$-$")

def _ci(lo, hi, nd=3):
    return f"[{_f(lo, nd)}, {_f(hi, nd)}]"

NOMES = {"gliner-base": r"\gliner{} base", "gliner-large": r"\gliner{} large", "gliner_base-ft-bc5cdr": r"\gliner{} fine-tuned",
         "qwen05b-ft-bc5cdr": "Qwen2.5-0.5B", "qwen15b-ft-bc5cdr": "Qwen2.5-1.5B", "bert-base-cased-ft-bc5cdr": "BERT base"}
CH = {"gliner-base": "Base", "gliner-large": "Large", "gliner_base-ft-bc5cdr": "Tuned", "qwen05b-ft-bc5cdr": "Small",
      "qwen15b-ft-bc5cdr": "Big", "bert-base-cased-ft-bc5cdr": None}
lin = [r"\begin{tabular}{lrrcrcc}", r"\toprule",
       r"extractor & \multicolumn{2}{c}{risk removed (\%)} & P1 & $|\Delta\aurc|$ & P2 & prediction \\",
       r" & fraction & attention & sign & & $\le$\,\encMassGeoMaxGap & \\", r"\midrule"]
for mod, p1, p2, c2, c3 in prev:
    if CH[mod]:
        rg, rm = float(M[f"rrBc{CH[mod]}Geo"]), float(M[f"rrBc{CH[mod]}Mass"])
        gap = M.get(f"bc{CH[mod]}Gap", "---") if p2 is not None else "---"
    else:
        rg = 100 * (1 - float(esc["aurc_geometric_fraction"]) / float(esc["aurc_acaso"]))
        rm = 100 * (1 - float(esc["aurc_span_mass"]) / float(esc["aurc_acaso"])); gap = M["bcBertGap"]
    ok = p1 and (p2 is None or p2)
    lin.append(f"{NOMES[mod]} & {_f(rg,1)} & {_f(rm,1)} & {'yes' if p1 else 'no'} & {gap} & "
               f"{'---' if p2 is None else ('yes' if p2 else 'no')} & {'confirmed' if ok else 'not confirmed'} \\\\")
lin += [r"\bottomrule", r"\end{tabular}"]
(AQUI / "tab_previsao.tex").write_text("% GENERATED by numeros_revisao.py. Do not edit.\n" + "\n".join(lin) + "\n", encoding="utf-8")

PT_ORD = [("gliner-base", "genia", r"\gliner{} base", "GENIA"), ("gliner-base", "conll2003", r"\gliner{} base", "CoNLL"),
          ("gliner-large", "genia", r"\gliner{} large", "GENIA"), ("gliner-large", "conll2003", r"\gliner{} large", "CoNLL"),
          ("gliner_base-ft-genia", "genia", r"\gliner{} fine-tuned", "GENIA"), ("gliner_base-ft-conll2003", "conll2003", r"\gliner{} fine-tuned", "CoNLL"),
          ("qwen05b-ft-genia", "genia", "Qwen2.5-0.5B", "GENIA"), ("qwen05b-ft-conll2003", "conll2003", "Qwen2.5-0.5B", "CoNLL"),
          ("qwen15b-ft-genia", "genia", "Qwen2.5-1.5B", "GENIA"), ("qwen15b-ft-conll2003", "conll2003", "Qwen2.5-1.5B", "CoNLL")]

def _cel(r):
    if r is None: return "---"
    d = round(float(r.delta), 3)
    x = "0" if r.delta == 0 else ("0.000" if d == 0 else f"{d:+.3f}".replace("-", "$-$").replace("+", "$+$"))
    if r.ci_high < 0: return r"\textbf{" + x + r"}$^{\downarrow}$"
    if r.ci_low > 0: return x + r"$^{\uparrow}$"
    return x

lin = [r"\begin{tabular}{llrrrccc}", r"\toprule",
       r" & & \multicolumn{3}{c}{errors of each type} & \multicolumn{3}{c}{attention mass minus geometric fraction} \\",
       r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}",
       r"extractor & corpus & label & boundary & spurious & label & boundary & spurious \\", r"\midrule"]
for pt, corp, nm, cn in PT_ORD:
    q = c4[(c4.ponto == pt) & (c4.corpus == corp)]
    ns = [int(q[q.tipo == t].n_erros.iloc[0]) for t in TIP]
    cs = [_cel(q[(q.tipo == t) & (q.teste == "E1")].iloc[0]) for t in TIP]
    lin.append(f"{nm} & {cn} & {ns[0]:,} & {ns[1]:,} & {ns[2]:,} & " + " & ".join(cs) + r" \\")
lin += [r"\bottomrule", r"\end{tabular}"]
(AQUI / "tab_tipos_erro.tex").write_text("% GENERATED by numeros_revisao.py. Do not edit.\n" + "\n".join(lin) + "\n", encoding="utf-8")

lin = [r"\begin{tabular}{llrcc}", r"\toprule",
       r"extractor & corpus & median $\log r$ & 95\% interval & errors $-$ correct \\", r"\midrule"]
for pt, corp, nm, cn in PT_ORD:
    q = c1[(c1.ponto == pt) & (c1.corpus == corp)].iloc[0]
    j3 = _f(q.J3_dif_erros_menos_acertos, 3)
    if q.J3_ci_low > 0 or q.J3_ci_high < 0:
        j3 = r"\textbf{" + j3 + "}"
    lin.append(f"{nm} & {cn} & {_f(q.J1_mediana_log_r)} & {_ci(q.J1_ci_low, q.J1_ci_high)} & {j3} {_ci(q.J3_ci_low, q.J3_ci_high)} \\\\")
lin += [r"\bottomrule", r"\end{tabular}"]
(AQUI / "tab_janelas.tex").write_text("% GENERATED by numeros_revisao.py. Do not edit.\n" + "\n".join(lin) + "\n", encoding="utf-8")

# magnitudes for prose that states the direction in words (a hyphen is not a minus sign in text)
put("decRawGapAbsMin", abs(float(M["decRawGapMax"])), "{:.3f}")
put("decRawGapAbsMax", abs(float(M["decRawGapMin"])), "{:.3f}")
put("errLabelAbsMin", abs(float(M["errLabelMin"])), "{:.3f}")
put("errLabelAbsMax", abs(float(M["errLabelMax"])), "{:.3f}")
