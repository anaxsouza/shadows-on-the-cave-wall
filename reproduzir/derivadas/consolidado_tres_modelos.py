# Recuperado da linhagem do artefato que produziu a peça publicada; só os CAMINHOS
# foram trocados para o repositório. Rode por reproduzir/derivadas.py, que o executa
# em saida/derivadas/ e compara o que ele escreve com o publicado.
from pathlib import Path as _P
RAIZ = _P(__file__).resolve().parents[2]
import csv

FONTES = {
  "gliner-base":   (str(RAIZ / "docs/tese/resultados" / "confirmatorio_decl02.csv"),             str(RAIZ / "docs/tese/resultados" / "confirmatorio_decl03_tarefa.csv")),
  "gliner-large":  (str(RAIZ / "docs/tese/resultados" / "confirmatorio_decl04_geometria.csv"),   str(RAIZ / "docs/tese/resultados" / "confirmatorio_decl04_tarefa.csv")),
  "ajustado":      (str(RAIZ / "docs/tese/resultados" / "confirmatorio_ajustado_geometria.csv"), str(RAIZ / "docs/tese/resultados" / "confirmatorio_ajustado_tarefa.csv")),
}

geo, tar = {}, {}
for m, (fg, ft) in FONTES.items():
    geo[m] = list(csv.DictReader(open(fg, encoding="utf-8-sig")))
    tar[m] = list(csv.DictReader(open(ft, encoding="utf-8-sig")))

erro = {
    ("gliner-base", "genia"): 0.4818,
    ("gliner-base", "conll2003"): 0.5310,
    ("gliner-large", "genia"): 0.4224,
    ("gliner-large", "conll2003"): 0.4667,
    ("ajustado", "genia"): 0.2458,
    ("ajustado", "conll2003"): 0.1028,
}

def g1(m, corpus, comp, metrica=None, estrato="todos"):
    for r in geo[m]:
        if r["corpus"] == corpus and r["comparacao"] == comp and r.get("estrato", "todos") == estrato:
            if metrica is None or r.get("metrica") == metrica:
                return r

def lt(m, corpus, comp, meta):
    for r in tar[m]:
        if r["corpus"] == corpus and r["comparacao"] == comp and r["meta"] == meta:
            return r

def num(v):
    try: return f"{float(v):.4f}"
    except (TypeError, ValueError): return "INALC."

linhas = []
for m in FONTES:
    for corpus in ("genia", "conll2003"):
        r2 = g1(m, corpus, "C1", "R2_massa_vs_fracao_geometrica")
        md = g1(m, corpus, "C1", "mediana_observado_sobre_esperado")
        for comp in ("C2", "C3"):
            r = g1(m, corpus, comp)
            linhas.append(dict(modelo=m, corpus=corpus, comparacao=comp,
                               R2_geometria=r2["valor"] if r2 else "",
                               mediana_obs_esp=md["valor"] if md else "",
                               delta=r["delta"], ci_low=r["ci_low"], ci_high=r["ci_high"],
                               veredito=r["veredito"]))

tarefa_linhas = []
for m in FONTES:
    for corpus in ("genia", "conll2003"):
        for meta in ("0.70", "0.80", "0.90", "0.95"):
            r = lt(m, corpus, "T2", meta)
            if not r: continue
            sup = num(r.get("review_load"))
            base = num(r.get("review_load_contra"))
            dl = num(r.get("delta")) if sup != "INALC." and base != "INALC." else "—"
            tarefa_linhas.append(dict(modelo=m, corpus=corpus, comparacao="T2", meta=meta,
                                      carga_massa=sup, carga_confianca=base, delta=dl,
                                      veredito=r["veredito"]))

campos = ["modelo","corpus","erro_base","R2_geometria","mediana_obs_esp","comparacao",
          "meta","valor_supervisor","valor_confianca","delta","ci_low","ci_high","veredito"]
rows = []
for l in linhas:
    rows.append({"modelo": l["modelo"], "corpus": l["corpus"],
                 "erro_base": f"{erro.get((l['modelo'],l['corpus']), float('nan')):.4f}",
                 "R2_geometria": l["R2_geometria"], "mediana_obs_esp": l["mediana_obs_esp"],
                 "comparacao": l["comparacao"], "meta": "—",
                 "valor_supervisor": "—", "valor_confianca": "—",
                 "delta": l["delta"], "ci_low": l["ci_low"], "ci_high": l["ci_high"],
                 "veredito": l["veredito"]})
for t in tarefa_linhas:
    r = lt(t["modelo"], t["corpus"], "T2", t["meta"])
    rows.append({"modelo": t["modelo"], "corpus": t["corpus"],
                 "erro_base": f"{erro.get((t['modelo'],t['corpus']), float('nan')):.4f}",
                 "R2_geometria": "—", "mediana_obs_esp": "—", "comparacao": "T2",
                 "meta": t["meta"], "valor_supervisor": t["carga_massa"],
                 "valor_confianca": t["carga_confianca"], "delta": t["delta"],
                 "ci_low": r.get("ci_low","—"), "ci_high": r.get("ci_high","—"),
                 "veredito": t["veredito"]})

with open("consolidado_tres_modelos.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=campos, extrasaction="ignore")
    w.writeheader()
    w.writerows(rows)