
import sys, csv, json; sys.path.insert(0, ".")
from pathlib import Path
from src.selective.preregistration import load_preregistration
from src.selective.task_impact import run_task_impact

ALVO = {
 "gliner-large/genia":     ("relato-vao-gliner-large", "results/gliner-large/genia/test", "genia", 5506),
 "gliner-large/conll2003": ("relato-vao-gliner-large", "results/gliner-large/conll2003/test", "conll2003", 5648),
 "ajustado/genia":         ("relato-vao-ft-genia", "results/gliner_base-ft-genia/genia/test", "genia", 5506),
 "ajustado/conll2003":     ("relato-vao-ft-conll", "results/gliner_base-ft-conll2003/conll2003/test", "conll2003", 5648),
}
linhas, notas = [], []
for rot, (cfg, d, corpus, ouro) in ALVO.items():
    p = load_preregistration(f"configs/{cfg}.yaml")
    r = run_task_impact(Path(d), p, corpus, n_gold=ouro)
    for l in r.linhas:
        linhas.append({"arranjo": rot, "estatuto": "exploratorio", **l})
    notas += [f"[{rot}] {n}" for n in r.notas]
    print(f"{rot}: {len(r.linhas)} linhas", flush=True)

campos = list({k: None for l in linhas for k in l})
alvo = Path("docs/tese/resultados/relato_vao_exploratorio.csv")
with alvo.open("w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=campos, extrasaction="ignore")
    w.writeheader(); w.writerows(linhas)
Path("docs/tese/resultados/relato_vao_notas.txt").write_text("\n".join(notas) + "\n", encoding="utf-8")
print(f"escrito {alvo}: {len(linhas)} linhas", flush=True)
for n in notas:
    if "grade do VAO" in n or "grade do VÃO" in n:
        print("  ", n[:190], flush=True)
