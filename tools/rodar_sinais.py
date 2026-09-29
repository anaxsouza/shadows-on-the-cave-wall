"""As dezesseis comparações da decl-09, nos dois corpora.

Roteamento por tipo: as de AURC vão ao runner de geometria, as de tarefa ao de
impacto. O estatuto do nulo de cada sinal vem do registro de signals.py e entra
como coluna, porque um nulo exato refutado e um empírico refutado não se leem
igual.
"""
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, ".")
from src.selective.comparisons import run_declared_comparisons  # noqa: E402
from src.selective.preregistration import load_preregistration  # noqa: E402
from src.selective.signals import SignalError, por_nome  # noqa: E402
from src.selective.task_impact import run_task_impact  # noqa: E402

OURO = {"genia": 5506, "conll2003": 5648}
p = load_preregistration("configs/decl-09-sinais-large.yaml")
print(f"declaração {p.declaration_id} | {p.declaration_hash} | medição {p.measurement_hash}",
      flush=True)


def estatuto(nome: str) -> tuple[str, str]:
    try:
        s = por_nome(nome)
        return s.familia, s.estatuto
    except SignalError:
        return "geometrico", "exato" if nome in ("span_size", "geometric_fraction",
                                                 "sentence_length") else "—"


geo, tar, notas = [], [], []
for corpus in ("genia", "conll2003"):
    d = Path(f"results/gliner-large/{corpus}/test")
    rg = run_declared_comparisons(d, p, corpus)
    rt = run_task_impact(d, p, corpus, n_gold=OURO[corpus])
    for l in rg.linhas:
        fam, est = estatuto(str(l.get("escore") or l.get("supervisor") or ""))
        geo.append({**l, "familia": fam, "estatuto_nulo": est})
    for l in rt.linhas:
        fam, est = estatuto(str(l.get("supervisor") or ""))
        tar.append({**l, "familia": fam, "estatuto_nulo": est})
    notas += [f"[{corpus}] {n}" for n in rg.notas] + [f"[{corpus}] {n}" for n in rt.notas]
    print(f"{corpus}: {len(rg.linhas)} linhas de geometria, {len(rt.linhas)} de tarefa",
          flush=True)

RES = Path("docs/tese/resultados")
for nome, linhas in (("confirmatorio_sinais_geometria.csv", geo),
                     ("confirmatorio_sinais_tarefa.csv", tar)):
    if not linhas:
        continue
    campos = list({k: None for l in linhas for k in l})
    with (RES / nome).open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=campos, extrasaction="ignore")
        w.writeheader()
        w.writerows(linhas)
    print(f"escrito {RES / nome}: {len(linhas)} linhas", flush=True)
(RES / "confirmatorio_sinais_notas.txt").write_text("\n".join(notas) + "\n", encoding="utf-8")
print(f"ressalvas: {len(notas)}", flush=True)
