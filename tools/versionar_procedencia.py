"""Copia os MEDIDA.json de results/ para uma pasta VERSIONADA.

POR QUE EXISTE

`results/` esta inteiramente no .gitignore (linha 78) — decisao correta, porque
la moram gigabytes de tensores. Mas o `MEDIDA.json` mora la tambem, e ele e o
que PROVA sob qual declaracao, com quais camadas e com quais pesos cada tabela
foi medida. As tabelas confirmatorias sao copiadas para docs/tese/resultados/;
a procedencia delas nao era.

Consequencia da lacuna, antes deste script: perder results/ perdia o registro de
COMO cada numero do paper foi obtido, e nenhum commit continha essa informacao.
Depois: perder results/ custa remedir, que e caro mas recuperavel.

Rode depois de qualquer medicao. Sao ~2 KB por medicao.
"""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEST = REPO / "docs/tese/resultados/procedencia"

NOTA = (
    "results/ esta inteiramente no .gitignore (linha 78), entao o MEDIDA.json — que e o "
    "que PROVA sob qual declaracao e com quais pesos cada tabela foi medida — vivia so no "
    "disco local. As tabelas confirmatorias sao copiadas para docs/tese/resultados/, mas a "
    "procedencia delas nao era. Esta copia versionada fecha a lacuna: perder results/ passa "
    "a custar remedir, e nao perder o registro de como foi medido."
)


def main() -> int:
    DEST.mkdir(parents=True, exist_ok=True)
    n = 0
    for p in sorted(REPO.glob("results/*/*/*/MEDIDA.json")):
        modelo, corpus, split = p.parts[-4], p.parts[-3], p.parts[-2]
        d = json.loads(p.read_text(encoding="utf-8"))
        d["_origem"] = str(p.relative_to(REPO))
        d["_por_que_esta_copia_existe"] = NOTA
        (DEST / f"{modelo}__{corpus}__{split}.json").write_text(
            json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
        n += 1
        print(f"  {modelo}/{corpus}/{split}")
    print(f"{n} arquivos de procedencia em {DEST.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
