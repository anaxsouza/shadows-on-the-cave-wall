"""Refaz as tabelas derivadas e as figuras, e compara as tabelas com as publicadas.

    python reproduzir/derivadas.py

Cada script de `reproduzir/derivadas/` foi recuperado do registro que produziu a peça
publicada, com só os caminhos trocados. Aqui cada um roda em `saida/derivadas/`, e cada
CSV que ele escreve é comparado com o de mesmo nome em `docs/tese/resultados/`. As
figuras são regeradas e não comparadas byte a byte: o PNG muda com a versão da fonte e
do matplotlib, sem que o dado mude.
"""
import subprocess
import sys
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "reproduzir"))
from analise import comparar  # noqa: E402

PUB = RAIZ / "docs" / "tese" / "resultados"
OUT = RAIZ / "saida" / "derivadas"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    falhas = 0
    for s in sorted((RAIZ / "reproduzir" / "derivadas").glob("*.py")):
        antes = {p.name: p.stat().st_mtime_ns for p in OUT.iterdir()}
        r = subprocess.run([sys.executable, str(s)], cwd=OUT, capture_output=True, text=True,
                           env={"MPLBACKEND": "Agg", "PATH": "", "KMP_DUPLICATE_LIB_OK": "TRUE"})
        if r.returncode:
            print(f"{s.name:32s} FALHOU: {r.stderr.strip().splitlines()[-1][:150]}")
            falhas += 1
            continue
        # o que ESTE script escreveu: arquivo novo ou reescrito durante a execução
        novos = sorted(p.name for p in OUT.iterdir()
                       if antes.get(p.name) != p.stat().st_mtime_ns)
        for arq in novos:
            if not arq.endswith(".csv"):
                print(f"{s.name:32s} {arq:36s} regerada")
                continue
            difs = comparar(pd.read_csv(OUT / arq, dtype=str, keep_default_na=False),
                            pd.read_csv(PUB / arq, dtype=str, keep_default_na=False))
            falhas += bool(difs)
            print(f"{s.name:32s} {arq:36s} {'IDÊNTICA' if not difs else '; '.join(difs)}")
    return 1 if falhas else 0


if __name__ == "__main__":
    sys.exit(main())
