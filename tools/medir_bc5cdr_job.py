"""Programa de um job de MEDIÇÃO do BC5CDR: `medir_decoder.py` (com as features da sonda) e, em seguida,
`medir_aggseq.py`, sobre o MESMO peso ajustado, no mesmo contêiner.

POR QUE JUNTOS. Em GENIA/CoNLL foram jobs separados (`medir-chave` e `aggseq`). No BC5CDR o teste tem
5.865 sentenças e cada passagem leva ~1,5 h na A10G; juntar as duas poupa duas partidas de instância e
o recarregamento do peso, e mantém a instância de 1 slot ocupada por uma submissão só. Os dois
programas NÃO são alterados: este arquivo só os invoca como subprocessos, com o mesmo ambiente que a
SageMaker lhes daria, e as saídas não colidem (entities.csv, features_sonda.npz, MEDIDA.json de um;
aggseq.csv e AGGSEQ.json do outro). Falha de qualquer um derruba o job (check=True): nada de
resultado parcial apresentado como completo.

`medir_decoder.py` recusa `--corpus bc5cdr` por `choices` (arquivo da tarefa Remedição); o corpus entra
por SENTINEL_CORPORA, que o argparse aceita como padrão sem checar `choices`.
"""
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

AQUI = Path(__file__).resolve().parent
saida = Path(os.environ["SENTINEL_SAIDA"])
saida.mkdir(parents=True, exist_ok=True)
prov = {p: hashlib.sha256((AQUI / p).read_bytes()).hexdigest()
        for p in ("medir_decoder.py", "medir_aggseq.py")}
(saida / "FONTE_SHA256.json").write_text(json.dumps(prov, indent=1), encoding="utf-8")
print("fontes:", prov, flush=True)

for prog, extra in (("medir_decoder.py", {"SENTINEL_FEATURES": "1"}), ("medir_aggseq.py", {})):
    print(f"=== {prog} ===", flush=True)
    env = {**os.environ, **extra}
    subprocess.run([sys.executable, "-u", str(AQUI / prog)], check=True, env=env, cwd=AQUI)
print("FIM", flush=True)
