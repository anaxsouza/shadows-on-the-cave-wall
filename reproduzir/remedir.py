"""NÍVEL 2 DE REPRODUÇÃO: mede UM ponto a partir do peso publicado e compara a tabela.

    python reproduzir/remedir.py <ponto> <diretorio_do_peso> [--max-samples N]

<ponto> é uma pasta de `dados_medidos/`, por exemplo `gliner_base-ft-genia__sinais` ou
`qwen05b-ft-genia`. A ordem é fixa, e cada passo pode parar tudo:

1. CONFERE O PESO contra PESOS.json. Um peso que não confere não é medido, porque
   a tabela sairia de outro modelo carregando o nome deste.
2. MEDE com o mesmo comando que produziu a tabela publicada: `main.py selective
   --measure` para o GLiNER, `tools/medir_decoder.py` para o Qwen. O segundo lê o
   teste de `dados_decoder/`, que `tools/exportar_corpus_teste.py` reconstrói.
3. COMPARA com `dados_medidos/<ponto>`. As colunas discretas (sentença, acerto,
   índices de token, rótulo, trecho) têm de sair IDÊNTICAS: são as predições. As
   contínuas são reportadas pela maior diferença absoluta, porque CPU e GPU, ou
   duas GPUs, arredondam diferente na última casa. Foi medido: entre T4 e CPU, a
   confiança do GLiNER difere em até 3×10⁻⁵ com as predições idênticas.

Com --max-samples, compara só as primeiras N sentenças. É o modo de conferir o
encanamento em minutos; a conferência que conta é a completa.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
DADOS = RAIZ / "dados_medidos"
DISCRETAS = ["sentence_id", "loss", "is_nested", "token_indices", "rotulo", "mencao",
             "ancorada", "span_size", "span_position", "n_tokens"]
DECL = {"gliner_base-ft-genia__sinais": "decl-07-sinais-genia",
        "gliner_base-ft-conll2003__sinais": "decl-08-sinais-conll",
        "gliner_base-ft-genia": "decl-05-ajustado-genia",
        "gliner_base-ft-conll2003": "decl-06-ajustado-conll"}


def conferir_peso(d: Path, nome: str) -> None:
    sys.path.insert(0, str(RAIZ / "tools"))
    from impressao_digital import impressao_digital
    esp = {p["nome"]: p["impressao"] for p in
           json.loads((RAIZ / "PESOS.json").read_text())["pesos"]}[nome]
    got = impressao_digital(d)["checkpoint_sha256_v1"]
    if got != esp:
        sys.exit(f"PESO NÃO CONFERE: {d} tem {got[:16]}, PESOS.json diz {esp[:16]}")
    print(f"peso {nome}: CONFERE {got[:16]}")


def medir(ponto: str, peso: Path, corpus: str, saida: Path, n: int) -> None:
    env = {**os.environ, "KMP_DUPLICATE_LIB_OK": "TRUE"}
    lim = ["--max-samples", str(n)] if n > 0 else []
    if ponto.startswith("gliner"):
        mod = ponto.replace("__sinais", "")
        link = RAIZ / mod                      # o registro resolve o modelo pelo nome
        if not link.exists():
            link.symlink_to(peso.resolve())
        cmd = [sys.executable, "main.py", "selective", "--measure", "--model", mod,
               "--dataset", corpus, "--config", f"configs/{DECL[ponto]}.yaml",
               "--output-dir", str(saida)] + lim
    else:
        env.update(SENTINEL_DADOS=str(RAIZ / "dados_decoder"), SENTINEL_FEATURES="0")
        cmd = [sys.executable, "tools/medir_decoder.py", "--corpus", corpus, "--split", "test",
               "--modelo", str(peso), "--saida", str(saida)] + lim
    print("medindo:", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=RAIZ, env=env, check=True)


def comparar(ponto: str, corpus: str, saida: Path, n: int) -> int:
    pub = pd.read_csv(DADOS / ponto / corpus / "test" / "entities.csv", dtype=str,
                      keep_default_na=False)
    nov = pd.read_csv(saida / "entities.csv", dtype=str, keep_default_na=False)
    if "mencao" in nov:        # o publicado traz o hash do trecho, não o trecho
        nov["mencao"] = nov["mencao"].map(
            lambda t: hashlib.sha256(t.encode("utf-8")).hexdigest()[:16])
    if n > 0:
        pub = pub[pub["sentence_id"].isin(nov["sentence_id"].unique())].reset_index(drop=True)
    if len(pub) != len(nov):
        print(f"DIVERGE: {len(nov)} linhas medidas contra {len(pub)} publicadas")
        return 1
    falhou = 0
    for c in pub.columns:
        if c not in nov:
            continue
        if c in DISCRETAS:
            k = int((pub[c].values != nov[c].values).sum())
            print(f"  {c:24s} {'IDÊNTICA' if k == 0 else f'DIVERGE em {k} linhas'}")
            falhou |= k > 0
        else:
            a = pd.to_numeric(pub[c], errors="coerce").to_numpy()
            b = pd.to_numeric(nov[c], errors="coerce").to_numpy()
            ok = ~np.isnan(a) & ~np.isnan(b)
            if ok.any():
                print(f"  {c:24s} maior diferença {np.abs(a[ok] - b[ok]).max():.1e}")
    print(f"\n{len(nov)} linhas | predições {'IDÊNTICAS' if not falhou else 'DIVERGEM'}")
    return int(falhou)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("ponto"); ap.add_argument("peso", type=Path)
    ap.add_argument("--max-samples", type=int, default=-1)
    a = ap.parse_args()
    corpus = next(p.name for p in (DADOS / a.ponto).iterdir() if p.is_dir())
    conferir_peso(a.peso, a.ponto.replace("__sinais", ""))
    saida = RAIZ / "saida" / "remedicao" / a.ponto / corpus / "test"
    medir(a.ponto, a.peso, corpus, saida, a.max_samples)
    return comparar(a.ponto, corpus, saida, a.max_samples)


if __name__ == "__main__":
    sys.exit(main())
