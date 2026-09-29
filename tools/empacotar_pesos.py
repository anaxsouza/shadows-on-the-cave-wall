"""Package the six fine-tuned weights for the Zenodo deposit, verifying each fingerprint.

    python tools/empacotar_pesos.py <saida> [--gliner <dir com gliner_base-ft-*>] [--qwen <dir com qwen*-ft-*>]

Writes one `<nome>.tar` per model (uncompressed: safetensors do not compress) and a
`SHA256SUMS` in <saida>. Every model is fingerprinted with the SAME recipe as PESOS.json
before it is packed, and packing STOPS on any mismatch, so nothing unverified can reach
the deposit. The Qwen weights not found locally are fetched from the authors' bucket
(`SENTINEL_BUCKET`), which only the authors can read; an auditor starts from the deposit.

Resume state (optimizer, scheduler, RNG) is never in these directories; see
tools/impressao_digital.py for why it is excluded from the fingerprint.

The 1.5B jobs wrote their directory under a `qwen05b-` label (a literal in
treinar_decoder.py fixed after those runs). The fingerprint covers the BYTES of the
weights, not the folder name, so the folder is renamed and then verified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tarfile
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "tools"))
from impressao_digital import impressao_digital  # noqa: E402

# nome publicado -> (job de treino, nome da pasta dentro do model.tar.gz do job)
JOBS = {
    "qwen05b-ft-genia": ("sentinel-dec-05b-bf16-genia-20260918", "qwen05b-ft-genia"),
    "qwen05b-ft-conll2003": ("sentinel-dec-05b-bf16-conll-20260918", "qwen05b-ft-conll2003"),
    "qwen15b-ft-genia": ("sentinel-dec-15b-lote1-genia-20260918", "qwen05b-ft-genia"),
    "qwen15b-ft-conll2003": ("sentinel-dec-15b-lote1-conll-20260918", "qwen05b-ft-conll2003"),
}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        while b := f.read(8 << 20):
            h.update(b)
    return h.hexdigest()


def buscar_qwen(nome: str, tmp: Path) -> Path:
    import boto3
    job, pasta = JOBS[nome]
    bucket = os.environ["SENTINEL_BUCKET"]
    tgz = tmp / f"{nome}.tar.gz"
    print(f"  {nome}: baixando {job}", flush=True)
    boto3.client("s3").download_file(bucket, f"sentinel/saida/{job}/output/model.tar.gz", str(tgz))
    with tarfile.open(tgz) as tf:
        tf.extractall(tmp / nome / "_x", filter="data")
    achado = next((tmp / nome / "_x").rglob(f"{pasta}/model.safetensors")).parent
    destino = tmp / nome / nome
    shutil.move(str(achado), destino)
    tgz.unlink()
    return destino


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("saida", type=Path)
    ap.add_argument("--gliner", type=Path, default=Path("/tmp/gliner_ft"))
    ap.add_argument("--qwen", type=Path, default=Path("/tmp/sentinel_eixo"))
    a = ap.parse_args()
    a.saida.mkdir(parents=True, exist_ok=True)
    tmp = a.saida / "_trabalho"
    tmp.mkdir(exist_ok=True)
    esperado = {p["nome"]: p for p in json.loads((RAIZ / "PESOS.json").read_text())["pesos"]}
    somas = []
    for nome, reg in esperado.items():
        alvo = a.saida / f"{nome}.tar"
        if alvo.exists():
            print(f"{nome}: já empacotado, conferindo a soma", flush=True)
        else:
            local = (a.gliner if nome.startswith("gliner") else a.qwen) / nome
            d = local if (local / "model.safetensors").exists() or any(local.glob("*.bin")) else None
            if d is None:
                if nome not in JOBS:
                    print(f"{nome}: AUSENTE em {local}"); return 1
                d = buscar_qwen(nome, tmp)
            got = impressao_digital(d)["checkpoint_sha256_v1"]
            if got != reg["impressao"]:
                print(f"{nome}: DIVERGE ({got[:16]} contra {reg['impressao'][:16]})"); return 1
            print(f"{nome}: impressão CONFERE {got[:16]}", flush=True)
            with tarfile.open(alvo.with_suffix(".tar.part"), "w") as tf:
                tf.add(d, arcname=nome)
            alvo.with_suffix(".tar.part").rename(alvo)
            if d.is_relative_to(tmp):
                shutil.rmtree(tmp / nome)
        somas.append(f"{sha256(alvo)}  {alvo.name}")
    (a.saida / "SHA256SUMS").write_text("\n".join(somas) + "\n")
    shutil.copy2(RAIZ / "PESOS.json", a.saida / "PESOS.json")
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"\n{len(somas)} pacotes em {a.saida}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
