"""Baixa os quatro pesos do eixo e mede cada um no corpus dele.

ORDEM: 0,5B primeiro. Não é preferência — é que o 0,5B termina antes, e ter um
ponto do eixo medido cedo permite achar defeito de análise sem esperar o eixo
inteiro. Se algo estiver errado no formato de saída, é melhor descobrir em 6 h
do que em 26 h.

CONTAGEM DE THREADS FIXA E REGISTRADA. `OMP_NUM_THREADS` muda a ORDEM das
reduções em ponto flutuante, logo muda os últimos dígitos. Não muda veredito
nenhum, mas muda reprodutibilidade bit-a-bit — então o valor fica fixo aqui e
gravado no MEDIDA.json de cada corrida, em vez de depender do que a máquina
resolver usar. As medições de tempo de 18/09 (4,43 s/sentença) foram com 1
thread; com 8 a taxa é outra e a corrida a reporta.

O RENOMEIO DOS PESOS DE 1,5B. Os tarballs dos jobs de 1,5B trazem o diretório
rotulado `qwen05b-...` por um literal em treinar_decoder.py, corrigido DEPOIS
daqueles treinos. Os pesos são de Qwen2.5-1.5B e a impressão digital cobre os
BYTES de model.safetensors, não o nome da pasta — por isso renomear é seguro, e
a conferência da impressão depois do renomeio prova que é.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "tools"))
from impressao_digital import impressao_digital  # noqa: E402

B = os.environ.get("SENTINEL_BUCKET", "")  # bucket com os pesos publicados
THREADS = "8"
BASE = Path(os.environ.get("SENTINEL_BASE", "/tmp/sentinel_eixo"))

PONTOS = [
    ("qwen05b-ft-genia",      "genia",     "sentinel-dec-05b-bf16-genia-20260918",  "qwen05b-ft-genia"),
    ("qwen05b-ft-conll2003",  "conll2003", "sentinel-dec-05b-bf16-conll-20260918",  "qwen05b-ft-conll2003"),
    ("qwen15b-ft-genia",      "genia",     "sentinel-dec-15b-lote1-genia-20260918", "qwen05b-ft-genia"),
    ("qwen15b-ft-conll2003",  "conll2003", "sentinel-dec-15b-lote1-conll-20260918", "qwen05b-ft-conll2003"),
]


def baixar(nome: str, corpus: str, job: str, no_tar: str) -> Path:
    import boto3
    d = BASE / nome
    if (d / "model.safetensors").exists():
        print(f"  {nome}: já em disco", flush=True)
        return d
    d.parent.mkdir(parents=True, exist_ok=True)
    tgz = BASE / f"{nome}.tar.gz"
    if not tgz.exists():
        t0 = time.time()
        boto3.client("s3").download_file(B, f"sentinel/saida/{job}/output/model.tar.gz", str(tgz))
        print(f"  {nome}: baixado {tgz.stat().st_size/1e9:.2f} GB em {time.time()-t0:.0f}s", flush=True)
    tmp = BASE / f"_x_{nome}"
    shutil.rmtree(tmp, ignore_errors=True)
    with tarfile.open(tgz) as tf:
        tf.extractall(tmp, filter="data")
    shutil.rmtree(d, ignore_errors=True)
    shutil.move(str(tmp / no_tar), str(d))
    proc = next(tmp.glob("ajuste_decoder_*.json"))
    p = json.loads(proc.read_text(encoding="utf-8"))
    r = impressao_digital(d)
    ok = r["checkpoint_sha256_v1"] == p["checkpoint_sha256_v1"]
    print(f"  {nome}: base={p['modelo_base']} ckpt={p['checkpoint']} "
          f"impressao={'CONFERE' if ok else 'DIVERGE'}", flush=True)
    if not ok:
        raise SystemExit(f"{nome}: impressao digital NAO confere — nao medir sobre peso duvidoso")
    shutil.copy(proc, d.parent / f"procedencia_{nome}.json")
    shutil.rmtree(tmp, ignore_errors=True)
    return d


def medir(nome: str, corpus: str, modelo: Path) -> None:
    saida = BASE / "resultados" / nome / corpus / "test"
    if (saida / "entities.csv").exists():
        print(f"  {nome}: medição já existe em {saida}", flush=True)
        return
    saida.mkdir(parents=True, exist_ok=True)
    env = {**os.environ,
           "SENTINEL_DISPOSITIVO": "cpu", "SENTINEL_FEATURES": "1",
           "OMP_NUM_THREADS": THREADS, "MKL_NUM_THREADS": THREADS,
           "TOKENIZERS_PARALLELISM": "false", "KMP_DUPLICATE_LIB_OK": "TRUE",
           # SEM ISTO A MEDIÇÃO NÃO RODA, e a falha é silenciosa do ponto de
           # vista do laço: o carregador de corpus tenta buscar o dataset no
           # servidor Xet do HuggingFace, que este sandbox não alcança, morre em
           # segundos, e o ponto seguinte começa como se nada fosse. Foi o que
           # aconteceu em 22/09/2026 com três pontos seguidos.
           "HF_HUB_DISABLE_XET": "1"}
    t0 = time.time()
    r = subprocess.run([sys.executable, "-u", str(RAIZ / "tools" / "medir_decoder.py"),
                        "--corpus", corpus, "--split", "test",
                        "--modelo", str(modelo), "--saida", str(saida)],
                       cwd=RAIZ, env=env, capture_output=True, text=True)
    dt = time.time() - t0
    cauda = [l for l in r.stdout.splitlines() if "linhas (" in l or "erro base" in l]
    print(f"  {nome}/{corpus}: {'OK' if r.returncode == 0 else 'FALHOU'} em {dt/3600:.2f} h "
          f"| {cauda[-1] if cauda else r.stderr.strip()[-300:]}", flush=True)
    # PARAR no primeiro erro. A versão anterior seguia para o ponto seguinte, e
    # com isso três falhas de segundos pareceram progresso: os pesos baixavam,
    # os diretórios de saída apareciam vazios, e só ao olhar o disco é que se
    # via que nada tinha sido medido. Falhar alto é mais barato que descobrir
    # depois de 10 h que a corrida inteira estava vazia.
    if r.returncode != 0:
        raise SystemExit(f"{nome}/{corpus} falhou — parando a corrida em vez de "
                         f"seguir para o ponto seguinte")
    if True:
        m = saida / "MEDIDA.json"
        if m.exists():
            d = json.loads(m.read_text(encoding="utf-8"))
            d["omp_num_threads"] = THREADS
            d["segundos_de_parede"] = round(dt, 1)
            m.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")


def main() -> None:
    for nome, corpus, job, no_tar in PONTOS:
        print(f"=== {nome} ({corpus}) ===", flush=True)
        medir(nome, corpus, baixar(nome, corpus, job, no_tar))
    print("\nEIXO MEDIDO. Saídas em", BASE / "resultados", flush=True)


if __name__ == "__main__":
    main()
