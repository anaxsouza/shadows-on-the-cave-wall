"""Submete a REMEDIÇÃO de um ponto Qwen (decl-14 e decl-15) como SageMaker Training Job.

Cópia enxuta de `submeter_sagemaker.py` (que é de outra tarefa e não é editado):
mesma imagem, mesmos três canais (dados, base, ajustado), mesma instância
(ml.g5.2xlarge), mesmo `medir_decoder.py` — agora com SENTINEL_REMEDICAO=1, que
escreve `remedicao.csv` ao lado do `entities.csv`. Um job por ponto.

DIFERENÇAS em relação ao job `sentinel-medir-chave-*`:
- `SENTINEL_REMEDICAO=1`; sem `SENTINEL_FEATURES` (a sonda não é insumo daqui);
- o pacote de fonte leva também `src/selective/janelas.py`;
- `MaxRuntimeInSeconds` = TETO_S (2.700 s), dimensionado para o teto de gasto da
  tarefa (US$ 8 para quatro jobs a US$ 2,5752/h => no máximo US$ 1,93 por job);
  os jobs anteriores levaram 1.464 a 1.795 s.

Uso (as variáveis de ambiente de conta/papel são as de submeter_sagemaker.py):
    python tools/remedir_sagemaker.py --seco 05b-genia
    python tools/remedir_sagemaker.py 05b-genia
    python tools/remedir_sagemaker.py situacao <nome-do-job>
"""
from __future__ import annotations

import io
import json
import os
import sys
import tarfile
from pathlib import Path

import boto3

CONTA = os.environ.get("SENTINEL_AWS_CONTA", "")
REGIAO = os.environ.get("SENTINEL_AWS_REGIAO", "sa-east-1")
BUCKET = os.environ.get("SENTINEL_BUCKET", "")
PREFIXO = "sentinel"
IMAGEM = f"763104351884.dkr.ecr.{REGIAO}.amazonaws.com/pytorch-training:2.9.0-gpu-py312"
PAPEL = os.environ.get("SENTINEL_PAPEL_SAGEMAKER", "")
INSTANCIA = "ml.g5.2xlarge"
PRECO_HORA = 2.5752
VOLUME_GB = 100
TETO_S = 2_700
RAIZ = Path(__file__).resolve().parent

ARQUIVOS_FONTE = ("impressao_digital.py", "rotulos.py", "medir_decoder.py", "features_sonda.py")
ARQUIVOS_EXTRA = (
    ("../configs/config.yaml", "config.yaml"),
    ("../src/selective/decoder_adapter.py", "src/selective/decoder_adapter.py"),
    ("../src/selective/geometry.py", "src/selective/geometry.py"),
    ("../src/selective/signals.py", "src/selective/signals.py"),
    ("../src/selective/janelas.py", "src/selective/janelas.py"),
)
# Mesmos pacotes pinados dos jobs de medição (REQUISITOS de submeter_sagemaker.py).
REQUISITOS = """\
transformers==5.16.1
tokenizers==0.23.2
accelerate==1.15.0
safetensors==0.8.0
huggingface_hub==1.31.0
sentencepiece==0.2.2
protobuf==7.36.1
tiktoken==0.14.0
PyYAML==6.0.2
bitsandbytes==0.49.0
"""

# ponto -> (corpus, modelo HF, prefixo do peso de partida, prefixo do ajustado)
PONTOS = {
    "05b-genia": ("genia", "Qwen/Qwen2.5-0.5B", "base/qwen2.5-0.5b",
                  "saida/sentinel-dec-05b-bf16-genia-20260918/output/"),
    "05b-conll": ("conll2003", "Qwen/Qwen2.5-0.5B", "base/qwen2.5-0.5b",
                  "saida/sentinel-dec-05b-bf16-conll-20260918/output/"),
    "15b-genia": ("genia", "Qwen/Qwen2.5-1.5B", "base/qwen2.5-1.5b",
                  "saida/sentinel-dec-15b-lote1-genia-20260918/output/"),
    "15b-conll": ("conll2003", "Qwen/Qwen2.5-1.5B", "base/qwen2.5-1.5b",
                  "saida/sentinel-dec-15b-lote1-conll-20260918/output/"),
}


def montar_fonte() -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for nome in ARQUIVOS_FONTE:
            dados = (RAIZ / nome).read_bytes()
            info = tarfile.TarInfo(nome)
            info.size, info.mtime, info.mode = len(dados), 0, 0o644
            tf.addfile(info, io.BytesIO(dados))
        for origem, destino in ARQUIVOS_EXTRA:
            dados = (RAIZ / origem).resolve().read_bytes()
            info = tarfile.TarInfo(destino)
            info.size, info.mtime, info.mode = len(dados), 0, 0o644
            tf.addfile(info, io.BytesIO(dados))
        req = REQUISITOS.encode()
        info = tarfile.TarInfo("requirements.txt")
        info.size, info.mtime, info.mode = len(req), 0, 0o644
        tf.addfile(info, io.BytesIO(req))
    return buf.getvalue()


def submeter(ponto: str, seco: bool = False) -> dict:
    corpus, modelo, base, ajustado = PONTOS[ponto]
    nome = f"remedir-{ponto}"
    fonte = montar_fonte()
    chave = f"{PREFIXO}/fonte/{nome}/source.tar.gz"
    if not seco:
        boto3.client("s3", region_name=REGIAO).put_object(Bucket=BUCKET, Key=chave, Body=fonte)

    def canal(n, uri):
        return {"ChannelName": n, "InputMode": "File", "DataSource": {"S3DataSource": {
            "S3DataType": "S3Prefix", "S3Uri": uri, "S3DataDistributionType": "FullyReplicated"}}}

    pedido = {
        "TrainingJobName": nome,
        "AlgorithmSpecification": {"TrainingImage": IMAGEM, "TrainingInputMode": "File"},
        "RoleArn": PAPEL,
        "HyperParameters": {
            "sagemaker_program": json.dumps("medir_decoder.py"),
            "sagemaker_submit_directory": json.dumps(f"s3://{BUCKET}/{chave}"),
            "sagemaker_container_log_level": json.dumps(20),
            "sagemaker_region": json.dumps(REGIAO),
        },
        "InputDataConfig": [
            canal("dados", f"s3://{BUCKET}/{PREFIXO}/dados/"),
            canal("base", f"s3://{BUCKET}/{PREFIXO}/{base}/"),
            canal("ajustado", f"s3://{BUCKET}/{PREFIXO}/{ajustado}"),
        ],
        "OutputDataConfig": {"S3OutputPath": f"s3://{BUCKET}/{PREFIXO}/saida/"},
        "ResourceConfig": {"InstanceType": INSTANCIA, "InstanceCount": 1,
                           "VolumeSizeInGB": VOLUME_GB},
        "StoppingCondition": {"MaxRuntimeInSeconds": TETO_S},
        "Environment": {
            "SENTINEL_DADOS": "/opt/ml/input/data/dados",
            "SENTINEL_MODELO_BASE_CAMINHO": "/opt/ml/input/data/base",
            "SENTINEL_MODELOS": "/opt/ml/model",
            "SENTINEL_CKPT": "/opt/ml/scratch",
            "SENTINEL_DISPOSITIVO": "cuda",
            "SENTINEL_CORPORA": corpus,
            "SENTINEL_MODELO": modelo,
            "SENTINEL_AJUSTADO": "/opt/ml/input/data/ajustado",
            "SENTINEL_SAIDA": "/opt/ml/model",
            "SENTINEL_REMEDICAO": "1",
            "SENTINEL_PRECISAO": "fp32",
            "SENTINEL_OTIMIZADOR": "adamw_torch",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "HF_HOME": "/opt/ml/scratch/hf",
            "TOKENIZERS_PARALLELISM": "false",
        },
        "Tags": [{"Key": "projeto", "Value": "sentinel"}, {"Key": "braco", "Value": "decoder"},
                 {"Key": "etapa", "Value": "remedicao"}, {"Key": "ponto", "Value": ponto}],
    }
    if seco:
        return {"seco": True, "pedido": pedido, "bytes_fonte": len(fonte)}
    sm = boto3.client("sagemaker", region_name=REGIAO)
    resp = sm.create_training_job(**pedido)
    return {"nome": nome, "arn": resp["TrainingJobArn"], "chave_fonte": chave}


def situacao(nome: str) -> dict:
    d = boto3.client("sagemaker", region_name=REGIAO).describe_training_job(TrainingJobName=nome)
    f = {k: d.get(k) for k in ("TrainingJobStatus", "SecondaryStatus", "FailureReason",
                               "TrainingStartTime", "TrainingEndTime", "BillableTimeInSeconds")}
    f["ModelArtifacts"] = (d.get("ModelArtifacts") or {}).get("S3ModelArtifacts")
    if f["BillableTimeInSeconds"]:
        f["custo_usd"] = round(f["BillableTimeInSeconds"] / 3600 * PRECO_HORA, 4)
    return f


if __name__ == "__main__":
    a = sys.argv[1:]
    if a and a[0] == "situacao":
        print(json.dumps(situacao(a[1]), indent=1, default=str))
    elif a and a[0] == "--seco":
        r = submeter(a[1], seco=True)
        print(json.dumps(r["pedido"], indent=1, ensure_ascii=False))
        print(f"source.tar.gz: {r['bytes_fonte']} bytes", file=sys.stderr)
    else:
        print(json.dumps(submeter(a[0]), indent=1))
