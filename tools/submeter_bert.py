"""Submete o treino do BERT (decl-22) como SageMaker Training Job, um job por corpus.

Mesmo caminho verificado de `submeter_sagemaker.py` (imagem pytorch-training
2.9.0-gpu-py312 em sa-east-1, ml.g4dn.xlarge, 1 instância, sem endpoint), com
três diferenças declaradas:

- canais próprios: `dados` (s3://<bucket>/sentinel/dados-bert/) e `base`
  (s3://<bucket>/sentinel/base/bert-base-cased/, com a impressão digital do peso
  de partida gravada em BASE.json), para o job não depender da rede do hub;
- `MaxRuntimeInSeconds` = 5 400 (1,5 h = US$ 1,88 a 1,252/h): o treino estimado é
  de 10 a 25 min por corpus; o teto só impede um travamento de faturar;
- fonte mínima (treinar_bert.py, impressao_digital.py, bert_adapter.py).

Uso:  python tools/submeter_bert.py <corpus> [--seco] [--limite N]
      python tools/submeter_bert.py situacao <nome-do-job>
Variáveis: SENTINEL_BUCKET, SENTINEL_PAPEL_SAGEMAKER (obrigatórias, sem padrão).
"""
from __future__ import annotations

import io
import json
import os
import sys
import tarfile
import time
from pathlib import Path

REGIAO = os.environ.get("SENTINEL_AWS_REGIAO", "sa-east-1")
BUCKET = os.environ.get("SENTINEL_BUCKET", "")
PAPEL = os.environ.get("SENTINEL_PAPEL_SAGEMAKER", "")
PREFIXO = "sentinel"
IMAGEM = f"763104351884.dkr.ecr.{REGIAO}.amazonaws.com/pytorch-training:2.9.0-gpu-py312"
INSTANCIA = "ml.g4dn.xlarge"
VOLUME_GB = 30
TETO_S = 5_400
PRECO_HORA = 1.252

RAIZ = Path(__file__).resolve().parent
FONTE = ((RAIZ / "treinar_bert.py", "treinar_bert.py"),
         (RAIZ / "impressao_digital.py", "impressao_digital.py"),
         (RAIZ.parent / "src/selective/bert_adapter.py", "bert_adapter.py"))
REQUISITOS = "transformers==5.16.1\ntokenizers==0.23.2\nsafetensors==0.8.0\nhuggingface_hub==1.31.0\n"


def montar_fonte() -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for origem, destino in FONTE:
            dados = origem.read_bytes()
            info = tarfile.TarInfo(destino)
            info.size, info.mtime, info.mode = len(dados), 0, 0o644
            tf.addfile(info, io.BytesIO(dados))
        req = REQUISITOS.encode()
        info = tarfile.TarInfo("requirements.txt")
        info.size, info.mtime, info.mode = len(req), 0, 0o644
        tf.addfile(info, io.BytesIO(req))
    return buf.getvalue()


def pedido_de(corpus: str, nome: str, chave: str, limite: int = 0) -> dict:
    def canal(nome_, prefixo):
        return {"ChannelName": nome_, "InputMode": "File", "DataSource": {"S3DataSource": {
            "S3DataType": "S3Prefix", "S3Uri": f"s3://{BUCKET}/{prefixo}/",
            "S3DataDistributionType": "FullyReplicated"}}}
    env = {
        "SENTINEL_DADOS": "/opt/ml/input/data/dados",
        "SENTINEL_BERT_BASE": "/opt/ml/input/data/base",
        "SENTINEL_SAIDA": "/opt/ml/model",
        "SENTINEL_CORPORA": corpus,
        "SENTINEL_DISPOSITIVO": "cuda",
        "HF_HOME": "/opt/ml/scratch/hf", "TOKENIZERS_PARALLELISM": "false",
    }
    tags = [{"Key": "projeto", "Value": "sentinel"}, {"Key": "braco", "Value": "bert"},
            {"Key": "corpus", "Value": corpus}]
    if limite:
        env["SENTINEL_LIMITE"] = str(limite)
        tags.append({"Key": "fumaca", "Value": "sim"})
    return {
        "TrainingJobName": nome,
        "AlgorithmSpecification": {"TrainingImage": IMAGEM, "TrainingInputMode": "File"},
        "RoleArn": PAPEL,
        "HyperParameters": {
            "sagemaker_program": json.dumps("treinar_bert.py"),
            "sagemaker_submit_directory": json.dumps(f"s3://{BUCKET}/{chave}"),
            "sagemaker_container_log_level": json.dumps(20),
            "sagemaker_region": json.dumps(REGIAO),
        },
        "InputDataConfig": [canal("dados", f"{PREFIXO}/dados-bert"),
                            canal("base", f"{PREFIXO}/base/bert-base-cased")],
        "OutputDataConfig": {"S3OutputPath": f"s3://{BUCKET}/{PREFIXO}/saida/"},
        "ResourceConfig": {"InstanceType": INSTANCIA, "InstanceCount": 1, "VolumeSizeInGB": VOLUME_GB},
        "StoppingCondition": {"MaxRuntimeInSeconds": TETO_S},
        "Environment": env, "Tags": tags,
    }


def submeter(corpus: str, limite: int = 0, seco: bool = False) -> dict:
    nome = f"sentinel-bert-{corpus}-{time.strftime('%Y%m%d-%H%M%S')}"
    chave = f"{PREFIXO}/fonte/{nome}/source.tar.gz"
    fonte = montar_fonte()
    pedido = pedido_de(corpus, nome, chave, limite)
    if seco:
        return {"seco": True, "pedido": pedido, "bytes_fonte": len(fonte)}
    if not BUCKET or not PAPEL:
        raise SystemExit("defina SENTINEL_BUCKET e SENTINEL_PAPEL_SAGEMAKER")
    import boto3
    boto3.client("s3", region_name=REGIAO).put_object(Bucket=BUCKET, Key=chave, Body=fonte)
    resp = boto3.client("sagemaker", region_name=REGIAO).create_training_job(**pedido)
    return {"nome": nome, "arn": resp["TrainingJobArn"], "chave_fonte": chave,
            "teto_usd": round(TETO_S / 3600 * PRECO_HORA, 2)}


def situacao(nome: str) -> dict:
    import boto3
    d = boto3.client("sagemaker", region_name=REGIAO).describe_training_job(TrainingJobName=nome)
    fora = {k: d.get(k) for k in ("TrainingJobStatus", "SecondaryStatus", "FailureReason",
                                  "TrainingStartTime", "TrainingEndTime", "BillableTimeInSeconds")}
    fora["ModelArtifacts"] = (d.get("ModelArtifacts") or {}).get("S3ModelArtifacts")
    if fora["BillableTimeInSeconds"]:
        fora["custo_usd"] = round(fora["BillableTimeInSeconds"] / 3600 * PRECO_HORA, 3)
    return fora


if __name__ == "__main__":
    a = sys.argv[1:]
    if a and a[0] == "situacao":
        print(json.dumps(situacao(a[1]), indent=1, default=str))
    else:
        lim = int(a[a.index("--limite") + 1]) if "--limite" in a else 0
        print(json.dumps(submeter(a[0], lim, "--seco" in a), indent=1, default=str))
