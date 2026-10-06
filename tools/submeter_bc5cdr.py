"""Submissões do BC5CDR (decl-16 a decl-21) à SageMaker: reutiliza `submeter_sagemaker.py` sem alterá-lo.

Este arquivo NÃO muda nenhum comportamento existente: importa `submeter` e fixa, por job, o que é
específico do BC5CDR — o corpus, o tempo máximo (dimensionado para o teto de gasto da tarefa, e não o
padrão de 4,5 h) e o volume. Conta, bucket e papel vêm do ambiente, como no original.

Uso (em célula com a credencial AWS declarada):
    from submeter_bc5cdr import treinar_decoder_bc5cdr, treinar_gliner_bc5cdr, medir_bc5cdr
"""
from __future__ import annotations

import io
import json
import os
import sys
import tarfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("SENTINEL_AWS_REGIAO", "sa-east-1")
# SENTINEL_AWS_CONTA, SENTINEL_BUCKET and SENTINEL_PAPEL_SAGEMAKER come from the environment
import submeter_sagemaker as S  # noqa: E402

PRECO_G5_2XL = 2.5752     # US$/h on-demand
PRECO_G4DN_XL = 1.252


def _com_teto(segundos: int, volume: int = 100):
    S.TETO_S, S.VOLUME_GB = segundos, volume


def treinar_decoder_bc5cdr(ponto: str, teto_s: int, seco: bool = False, limite: int = 0) -> dict:
    """Treino de um Qwen no BC5CDR: bf16 + Adam 8 bits em g5.2xlarge, a receita de GENIA/CoNLL."""
    _com_teto(teto_s)
    tag = {"qwen2.5-0.5b": "05b", "qwen2.5-1.5b": "15b"}[ponto]
    nome = f"sentinel-bc5cdr-dec-{tag}-{time.strftime('%m%d%H%M%S')}"
    return S.submeter(nome=nome, corpora="bc5cdr", programa="treinar_decoder.py", ponto=ponto,
                      precisao="bf16", otimizador="adamw_bnb_8bit", limite=limite, seco=seco)


def medir_bc5cdr(ponto: str, modelo_s3: str, programa: str, teto_s: int, seco: bool = False) -> dict:
    """`medir_decoder.py` (com as features da sonda) ou `medir_aggseq.py` sobre o peso ajustado."""
    assert programa in ("medir_decoder.py", "medir_aggseq.py", "medir_bc5cdr_job.py")
    _com_teto(teto_s)
    # `medir_decoder.py` passou a importar `src.selective.janelas` (tarefa Remedição); o tarball do
    # submissor original não a leva. O primeiro job de medição (sentinel-bc5cdr-medir-05b-1001172308)
    # morreu em 209 s por isso. Acrescentada aqui, sem editar o submissor.
    extra = ("../src/selective/janelas.py", "src/selective/janelas.py")
    if extra not in S.ARQUIVOS_EXTRA:
        S.ARQUIVOS_EXTRA = tuple(S.ARQUIVOS_EXTRA) + (extra,)
    if "medir_bc5cdr_job.py" not in S.ARQUIVOS_FONTE:   # o wrapper viaja no tarball, sem editar o submissor
        S.ARQUIVOS_FONTE = tuple(S.ARQUIVOS_FONTE) + ("medir_bc5cdr_job.py",)
    tag = {"qwen2.5-0.5b": "05b", "qwen2.5-1.5b": "15b"}[ponto]
    k = {"medir_decoder.py": "chave", "medir_aggseq.py": "aggseq", "medir_bc5cdr_job.py": "medir"}[programa]
    nome = f"sentinel-bc5cdr-{k}-{tag}-{time.strftime('%m%d%H%M%S')}"
    return S.submeter(nome=nome, corpora="bc5cdr", programa=programa, ponto=ponto,
                      modelo_s3=modelo_s3, seco=seco)


# --- GLiNER: treinar_extrator.py na g4dn.xlarge ---------------------------------------------------
REQ_GLINER = """\
gliner==0.2.26
transformers==5.1.0
tokenizers==0.22.2
accelerate==1.13.0
safetensors==0.7.0
huggingface-hub==1.16.4
sentencepiece==0.2.1
protobuf==7.35.0
onnxruntime==1.26.0
PyYAML==6.0.3
"""


def _fonte_gliner() -> bytes:
    buf = io.BytesIO()
    raiz = Path(__file__).resolve().parent
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for nome, dados in (("treinar_extrator.py", (raiz / "treinar_extrator.py").read_bytes()),
                            ("requirements.txt", REQ_GLINER.encode())):
            info = tarfile.TarInfo(nome)
            info.size, info.mtime, info.mode = len(dados), 0, 0o644
            tf.addfile(info, io.BytesIO(dados))
    return buf.getvalue()


def treinar_gliner_bc5cdr(teto_s: int = 7200, seco: bool = False, limite: int = 0) -> dict:
    import boto3
    nome = f"sentinel-bc5cdr-gliner-{time.strftime('%m%d%H%M%S')}"
    s3 = boto3.client("s3", region_name=S.REGIAO)
    chave = f"{S.PREFIXO}/fonte/{nome}/source.tar.gz"
    fonte = _fonte_gliner()
    pedido = {
        "TrainingJobName": nome,
        "AlgorithmSpecification": {"TrainingImage": S.IMAGEM, "TrainingInputMode": "File"},
        "RoleArn": S.PAPEL,
        "HyperParameters": {
            "sagemaker_program": json.dumps("treinar_extrator.py"),
            "sagemaker_submit_directory": json.dumps(f"s3://{S.BUCKET}/{chave}"),
            "sagemaker_container_log_level": json.dumps(20),
            "sagemaker_region": json.dumps(S.REGIAO),
        },
        "InputDataConfig": [{
            "ChannelName": "dados",
            "DataSource": {"S3DataSource": {"S3DataType": "S3Prefix",
                                            "S3Uri": f"s3://{S.BUCKET}/{S.PREFIXO}/dados/",
                                            "S3DataDistributionType": "FullyReplicated"}},
            "InputMode": "File"}],
        "OutputDataConfig": {"S3OutputPath": f"s3://{S.BUCKET}/{S.PREFIXO}/saida/"},
        "ResourceConfig": {"InstanceType": "ml.g4dn.xlarge", "InstanceCount": 1, "VolumeSizeInGB": 60},
        "StoppingCondition": {"MaxRuntimeInSeconds": teto_s},
        "Environment": {
            "SENTINEL_CORPORA": "bc5cdr", "SENTINEL_DADOS": "/opt/ml/input/data/dados",
            "SENTINEL_MODELOS": "/opt/ml/model", "SENTINEL_CKPT": "/opt/ml/scratch",
            "SENTINEL_DISPOSITIVO": "cuda", "HF_HOME": "/opt/ml/scratch/hf",
            "TOKENIZERS_PARALLELISM": "false",
            **({"SENTINEL_LIMITE": str(limite)} if limite else {}),
        },
        "Tags": [{"Key": "projeto", "Value": "sentinel"}, {"Key": "braco", "Value": "gliner-bc5cdr"}],
    }
    if seco:
        return {"seco": True, "pedido": pedido, "bytes_fonte": len(fonte)}
    s3.put_object(Bucket=S.BUCKET, Key=chave, Body=fonte)
    r = S.boto3.client("sagemaker", region_name=S.REGIAO).create_training_job(**pedido)
    return {"nome": nome, "arn": r["TrainingJobArn"], "chave_fonte": chave}
