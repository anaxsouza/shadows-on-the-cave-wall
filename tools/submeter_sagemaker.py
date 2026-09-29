"""Submete o treino do braço decoder como SageMaker Training Job.

POR QUE ESTE ARQUIVO EXISTE

A sessão de 16/09/2026 mediu que a instância com GPU ficou 86% do tempo ociosa e
faturada, e que 11% do tempo DENTRO de job foi gasto em execuções que falharam
por coisas que um teste local pega. Um training job gerenciado tem a propriedade
que a instância não tinha: carrega a entrada, roda, devolve a saída e morre.
Não existe estado que só ele tenha, e não existe minuto faturado depois do fim.

A GARANTIA NÃO É CONFIANÇA, É FORMATO DE RECURSO

A política de submissão usada pelo autor permite `CreateTrainingJob` e nega
`CreateEndpoint`, `CreateNotebookInstance` e `ec2:RunInstances`. Um training job
não tem como sobreviver ao próprio fim. É por isso que este caminho satisfaz o
requisito literal — "jobs que eu submeta, uma vez que acabou, simplesmente para
de gastar" — sem depender de ninguém se lembrar de desligar nada.

AS ESCOLHAS, E O QUE CADA UMA CUSTARIA DIFERENTE

1. UM job POR CORPUS. Esta escolha começou ao contrário — um job para os dois,
   com o risco declarado de que a falha do segundo levaria o peso do primeiro,
   porque a SageMaker só envia `/opt/ml/model` ao terminar BEM. O risco não era
   hipotético: medido no log do primeiro job, o passo custa 2,04 s, a avaliação
   de 300 sentenças de validação custa 17,4 min, e são 4 avaliações por corpus.
   Isso dá 155 min no GENIA e ~140 min no CoNLL — 4,9 h somadas, contra um teto
   de 3 h. O job morreria no teto durante o segundo corpus e devolveria NADA,
   inclusive o primeiro modelo já escolhido.

   Por corpus, cada job cabe com folga e a falha de um não leva o outro. A
   quota de 1 instância simultânea os torna sequenciais de todo modo; o que se
   paga a mais é uma segunda puxada de imagem (~4 min). Custou ~1 h de GPU
   descobrir isso por medição em vez de por estimativa.

2. VERSÕES PINADAS pelo `pip freeze` da máquina onde o script foi escrito e
   testado, e não pelo que a imagem traz. `transformers` 5.16.1 é exigência de
   API e não preferência: o script usa `processing_class=` e `warmup_steps`,
   que mudaram de nome entre versões. A imagem entra com o torch dela (2.9,
   CUDA pronta) porque pinar torch aqui baixaria uma roda de 2,5 GB e poderia
   trocar a compilação de CUDA pela de CPU.

3. `MaxRuntimeInSeconds` em 4,5 h (`TETO_S = 16_200`), e não no teto de 6 h que a
   política permite. ESTE ITEM DIZIA 3 h e estava desatualizado: 3 h era o valor
   do desenho de UM job para os dois corpora, e a inversão para um job por corpus
   o alargou para 4,5 h — a docstring não acompanhou, e foi corrigida em
   18/09/2026. O
   treino estimado é de ~1 h a 1,5 h. O teto não é orçamento: é o que impede um
   travamento de faturar seis horas. Folga de 2x, não de 6x.

O QUE ESTE ARQUIVO NÃO FAZ

Não mede. A medição roda local, em CPU, porque foi MEDIDO que ela não precisa de
GPU. ESTE NÚMERO JÁ ERROU DUAS VEZES, e a segunda foi minha no mesmo dia. Dizia
`2,53 s por sentença, 3,7 h`, herdado da sessão anterior sob outro regime de
geração e apresentado como MEDIDO quando eu mesmo o havia posto em dúvida. Ao
corrigi-lo escrevi `3,49 s por sentença`, que saiu de dividir 69,8 s por 20
sentenças — mas aqueles 69,8 s INCLUEM o carregamento do modelo, então o
resultado superestima o custo marginal, e a divisão nunca foi rodada em célula
nenhuma: era aritmética de cabeça apresentada como medição.

O valor abaixo é DERIVADO de duas corridas do medidor, em 18/09/2026, em CPU de 1
thread e com as features da sonda ligadas:

    n=20  ->  104,0 s          custo fixo de carga:      15,5 s
    n=60  ->  281,1 s          custo MARGINAL/sentença:   4,43 s

O coeficiente entre os dois tamanhos é o que elimina a carga do modelo, e é ele
que se projeta: 5.307 sentenças dão 6,5 h por ponto do eixo e 13,1 h nos dois.
Com a passagem de feixe do AggSeq (+7,25 s/sentença, medida à parte), 17,2 h por
ponto — e é por isso que o feixe foi para a GPU. Em máquina parada que custa
zero. Misturar as duas coisas num job é o que produziu os 86%.
"""
from __future__ import annotations

import io
import json
import tarfile
import time
from pathlib import Path

import os

import boto3

# Conta, região, bucket e papel são DO AUDITOR, não do autor: vêm do ambiente,
# e a falta de qualquer um é erro na hora de submeter, não padrão silencioso.
CONTA = os.environ.get("SENTINEL_AWS_CONTA", "")
REGIAO = os.environ.get("SENTINEL_AWS_REGIAO", "sa-east-1")
BUCKET = os.environ.get("SENTINEL_BUCKET", "")
PREFIXO = "sentinel"

# Resolvida por `sagemaker.core.image_uris.retrieve`, não escrita à mão.
IMAGEM = f"763104351884.dkr.ecr.{REGIAO}.amazonaws.com/pytorch-training:2.9.0-gpu-py312"
PAPEL = os.environ.get("SENTINEL_PAPEL_SAGEMAKER", "")  # ARN do papel de execução

INSTANCIA = "ml.g4dn.xlarge"      # Tesla T4 — a MESMA GPU da instância anterior
# 100, e o valor foi ENCONTRADO por duas recusas da API, não escolhido: a
# g4dn.xlarge traz armazenamento local fixo de 125 GB, então 150 é recusado
# ("please reduce"); omitir o campo também é recusado, porque a ausência é lida
# como 0 e o mínimo é 1. O teto útil é 125 e o necessário são ~42 GB (3
# checkpoints x ~6 GB x 2 corpora, mais modelo, base e cache).
VOLUME_GB = 100
# 4,5 h para um corpus cuja execução medida é de ~2,6 h. A política permite
# 21.600 s. O teto não é orçamento — é o que impede um travamento de faturar
# até o limite; 3 h davam 14% de folga sobre a medição, que é pouco para
# absorver um corpus mais lento.
TETO_S = 16_200

RAIZ = Path(__file__).resolve().parent
# config.yaml viaja no pacote porque tools/rotulos.py lê dele o vocabulário de
# rótulos, que é a fonte Única do projeto. Sem ele o contêiner teria de
# redeclarar a tabela, que é exatamente o defeito medido em 16/09/2026.
ARQUIVOS_FONTE = ("treinar_decoder.py", "impressao_digital.py", "rotulos.py",
                  "medir_aggseq.py", "medir_decoder.py", "features_sonda.py")
ARQUIVOS_EXTRA = (
    ("../configs/config.yaml", "config.yaml"),
    # O ADAPTADOR VIAJA JUNTO em vez de ser duplicado. `medir_aggseq.py` precisa
    # de `prompt_de` e `parse_saida_contando`, e reescrevê-los aqui criaria duas
    # definições do FORMATO DO PROMPT — treino e medição passariam a usar
    # formatos diferentes na primeira correção de um só lado, que é exatamente o
    # defeito que o vocabulário de rótulo custou a este projeto em 16/09/2026.
    # Ele só importa numpy e biblioteca padrão, então embarcá-lo não arrasta o
    # resto do pacote; sem `__init__.py`, de propósito, para não puxar o módulo
    # inteiro — `src.selective` resolve como pacote de espaço de nomes.
    ("../src/selective/decoder_adapter.py", "src/selective/decoder_adapter.py"),
    # `medir_decoder.py` puxa mais dois, e os dois são numpy puro — conferido
    # antes de embarcar. `signals` importa `.geometry` por import relativo, que
    # funciona porque `src.selective` resolve como pacote de espaço de nomes.
    ("../src/selective/geometry.py", "src/selective/geometry.py"),
    ("../src/selective/signals.py", "src/selective/signals.py"),
)
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
# Adam de 8 bits: e o que faz o 1,54B caber em 14,4 GiB preservando os
# pesos-mestres em fp32. Sem ele o unico regime que cabe e bf16 puro, que
# arredonda a atualizacao para zero com lr 2e-5.
bitsandbytes==0.49.0
"""


def montar_fonte() -> bytes:
    """O tarball que a SageMaker extrai e roda. Determinístico: sem mtime."""
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


# Os DOIS pontos do eixo de escala, cada um com o prefixo do seu peso de partida
# no S3 e a instância que o regime exige. O 0,5B cabe em fp32 no T4, mas ele é
# REFEITO em bf16+8-bit no A10G para ser ponto comparável do eixo — foi decisão
# de escopo do autor, e um eixo com regimes diferentes nos dois pontos não é um
# eixo. O ajuste em fp32 já existente fica como controle de precisão de graça.
PONTOS = {
    "qwen2.5-0.5b": {"modelo": "Qwen/Qwen2.5-0.5B", "base": "base/qwen2.5-0.5b",
                     "instancia": "ml.g5.2xlarge"},
    "qwen2.5-1.5b": {"modelo": "Qwen/Qwen2.5-1.5B", "base": "base/qwen2.5-1.5b",
                     "instancia": "ml.g5.2xlarge"},
}


def submeter(nome: str | None = None, corpora: str = "genia,conll2003",
             programa: str = "treinar_decoder.py", modelo_s3: str | None = None,
             limite: int = 0, seco: bool = False, ponto: str = "qwen2.5-0.5b",
             precisao: str = "fp32", otimizador: str = "adamw_torch") -> dict:
    if ponto not in PONTOS:
        raise SystemExit(f"ponto {ponto!r} desconhecido; use um de {list(PONTOS)}")
    cfg = PONTOS[ponto]
    nome = nome or f"sentinel-decoder-{time.strftime('%Y%m%d-%H%M%S')}"
    s3 = boto3.client("s3", region_name=REGIAO)
    fonte = montar_fonte()
    chave = f"{PREFIXO}/fonte/{nome}/source.tar.gz"
    if not seco:
        s3.put_object(Bucket=BUCKET, Key=chave, Body=fonte)

    pedido = {
        "TrainingJobName": nome,
        "AlgorithmSpecification": {
            "TrainingImage": IMAGEM,
            "TrainingInputMode": "File",
        },
        "RoleArn": PAPEL,
        # Os `sagemaker_*` são consumidos pelo toolkit da imagem e NÃO chegam ao
        # script como argumento. Nenhum hiperparâmetro do estudo passa por aqui:
        # eles são o pré-registro e vivem no script.
        "HyperParameters": {
            # O PROGRAMA é parâmetro e não literal desde 22/09/2026: a mesma
            # submissão serve o treino e a passagem de feixe do AggSeq, que roda
            # na GPU por decisão do autor em 18/09. Um literal aqui obrigaria a
            # duplicar a função inteira, criando duas fontes para uma submissão.
            "sagemaker_program": json.dumps(programa),
            "sagemaker_submit_directory": json.dumps(f"s3://{BUCKET}/{chave}"),
            "sagemaker_container_log_level": json.dumps(20),
            "sagemaker_region": json.dumps(REGIAO),
        },
        # DOIS canais, e o segundo é decisão de procedência e não de conveniência.
        # O peso de partida entra pelo S3 com impressão digital conferida
        # (`7e3b2f82…`, revisão HF `060db649…`) em vez de ser baixado do hub
        # DENTRO de um job faturado. Duas coisas se ganham: o job não depende de
        # rede que eu não controlo depois de o relógio começar, e "mesmo peso de
        # partida" passa a ser verificável em vez de afirmado pelo nome.
        "InputDataConfig": [
            {
                "ChannelName": "dados",
                "DataSource": {"S3DataSource": {
                    "S3DataType": "S3Prefix",
                    "S3Uri": f"s3://{BUCKET}/{PREFIXO}/dados/",
                    "S3DataDistributionType": "FullyReplicated",
                }},
                "InputMode": "File",
            },
            {
                "ChannelName": "base",
                "DataSource": {"S3DataSource": {
                    "S3DataType": "S3Prefix",
                    "S3Uri": f"s3://{BUCKET}/{PREFIXO}/{cfg['base']}/",
                    "S3DataDistributionType": "FullyReplicated",
                }},
                "InputMode": "File",
            },
            # O TERCEIRO canal só existe quando o job NÃO é de treino. O AggSeq
            # mede um peso JÁ AJUSTADO, e ele entra pelo S3 pela mesma razão que
            # o peso de partida: o job não depende de rede depois de o relógio
            # começar, e qual peso foi medido fica verificável em vez de
            # afirmado. `modelo_s3` é o prefixo do tarball de saída do job de
            # treino que produziu aquele peso.
            *([{
                "ChannelName": "ajustado",
                "DataSource": {"S3DataSource": {
                    "S3DataType": "S3Prefix",
                    "S3Uri": modelo_s3,
                    "S3DataDistributionType": "FullyReplicated",
                }},
                "InputMode": "File",
            }] if modelo_s3 else []),
        ],
        "OutputDataConfig": {"S3OutputPath": f"s3://{BUCKET}/{PREFIXO}/saida/"},
        "ResourceConfig": {
            "InstanceType": cfg["instancia"],
            "InstanceCount": 1,
            **({"VolumeSizeInGB": VOLUME_GB} if VOLUME_GB else {}),
        },
        "StoppingCondition": {"MaxRuntimeInSeconds": TETO_S},
        "Environment": {
            # `/opt/ml` é onde o volume de EBS é montado; `/tmp` é volume raiz e
            # não caberiam 36 GB de checkpoint.
            "SENTINEL_DADOS": "/opt/ml/input/data/dados",
            "SENTINEL_MODELO_BASE_CAMINHO": "/opt/ml/input/data/base",
            "SENTINEL_MODELOS": "/opt/ml/model",
            "SENTINEL_CKPT": "/opt/ml/scratch",
            "SENTINEL_DISPOSITIVO": "cuda",
            "SENTINEL_CORPORA": corpora,
            "SENTINEL_MODELO": cfg["modelo"],
            # O medidor de feixe lê daqui. Ausente nos jobs de treino, que não
            # têm o canal — e o medir_aggseq.py exige o caminho por argumento,
            # então um job mal configurado FALHA em vez de medir outra coisa.
            **({"SENTINEL_AJUSTADO": "/opt/ml/input/data/ajustado"} if modelo_s3 else {}),
            "SENTINEL_SAIDA": "/opt/ml/model",
            # AS FEATURES DA SONDA ligadas em todo job de medição. Sem isto os
            # quatro jobs de 23/09/2026 mediram todos os escalares e NÃO
            # gravaram `features_sonda.npz` — e a sonda é sinal DECLARADO em
            # decl-10 a decl-13, então a análise assinada não rodaria. O custo
            # medido da extração é essencialmente zero (69,0 s contra 69,8 s em
            # 20 sentenças), e os escalares saem bit a bit iguais com e sem ela.
            **({"SENTINEL_FEATURES": "1"} if programa == "medir_decoder.py" else {}),
            # Recomendado pela PRÓPRIA mensagem do OutOfMemory do job que falhou:
            # 1,21 GiB estavam "reserved but unallocated", isto é, fragmentação.
            # Segmentos expansíveis deixam o alocador reaproveitá-los em vez de
            # pedir um bloco novo. Não altera nenhum número do treino.
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "SENTINEL_PRECISAO": precisao,
            "SENTINEL_OTIMIZADOR": otimizador,
            "HF_HOME": "/opt/ml/scratch/hf",
            "TOKENIZERS_PARALLELISM": "false",
        },
        "Tags": [
            {"Key": "projeto", "Value": "sentinel"},
            {"Key": "braco", "Value": "decoder"},
            {"Key": "ponto", "Value": ponto},
            {"Key": "precisao", "Value": precisao},
        ],
    }
    if limite:
        pedido["Environment"]["SENTINEL_LIMITE"] = str(limite)
        pedido["Tags"].append({"Key": "fumaca", "Value": "sim"})
    if seco:
        return {"seco": True, "pedido": pedido, "bytes_fonte": len(fonte)}
    sm = boto3.client("sagemaker", region_name=REGIAO)
    resp = sm.create_training_job(**pedido)
    return {"nome": nome, "arn": resp["TrainingJobArn"], "chave_fonte": chave}


def situacao(nome: str) -> dict:
    """Estado do job, sem manter nada no ar para perguntar."""
    sm = boto3.client("sagemaker", region_name=REGIAO)
    d = sm.describe_training_job(TrainingJobName=nome)
    fora = {k: d.get(k) for k in (
        "TrainingJobStatus", "SecondaryStatus", "FailureReason",
        "TrainingStartTime", "TrainingEndTime", "BillableTimeInSeconds")}
    fora["ModelArtifacts"] = (d.get("ModelArtifacts") or {}).get("S3ModelArtifacts")
    return fora


if __name__ == "__main__":
    import sys
    if "--seco" in sys.argv:
        r = submeter(seco=True)
        print(json.dumps(r["pedido"], indent=1, ensure_ascii=False))
        print(f"\nsource.tar.gz: {r['bytes_fonte']} bytes", file=sys.stderr)
    elif len(sys.argv) > 2 and sys.argv[1] == "situacao":
        print(json.dumps(situacao(sys.argv[2]), indent=1, default=str))
    else:
        print(json.dumps(submeter(), indent=1))
