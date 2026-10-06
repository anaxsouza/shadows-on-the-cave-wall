"""Ajuste fino de um decoder autorregressivo para NER generativo.

POR QUE ESTE SCRIPT EXISTE, e por que o decoder importa mais que o encoder

O confundidor geométrico está estabelecido no encoder: a massa de atenção sobre
um trecho tem esperança exata `k/|K|` sob permutabilidade das chaves, e nenhuma
variante da fórmula escapa, porque todas são agregados do mesmo orçamento fixo.

No decoder a álgebra não desaparece — ela PIORA, e isso é derivável. A máscara
causal faz o número de chaves permitidas depender da POSIÇÃO da consulta: a
linha `i` só pode olhar `i+1` chaves. Então a esperança da massa deixa de
depender de dois números geométricos (tamanho do trecho e da sentença) e passa
a depender de três, com a posição entrando como dimensão própria. Um trecho de
1 token no começo da sentença e um de 4 tokens no fim têm esperanças muito
diferentes, e num encoder bidirecional teriam razão exatamente 4:1.

Isso é o que torna o braço decoder mais importante, e não apenas mais um
arranjo: decoders autorregressivos são a base dos sistemas agênticos atuais,
inclusive para extração de entidades. Um confundidor que já engana num encoder
engana de uma maneira a MAIS num decoder.

AS TRÊS ESCOLHAS DE MÉTODO, declaradas aqui e não escolhidas depois

1. EXTRAÇÃO GENERATIVA, e não uma cabeça de pontuação de trechos colada num
   encoder causal. É como decoders são de fato usados — o modelo lê o texto e
   ESCREVE as entidades. Colar uma cabeça de trechos daria um comparativo mais
   limpo, mas mediria um arranjo que ninguém usa, e a pergunta do capítulo é
   sobre o confundidor no arranjo real.

2. CONFIANÇA = média dos log-probs dos tokens da menção gerada, exponenciada.
   É a quantidade que o próprio modelo reporta sobre aquela saída, e é o
   análogo direto do escore de trecho do GLiNER: o número contra o qual todo
   sinal interno tem de se provar. Não é escolha livre — qualquer outra
   definição não seria "a confiança do próprio modelo".

3. REGRA DE CHECKPOINT: melhor F1 ESTRITO na validação, com o mesmo casamento
   (mesma fronteira, mesmo rótulo) usado pela tabela por entidade. Declarada
   aqui, antes do treino, e NUNCA olha nenhuma quantidade da hipótese. Escolher
   o checkpoint depois de ver ΔAURC seria o atirador pintando o alvo com outro
   nome.

HIPERPARÂMETROS: a receita padrão, não varrida. Varrer abriria graus de
liberdade que o pré-registro existe para fechar, e o objetivo aqui não é o
melhor extrator possível — é um extrator competente o bastante para que o nulo
não seja atribuível ao arranjo.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from pathlib import Path

import torch
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from impressao_digital import impressao_digital  # noqa: E402
from rotulos import descricoes  # noqa: E402
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

MODELO = "Qwen/Qwen2.5-0.5B"
SEED = 42
EPOCAS = 3

# O LOTE EFETIVO É 16 E É INVARIANTE — é ele que entra no gradiente, e mudá-lo
# entre os pontos do eixo tornaria os dois pesos incomparáveis. O que varia é
# como esses 16 são fatiados na GPU, e isso NÃO é hiperparâmetro: mesmo
# gradiente, mesma ordem de exemplos, mesma semente.
#
# POR QUE A FATIA MUDOU. O 1,5B com `LOTE=4` estourou a A10G: 21,46 GiB em uso
# de 22,06 disponíveis, falhando ao alocar 742 MiB no passo 31 de 2.535. Eu
# havia projetado 14,4 GiB para este regime e errei — a projeção omitia a
# matriz de logits (`lote × seqlen × 151.936` promovida a fp32, mais o gradiente
# dela) e, mesmo com esse termo, ainda subestima em ~2,8 GiB, que são
# fragmentação (1,21 GiB pelo próprio relatório do erro), contexto de CUDA e
# espaços de trabalho do cuBLAS. Fórmula fechada não prevê isso; medir prevê.
LOTE_EFETIVO = 16
FATIA = {"qwen2.5-0.5b": 4, "qwen2.5-1.5b": 1}
# CORREÇÃO DE 01/10/2026 (BC5CDR), sem efeito sobre o que já foi treinado: `submeter_sagemaker.py` passa
# em SENTINEL_MODELO o NOME DO HUB ("Qwen/Qwen2.5-1.5B"), enquanto `FATIA` e `curto` (mais abaixo)
# são indexados pela CHAVE CURTA ("qwen2.5-1.5b"). Sem a tradução, o 1,5B caía no padrão 4 (a fatia
# que estourou a A10G) e o nome do peso não era resolvido. A tradução aceita as duas grafias.
CHAVE_DO_MODELO = {"Qwen/Qwen2.5-0.5B": "qwen2.5-0.5b", "Qwen/Qwen2.5-1.5B": "qwen2.5-1.5b"}
_MODELO_ENV = os.environ.get("SENTINEL_MODELO", "qwen2.5-0.5b")
CHAVE = CHAVE_DO_MODELO.get(_MODELO_ENV, _MODELO_ENV)
LOTE = FATIA.get(CHAVE, 4)
ACUM = LOTE_EFETIVO // LOTE
assert LOTE * ACUM == LOTE_EFETIVO, (LOTE, ACUM)
MAX_LEN = 320
N_VAL = 300          # sentenças de validação avaliadas por checkpoint

# CAMINHOS E DISPOSITIVO VÊM DO AMBIENTE, e os padrões são os da máquina
# efêmera original — quem rodava antes continua rodando igual. A razão de
# parametrizar em vez de forkar: um segundo script divergiria deste no primeiro
# ajuste, e a divergência apareceria como número diferente, não como erro.
#
# O QUE O AMBIENTE MOVE: onde o dado entra, onde o peso sai, em que dispositivo
# roda, e quais corpora. Nada disso é do pré-registro.
DADOS = Path(os.environ.get("SENTINEL_DADOS", "dados_decoder"))
MODELOS = Path(os.environ.get("SENTINEL_MODELOS", "modelos"))
CKPT = Path(os.environ.get("SENTINEL_CKPT", "."))
DISPOSITIVO = os.environ.get("SENTINEL_DISPOSITIVO", "cuda")
CORPORA = tuple(os.environ.get("SENTINEL_CORPORA", "genia,conll2003").split(","))

# A IDENTIDADE do peso de partida passa a vir do ambiente em 17/09/2026, porque
# o braço ganhou um segundo tamanho por decisão de escopo do autor. Não é
# afrouxamento: o valor efetivo vai para o JSON de procedência em `modelo_base`,
# e o padrão continua o do primeiro ajuste, então uma corrida que não declare
# nada roda o que rodava. O que seria afrouxamento é o caminho dos bytes mudar
# sem a identidade mudar — e isso o `modelo_base_sha256_v1` pega.
MODELO = os.environ.get("SENTINEL_MODELO", MODELO)

# PRECISÃO E OTIMIZADOR — a escolha é de ENGENHARIA e foi decidida por
# aritmética de memória, não por preferência. Registrada aqui porque entra na
# procedência e porque a razão não é óbvia:
#
#   fp32 + AdamW fp32 ........ 16 B/parâmetro = 23,0 GiB no 1,54B. NÃO CABE em
#                              nenhuma instância que a política permite.
#   bf16=True do HF .......... precisão MISTA com pesos-mestres em fp32. Não
#                              economiza estado de otimizador: também 23,0 GiB.
#   bf16 puro ................ 8 B = 11,5 GiB, cabe — e está ERRADO aqui: com
#                              lr 2e-5 a atualização relativa é ~1e-7 e a
#                              resolução do bf16 é 3,9e-3, quatro ordens de
#                              grandeza mais grossa. A atualização é arredondada
#                              para zero e o treino não anda.
#   bf16 + Adam 8-bit ........ 10 B = 14,4 GiB, CABE no A10G, preserva os
#                              mestres em fp32 e mantém o AdamW como algoritmo.
#                              Só o armazenamento do momento é comprimido.
#
# Por isso os pesos continuam carregando em fp32: quem faz a conta em bf16 é o
# autocast, e os mestres têm de ser fp32 para a atualização existir.
PRECISAO = os.environ.get("SENTINEL_PRECISAO", "fp32")
OTIMIZADOR = os.environ.get("SENTINEL_OTIMIZADOR", "adamw_torch")
if PRECISAO not in ("fp32", "bf16"):
    raise SystemExit(f"SENTINEL_PRECISAO tem de ser fp32 ou bf16, não {PRECISAO!r}")

# A IDENTIDADE do peso de partida é `MODELO`, declarada acima e não movível. O
# que o ambiente move é DE ONDE aqueles bytes são lidos — um diretório local em
# vez do hub. Caminho não é hipótese, e a distinção não é retórica: sem ela,
# `"modelo_base": "Qwen/Qwen2.5-0.5B"` identifica o peso de partida pelo NOME,
# que é exatamente o defeito que o `checkpoint_sha256` corrigiu para o peso de
# chegada. Se o repositório for republicado, a procedência afirmaria mesmidade
# onde não há. A igualdade passa a ser PROVADA: quando o caminho é um
# diretório, a impressão digital dos pesos de partida entra na procedência.
MODELO_CAMINHO = os.environ.get("SENTINEL_MODELO_BASE_CAMINHO", MODELO)

# O QUE O AMBIENTE NÃO MOVE: a semente, os hiperparâmetros de treino e o tamanho
# da validação que seleciona o checkpoint. São o pré-registro, e um override
# deles não é configuração — é grau de liberdade sobre a hipótese.
#
# `SENTINEL_LIMITE` existe para UMA finalidade: o teste de fumaça, que prova que
# o encanamento roda antes de a GPU ser ligada. Ele é o ÚNICO portão, e só com
# ele ligado época e N_VAL podem encolher — porque um teste de fumaça com 3
# épocas sobre 12 exemplos gastaria o tempo que ele existe para poupar. Sem
# LIMITE, os dois valores são as constantes acima e não há caminho de código
# que os altere. `FUMACA` vai para o JSON de procedência, então uma corrida de
# fumaça é impossível de confundir com uma corrida de resultado.
LIMITE = int(os.environ.get("SENTINEL_LIMITE", "0"))     # 0 = corpus inteiro
FUMACA = bool(LIMITE)
if FUMACA:
    EPOCAS = int(os.environ.get("SENTINEL_EPOCAS", "1"))
    N_VAL = int(os.environ.get("SENTINEL_N_VAL", "4"))

# O vocabulário de rótulos mora em tools/rotulos.py e em nenhum outro lugar. A
# cópia que ficava aqui era a MESMA tabela, e ter duas definições da mesma coisa
# em arquivos diferentes é como a divergência entre treino e medição passou
# despercebida: cada lado parecia ter o seu mapa correto.
SEP = " ## "


def prompt(texto: str, rotulos: list[str]) -> str:
    return (f"Text: {texto}\n"
            f"Entity types: {', '.join(rotulos)}\n"
            f"Entities:\n")


def alvo(entidades: list[tuple[str, str]]) -> str:
    if not entidades:
        return "none"
    return "\n".join(f"{t}{SEP}{r}" for t, r in entidades)


def carregar(corpus: str, split: str) -> list[dict]:
    """Converte o formato de trechos por índice de palavra em prompt e alvo."""
    fora = []
    with (DADOS / f"{corpus}_{split}.jsonl").open(encoding="utf-8") as fh:
        for linha in fh:
            d = json.loads(linha)
            toks = d["tokenized_text"]
            texto = " ".join(toks)
            ents = []
            for ini, fim, rot in d["ner"]:
                ents.append((" ".join(toks[int(ini):int(fim) + 1]), str(rot)))
            fora.append({"texto": texto, "entidades": ents,
                         "rotulos": sorted({r for _, r in ents}) or []})
    return fora


class DadosNER(Dataset):
    """Perda calculada SÓ no alvo: o prompt é entrada, não é o que se aprende."""

    def __init__(self, exemplos, tok, rotulos_corpus):
        self.ex, self.tok = exemplos, tok
        self.rot = sorted(rotulos_corpus)

    def __len__(self):
        return len(self.ex)

    def __getitem__(self, i):
        e = self.ex[i]
        p = prompt(e["texto"], self.rot)
        a = alvo(e["entidades"]) + self.tok.eos_token
        ids_p = self.tok(p, add_special_tokens=False)["input_ids"]
        ids_a = self.tok(a, add_special_tokens=False)["input_ids"]
        ids = (ids_p + ids_a)[:MAX_LEN]
        rot = ([-100] * len(ids_p) + ids_a)[:MAX_LEN]
        return {"input_ids": ids, "labels": rot}


class Colador:
    """O colador como objeto de módulo, e não `lambda` dentro de `main`.

    Com `dataloader_num_workers=2` o colador é enviado aos processos filhos.
    Onde o método de partida é `fork` (Linux, e portanto o contêiner da
    SageMaker) uma `lambda` local passa; onde é `spawn` (macOS) ela tem de ser
    serializada e não pode. O teste de fumaça local expôs isso. Não é mudança
    de comportamento: é a mesma função, alcançável por nome.
    """

    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, lote):
        return colar(lote, self.pad_id)


def colar(lote, pad_id):
    n = max(len(x["input_ids"]) for x in lote)
    return {
        "input_ids": torch.tensor([x["input_ids"] + [pad_id] * (n - len(x["input_ids"]))
                                   for x in lote]),
        "attention_mask": torch.tensor([[1] * len(x["input_ids"]) + [0] * (n - len(x["input_ids"]))
                                        for x in lote]),
        "labels": torch.tensor([x["labels"] + [-100] * (n - len(x["labels"])) for x in lote]),
    }


def parse_saida(texto: str) -> set[tuple[str, str]]:
    """As menções geradas, como conjunto. Linha sem separador é descartada."""
    fora = set()
    for linha in texto.strip().splitlines():
        if SEP not in linha:
            continue
        m, _, r = linha.partition(SEP)
        m, r = m.strip(), r.strip()
        if m and r and m.lower() != "none":
            fora.add((m, r))
    return fora


@torch.no_grad()
def f1_estrito(modelo, tok, exemplos, rotulos_corpus, n=N_VAL):
    """F1 sobre pares (menção, rótulo), casamento ESTRITO.

    Não é o F1 por posição de caractere da tabela por entidade — aqui a saída é
    texto gerado, e a menção é casada por string. A diferença fica declarada: é
    regra de SELEÇÃO de checkpoint, não número de resultado.
    """
    rot = sorted(rotulos_corpus)
    tp = fp = fn = 0
    modelo.eval()
    for e in exemplos[:n]:
        p = prompt(e["texto"], rot)
        ids = tok(p, return_tensors="pt", add_special_tokens=False).to(modelo.device)
        saida = modelo.generate(**ids, max_new_tokens=96, do_sample=False,
                                pad_token_id=tok.pad_token_id)
        gerado = tok.decode(saida[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
        pred = parse_saida(gerado)
        ouro = {(m.strip(), r.strip()) for m, r in e["entidades"]}
        tp += len(pred & ouro)
        fp += len(pred - ouro)
        fn += len(ouro - pred)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return (2 * prec * rec / (prec + rec) if prec + rec else 0.0), prec, rec


def main():
    torch.manual_seed(SEED)
    for corpus in CORPORA:
        print(f"\n{'=' * 70}\n{corpus}\n{'=' * 70}", flush=True)
        treino = carregar(corpus, "train")
        val = carregar(corpus, "validation")
        if LIMITE:
            treino, val = treino[:LIMITE], val[:LIMITE]
            print(f"*** FUMAÇA: {LIMITE} exemplos, {EPOCAS} época(s). "
                  f"NÃO é corrida de resultado.", flush=True)
        # Os rótulos vêm do jsonl, que é a verdade do que o treino vai ver. O
        # `if False else` que ficava aqui desviava de `ROTULOS` e deixava o mapa
        # como documentação de algo que o código não fazia — e foi por isso que
        # a divergência com a medição sobreviveu até 16/09/2026.
        rotulos = sorted({r for e in treino for _, r in e["entidades"]})
        # GUARDA: o vocabulário do export tem de ser o declarado em
        # tools/rotulos.py, porque é contra ele que a MEDIÇÃO vai comparar. Se o
        # export for regerado com outra grafia, isto falha alto aqui em vez de
        # produzir um modelo que só erra na hora de medir. Sob fumaça a amostra
        # é pequena e o vocabulário é subconjunto, então só se exige contenção.
        esperados = set(descricoes(corpus))
        if FUMACA:
            assert set(rotulos) <= esperados, (
                f"rótulos fora do declarado: {sorted(set(rotulos) - esperados)}")
        else:
            assert set(rotulos) == esperados, (
                f"vocabulário do export != declarado em tools/rotulos.py. "
                f"export={rotulos} declarado={sorted(esperados)}. A medição "
                f"compara contra o declarado; treinar assim daria um modelo que "
                f"só erra na medição.")
        print(f"{len(treino)} treino, {len(val)} validação | rótulos {rotulos}", flush=True)

        tok = AutoTokenizer.from_pretrained(MODELO_CAMINHO)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        # ATENÇÃO EAGER É OBRIGATÓRIA e não preferência: com o padrão `sdpa`,
        # `output_attentions=True` devolve tupla VAZIA — zero camadas, sem erro
        # nenhum. Uma medição feita sem isto produziria nada em silêncio.
        m = AutoModelForCausalLM.from_pretrained(
            MODELO_CAMINHO, dtype=torch.float32,
            attn_implementation="eager").to(DISPOSITIVO)
        m.gradient_checkpointing_enable()
        m.config.use_cache = False

        # MEDIDO em 16/09/2026, e vale registrar em vez de descartar: o
        # Qwen2.5-0.5B BASE dá F1 0,0000 aqui. Não é defeito do prompt — é que
        # ele não é modelo de instrução, e sem ajuste não produz nada no formato
        # `menção ## rótulo`. A consequência para o relato é que o ganho do
        # ajuste fino neste braço é o valor ABSOLUTO do F1 final, e não uma
        # diferença contra uma linha de base informativa como a do encoder
        # (que já extraía sem treino, com F1 0,56 e 0,67).
        f0, p0, r0 = f1_estrito(m, tok, val, rotulos)
        print(f"ANTES do ajuste: F1 {f0:.4f} P {p0:.4f} R {r0:.4f}", flush=True)

        ds = DadosNER(treino, tok, rotulos)
        args = TrainingArguments(
            output_dir=str(CKPT / f"ckpt-{corpus}"), num_train_epochs=EPOCAS,
            per_device_train_batch_size=LOTE, gradient_accumulation_steps=ACUM,
            # `warmup_steps` e não `warmup_ratio`: esta versão do transformers
            # (5.16.1) não tem a segunda, e a assinatura foi LIDA do pacote
            # instalado em vez de suposta. 300 passos são ~3% dos ~10.100 do
            # corpus maior, que é a proporção que a receita padrão usa.
            learning_rate=2e-5, warmup_steps=300, logging_steps=200,
            save_strategy="epoch", save_total_limit=EPOCAS, seed=SEED,
            # `fp16` NUNCA: no T4 ele precisaria de escalonamento de perda, e a
            # faixa de expoente estreita é risco sem ganho quando existe bf16.
            bf16=(PRECISAO == "bf16"), fp16=False, optim=OTIMIZADOR, report_to=[],
            # 2 no treino de verdade; 0 no teste de fumaça. Não é hiperparâmetro
            # — é como o lote chega à GPU, sem efeito no resultado. A concessão
            # existe porque no macOS o método de partida é `spawn` e o
            # `DataLoader` com filhos trava aqui: a corrida foi interrompida com
            # ~22 min decorridos e nenhum passo completo, pelo registro de
            # execução. O `~35 min` que este comentário dizia antes era minha
            # estimativa de relógio, não medição, e foi corrigido em 18/09/2026.
            # No Linux do contêiner é `fork` e os 2 valem.
            dataloader_num_workers=0 if FUMACA else 2,
        )
        Trainer(model=m, args=args, train_dataset=ds,
                data_collator=Colador(tok.pad_token_id),
                processing_class=tok).train()

        # SELEÇÃO: melhor F1 de validação entre as épocas. A regra está acima,
        # declarada antes do treino, e não olha nenhuma quantidade da hipótese.
        pontos = []
        for ck in sorted((CKPT / f"ckpt-{corpus}").glob("checkpoint-*"),
                         key=lambda p: int(p.name.split("-")[1])):
            # fp32 SEMPRE aqui, mesmo quando o treino foi em precisão mista, e a
            # razão é coerência com a medição: `medir_decoder.py` carrega em
            # fp32, então selecionar o checkpoint numa precisão e medi-lo em
            # outra faria a regra de seleção olhar um modelo diferente daquele
            # que responde a hipótese. Subir de bf16 para fp32 é determinístico
            # e não perde informação.
            mm = AutoModelForCausalLM.from_pretrained(
                str(ck), dtype=torch.float32, attn_implementation="eager").to(DISPOSITIVO)
            f_, p_, r_ = f1_estrito(mm, tok, val, rotulos)
            pontos.append({"checkpoint": ck.name, "f1": f_, "p": p_, "r": r_})
            print(f"  {ck.name}: F1 {f_:.4f} P {p_:.4f} R {r_:.4f}", flush=True)
            del mm
            if DISPOSITIVO == "cuda":
                torch.cuda.empty_cache()
        melhor = max(pontos, key=lambda x: x["f1"])
        # O NOME DERIVA DO MODELO, e o literal `qwen05b` era um defeito que só
        # apareceu quando o eixo de escala existiu: os dois pesos de 1,5B saíram
        # nomeados `qwen05b-ft-*`, dando DOIS nomes para QUATRO pesos. As
        # declarações decl-12 e decl-13 declaram `qwen15b-ft-*`, que não existiria
        # — a medição falharia ao procurar o modelo ou, pior, mediria o peso
        # errado se os dois tivessem sido extraídos no mesmo diretório.
        #
        # `SENTINEL_MODELO` é a mesma variável que escolhe a fatia de lote, então
        # nome e configuração não podem divergir.
        curto = {"qwen2.5-0.5b": "qwen05b", "qwen2.5-1.5b": "qwen15b"}
        ponto = CHAVE
        if ponto not in curto:
            raise SystemExit(
                f"SENTINEL_MODELO={ponto!r} sem nome curto registrado. Acrescente-o "
                f"a `curto` em vez de deixar o nome do peso ser adivinhado: "
                f"conhecidos {sorted(curto)}")
        destino = MODELOS / f"{curto[ponto]}-ft-{corpus}"
        destino.parent.mkdir(parents=True, exist_ok=True)
        import shutil
        if destino.exists():
            shutil.rmtree(destino)
        shutil.copytree(CKPT / f"ckpt-{corpus}" / melhor["checkpoint"], destino)
        # O ESTADO DE RETOMADA SAI. `optimizer.pt` são os momentos do Adam e
        # pesam o DOBRO do modelo (3,95 GB contra 1,98 GB, medido). Guardá-lo
        # triplicaria o que a SageMaker sobe ao terminar e o que eu baixo para
        # medir, e a medição nunca o lê — nada aqui retoma treino. O que fica é
        # o modelo, o tokenizador e o `trainer_state.json`, que é a prova de
        # qual passo produziu este checkpoint.
        for lixo in ("optimizer.pt", "scheduler.pt", "rng_state.pth"):
            (destino / lixo).unlink(missing_ok=True)
        tok.save_pretrained(destino)
        print(f"ESCOLHIDO {melhor['checkpoint']} -> F1 {melhor['f1']:.4f} "
              f"(ganho {melhor['f1'] - f0:+.4f}) -> {destino}", flush=True)
        # A impressão digital é calculada AQUI, e não depois em outra máquina:
        # depois é onde ela deixou de existir na vez anterior.
        digital = impressao_digital(destino)
        print(f"IMPRESSÃO {digital['checkpoint_sha256_v1']}", flush=True)
        # Os pesos de PARTIDA, por conteúdo. `None` quando vieram do hub: aí a
        # identidade é só o nome, e dizer isso é melhor que gravar um campo
        # ausente que se leria como "conferido".
        base_digital = (impressao_digital(MODELO_CAMINHO)["checkpoint_sha256_v1"]
                        if Path(MODELO_CAMINHO).is_dir() else None)
        json.dump({"corpus": corpus, "modelo_base": MODELO,
                   "modelo_base_caminho": str(MODELO_CAMINHO),
                   "modelo_base_sha256_v1": base_digital,
                   **digital,
                   # O REGIME NUMÉRICO entra na procedência porque dois ajustes
                   # sob regimes diferentes não são pontos comparáveis de um
                   # eixo de escala, e sem este campo a incomparabilidade ficaria
                   # invisível na tabela.
                   "precisao": PRECISAO, "otimizador": OTIMIZADOR,
                   "selecao_e_medicao_em": "fp32",
                   "fumaca": FUMACA,
                   "f1_val": melhor["f1"], "p_val": melhor["p"], "r_val": melhor["r"],
                   "f1_val_antes": f0, "p_antes": p0, "r_antes": r0,
                   "checkpoint": melhor["checkpoint"], "por_epoca": pontos,
                   "rotulos": rotulos, "seed": SEED, "epocas": EPOCAS, "lote": LOTE,
                   "acumulacao": ACUM, "n_val_avaliadas": N_VAL,
                   "regra_checkpoint": "melhor F1 estrito na validacao; nunca olha nenhuma "
                                       "quantidade da hipotese",
                   "hiperparametros": "receita padrao, nao varrida",
                   "attn_implementation": "eager (obrigatorio: sdpa devolve atencao vazia)"},
                  open(MODELOS / f"ajuste_decoder_{corpus}.json", "w"),
                  ensure_ascii=False, indent=1)
        del m
        torch.cuda.empty_cache()
    print("\nFIM", flush=True)


if __name__ == "__main__":
    main()
