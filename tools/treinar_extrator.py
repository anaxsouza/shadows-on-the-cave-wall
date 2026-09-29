"""Ajuste fino do extrator, para fechar a objeção do Perímetro 2.

POR QUE ESTE SCRIPT EXISTE

O resultado confirmatório é negativo em duas escalas, e a única objeção que
continua de pé não é sobre atenção: é sobre o extrator. Os dois modelos medidos
são de prateleira, nenhum treinado nos corpora, e erram 42% e 47%. Um revisor
pode alegar que a faixa dinâmica dos sinais estava comprimida e que o nulo é do
arranjo. Um extrator ajustado ao corpus é a versão forte do teste.

AS TRÊS ESCOLHAS QUE PODERIAM SER GRAU DE LIBERDADE, E COMO FICAM FECHADAS

1. Hiperparâmetros: a receita padrão do gliner (encoder 5e-6, resto 5e-5, 3
   épocas, lote 8, aquecimento 10%, semente 42). NÃO varridos. Varrer e reportar
   o melhor daria N chances ao acaso.
2. Regra de escolha de checkpoint: melhor F1 no split de VALIDAÇÃO, declarada
   aqui e não escolhida depois de ver as curvas. A seleção nunca olha ΔAURC nem
   qualquer quantidade da hipótese — só a qualidade do extrator, que é o arranjo
   e não o que está sob teste.
3. Regra de acerto: casamento ESTRITO (mesma fronteira e mesmo rótulo), a mesma
   do `loss` da tabela por entidade. Usar outra regra aqui faria o F1 do treino
   incomparável com a taxa de erro base da medição.

O split de TESTE não é tocado por nada aqui.
"""

from __future__ import annotations

import json

import numpy as np
import torch
from gliner import GLiNER
from gliner.data_processing.collator import UniEncoderSpanDataCollator
from gliner.training import Trainer, TrainingArguments
from pathlib import Path

SEED, EPOCAS, LOTE = 42, 3, 8
N_VAL = 400  # sentenças de validação usadas na avaliação: F1 em 400 já separa épocas


def carregar(caminho: str) -> list[dict]:
    return [json.loads(l) for l in open(caminho, encoding="utf-8")]


def _offsets(tokens: list[str], texto: str) -> list[tuple[int, int]]:
    pos, out = 0, []
    for t in tokens:
        i = texto.find(t, pos)
        out.append((i, i + len(t)))
        pos = i + len(t)
    return out


def f1_estrito(modelo, dados: list[dict], rotulos: list[str], limiar: float = 0.5):
    tp = fp = fn = 0
    for ex in dados:
        ouro = {(a, b, l) for a, b, l in ex["ner"]}
        texto = " ".join(ex["tokenized_text"])
        offs = _offsets(ex["tokenized_text"], texto)
        obtido = set()
        for e in modelo.predict_entities(texto, rotulos, threshold=limiar):
            i0 = next((i for i, (a, b) in enumerate(offs) if a <= e["start"] < b), None)
            i1 = next((i for i, (a, b) in enumerate(offs) if a < e["end"] <= b), None)
            if i0 is not None and i1 is not None:
                obtido.add((i0, i1, e["label"]))
        tp += len(ouro & obtido)
        fp += len(obtido - ouro)
        fn += len(ouro - obtido)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return (2 * p * r / (p + r) if p + r else 0.0), p, r


def main() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    for corpus in ("genia", "conll2003"):
        treino = carregar(f"{corpus}_train.jsonl")
        val = carregar(f"{corpus}_validation.jsonl")
        rotulos = sorted({l for ex in treino for _, _, l in ex["ner"]})
        print(f"\n===== {corpus}: {len(treino)} treino, {len(val)} validacao, "
              f"rotulos {rotulos}", flush=True)

        m = GLiNER.from_pretrained("urchade/gliner_base")
        f0, p0, r0 = f1_estrito(m.eval().to("cuda"), val[:N_VAL], rotulos)
        print(f"ANTES do ajuste  F1 {f0:.4f}  P {p0:.4f}  R {r0:.4f}", flush=True)

        m = m.train()
        col = UniEncoderSpanDataCollator(
            m.config, data_processor=m.data_processor, prepare_labels=True)
        args = TrainingArguments(
            output_dir=f"ckpt/{corpus}", learning_rate=5e-6, others_lr=5e-5,
            weight_decay=0.01, others_weight_decay=0.01, lr_scheduler_type="linear",
            warmup_ratio=0.1, per_device_train_batch_size=LOTE,
            num_train_epochs=EPOCAS, eval_strategy="no", save_strategy="epoch",
            save_total_limit=EPOCAS, dataloader_num_workers=0, use_cpu=False,
            report_to="none", seed=SEED, logging_steps=200, fp16=True,
        )
        # `processing_class` e nao `tokenizer`: o transformers novo renomeou o
        # argumento, e a assinatura foi lida do pacote instalado em vez de
        # suposta pela documentacao.
        Trainer(model=m, args=args, train_dataset=treino, data_collator=col,
                processing_class=m.data_processor.transformer_tokenizer).train()

        melhor, escolhido = -1.0, None
        pontos = []
        for ep in sorted(Path(f"ckpt/{corpus}").glob("checkpoint-*"),
                         key=lambda p: int(p.name.split("-")[1])):
            mm = GLiNER.from_pretrained(str(ep), local_files_only=True).eval().to("cuda")
            f, p_, r_ = f1_estrito(mm, val[:N_VAL], rotulos)
            pontos.append({"checkpoint": ep.name, "f1": round(f, 4),
                           "p": round(p_, 4), "r": round(r_, 4)})
            print(f"  {ep.name}: F1 {f:.4f}  P {p_:.4f}  R {r_:.4f}", flush=True)
            if f > melhor:
                melhor, escolhido = f, ep
            del mm
            torch.cuda.empty_cache()

        print(f"ESCOLHIDO {escolhido.name} por F1 de validacao {melhor:.4f} "
              f"(antes do ajuste {f0:.4f}, ganho {melhor - f0:+.4f})", flush=True)
        GLiNER.from_pretrained(str(escolhido), local_files_only=True).save_pretrained(
            f"ajustado-{corpus}")
        json.dump({"corpus": corpus, "checkpoint": escolhido.name,
                   "f1_val": round(melhor, 4), "f1_val_antes": round(f0, 4),
                   "p_antes": round(p0, 4), "r_antes": round(r0, 4),
                   "por_epoca": pontos, "rotulos": rotulos, "seed": SEED,
                   "epocas": EPOCAS, "lote": LOTE, "n_val_avaliadas": N_VAL,
                   "regra_checkpoint": "melhor F1 estrito na validacao; nunca olha ΔAURC",
                   "hiperparametros": "receita padrao do gliner, nao varrida"},
                  open(f"ajuste_{corpus}.json", "w"), ensure_ascii=False, indent=1)
        del m
        torch.cuda.empty_cache()

    print("\nFIM", flush=True)


if __name__ == "__main__":
    main()
