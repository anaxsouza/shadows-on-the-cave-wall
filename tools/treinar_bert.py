"""Treino do classificador por token da decl-22 (bert-base-cased + softmax).

A receita é a da decl-22 e não tem parâmetro livre: lr 5e-5, lote 32, 3 épocas,
comprimento máximo 256, decaimento linear sem aquecimento, weight_decay 0,01,
AdamW, seed 42, checkpoint da ÚLTIMA época (sem seleção), rótulo no primeiro
sub-token e -100 nos demais. Laço de treino próprio, em PyTorch puro, para que
nenhum padrão de uma versão do Trainer entre sem ser declarado.

Variáveis de ambiente (as mesmas convenções dos outros treinos do projeto):
  SENTINEL_DADOS       diretório com {corpus}_{train,validation,test}.jsonl (GENIA, CoNLL)
                       ou train/valid/test.json + label.json (BC5CDR, em <dir>/bc5cdr/)
  SENTINEL_BERT_BASE   diretório do bert-base-cased
  SENTINEL_SAIDA       onde gravar bert-base-cased-ft-<corpus>/
  SENTINEL_CORPORA     lista separada por vírgula (padrão: genia)
  SENTINEL_LIMITE      n sentenças de treino/val/teste (SÓ teste de fumaça)
  SENTINEL_DISPOSITIVO cuda | cpu (padrão: cuda se disponível)
"""
from __future__ import annotations

import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
sys.path.insert(0, str(AQUI.parent))
try:
    from src.selective.bert_adapter import (  # noqa: E402
        MAX_LEN, decodifica_conlleval, f1_estrito, ler_corpus, rotulos_bio)
except ImportError:                                  # contêiner: arquivo solto na raiz
    from bert_adapter import (  # noqa: E402
        MAX_LEN, decodifica_conlleval, f1_estrito, ler_corpus, rotulos_bio)
from impressao_digital import impressao_digital  # noqa: E402

RECEITA = {
    "modelo_de_partida": "google-bert/bert-base-cased",
    "lr": 5e-5, "lote": 32, "epocas": 3, "max_len": MAX_LEN,
    "decaimento": "linear até 0, sem aquecimento", "weight_decay": 0.01,
    "otimizador": "AdamW", "seed": 42, "checkpoint": "última época, sem seleção",
    "precisao": "fp32",
}


def prever_lote(modelo, tokenizador, lote_tokens: Sequence[Sequence[str]], dispositivo: str,
                id2label: dict[int, str]) -> list[dict]:
    """Previsão por palavra (rótulo do primeiro sub-token). Sem atenção.

    Devolve, por sentença, `tags` (uma por palavra; 'O' para palavra sem
    sub-token dentro do limite) e `pmax` (prob. máxima do softmax, ou None).
    """
    import torch

    enc = tokenizador(list(map(list, lote_tokens)), is_split_into_words=True, truncation=True,
                      max_length=MAX_LEN, padding=True, return_tensors="pt")
    with torch.no_grad():
        logits = modelo(**{k: v.to(dispositivo) for k, v in enc.items()}).logits
        probs = torch.softmax(logits.float(), dim=-1).cpu().numpy()
    saida = []
    for b, toks in enumerate(lote_tokens):
        tags, pmax = ["O"] * len(toks), [None] * len(toks)
        visto = set()
        for pos, w in enumerate(enc.word_ids(b)):
            if w is None or w in visto:
                continue
            visto.add(w)
            k = int(probs[b, pos].argmax())
            tags[w], pmax[w] = id2label[k], float(probs[b, pos, k])
        saida.append({"tags": tags, "pmax": pmax})
    return saida


def dados_do_corpus(corpus: str) -> str:
    raiz = os.environ.get("SENTINEL_DADOS", "")
    return str(Path(raiz) / "bc5cdr") if corpus == "bc5cdr" else raiz


def codificar(tok, tokens, tags, rotulo2id):
    enc = tok(list(tokens), is_split_into_words=True, truncation=True, max_length=MAX_LEN)
    ids, visto = [], set()
    for w in enc.word_ids():
        if w is None or w in visto:
            ids.append(-100)
        else:
            visto.add(w)
            ids.append(rotulo2id[tags[w]])
    return enc["input_ids"], enc["attention_mask"], ids


def lote_tensores(itens, pad_id):
    import torch
    n = max(len(i[0]) for i in itens)
    ids = torch.full((len(itens), n), pad_id, dtype=torch.long)
    mask = torch.zeros((len(itens), n), dtype=torch.long)
    lab = torch.full((len(itens), n), -100, dtype=torch.long)
    for k, (a, m, l) in enumerate(itens):
        ids[k, :len(a)], mask[k, :len(m)], lab[k, :len(l)] = (
            torch.tensor(a), torch.tensor(m), torch.tensor(l))
    return ids, mask, lab


def avaliar(modelo, tok, exemplos, dispositivo, id2label, bs=64):
    modelo.eval()
    prev, ouro = [], []
    ordem = sorted(range(len(exemplos)), key=lambda i: len(exemplos[i]["tokens"]))
    por = {}
    for ini in range(0, len(ordem), bs):
        idx = ordem[ini:ini + bs]
        out = prever_lote(modelo, tok, [exemplos[i]["tokens"] for i in idx], dispositivo, id2label)
        for i, o in zip(idx, out):
            por[i] = o
    for i, ex in enumerate(exemplos):
        prev.append(decodifica_conlleval(por[i]["tags"]))
        ouro.append(ex["ouro"])
    return f1_estrito(prev, ouro)


def treinar(corpus: str, saida: Path, limite: int = 0) -> dict:
    import torch
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    base = os.environ["SENTINEL_BERT_BASE"]
    disp = os.environ.get("SENTINEL_DISPOSITIVO") or ("cuda" if torch.cuda.is_available() else "cpu")
    seed = RECEITA["seed"]
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    d = dados_do_corpus(corpus)
    treino = ler_corpus(corpus, "train", d)
    val = ler_corpus(corpus, "validation", d)
    teste = ler_corpus(corpus, "test", d)
    if limite:
        treino, val, teste = treino[:limite], val[:limite], teste[:limite]
    rotulos = rotulos_bio(corpus)
    r2i = {r: i for i, r in enumerate(rotulos)}
    i2r = dict(enumerate(rotulos))
    print(f"{corpus}: treino {len(treino)} | val {len(val)} | teste {len(teste)} | "
          f"{len(rotulos)} rótulos | dispositivo {disp}", flush=True)

    tok = AutoTokenizer.from_pretrained(base)
    modelo = AutoModelForTokenClassification.from_pretrained(
        base, num_labels=len(rotulos), id2label=i2r, label2id=r2i).to(disp)
    cod = [codificar(tok, e["tokens"], e["tags_planas"], r2i) for e in treino]
    n_trunc = sum(1 for e, c in zip(treino, cod)
                  if sum(1 for x in c[2] if x != -100) < len(e["tokens"]))

    sem_decaimento = ("bias", "LayerNorm.weight")
    params = [
        {"params": [p for n, p in modelo.named_parameters()
                    if not any(s in n for s in sem_decaimento)],
         "weight_decay": RECEITA["weight_decay"]},
        {"params": [p for n, p in modelo.named_parameters()
                    if any(s in n for s in sem_decaimento)], "weight_decay": 0.0},
    ]
    opt = torch.optim.AdamW(params, lr=RECEITA["lr"])
    bs, ep = RECEITA["lote"], RECEITA["epocas"]
    passos_ep = (len(cod) + bs - 1) // bs
    total = passos_ep * ep
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: max(0.0, 1.0 - s / total))
    g = torch.Generator().manual_seed(seed)

    historico, passo, t0 = [], 0, time.time()
    for e in range(ep):
        modelo.train()
        perm = torch.randperm(len(cod), generator=g).tolist()
        soma = 0.0
        for ini in range(0, len(perm), bs):
            ids, mask, lab = lote_tensores([cod[i] for i in perm[ini:ini + bs]], tok.pad_token_id)
            out = modelo(input_ids=ids.to(disp), attention_mask=mask.to(disp), labels=lab.to(disp))
            out.loss.backward()
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            soma += float(out.loss); passo += 1
            if passo % 100 == 0:
                print(f"  época {e + 1} passo {passo}/{total} perda {float(out.loss):.4f} "
                      f"{time.time() - t0:.0f}s", flush=True)
        historico.append({"epoca": e + 1, "perda_media": soma / passos_ep})
        print(f"época {e + 1}: perda média {soma / passos_ep:.4f}", flush=True)

    # CHECKPOINT DA ÚLTIMA ÉPOCA: salvo sem avaliar antes, para a avaliação não poder escolher.
    nome = f"bert-base-cased-ft-{corpus}"
    d_saida = saida / nome
    d_saida.mkdir(parents=True, exist_ok=True)
    modelo.save_pretrained(d_saida, safe_serialization=True)
    tok.save_pretrained(d_saida)
    f_val = avaliar(modelo, tok, val, disp, i2r)
    f_teste = avaliar(modelo, tok, teste, disp, i2r)
    res = {
        "corpus": corpus, "modelo": nome, "receita": RECEITA, "limite_fumaca": limite or None,
        "n_treino": len(treino), "n_sentencas_truncadas_no_treino": n_trunc,
        "passos": passo, "segundos_de_treino": round(time.time() - t0, 1),
        "historico": historico, "validacao_estrito": f_val, "teste_estrito": f_teste,
        "impressao_digital": impressao_digital(d_saida),
        "torch": torch.__version__,
    }
    import transformers
    res["transformers"] = transformers.__version__
    (d_saida / "TREINO.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"{corpus}: F1 estrito val {f_val['f1']:.4f} | teste {f_teste['f1']:.4f}", flush=True)
    return res


if __name__ == "__main__":
    saida = Path(os.environ.get("SENTINEL_SAIDA", "saida/bert_pesos"))
    limite = int(os.environ.get("SENTINEL_LIMITE", "0") or 0)
    for c in os.environ.get("SENTINEL_CORPORA", "genia").split(","):
        treinar(c.strip(), saida, limite)
