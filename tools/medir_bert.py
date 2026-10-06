"""Medição e análise do classificador por token (decl-22 a decl-25).

  python tools/medir_bert.py medir   <corpus> <dir_do_modelo> [--max-samples N] [--saida DIR]
  python tools/medir_bert.py analisar [corpus ...]

MEDIR produz `saida/bert/<modelo>/<corpus>/test/entities.csv` no formato das
tabelas do encoder (sentence_id, loss, model_confidence, span_mass, is_nested,
token_indices), mais `sentence_lengths.csv` (T por sentença) e `MEDIDA.json` com a
procedência que o runner confere (declaração, hash de medição, modelo, impressão
digital dos pesos). Colunas acrescentadas ao final, que nenhum teste consome:
palavra_ini, palavra_fim, rotulo, n_tokens. NENHUM texto do corpus é gravado.

Definições (decl-22):
- um trecho é uma sequência máxima B-X I-X… (conlleval); `loss` = 0 se (palavra
  inicial, palavra final, rótulo) está no ouro, 1 se não; o ouro do GENIA é o
  aninhado e a previsão é plana (critério estrito de sempre);
- `model_confidence` = média geométrica, sobre as palavras do trecho, da
  probabilidade máxima do softmax no primeiro sub-token;
- `span_mass` = fração da atenção média (todas as camadas e cabeças, ANTES da razão)
  recebida pelos sub-tokens do trecho, com o [CLS] (posição 0) fora do denominador
  (`span_attention_mass`, política drop_from_denominator); os sub-tokens do trecho
  vêm dos deslocamentos de caractere do texto unido por espaço;
- T (`n_tokens`) = comprimento da sequência de sub-tokens incluindo [CLS] e [SEP],
  que é a dimensão da matriz de atenção, truncada em 256;
- `is_nested` vem da ANOTAÇÃO (invariante do protocolo): o trecho previsto está
  contido, sem ser igual, em outro trecho anotado.
"""
from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parent
sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(AQUI))

from src.selective.attention_mass import span_attention_mass  # noqa: E402
from src.selective.bert_adapter import (  # noqa: E402
    MAX_LEN, confianca_geometrica, decodifica_conlleval, f1_estrito, ler_corpus)

PONTOS = {
    "genia": ("configs/decl-23-bert-genia.yaml", "bert-base-cased-ft-genia"),
    "conll2003": ("configs/decl-24-bert-conll.yaml", "bert-base-cased-ft-conll2003"),
    "bc5cdr": ("configs/decl-25-bert-bc5cdr.yaml", "bert-base-cased-ft-bc5cdr"),
}
COLUNAS = ("sentence_id", "loss", "model_confidence", "span_mass", "is_nested", "token_indices",
           "palavra_ini", "palavra_fim", "rotulo", "n_tokens")


def dados_do_corpus(corpus: str) -> str:
    raiz = os.environ.get("SENTINEL_DADOS", str(RAIZ / "dados_bert"))
    return str(Path(raiz) / "bc5cdr") if corpus == "bc5cdr" else raiz


def aninhado(intervalo, todos) -> bool:
    a, b = intervalo
    return any(c <= a and b <= d and (c, d) != (a, b) for c, d in todos)


def indices_de_subtokens(offsets, ini_char: int, fim_char: int) -> list[int]:
    """Sub-tokens cujo intervalo de caracteres cai dentro de [ini_char, fim_char)."""
    return [j for j, (a, b) in enumerate(offsets) if b > a and a >= ini_char and b <= fim_char]


def medir_sentenca(modelo, tok, tokens, id2label):
    """Previsões por palavra + atenção média (L,H,T,T) de UMA sentença."""
    import torch

    texto = " ".join(tokens)
    inicios, cur = [], 0
    for t in tokens:
        inicios.append(cur)
        cur += len(t) + 1
    enc = tok(texto, return_offsets_mapping=True, truncation=True, max_length=MAX_LEN,
              return_tensors="pt")
    offsets = [tuple(o) for o in enc.pop("offset_mapping")[0].tolist()]
    with torch.no_grad():
        saida = modelo(**enc, output_attentions=True)
    if not saida.attentions:
        raise RuntimeError("o modelo não devolveu atenção: carregue com attn_implementation='eager' "
                           "(com sdpa, output_attentions devolve vazio sem erro)")
    probs = torch.softmax(saida.logits[0].float(), dim=-1).numpy()
    att = np.stack([a[0].float().numpy() for a in saida.attentions])      # [L,H,T,T]
    tags, pmax = ["O"] * len(tokens), [None] * len(tokens)
    for w, (s, t) in enumerate(zip(inicios, tokens)):
        toks = indices_de_subtokens(offsets, s, s + len(t))
        if not toks:
            continue
        j = toks[0]
        k = int(probs[j].argmax())
        tags[w], pmax[w] = id2label[k], float(probs[j, k])
    return {"tags": tags, "pmax": pmax, "att": att, "offsets": offsets, "inicios": inicios,
            "T": len(offsets)}


def medir(corpus: str, dir_modelo: str, saida: Path | None = None, max_samples: int = 0) -> dict:
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    from src.selective.preregistration import load_preregistration
    from impressao_digital import impressao_digital

    decl, nome = PONTOS[corpus]
    prereg = load_preregistration(str(RAIZ / decl))
    saida = Path(saida) if saida else RAIZ / "saida" / "bert" / nome / corpus / "test"
    saida.mkdir(parents=True, exist_ok=True)
    exemplos = ler_corpus(corpus, "test", dados_do_corpus(corpus))
    if max_samples:
        exemplos = exemplos[:max_samples]

    tok = AutoTokenizer.from_pretrained(dir_modelo)
    modelo = AutoModelForTokenClassification.from_pretrained(
        dir_modelo, attn_implementation="eager").eval()
    id2label = {int(k): v for k, v in modelo.config.id2label.items()}

    linhas, comprimentos, previstos, ouros = [], [], [], []
    for i, ex in enumerate(exemplos):
        r = medir_sentenca(modelo, tok, ex["tokens"], id2label)
        sid = f"s{i}"
        comprimentos.append((sid, r["T"]))
        spans = decodifica_conlleval(r["tags"])
        previstos.append(spans)
        ouros.append(ex["ouro"])
        ouro = set(ex["ouro"])
        intervalos = {(a, b) for a, b, _ in ex["ouro"]}
        for a, b, rot in spans:
            ci = r["inicios"][a]
            cf = r["inicios"][b] + len(ex["tokens"][b])
            idx = indices_de_subtokens(r["offsets"], ci, cf)
            if not idx:
                continue
            conf = confianca_geometrica([r["pmax"][w] for w in range(a, b + 1)
                                         if r["pmax"][w] is not None])
            massa = span_attention_mass(
                r["att"], idx, sink_policy=prereg.sink_policy, layers=prereg.layers,
                heads=prereg.heads, sink_index=0).mass
            linhas.append({
                "sentence_id": sid, "loss": 0 if (a, b, rot) in ouro else 1,
                "model_confidence": round(conf, 6), "span_mass": round(massa, 8),
                "is_nested": int(aninhado((a, b), intervalos)),
                "token_indices": "|".join(map(str, idx)),
                "palavra_ini": a, "palavra_fim": b, "rotulo": rot, "n_tokens": r["T"]})
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(exemplos)} sentenças, {len(linhas)} linhas", flush=True)

    with (saida / "entities.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(COLUNAS))
        w.writeheader()
        w.writerows(linhas)
    with (saida / "sentence_lengths.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["sentence_id", "n_tokens"])
        w.writerows(comprimentos)
    f1 = f1_estrito(previstos, ouros)
    erro = float(np.mean([l["loss"] for l in linhas])) if linhas else float("nan")
    fp = impressao_digital(dir_modelo)
    meta = {
        "modelo": {"id": nome, "camadas": int(modelo.config.num_hidden_layers),
                   "cabecas": int(modelo.config.num_attention_heads)},
        "colunas": list(COLUNAS), "corpus": corpus, "split": "test",
        "n_sentencas": len(exemplos), "n_entidades_preditas": len(linhas),
        "n_entidades_ouro": f1["n_ouro"], "taxa_de_erro_base": erro,
        "f1_estrito_teste": f1,
        "max_samples": max_samples or None,
        "attn_implementation": "eager",
        "impressao_digital": fp,
        "texto_medido": "palavras unidas por espaco; sub-tokens do trecho pelos offsets de caractere",
        "preregistro": {
            "layers": list(prereg.layers), "heads": prereg.heads,
            "sink_policy": prereg.sink_policy, "declaration_id": prereg.declaration_id,
            "declaration_hash": prereg.declaration_hash,
            "measurement_hash": prereg.measurement_hash, "source": decl, "model": prereg.model},
    }
    (saida / "MEDIDA.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False, default=list),
                                       encoding="utf-8")
    print(f"{corpus}: {len(linhas)} trechos previstos | erro {erro:.4f} | F1 estrito {f1['f1']:.4f}"
          f" | {saida}", flush=True)
    return meta


def analisar(corpora: list[str], medidas: Path | None = None, destino: Path | None = None) -> dict:
    """Roda as comparações declaradas (C1, C2, C3, T1-T4) em cada ponto."""
    from src.selective.comparisons import run_declared_comparisons
    from src.selective.preregistration import load_preregistration
    from src.selective.task_impact import run_task_impact

    medidas = Path(medidas) if medidas else RAIZ / "saida" / "bert"
    destino = Path(destino) if destino else RAIZ / "docs" / "tese" / "resultados"
    geo, tar, notas = [], [], []
    for corpus in corpora:
        decl, nome = PONTOS[corpus]
        p = load_preregistration(str(RAIZ / decl))
        d = medidas / nome / corpus / "test"
        meta = json.loads((d / "MEDIDA.json").read_text(encoding="utf-8"))
        n_gold = int(meta["n_entidades_ouro"])
        print(f"declaração {p.declaration_id} | {p.declaration_hash} | medição {p.measurement_hash}"
              f" | n_ouro {n_gold}", flush=True)
        rg = run_declared_comparisons(d, p, corpus)
        rt = run_task_impact(d, p, corpus, n_gold=n_gold)
        geo += [{"modelo": nome, "declaracao": p.declaration_id, **l} for l in rg.linhas]
        tar += [{"modelo": nome, "declaracao": p.declaration_id, **l} for l in rt.linhas]
        notas += [f"[{corpus}] {n}" for n in rg.notas + rt.notas]
        print(f"{corpus}: {len(rg.linhas)} linhas de geometria, {len(rt.linhas)} de tarefa", flush=True)
    for arq, linhas in (("confirmatorio_bert_geometria.csv", geo), ("confirmatorio_bert_tarefa.csv", tar)):
        alvo = destino / arq
        if alvo.exists() and not os.environ.get("SENTINEL_PERMITE_REESCREVER"):
            raise SystemExit(f"{alvo} já existe; recuso sobrescrever")
        campos = list({k: None for l in linhas for k in l})
        with alvo.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=campos, extrasaction="ignore")
            w.writeheader()
            w.writerows(linhas)
        print(f"escrito {alvo}: {len(linhas)} linhas", flush=True)
    (destino / "confirmatorio_bert_notas.txt").write_text("\n".join(notas) + "\n", encoding="utf-8")
    return {"geo": geo, "tar": tar, "notas": notas}


if __name__ == "__main__":
    a = sys.argv[1:]
    if not a:
        raise SystemExit(__doc__)
    if a[0] == "medir":
        ms = int(a[a.index("--max-samples") + 1]) if "--max-samples" in a else 0
        sd = Path(a[a.index("--saida") + 1]) if "--saida" in a else None
        medir(a[1], a[2], sd, ms)
    elif a[0] == "analisar":
        analisar(a[1:] or list(PONTOS))
