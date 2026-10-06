"""Adaptador do classificador por token (decl-22): BERT + softmax, rótulo por palavra.

Este módulo reúne o que treino e medição PRECISAM compartilhar, para que as duas
passagens não tenham duas definições da mesma coisa:

1. leitura dos corpora e esquema de rótulos (BIO por palavra);
2. decodificação conlleval (um I-X sem B-X/I-X antes abre trecho);
3. a confiança do trecho (média geométrica, sobre as palavras do trecho, da
   probabilidade máxima do softmax no primeiro sub-token);
4. o F1 estrito (trecho, rótulo), que é o critério de sempre.

Só biblioteca padrão e numpy: nenhum torch aqui (o guarda dos testes só admite torch
nos adaptadores de modelo). A parte que toca o modelo — `prever_lote` e
`medir_sentenca` — vive em tools/treinar_bert.py e tools/medir_bert.py.

DECISÕES DESTE ARQUIVO QUE NÃO ESTÃO NA DECL-22 (declaradas, não escondidas)

- GENIA, etiquetas planas. A decl-22 manda treinar com "as etiquetas planas do
  carregador". O carregador não roda dentro do contêiner (puxa `datasets` e rede),
  então as etiquetas são reconstruídas dos `*_train.jsonl` do canal de dados, que
  vieram dos mesmos carregadores, aplicando a regra do `_create_bio_labels` do
  carregador: para cada entidade, NA ORDEM DO ARQUIVO, B no início e I até o fim
  (fim inclusivo em índice de palavra), sobrescrevendo. Isto presume que a ordem
  das entidades do jsonl é a do carregador; não foi possível conferir porque o
  carregador depende de rede (ver BERT.md).
- `weight_decay` 0,01 NÃO se aplica a bias e LayerNorm (convenção do Trainer do
  Hugging Face, a que a "receita padrão" subentende).
- Palavras além de 256 sub-tokens ficam sem rótulo de treino e são previstas como
  `O` na avaliação.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

__all__ = [
    "MAX_LEN", "CLASSES", "normaliza_rotulo", "ler_corpus", "bio_plano",
    "rotulos_bio", "decodifica_conlleval", "confianca_geometrica", "f1_estrito",
    "spans_de_caracteres",
]

MAX_LEN = 256

# Classes por corpus, no código do carregador (as do teste).
CLASSES = {
    "genia": ("CELL_LINE", "CELL_TYPE", "DNA", "PROTEIN", "RNA"),
    "conll2003": ("LOC", "MISC", "ORG", "PER"),
    "bc5cdr": ("Chemical", "Disease"),
}
_ALIAS_CONLL = {"PERSON": "PER", "ORGANIZATION": "ORG", "LOCATION": "LOC",
                "MISCELLANEOUS": "MISC"}


def normaliza_rotulo(corpus: str, rotulo: str) -> str:
    """Rótulo do arquivo (descrição minúscula no treino, sigla no teste) -> código."""
    if corpus == "bc5cdr":
        r = str(rotulo).strip().capitalize()
        if r not in CLASSES["bc5cdr"]:
            raise KeyError(f"rótulo {rotulo!r} fora de {CLASSES['bc5cdr']}")
        return r
    r = str(rotulo).strip().upper().replace(" ", "_").replace("-", "_")
    r = _ALIAS_CONLL.get(r, r) if corpus == "conll2003" else r
    if r not in CLASSES[corpus]:
        raise KeyError(f"rótulo {rotulo!r} do corpus {corpus!r} fora de {CLASSES[corpus]}")
    return r


def rotulos_bio(corpus: str) -> list[str]:
    """['O', 'B-X', 'I-X', ...]: a ordem é a do índice de saída do classificador."""
    out = ["O"]
    for c in CLASSES[corpus]:
        out += [f"B-{c}", f"I-{c}"]
    return out


def bio_plano(n: int, spans_em_ordem: Iterable[tuple[int, int, str]]) -> list[str]:
    """Etiquetas planas por sobrescrita, na ordem dada (regra do carregador)."""
    tags = ["O"] * n
    for a, b, r in spans_em_ordem:
        if not (0 <= a <= b < n):
            continue
        tags[a] = f"B-{r}"
        for i in range(a + 1, b + 1):
            tags[i] = f"I-{r}"
    return tags


def decodifica_conlleval(tags: Sequence[str]) -> list[tuple[int, int, str]]:
    """Trechos [a, b] (b inclusivo, por palavra). I-X sem X antes abre trecho."""
    spans, ini, tipo = [], None, None
    for i, t in enumerate(list(tags) + ["O"]):
        if t == "O":
            novo, r = False, None
        else:
            pref, r = t[0], t[2:]
            novo = pref == "B" or tipo != r
        if ini is not None and (t == "O" or novo):
            spans.append((ini, i - 1, tipo))
            ini, tipo = None, None
        if t != "O" and ini is None:
            ini, tipo = i, r
    return spans


def confianca_geometrica(probs_max: Sequence[float]) -> float:
    """Média geométrica das probabilidades máximas (a definição da decl-22)."""
    p = np.asarray(probs_max, dtype=float)
    return float(np.exp(np.mean(np.log(np.clip(p, 1e-300, 1.0)))))


def f1_estrito(previstos: Iterable[tuple], ouro: Iterable[tuple]) -> dict:
    """Critério estrito: (início, fim, rótulo) iguais. Conjuntos por sentença."""
    tp = n_prev = n_ouro = 0
    for pv, ou in zip(previstos, ouro):
        pv, ou = set(pv), set(ou)
        tp += len(pv & ou)
        n_prev += len(pv)
        n_ouro += len(ou)
    p = tp / n_prev if n_prev else 0.0
    r = tp / n_ouro if n_ouro else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"precisao": p, "recall": r, "f1": f, "tp": tp,
            "n_previstos": n_prev, "n_ouro": n_ouro}


def spans_de_caracteres(tokens: Sequence[str], ner_char: Iterable) -> list[tuple[int, int, str]]:
    """ner_char (texto unido por espaço) -> trechos por palavra, como `trechos_de_ouro`."""
    inicios, cur = [], 0
    for t in tokens:
        inicios.append(cur)
        cur += len(t) + 1
    out = []
    for a, b, r in ner_char:
        i0 = next((i for i, (s, t) in enumerate(zip(inicios, tokens)) if s <= a < s + len(t)), None)
        i1 = next((i for i, (s, t) in enumerate(zip(inicios, tokens)) if s < b <= s + len(t)), None)
        if i0 is None or i1 is None or i1 < i0:
            continue
        out.append((i0, i1, str(r)))
    return out


def ler_corpus(corpus: str, particao: str, dados: str | Path) -> list[dict]:
    """Sentenças como {tokens, ouro (aninhado, por palavra), tags_planas}.

    Partições: 'train', 'validation', 'test' (BC5CDR: valid no disco).
    """
    d = Path(dados)
    exemplos = []
    if corpus == "bc5cdr":
        arq = d / f"{ {'validation': 'valid'}.get(particao, particao) }.json"
        inv = {v: k for k, v in json.loads((d / "label.json").read_text()).items()}
        for linha in arq.read_text(encoding="utf-8").splitlines():
            if not linha.strip():
                continue
            j = json.loads(linha)
            tags = [inv[int(t)] for t in j["tags"]]
            ouro = decodifica_conlleval(tags)
            exemplos.append({"tokens": [str(t) for t in j["tokens"]], "ouro": ouro,
                             "tags_planas": tags})
        return exemplos
    arq = d / f"{corpus}_{particao}.jsonl"
    for linha in arq.read_text(encoding="utf-8").splitlines():
        if not linha.strip():
            continue
        j = json.loads(linha)
        toks = [str(t) for t in j["tokenized_text"]]
        if "ner_char" in j:
            bruto = spans_de_caracteres(toks, j["ner_char"])
        elif "ner" in j:
            bruto = [(int(a), int(b), str(r)) for a, b, r in j["ner"]]
        else:
            raise KeyError(f"{arq}: nem `ner` nem `ner_char`")
        em_ordem = [(a, b, normaliza_rotulo(corpus, r)) for a, b, r in bruto]
        exemplos.append({"tokens": toks, "ouro": em_ordem,
                         "tags_planas": bio_plano(len(toks), em_ordem)})
    return exemplos
