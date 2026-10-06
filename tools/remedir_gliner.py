"""Remedição dos pontos GLiNER (decl-14 e decl-15): refaz a previsão e acrescenta
o tipo de erro e as janelas de controle.

Uso:
    python tools/remedir_gliner.py --ponto gliner-base --corpus genia \
        --modelo urchade/gliner_base --preregistro configs/config.yaml \
        --tabela-medida /caminho/results/gliner-base/genia/test/entities.csv

Escreve em saida/remedicao/<ponto>/<corpus>/test/: remedicao.csv, guarda.json e
MEDIDA_remedicao.json. A GUARDA roda no fim (`tools/remedir_guarda.py`): se a
`loss` remedida não sair idêntica linha a linha à da tabela já medida, o ponto
NÃO é válido e o guarda.json diz isso.

O corpus vem de `dados_decoder/<corpus>_test.jsonl`, que é a exportação dos MESMOS
carregadores do projeto (mesma ordem, mesmo texto `' '.join(tokens)`, mesmos
offsets de caractere), conferida contra o SHA-256 de MANIFESTO_test.json. Os
carregadores baixariam o corpus da rede; o jsonl evita isso sem mudar o dado.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(Path(__file__).resolve().parent))

COLUNAS = (
    "sentence_id", "loss", "pred_start", "pred_end", "pred_label", "tipo_erro",
    "janelas_n", "janelas_enr_media", "enr_entidade",
    "model_confidence", "span_mass", "fracao_geometrica",
    "span_size", "n_tokens", "span_position", "token_indices", "is_nested",
)


def carregar_corpus(corpus: str) -> list[dict]:
    """O teste, em dicionários que o adaptador aceita; confere o SHA-256."""
    d = RAIZ / "dados_decoder"
    arq = d / f"{corpus}_test.jsonl"
    man = json.loads((d / "MANIFESTO_test.json").read_text(encoding="utf-8"))["arquivos"]
    h = hashlib.sha256(arq.read_bytes()).hexdigest()
    if h != man[arq.name]["sha256"]:
        raise SystemExit(f"{arq.name}: SHA-256 {h[:16]} diverge do manifesto")
    exemplos = []
    for linha in arq.open(encoding="utf-8"):
        j = json.loads(linha)
        toks = [str(t) for t in j["tokenized_text"]]
        exemplos.append({
            "text": " ".join(toks), "tokens": toks,
            "entities": [{"start": int(a), "end": int(b), "label": str(r)}
                         for a, b, r in j["ner_char"]],
        })
    return exemplos


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ponto", required=True)
    ap.add_argument("--corpus", required=True, choices=["genia", "conll2003"])
    ap.add_argument("--modelo", required=True, help="id do hub ou diretório local do peso")
    ap.add_argument("--preregistro", required=True)
    ap.add_argument("--tabela-medida", required=True, help="entities.csv já medido (guarda)")
    ap.add_argument("--saida", default=None)
    ap.add_argument("--limite", type=int, default=0)
    ap.add_argument("--threads", type=int, default=0)
    a = ap.parse_args(argv)

    # O modelo carrega ANTES do corpus e dos pacotes pesados (SIGSEGV de OpenMP
    # documentado em src/cli/commands/selective.py).
    import torch
    if a.threads:
        torch.set_num_threads(a.threads)
    from gliner import GLiNER

    modelo = GLiNER.from_pretrained(a.modelo)
    modelo.eval()

    import numpy as np
    import yaml
    from src.selective.attention_mass import span_attention_mass
    from src.selective.geometry import expected_mass
    from src.selective.gliner_adapter import GLiNERAdapter, _tokens_do_span
    from src.selective.janelas import (
        enriquecimento_encoder, janelas_da_entidade, media_encoder, tipo_erro)
    from src.selective.measurement import _stack_attention
    from src.selective.preregistration import load_preregistration

    prereg = load_preregistration(a.preregistro)
    cfg = yaml.safe_load(open(RAIZ / "configs/config.yaml", encoding="utf-8"))
    mapa = (cfg.get("gliner_labels") or {})[a.corpus]
    ad = GLiNERAdapter(modelo, prereg=prereg, labels=list(mapa), label_to_corpus=mapa)

    exemplos = carregar_corpus(a.corpus)
    if a.limite:
        exemplos = exemplos[: a.limite]
    linhas, n_sem_idx = [], 0
    for i, ex in enumerate(exemplos):
        sid = f"s{i}"
        r = ad.para_remedicao(ex)
        attn = _stack_attention(r["attentions"])           # float16, como na medição
        T = int(attn.shape[-1])
        offsets = r["offsets"]
        media = media_encoder(attn, prereg.layers, prereg.heads)
        tokens_frase = [j for j, (x, y) in enumerate(offsets) if x != y]
        ocupados = set()
        for g0, g1, _ in r["gold"]:
            ocupados.update(_tokens_do_span(offsets, g0, g1))
        for p in r["predicted"]:
            ocupados.update(p["token_indices"])
        for p in r["predicted"]:
            idx = list(p["token_indices"])
            if not idx:
                n_sem_idx += 1
                continue
            massa = span_attention_mass(
                attn, idx, sink_policy=prereg.sink_policy,
                layers=prereg.layers, heads=prereg.heads).mass
            esp = float(expected_mass(len(idx), T, sink_policy=prereg.sink_policy))
            n_j, enr_j = janelas_da_entidade(
                idx, tokens_frase, ocupados,
                lambda w: enriquecimento_encoder(media, w, T, prereg.sink_policy))
            ini, fim = p["char_span"]
            tipo = tipo_erro(ini, fim, p["label"], r["gold"])
            assert (tipo == "acerto") == bool(p["correct"]), (sid, ini, fim, tipo)
            linhas.append({
                "sentence_id": sid, "loss": 0 if p["correct"] else 1,
                "pred_start": ini, "pred_end": fim, "pred_label": p["label"],
                "tipo_erro": tipo, "janelas_n": n_j,
                "janelas_enr_media": enr_j, "enr_entidade": massa / esp,
                "model_confidence": p["confidence"], "span_mass": massa,
                "fracao_geometrica": esp, "span_size": len(idx), "n_tokens": T,
                "span_position": int(min(idx)),
                "token_indices": "|".join(str(int(v)) for v in idx),
                "is_nested": int(bool(p["is_nested"])),
            })
        if (i + 1) % 200 == 0:
            print(f"  {a.ponto}/{a.corpus}: {i + 1}/{len(exemplos)} sentenças, "
                  f"{len(linhas)} linhas", flush=True)

    saida = Path(a.saida or RAIZ / "saida/remedicao" / a.ponto / a.corpus / "test")
    saida.mkdir(parents=True, exist_ok=True)
    with (saida / "remedicao.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(COLUNAS))
        w.writeheader()
        for l in linhas:
            w.writerow({k: (repr(float(v)) if isinstance(v, float) else v) for k, v in l.items()})
    (saida / "MEDIDA_remedicao.json").write_text(json.dumps({
        "ponto": a.ponto, "corpus": a.corpus, "modelo": a.modelo,
        "preregistro": a.preregistro, "declaration_id": prereg.declaration_id,
        "layers": list(prereg.layers), "heads": prereg.heads,
        "sink_policy": prereg.sink_policy, "n_sentencas": len(exemplos),
        "n_linhas": len(linhas), "previsoes_sem_token_descartadas": n_sem_idx,
        "limite": a.limite or None,
    }, indent=1, ensure_ascii=False), encoding="utf-8")

    from remedir_guarda import guarda
    g = guarda(saida / "remedicao.csv", Path(a.tabela_medida), tipo="encoder",
               limite=a.limite or None)
    (saida / "guarda.json").write_text(json.dumps(g, indent=1, ensure_ascii=False),
                                       encoding="utf-8")
    print(json.dumps({k: g[k] for k in ("ok", "n_remedido", "n_medido", "linhas_loss_diferentes")}))
    return 0 if g["ok"] else 3


if __name__ == "__main__":
    sys.exit(main())
