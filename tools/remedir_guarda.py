"""A GUARDA de remedição (decl-14/decl-15, regras comuns).

A passada que refaz as previsões tem de reproduzir, linha a linha, a tabela já
medida: `loss` IDÊNTICA; `model_confidence` e a massa (`span_mass` nos encoders,
`causal_mass_prompt` nos decoders = `span_mass` da tabela derivada) a 1e-5, que é
a precisão de gravação (6 casas) da tabela original; e os índices de token
iguais. Qualquer diferença => ok = False e o ponto não é entregue como válido.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

TOL = 1e-5


def _ler(p: Path) -> list[dict]:
    with p.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def guarda(remedido: Path, medido: Path, *, tipo: str, limite: int | None = None) -> dict:
    rem, med = _ler(remedido), _ler(medido)
    col_massa_rem = "span_mass" if tipo == "encoder" else "causal_mass_prompt"
    if limite:
        # fumaça: compara só o prefixo de sentenças medido
        ids = {r["sentence_id"] for r in rem}
        med = [m for m in med if m["sentence_id"] in ids]
    g = {
        "tipo": tipo, "tolerancia": TOL, "n_remedido": len(rem), "n_medido": len(med),
        "limite_sentencas": limite,
    }
    g["mesmo_numero_de_linhas"] = len(rem) == len(med)
    n = min(len(rem), len(med))
    dif_loss, dif_sid, dif_tok, dif_conf, dif_massa = [], [], [], [], []
    max_conf = max_massa = 0.0
    for i in range(n):
        a, b = rem[i], med[i]
        if a["sentence_id"] != b["sentence_id"]:
            dif_sid.append(i)
        if int(a["loss"]) != int(b["loss"]):
            dif_loss.append(i)
        if a["token_indices"] != b["token_indices"]:
            dif_tok.append(i)
        dc = abs(float(a["model_confidence"]) - float(b["model_confidence"]))
        dm = abs(float(a[col_massa_rem]) - float(b["span_mass"]))
        max_conf, max_massa = max(max_conf, dc), max(max_massa, dm)
        if dc > TOL or math.isnan(dc):
            dif_conf.append(i)
        if dm > TOL or math.isnan(dm):
            dif_massa.append(i)
    # a taxonomia tem de ser coerente com a perda: acerto <=> loss == 0
    incoerentes = [i for i, r in enumerate(rem)
                   if (r["tipo_erro"] == "acerto") != (int(r["loss"]) == 0)]
    g.update({
        "linhas_loss_diferentes": len(dif_loss), "primeiras_loss_diferentes": dif_loss[:10],
        "linhas_sentence_id_diferentes": len(dif_sid),
        "linhas_token_indices_diferentes": len(dif_tok),
        "linhas_confianca_acima_da_tolerancia": len(dif_conf), "max_abs_confianca": max_conf,
        "linhas_massa_acima_da_tolerancia": len(dif_massa), "max_abs_massa": max_massa,
        "tipo_incoerente_com_loss": len(incoerentes),
    })
    g["ok"] = bool(
        g["mesmo_numero_de_linhas"] and not dif_loss and not dif_sid and not dif_tok
        and not dif_conf and not dif_massa and not incoerentes and n > 0)
    return g
