"""Resumo dos dez pontos da remedição: contagens por tipo, janelas, guarda. Escreve
saida/remedicao/RESUMO.json (insumo do REMEDICAO.md)."""
import json
from pathlib import Path
import pandas as pd

R = Path(__file__).resolve().parents[1] / "saida/remedicao"
PONTOS = [("gliner-base", "genia"), ("gliner-base", "conll2003"), ("gliner-large", "genia"),
          ("gliner-large", "conll2003"), ("gliner_base-ft-genia", "genia"),
          ("gliner_base-ft-conll2003", "conll2003"), ("qwen05b-ft-genia", "genia"),
          ("qwen05b-ft-conll2003", "conll2003"), ("qwen15b-ft-genia", "genia"),
          ("qwen15b-ft-conll2003", "conll2003")]
out = []
for p, c in PONTOS:
    d = R / p / c / "test"
    if not (d / "remedicao.csv").exists():
        out.append({"ponto": p, "corpus": c, "status": "ausente"}); continue
    df = pd.read_csv(d / "remedicao.csv")
    g = json.loads((d / "guarda.json").read_text())
    tipos = df.tipo_erro.value_counts().to_dict()
    out.append({
        "ponto": p, "corpus": c, "arquivos": str(d.relative_to(R.parents[1])),
        "guarda_ok": g["ok"], "guarda": {k: g[k] for k in (
            "linhas_loss_diferentes", "linhas_confianca_acima_da_tolerancia", "max_abs_confianca",
            "linhas_massa_acima_da_tolerancia", "max_abs_massa", "tipo_incoerente_com_loss")},
        "n_entidades": int(len(df)), "n_erros": int((df.loss == 1).sum()),
        "tipos": {t: int(tipos.get(t, 0)) for t in ("acerto", "rotulo", "fronteira", "sem_par")},
        "n_sem_janela": int((df.janelas_n == 0).sum()),
        "janelas_n_mediana": float(df.janelas_n.median()),
        "janelas_enr_media_global": float(df.janelas_enr_media.mean()),
        "enr_entidade_media": float(df.enr_entidade.mean()),
    })
(R / "RESUMO.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
for o in out:
    print(o["ponto"], o["corpus"], o.get("guarda_ok"), o.get("n_entidades"), o.get("tipos"), o.get("n_sem_janela"))
