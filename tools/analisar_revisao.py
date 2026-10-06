"""Analysis of the revision tests C4 (decl-14) and C1 (decl-15), mechanically from the declarations.

    python tools/analisar_revisao.py            # writes docs/tese/resultados/revisao/*.csv

Input: saida/remedicao/<ponto>/<corpus>/test/remedicao.csv and guarda.json (produced by the
remeasurement), plus dados_medidos/<ponto>/<corpus>/test/sentence_lengths.csv.

C4: for each point and error type t, the table "all correct + errors of type t" is written in the
format of entities.csv and handed, unchanged, to `run_declared_comparisons` under the ORIGIN
declaration of the point; E1 is its C2 and E2 its C3, pooled stratum ("todos").
C1: J1 median of log r(e) over entities, J2 the same split by correct/error and share r>1, J3 the
difference of medians error minus correct. Intervals: 2,000 bootstrap resamples over sentences,
seed 42, on the evaluation partition given by the same sentence split the runner uses.
"""
from __future__ import annotations

import csv
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
from src.selective.comparisons import run_declared_comparisons, _separar_por_sentenca  # noqa: E402
from src.selective.preregistration import load_preregistration  # noqa: E402

REM = RAIZ / "saida" / "remedicao"
MED = RAIZ / "dados_medidos"
MED_DEC = RAIZ / "saida" / "decoder"  # tabelas montadas por tools/analisar_decoder.py


def _med(pt: str) -> Path:
    return MED_DEC if pt.startswith("qwen") else MED
OUT = RAIZ / "docs" / "tese" / "resultados" / "revisao"
ORIGEM = {
    ("gliner-base", "genia"): "decl-02-geometria", ("gliner-base", "conll2003"): "decl-02-geometria",
    ("gliner-large", "genia"): "decl-04-escala", ("gliner-large", "conll2003"): "decl-04-escala",
    ("gliner_base-ft-genia", "genia"): "decl-05-ajustado-genia",
    ("gliner_base-ft-conll2003", "conll2003"): "decl-06-ajustado-conll",
    ("qwen05b-ft-genia", "genia"): "decl-10-decoder-05b-genia",
    ("qwen05b-ft-conll2003", "conll2003"): "decl-11-decoder-05b-conll",
    ("qwen15b-ft-genia", "genia"): "decl-12-decoder-15b-genia",
    ("qwen15b-ft-conll2003", "conll2003"): "decl-13-decoder-15b-conll",
}
TIPOS = ("rotulo", "fronteira", "sem_par")
B, NIVEL, SEED = 2000, 0.95, 42


def _carregar(pt: str, corp: str) -> pd.DataFrame | None:
    d = REM / pt / corp / "test"
    g = d / "guarda.json"
    if not (d / "remedicao.csv").is_file() or not g.is_file():
        return None
    gj = json.loads(g.read_text())
    # A guarda DECLARADA (decl-14/15, regras comuns) é a coluna `loss` idêntica linha a linha.
    # A tolerância de confiança/massa em guarda.json é conferência adicional, relatada à parte.
    if not (gj.get("mesmo_numero_de_linhas") and gj.get("linhas_loss_diferentes") == 0
            and gj.get("linhas_sentence_id_diferentes", 0) == 0):
        raise SystemExit(f"GUARDA FALHOU em {pt}/{corp}: a análise não roda (decl-14/15, regras comuns)")
    return pd.read_csv(d / "remedicao.csv", dtype={"sentence_id": str, "token_indices": str})


def _tabela_para_runner(mascara: np.ndarray, pt: str, corp: str, rot: str, destino: Path) -> Path:
    """A tabela MEDIDA da origem, só com as linhas da máscara: nenhuma coluna remedida entra."""
    orig = _med(pt) / pt / corp / "test"
    with (orig / "entities.csv").open(encoding="utf-8", newline="") as fh:
        linhas = fh.read().splitlines()
    cab, corpo = linhas[0], linhas[1:]
    assert len(corpo) == mascara.size, (pt, corp, len(corpo), mascara.size)
    d = destino / pt / f"{corp}__{rot}" / "test"
    d.mkdir(parents=True, exist_ok=True)
    (d / "entities.csv").write_text("\n".join([cab] + [l for l, m in zip(corpo, mascara) if m]) + "\n",
                                    encoding="utf-8")
    shutil.copy(orig / "sentence_lengths.csv", d / "sentence_lengths.csv")
    (d / "attention").mkdir(exist_ok=True)
    return d


def c4() -> pd.DataFrame:
    linhas = []
    with tempfile.TemporaryDirectory() as tmp:
        for (pt, corp), did in ORIGEM.items():
            df = _carregar(pt, corp)
            if df is None:
                linhas.append(dict(ponto=pt, corpus=corp, estado="sem dados remedidos"))
                continue
            p = load_preregistration(RAIZ / "configs" / f"{did}.yaml")
            for t in TIPOS:
                sub = df[(df.tipo_erro == "acerto") | (df.tipo_erro == t)].copy()
                n_err = int((sub.tipo_erro == t).sum())
                base = dict(ponto=pt, corpus=corp, declaracao_origem=did, tipo=t, n_erros=n_err,
                            n_acertos=int((sub.tipo_erro == "acerto").sum()))
                if n_err == 0:
                    linhas.append(dict(base, estado="nenhum erro deste tipo"))
                    continue
                orig = pd.read_csv(_med(pt) / pt / corp / "test" / "entities.csv", dtype={"sentence_id": str})
                assert (orig["loss"].to_numpy() == df["loss"].to_numpy()).all(), (pt, corp)
                assert (orig["sentence_id"].to_numpy() == df["sentence_id"].to_numpy()).all(), (pt, corp)
                m = ((df.tipo_erro == "acerto") | (df.tipo_erro == t)).to_numpy()
                d = _tabela_para_runner(m, pt, corp, t, Path(tmp))
                rel = run_declared_comparisons(d, p, corp)
                for r in rel.linhas:
                    if r.get("tipo") == "verdict" and r.get("comparacao") in ("C2", "C3") \
                            and r.get("estrato") == "todos":
                        linhas.append(dict(base, teste={"C2": "E1", "C3": "E2"}[r["comparacao"]],
                                           escore=r.get("escore"), contra=r.get("contra"),
                                           aurc_escore=r.get("aurc_escore"), aurc_contra=r.get("aurc_contra"),
                                           delta=r.get("delta"), ci_low=r.get("ci_low"), ci_high=r.get("ci_high"),
                                           n_degeneradas=r.get("n_degeneradas"), estado="ok"))
                    if r.get("tipo") == "descriptive" and r.get("comparacao") == "escores" \
                            and r.get("estrato") == "todos" and r.get("particao") == "avaliação":
                        linhas.append(dict(base, teste="E3", metrica=r.get("metrica"), valor=r.get("valor"),
                                           estado="ok"))
    return pd.DataFrame(linhas)


def _boot_mediana(v: np.ndarray, grupos: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    us = np.unique(grupos); idx = {u: np.flatnonzero(grupos == u) for u in us}
    est = []
    for _ in range(B):
        s = rng.choice(us, size=us.size, replace=True)
        est.append(np.median(v[np.concatenate([idx[u] for u in s])]))
    a = (1 - NIVEL) / 2
    return float(np.quantile(est, a)), float(np.quantile(est, 1 - a))


def _boot_dif(v, err, grupos, rng):
    us = np.unique(grupos); idx = {u: np.flatnonzero(grupos == u) for u in us}
    est = []
    for _ in range(B):
        s = rng.choice(us, size=us.size, replace=True)
        i = np.concatenate([idx[u] for u in s])
        e, c = v[i][err[i]], v[i][~err[i]]
        if e.size and c.size:
            est.append(np.median(e) - np.median(c))
    a = (1 - NIVEL) / 2
    return float(np.quantile(est, a)), float(np.quantile(est, 1 - a))


def c1() -> pd.DataFrame:
    linhas = []
    for (pt, corp), did in ORIGEM.items():
        df = _carregar(pt, corp)
        if df is None:
            linhas.append(dict(ponto=pt, corpus=corp, estado="sem dados remedidos"))
            continue
        p = load_preregistration(RAIZ / "configs" / f"{did}.yaml")
        t = {"sentence_id": df.sentence_id.to_numpy(), "loss": df["loss"].to_numpy(float),
             "i": np.arange(len(df))}
        _, aval = _separar_por_sentenca(t, p)
        a = df.iloc[aval["i"]]
        sem = int((a.janelas_n == 0).sum())
        a = a[(a.janelas_n > 0) & (a.janelas_enr_media > 0) & (a.enr_entidade > 0)]
        lr = np.log(a.enr_entidade.to_numpy() / a.janelas_enr_media.to_numpy())
        g = a.sentence_id.to_numpy(); err = a["loss"].to_numpy() > 0
        rng = np.random.default_rng(SEED)
        lo, hi = _boot_mediana(lr, g, rng)
        dlo, dhi = _boot_dif(lr, err, g, np.random.default_rng(SEED))
        linhas.append(dict(
            ponto=pt, corpus=corp, declaracao_origem=did, n=int(lr.size), n_sem_janela=sem,
            J1_mediana_log_r=float(np.median(lr)), J1_ci_low=lo, J1_ci_high=hi,
            J1_veredito=("destaca" if lo > 0 else "atenua" if hi < 0 else "nao distingue"),
            J2_mediana_acertos=float(np.median(lr[~err])), J2_mediana_erros=float(np.median(lr[err])),
            J2_frac_r_maior_1=float(np.mean(lr > 0)),
            J3_dif_erros_menos_acertos=float(np.median(lr[err]) - np.median(lr[~err])),
            J3_ci_low=dlo, J3_ci_high=dhi, estado="ok"))
    return pd.DataFrame(linhas)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    c4().to_csv(OUT / "c4_tipos_de_erro.csv", index=False)
    c1().to_csv(OUT / "c1_janelas.csv", index=False)
    print(f"-> {OUT.relative_to(RAIZ)}/c4_tipos_de_erro.csv, c1_janelas.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
