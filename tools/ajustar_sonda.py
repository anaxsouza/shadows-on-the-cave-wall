"""Ajusta as SONDAS supervisionadas e escreve os escores delas.

O QUE É UMA SONDA AQUI

Os sinais confirmatórios são resumos NÃO supervisionados: a norma do estado, a
entropia da linha de atenção. Ninguém os ensinou a prever erro — se eles preveem,
é porque a informação estava lá.

A sonda é o contrário: uma regressão logística que APRENDE a combinar as features
para prever acerto. Ela existe para responder a uma objeção específica — "vocês
não acharam sinal porque procuraram com instrumento cego" — e por isso é
EXPLORATÓRIA. Ela não decide veredito: pesos ajustados abandonam o nulo exato que
torna a afirmação sobre atenção imune à objeção de escala.

DUAS VERSÕES, POR DECISÃO DO AUTOR EM 22/09/2026

  `sonda_*`            regularização no valor convencional (L2, C=1,0), escolhida
                       sem olhar dado nenhum. É esta que leva o NOME DECLARADO,
                       porque é a que tem zero parâmetro livre — a mesma
                       propriedade que torna as declarações defensáveis.
  `sonda_*_ajustada`   regularização escolhida por validação cruzada DENTRO da
                       calibração. Coluna ADICIONAL, posterior à assinatura das
                       declarações, e marcada como tal: ela não está em
                       `comparison_scores` de decl-10 a decl-13 e a análise
                       declarada não a lê.

A DIFERENÇA entre as duas é informação própria: mede quanto de qualquer vantagem
da sonda vem da liberdade de ajuste e não do sinal.

A PARTIÇÃO É A MESMA DA ANÁLISE, E ISSO NÃO É DETALHE

A separação calibração/avaliação vem de `_separar_por_sentenca` do próprio
projeto, IMPORTADA e não reescrita. Reimplementá-la daria uma partição que
divergiria da análise na primeira mudança de semente ou de fração, e a sonda
estaria ajustada em sentenças que a análise usa para avaliar — vazamento que não
aparece como erro, aparece como resultado bom.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
from src.selective.comparisons import _separar_por_sentenca      # noqa: E402
from src.selective.preregistration import load_preregistration   # noqa: E402

CS = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]


def _achatar(a: np.ndarray) -> np.ndarray:
    return a.reshape(a.shape[0], -1) if a.ndim > 2 else a


def ajustar(X: np.ndarray, y: np.ndarray, i_cal: np.ndarray, semente: int) -> dict:
    """Escores das duas versões, ajustadas SÓ na calibração.

    `y` é ACERTO (1 - loss): a sonda prediz acerto, então escore alto significa
    entregar, na mesma orientação de `model_confidence`. Fixar a orientação por
    definição evita testar as duas e relatar a melhor, que dobraria as chances
    ao acaso.
    """
    Xc, yc = X[i_cal], y[i_cal]
    if len(np.unique(yc)) < 2:
        return {"erro": f"calibração tem uma classe só ({np.unique(yc).tolist()})"}
    out = {}
    fixa = make_pipeline(StandardScaler(),
                         LogisticRegression(C=1.0, max_iter=2000, random_state=semente))
    fixa.fit(Xc, yc)
    out["fixa"] = fixa.predict_proba(X)[:, 1]
    n_min = int(np.bincount(yc.astype(int)).min())
    cv = max(2, min(5, n_min))
    aj = make_pipeline(StandardScaler(),
                       LogisticRegressionCV(Cs=CS, cv=cv, scoring="neg_log_loss",
                                            max_iter=2000, random_state=semente))
    aj.fit(Xc, yc)
    out["ajustada"] = aj.predict_proba(X)[:, 1]
    out["C_escolhido"] = float(aj[-1].C_[0])
    out["cv_dobras"] = cv
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--medicao", required=True, help="diretório com entities.csv e features_sonda.npz")
    ap.add_argument("--declaracao", required=True)
    a = ap.parse_args()
    d = Path(a.medicao)
    ent = pd.read_csv(d / "entities.csv")
    npz = np.load(d / "features_sonda.npz", allow_pickle=False)
    prereg = load_preregistration(a.declaracao)

    lid = npz["linha_id"]
    ent = ent.iloc[lid].reset_index(drop=True)      # alinhar entities às features
    y = (1 - ent["loss"].to_numpy()).astype(int)

    # A MESMA partição da análise, pela MESMA função.
    t = {"sentence_id": ent["sentence_id"].to_numpy(),
         "idx": np.arange(len(ent))}
    cal, aval = _separar_por_sentenca(t, prereg)
    i_cal = cal["idx"]
    assert len(set(cal["sentence_id"]) & set(aval["sentence_id"])) == 0, "sentença nos dois lados"

    saida = pd.DataFrame({"linha_id": lid})
    meta = {"declaracao": prereg.declaration_id,
            "n_linhas": int(len(ent)), "n_calibracao": int(len(i_cal)),
            "n_avaliacao": int(len(aval["idx"])),
            "fracao_calibracao_declarada": prereg.calibration_fraction,
            "semente": prereg.seed, "alvo": "acerto = 1 - loss",
            "particao": "_separar_por_sentenca do projeto, importada",
            "coluna_ajustada": ("POSTERIOR à assinatura das declarações (decisão do autor em "
                                "22/09/2026). NÃO está em comparison_scores e a análise "
                                "declarada não a lê.")}
    for nome, chave in (("sonda_ocultos", "oculta"), ("sonda_atencao_cabecas", "cabeca")):
        A = npz[chave]
        if A.size == 0 or A.shape[0] != len(ent):
            meta[nome] = f"AUSENTE: forma {A.shape} contra {len(ent)} linhas"
            continue
        r = ajustar(_achatar(A), y, i_cal, prereg.seed)
        if "erro" in r:
            meta[nome] = r["erro"]; continue
        saida[nome] = r["fixa"]
        saida[f"{nome}_ajustada"] = r["ajustada"]
        meta[nome] = {"n_features": int(_achatar(A).shape[1]),
                      "C_fixo": 1.0, "C_escolhido_por_cv": r["C_escolhido"],
                      "cv_dobras": r["cv_dobras"]}
    saida.to_csv(d / "sondas.csv", index=False)
    (d / "SONDAS.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
