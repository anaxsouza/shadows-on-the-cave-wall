"""A análise declarada do braço decoder: decl-10 a decl-13, mais as adendas 01 e 02.

O QUE ESTE ARQUIVO FAZ, EM ORDEM, E POR QUE CADA PASSO PODE PARAR TUDO

1. CONFERE A REMEDIÇÃO. As tabelas `sentinel-medir-chave-*` só acrescentam
   `mencao` e `rotulo` às `sentinel-medir-feat-*`. A medição é determinística, então
   toda coluna anterior tem de sair idêntica como texto cru. Qualquer diferença
   PARA: significaria que as sondas, ajustadas sobre a tabela anterior, estão
   alinhadas a outra coisa.

2. MONTA A TABELA NA FORMA QUE O EXECUTOR LÊ, e é aqui que vive a adenda 02:
   `span_mass := causal_mass_prompt`; `sentence_length := n_tokens`; as linhas com
   `ancorada == 0` saem; a presença de `expected_causal` faz o executor usar o nulo
   causal. A tabela é ESCRITA em disco e fica como evidência do que foi julgado.

3. JUNTA O AggSeq por (sentence_id, mencao, rotulo). Chave repetida PARA. Um par
   guloso ausente de todo feixe recebe 0, e a contagem vai para as notas.

4. JUNTA AS SONDAS por `linha_id` e AJUSTA A LINHA DE BASE TREINADA da adenda 01,
   com a mesma função `ajustar` das sondas, na mesma partição.

5. RODA as comparações assinadas pelos executores do projeto, sem reimplementar
   nenhum critério, e depois as 32 da adenda 01, que usam os mesmos executores com
   a lista de comparações trocada.
"""
from __future__ import annotations

import csv
import dataclasses
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(RAIZ / "tools"))
from ajustar_sonda import ajustar                                    # noqa: E402
from src.selective.comparisons import (                               # noqa: E402
    _separar_por_sentenca, run_declared_comparisons)
from src.selective.preregistration import Comparison, load_preregistration  # noqa: E402
from src.selective.task_impact import run_task_impact                 # noqa: E402

import os
# `dados_medidos/` é o pacote publicado (tools/empacotar_medicoes.py): uma pasta por
# ponto com entities.csv, sondas.csv e aggseq.csv, sem texto de corpus.
DADOS = Path(os.environ.get("SENTINEL_DADOS_MEDIDOS", RAIZ / "dados_medidos"))
SAIDA = Path(os.environ.get("SENTINEL_SAIDA_ANALISE", RAIZ / "saida" / "decoder"))
# A remedição COM a chave de junção foi conferida contra a anterior em 23/09/2026, no
# repositório de desenvolvimento, e a conferência vai registrada em NOTAS.json. Aqui ela
# só roda se a tabela anterior for fornecida em SENTINEL_MEDICAO_ANTERIOR.
ANTERIOR = os.environ.get("SENTINEL_MEDICAO_ANTERIOR")
OURO = {"genia": 5506, "conll2003": 5648}          # adenda 02, item 6
PONTOS = [("qwen05b-ft-genia", "genia", "decl-10-decoder-05b-genia", "05b-genia"),
          ("qwen05b-ft-conll2003", "conll2003", "decl-11-decoder-05b-conll", "05b-conll"),
          ("qwen15b-ft-genia", "genia", "decl-12-decoder-15b-genia", "15b-genia"),
          ("qwen15b-ft-conll2003", "conll2003", "decl-13-decoder-15b-conll", "15b-conll")]
COLUNAS_TABELA = ["sentence_id", "loss", "model_confidence", "span_mass", "is_nested",
                  "token_indices", "span_position", "expected_causal",
                  "row_entropy_causal", "row_max_causal",
                  "hidden_norm", "hidden_dist_centroide", "hidden_delta_camadas",
                  "aggseq", "sonda_ocultos", "sonda_atencao_cabecas",
                  "sonda_ocultos_ajustada", "sonda_atencao_cabecas_ajustada",
                  "base_treinada", "base_treinada_ajustada"]
SONDAS = ["sonda_ocultos", "sonda_atencao_cabecas",
          "sonda_ocultos_ajustada", "sonda_atencao_cabecas_ajustada"]


class Parar(RuntimeError):
    pass


def conferir_remedicao(nome: str, corpus: str) -> pd.DataFrame:
    nova = pd.read_csv(DADOS / nome / corpus / "test" / "entities.csv",
                       keep_default_na=False, dtype=str)
    if not ANTERIOR:
        return nova
    velha = pd.read_csv(Path(ANTERIOR) / nome / corpus / "test" / "entities.csv",
                        keep_default_na=False, dtype=str)
    if len(nova) != len(velha):
        raise Parar(f"{nome}: {len(nova)} linhas contra {len(velha)}")
    for c in velha.columns:
        n = int((nova[c].values != velha[c].values).sum())
        if n:
            raise Parar(f"{nome}: coluna {c!r} difere em {n} linhas entre as remedições")
    return nova


def montar(nome: str, corpus: str, did: str) -> tuple[Path, dict]:
    notas: dict = {}
    e = conferir_remedicao(nome, corpus)
    notas["remedicao"] = ("idêntica em todas as colunas anteriores, como texto cru"
                          if ANTERIOR else "não reconferida aqui; ver NOTAS.json publicado")
    e["linha_id"] = np.arange(len(e))
    notas["inventadas_descartadas"] = int((e["ancorada"] == "0").sum())
    e = e[e["ancorada"] == "1"].copy()

    # AggSeq
    ag = pd.read_csv(DADOS / nome / corpus / "test" / "aggseq.csv", keep_default_na=False, dtype=str)
    rep = ag.duplicated(["sentence_id", "mencao", "rotulo"]).sum()
    if rep:
        raise Parar(f"{nome}: {rep} chaves repetidas na tabela do feixe")
    j = e.merge(ag[["sentence_id", "mencao", "rotulo", "aggseq"]],
                on=["sentence_id", "mencao", "rotulo"], how="left")
    if len(j) != len(e):
        raise Parar(f"{nome}: a junção com o feixe mudou o número de linhas")
    notas["aggseq_ausente_de_todo_feixe"] = int(j["aggseq"].isna().sum())
    j["aggseq"] = j["aggseq"].fillna("0")
    e = j

    # sondas, por linha_id
    s = pd.read_csv(DADOS / nome / corpus / "test" / "sondas.csv")
    s["linha_id"] = s["linha_id"].astype(int)
    e["linha_id"] = e["linha_id"].astype(int)
    e = e.merge(s[["linha_id"] + SONDAS], on="linha_id", how="left")
    if e[SONDAS].isna().any().any():
        raise Parar(f"{nome}: {int(e[SONDAS].isna().any(axis=1).sum())} linhas sem sonda")

    # linha de base treinada, adenda 01
    prereg = load_preregistration(RAIZ / "configs" / f"{did}.yaml")
    cal, aval = _separar_por_sentenca({"sentence_id": e["sentence_id"].to_numpy(),
                                       "idx": np.arange(len(e))}, prereg)
    X = e[["model_confidence", "span_size", "n_tokens", "span_position"]].astype(float).to_numpy()
    y = (1 - e["loss"].astype(float)).astype(int).to_numpy()
    r = ajustar(X, y, cal["idx"], prereg.seed)
    if "erro" in r:
        raise Parar(f"{nome}: base treinada: {r['erro']}")
    e["base_treinada"], e["base_treinada_ajustada"] = r["fixa"], r["ajustada"]
    notas["base_treinada_C_cv"] = r["C_escolhido"]

    e["span_mass"] = e["causal_mass_prompt"]
    d = SAIDA / nome / corpus / "test"
    (d / "attention").mkdir(parents=True, exist_ok=True)
    e[COLUNAS_TABELA].to_csv(d / "entities.csv", index=False)
    # adenda 02, item 5: T = n_tokens
    L = e.drop_duplicates("sentence_id")[["sentence_id", "n_tokens"]]
    # AO LADO de entities.csv, e não dentro de attention/: é onde
    # `sentence_lengths()` procura (Path(attention_dir).parent). O diretório
    # attention/ existe só porque o executor recebe o caminho dele.
    L.to_csv(d / "sentence_lengths.csv", index=False)
    notas["n_linhas_julgadas"] = int(len(e))
    notas["n_sentencas"] = int(L.shape[0])
    return d, notas


def adenda01(prereg):
    L = []
    for sonda in SONDAS:
        base = "base_treinada_ajustada" if sonda.endswith("_ajustada") else "base_treinada"
        for tipo, crit, rot in (("verdict", "paired_delta_aurc_ci_excludes_zero", "A"),
                                ("task_verdict", "paired_review_load_ci_excludes_zero", "R")):
            L.append(Comparison(id=f"AD1-{rot}-{sonda}", kind=tipo,
                                question=f"{sonda} contra {base}",
                                score=sonda, against=base, criterion=crit))
    return dataclasses.replace(prereg, comparisons=tuple(L))


def _linhas(rel, ponto, origem):
    out = []
    for l in rel.linhas:
        out.append({"ponto": ponto, "origem": origem, **l})
    return out


def main() -> None:
    geo, tar, notas_all = [], [], {}
    for nome, corpus, did, rot in PONTOS:
        print(f"=== {nome} ({did}) ===", flush=True)
        d, notas = montar(nome, corpus, did)
        p = load_preregistration(RAIZ / "configs" / f"{did}.yaml")
        notas["declaration_hash"] = p.declaration_hash
        notas["measurement_hash"] = p.measurement_hash
        rg = run_declared_comparisons(d, p, corpus)
        rt = run_task_impact(d, p, corpus, n_gold=OURO[corpus])
        geo += _linhas(rg, nome, did); tar += _linhas(rt, nome, did)
        pa = adenda01(p)
        ra = run_declared_comparisons(d, pa, corpus)
        rta = run_task_impact(d, pa, corpus, n_gold=OURO[corpus])
        geo += _linhas(ra, nome, "adenda-01"); tar += _linhas(rta, nome, "adenda-01")
        notas["executor"] = list(rg.notas) + list(rt.notas)
        notas_all[nome] = notas
        print("  ", json.dumps({k: v for k, v in notas.items() if k != "executor"},
                               ensure_ascii=False), flush=True)
    SAIDA.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(geo).to_csv(SAIDA / "decoder_aurc.csv", index=False)
    pd.DataFrame(tar).to_csv(SAIDA / "decoder_carga.csv", index=False)
    (SAIDA / "NOTAS.json").write_text(json.dumps(notas_all, ensure_ascii=False, indent=1),
                                      encoding="utf-8")
    print(f"\nlinhas AURC {len(geo)} | linhas carga {len(tar)} | {SAIDA}", flush=True)


if __name__ == "__main__":
    main()
