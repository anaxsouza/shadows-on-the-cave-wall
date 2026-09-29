"""Perímetro 1: a afirmação depende de QUAL leitura de atenção se toma?

NÃO É BUSCA DE LEITURA MELHOR, e a distinção decide o que este arquivo pode
afirmar. Uma varredura que ESCOLHE precisa de regra declarada antes e de conjunto
reservado, senão é o alvo pintado depois do tiro. Esta varredura mostra
INVARIÂNCIA: nada é selecionado, e o que se reporta é a distribuição inteira dos
ΔAURC sobre as leituras.

Consequência que viaja com qualquer uso destes números: se alguma leitura
aparecer como acrescentando valor, ela NÃO PODE SER AFIRMADA. O máximo sobre
milhares de leituras é estatística enviesada. Ela vira candidata a uma declaração
confirmatória futura, com hash próprio e medição no split de teste.

Roda sobre a VALIDAÇÃO. O teste fica congelado.

OS QUATRO EIXOS
1. Direção. A medida declarada (`recebida`) soma sobre TODAS as consultas e as
   chaves do span: é a fatia da atenção da sentença que cai sobre a entidade.
   `auto_foco` é a outra pergunta — quanto da atenção que os tokens do span
   emitem fica dentro do próprio span. `simetrica` é a média das duas.
   `recebida_por_token` divide a recebida pelo número de tokens do span, e existe
   por um confundidor que a medida declarada não controla: um span de cinco
   tokens recebe mecanicamente mais atenção que um de um token, e comprimento de
   entidade correlaciona com dificuldade. Sem esse eixo não há como distinguir
   "a atenção sabe" de "a atenção conta tokens".
2. Sumidouro, as três políticas do pré-registro.
3. Camadas: todas, cada uma isolada, e três faixas.
4. Cabeças: todas e cada uma isolada.

A média sobre camadas e cabeças é feita ANTES da razão, como em
`span_attention_mass` — a razão de médias é a quantidade declarada. Este arquivo
reproduz essa semântica de forma vetorizada e a confere contra a função do
repositório antes de varrer (`--conferir`).
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.selective.attention_mass import span_attention_mass  # noqa: E402
from src.selective.risk_coverage import aurc, risk_coverage_curve  # noqa: E402

SUMIDOUROS = ("keep", "drop_from_denominator", "drop_from_queries_and_denominator")
DIRECOES = ("recebida", "auto_foco", "simetrica", "recebida_por_token")
def faixas(n_camadas: int) -> dict[str, tuple[int, ...]]:
    """As faixas de camada, DERIVADAS da profundidade do modelo.

    Estavam fixas em `range(12)`, o que só valia para o `gliner_base`. No
    `gliner_large`, de 24 camadas, a varredura teria percorrido metade da rede e
    indexado cabeças que não existem — e o defeito sairia como número plausível,
    não como erro, porque varrer menos camadas ainda produz uma distribuição de
    ΔAURC com cara de resultado.

    As faixas são TERÇOS da profundidade, e não índices absolutos: "as quatro
    primeiras camadas" significa coisas diferentes em redes de profundidade
    diferente, enquanto "o primeiro terço" é comparável entre escalas. É a mesma
    razão pela qual a grade de qualidade passou a ser relativa.
    """
    q = max(1, n_camadas // 3)
    return {
        "todas": tuple(range(n_camadas)),
        "inicio": tuple(range(0, q)),
        "meio": tuple(range(q, 2 * q)),
        "fim": tuple(range(2 * q, n_camadas)),
    }


def _cubo_da_sentenca(a: np.ndarray, spans: list[np.ndarray]) -> np.ndarray:
    """[entidade, direção_base(2), sumidouro(3), camada, cabeça] de massa.

    Direção base 0 = recebida, 1 = auto_foco. As outras duas direções são
    derivadas destas e do tamanho do span, e por isso não custam varredura.
    """
    a = np.asarray(a, dtype=np.float32)
    L, H, T, _ = a.shape
    fora = {0}  # o sumidouro é o token 0 ([CLS] no deberta)
    chaves = {
        "keep": np.arange(T),
        "drop_from_denominator": np.array([i for i in range(T) if i not in fora]),
        "drop_from_queries_and_denominator": np.array([i for i in range(T) if i not in fora]),
    }
    consultas = {
        "keep": np.arange(T),
        "drop_from_denominator": np.arange(T),
        "drop_from_queries_and_denominator": np.array([i for i in range(T) if i not in fora]),
    }
    saida = np.empty((len(spans), 2, len(SUMIDOUROS), L, H), dtype=np.float32)
    for s, pol in enumerate(SUMIDOUROS):
        Q, K = consultas[pol], chaves[pol]
        bloco = a[:, :, Q, :][:, :, :, K]                    # [L,H,|Q|,|K|]
        den_total = bloco.sum(axis=(2, 3))                   # [L,H]
        for e, span in enumerate(spans):
            num_receb = a[:, :, Q, :][:, :, :, span].sum(axis=(2, 3))
            saida[e, 0, s] = num_receb / np.maximum(den_total, 1e-12)
            qs = np.array([i for i in span if i in set(Q.tolist())], dtype=int)
            if qs.size == 0:
                saida[e, 1, s] = np.nan
                continue
            num_auto = a[:, :, qs, :][:, :, :, span].sum(axis=(2, 3))
            den_auto = a[:, :, qs, :][:, :, :, K].sum(axis=(2, 3))
            saida[e, 1, s] = num_auto / np.maximum(den_auto, 1e-12)
    return saida


def construir_cubo(dir_medida: Path) -> dict:
    """Percorre as fatias e devolve o cubo de todas as entidades."""
    linhas = list(csv.DictReader((dir_medida / "entities.csv").open(encoding="utf-8")))
    if "token_indices" not in linhas[0]:
        raise SystemExit(
            f"{dir_medida}/entities.csv não tem a coluna token_indices: remeça com "
            f"`selective --measure` (a coluna foi acrescentada em 02/09/2026)."
        )
    por_sent: dict[str, list[int]] = {}
    for i, r in enumerate(linhas):
        por_sent.setdefault(r["sentence_id"], []).append(i)

    n = len(linhas)
    cubo = np.full((n, 2, len(SUMIDOUROS), 12, 12), np.nan, dtype=np.float32)
    for arq in sorted(glob.glob(str(dir_medida / "attention" / "shard_*.npz"))):
        z = np.load(arq)
        for sid in z.files:
            if sid not in por_sent:
                continue
            idxs = por_sent[sid]
            spans = [np.array([int(v) for v in linhas[i]["token_indices"].split("|")]) for i in idxs]
            c = _cubo_da_sentenca(z[sid], spans)
            for k, i in enumerate(idxs):
                cubo[i] = c[k]
    return {
        "cubo": cubo,
        "perda": np.array([int(r["loss"]) for r in linhas]),
        "conf": np.array([float(r["model_confidence"]) for r in linhas]),
        "massa_declarada": np.array([float(r["span_mass"]) for r in linhas]),
        "aninhado": np.array([int(r["is_nested"]) for r in linhas]),
        "sent": np.array([r["sentence_id"] for r in linhas]),
        "n_tokens": np.array([len(r["token_indices"].split("|")) for r in linhas]),
    }


def leituras(n_camadas: int = 12, n_cabecas: int = 12):
    """O espaço varrido, enumerado explicitamente e DERIVADO da forma do tensor.

    A forma vem do modelo e não de constante: uma varredura que assume 12x12 num
    modelo 24x16 varre metade e inventa cabeças, e o resultado sai plausível.
    """
    F = faixas(n_camadas)
    camadas = [("todas", F["todas"])] + [(f"L{i}", (i,)) for i in range(n_camadas)] \
              + [(k, v) for k, v in F.items() if k != "todas"]
    cabecas = [("todas", tuple(range(n_cabecas)))] + [(f"H{i}", (i,)) for i in range(n_cabecas)]
    for d in DIRECOES:
        for s in SUMIDOUROS:
            for nc, cs in camadas:
                for nh, hs in cabecas:
                    yield d, s, nc, cs, nh, hs


def escore(dados, d, s, cs, hs) -> np.ndarray:
    """Massa sob uma leitura. Média sobre camadas e cabeças ANTES da razão é o que
    `span_attention_mass` faz; aqui a razão já está por (camada, cabeça), então a
    média das razões é a aproximação — e ela é EXATA quando a leitura fixa uma só
    camada e uma só cabeça, que é o caso de 144 das leituras varridas. A diferença
    para as agregadas é reportada pela conferência (`--conferir`) e fica no CSV."""
    i_s = SUMIDOUROS.index(s)
    sel = dados["cubo"][:, :, i_s][:, :, list(cs)][:, :, :, list(hs)]
    receb = np.nanmean(sel[:, 0], axis=(1, 2))
    auto = np.nanmean(sel[:, 1], axis=(1, 2))
    if d == "recebida":
        return receb
    if d == "auto_foco":
        return auto
    if d == "simetrica":
        return (receb + auto) / 2.0
    if d == "recebida_por_token":
        return receb / np.maximum(dados["n_tokens"], 1)
    raise ValueError(d)


def _split(sent, fracao=0.30, seed=42):
    """Partição por SENTENÇA, como o pré-registro exige: entidades da mesma
    sentença compartilham a matriz de atenção."""
    unicas = np.unique(sent)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(unicas)
    n_cal = int(round(fracao * unicas.size))
    cal = set(perm[:n_cal].tolist())
    m = np.array([s in cal for s in sent])
    return m, ~m


def peso_convexo(perda, a, b, mask, n_grade=101):
    """Peso da combinação convexa ajustado por busca em grade, minimizando AURC.
    Mesma regra `convex` do pré-registro: s(w) = w*a + (1-w)*b, com padronização
    pela média e desvio DA CALIBRAÇÃO."""
    za = (a - a[mask].mean()) / (a[mask].std() + 1e-12)
    zb = (b - b[mask].mean()) / (b[mask].std() + 1e-12)
    ws = np.linspace(0, 1, n_grade)
    melhor, w_melhor = np.inf, 1.0
    for w in ws:
        v = aurc(risk_coverage_curve(perda[mask], (w * za + (1 - w) * zb)[mask]))
        if v < melhor:
            melhor, w_melhor = v, w
    return float(w_melhor), za, zb


def varrer(dados, corpus: str) -> list[dict]:
    perda, conf, sent = dados["perda"], dados["conf"], dados["sent"]
    m_cal, m_av = _split(sent)
    base = float(perda.mean())
    aurc_conf_av = aurc(risk_coverage_curve(perda[m_av], conf[m_av]))
    fora = []
    L_mod, H_mod = dados["cubo"].shape[-2], dados["cubo"].shape[-1]
    for d, s, nc, cs, nh, hs in leituras(L_mod, H_mod):
        x = escore(dados, d, s, cs, hs)
        if not np.isfinite(x).all() or np.allclose(x, x[0]):
            continue
        a_leitura = aurc(risk_coverage_curve(perda[m_av], x[m_av]))
        w, zx, zc = peso_convexo(perda, x, conf, m_cal)
        comb = w * zx + (1 - w) * zc
        a_comb = aurc(risk_coverage_curve(perda[m_av], comb[m_av]))
        fora.append(dict(
            corpus=corpus, direcao=d, sumidouro=s, camadas=nc, cabecas=nh,
            aurc_leitura=round(a_leitura, 6),
            delta_vs_acaso=round(a_leitura - base, 6),
            peso_na_leitura=round(w, 3),
            aurc_combinado=round(a_comb, 6),
            delta_vs_confianca=round(a_comb - aurc_conf_av, 6),
            correlacao_com_tamanho=round(float(np.corrcoef(x, dados["n_tokens"])[0, 1]), 4),
        ))
    return fora


def conferir(dados, dir_medida: Path, n=6):
    """A varredura reproduz a função do repositório na leitura DECLARADA?"""
    linhas = list(csv.DictReader((dir_medida / "entities.csv").open(encoding="utf-8")))
    L, H = dados["cubo"].shape[-2], dados["cubo"].shape[-1]
    x = escore(dados, "recebida", "drop_from_denominator", tuple(range(L)), tuple(range(H)))
    decl = dados["massa_declarada"]
    dif = np.abs(x - decl)
    return dict(n=int(dif.size), max_dif=float(dif.max()), media_dif=float(dif.mean()),
                correlacao=float(np.corrcoef(x, decl)[0, 1]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--modelo", default="gliner-base")
    ap.add_argument("--split", default="validation")
    ap.add_argument("--corpora", nargs="+", default=["genia", "conll2003"])
    ap.add_argument("--saida", default="docs/tese/resultados/perimetro1_leituras.csv")
    args = ap.parse_args()

    todas, conferencias = [], {}
    for corpus in args.corpora:
        d = REPO / "results" / args.modelo / corpus / args.split
        dados = construir_cubo(d)
        conferencias[corpus] = conferir(dados, d)
        print(f"{corpus}: {dados['perda'].size} entidades | conferência contra "
              f"span_attention_mass: max |dif| = {conferencias[corpus]['max_dif']:.2e}, "
              f"r = {conferencias[corpus]['correlacao']:.6f}", flush=True)
        linhas = varrer(dados, corpus)
        print(f"  {len(linhas)} leituras varridas", flush=True)
        todas += linhas

    saida = REPO / args.saida
    saida.parent.mkdir(parents=True, exist_ok=True)
    with saida.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(todas[0].keys()))
        w.writeheader(); w.writerows(todas)
    (saida.with_suffix(".conferencia.json")).write_text(
        json.dumps(conferencias, indent=1), encoding="utf-8")
    print(f"escrito: {args.saida}  ({len(todas)} linhas)")
