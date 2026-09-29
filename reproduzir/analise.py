"""NÍVEL 1 DE REPRODUÇÃO: refaz todos os vereditos a partir das tabelas medidas.

    python reproduzir/analise.py

Lê só `dados_medidos/` (publicado, sem texto de corpus) e `configs/decl-*.yaml`.
Não carrega modelo, não precisa de GPU, e roda em minutos numa CPU comum.

Para cada declaração, roda os executores do projeto — os mesmos que produziram as
tabelas publicadas, sem nenhum critério reimplementado aqui — e escreve o resultado
em `saida/`. Depois compara CÉLULA A CÉLULA cada tabela regerada com a publicada em
`docs/tese/resultados/`, e termina com código 1 se qualquer célula diferir.

Números são comparados como o MESMO float, e não por tolerância: "igual até a sexta
casa" deixaria passar exatamente a diferença que muda um veredito na fronteira, um
intervalo que toca zero.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
from src.selective.comparisons import run_declared_comparisons  # noqa: E402
from src.selective.preregistration import load_preregistration  # noqa: E402
from src.selective.task_impact import run_task_impact  # noqa: E402

DADOS = RAIZ / "dados_medidos"
PUB = RAIZ / "docs" / "tese" / "resultados"
SAIDA = RAIZ / "saida"
OURO = {"genia": 5506, "conll2003": 5648}   # entidades de ouro no teste de cada corpus

# declaração -> pontos medidos, tabela de AURC, tabela de carga de revisão, filtro
# da tabela publicada (as de decl-05 e decl-06 dividem o mesmo arquivo)
CASOS = [
    ("decl-02-geometria", [("gliner-base", "genia"), ("gliner-base", "conll2003")],
     "confirmatorio_decl02.csv", None, None),
    ("decl-03-tarefa", [("gliner-base", "genia"), ("gliner-base", "conll2003")],
     None, "confirmatorio_decl03_tarefa.csv", None),
    ("decl-04-escala", [("gliner-large", "genia"), ("gliner-large", "conll2003")],
     "confirmatorio_decl04_geometria.csv", "confirmatorio_decl04_tarefa.csv", None),
    ("decl-05-ajustado-genia", [("gliner_base-ft-genia", "genia")],
     "confirmatorio_ajustado_geometria.csv", "confirmatorio_ajustado_tarefa.csv", "genia"),
    ("decl-06-ajustado-conll", [("gliner_base-ft-conll2003", "conll2003")],
     "confirmatorio_ajustado_geometria.csv", "confirmatorio_ajustado_tarefa.csv", "conll"),
    ("decl-07-sinais-genia", [("gliner_base-ft-genia__sinais", "genia")],
     "confirmatorio_sinais_ajustado_geometria.csv", "confirmatorio_sinais_ajustado_tarefa.csv", "genia"),
    ("decl-08-sinais-conll", [("gliner_base-ft-conll2003__sinais", "conll2003")],
     "confirmatorio_sinais_ajustado_geometria.csv", "confirmatorio_sinais_ajustado_tarefa.csv", "conll"),
    ("decl-09-sinais-large", [("gliner-large", "genia"), ("gliner-large", "conll2003")],
     "confirmatorio_sinais_geometria.csv", "confirmatorio_sinais_tarefa.csv", None),
]
# Colunas que a tabela publicada tem e o executor não produz: acrescentadas pelo
# script que a escreveu (nome do modelo, família, estatuto do nulo, corpus). Não são
# resultado, e ficam fora da comparação.
ANOTACOES = {"modelo", "familia", "estatuto_nulo", "corpus"}


def _filtrar(pub: pd.DataFrame, filtro: str | None) -> pd.DataFrame:
    if filtro is None or "modelo" not in pub:
        return pub
    return pub[pub["modelo"].astype(str).str.contains(filtro)].reset_index(drop=True)


def _como_texto(df: pd.DataFrame) -> pd.DataFrame:
    """A tabela como o CSV a grava: é a forma em que a publicada existe.

    Passar pelo CSV não é cosmético. No pandas 3, `astype(str)` mantém o valor
    ausente como ausente, e uma célula vazia da tabela regerada passava a diferir do
    `""` da publicada. Foi o que fez 14 tabelas idênticas parecerem divergentes na
    primeira execução deste script, em 24/09/2026.
    """
    import io
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    return pd.read_csv(buf, dtype=str, keep_default_na=False)


def comparar(novo: pd.DataFrame, pub: pd.DataFrame) -> list[str]:
    novo, pub = _como_texto(novo), _como_texto(pub)
    if len(novo) != len(pub):
        return [f"{len(novo)} linhas regeradas contra {len(pub)} publicadas"]
    difs = []
    for c in pub.columns:
        if c in ANOTACOES:
            continue
        if c not in novo:
            difs.append(f"coluna {c!r} ausente na regerada")
            continue
        a = pd.to_numeric(novo[c], errors="coerce")
        b = pd.to_numeric(pub[c], errors="coerce")
        num = a.notna() & b.notna()
        n = int(((a != b) & num).sum()) + int(((novo[c] != pub[c]) & ~num).sum())
        if n:
            difs.append(f"{c}: {n} células")
    return difs


def main() -> int:
    SAIDA.mkdir(exist_ok=True)
    falhas, rel = 0, []
    for did, pontos, fgeo, ftar, filtro in CASOS:
        p = load_preregistration(RAIZ / "configs" / f"{did}.yaml")
        geo, tar = [], []
        for mod, corpus in pontos:
            d = DADOS / mod / corpus / "test"
            if fgeo:
                geo += run_declared_comparisons(d, p, corpus).linhas
            if ftar:
                tar += run_task_impact(d, p, corpus, n_gold=OURO[corpus]).linhas
        for arq, linhas in ((fgeo, geo), (ftar, tar)):
            if not arq:
                continue
            novo = pd.DataFrame(linhas)
            novo.to_csv(SAIDA / f"{did}__{arq}", index=False)
            pub = _filtrar(pd.read_csv(PUB / arq, dtype=str, keep_default_na=False), filtro)
            difs = comparar(novo, pub)
            falhas += bool(difs)
            rel.append((did, arq, len(novo), "IDÊNTICA" if not difs else "; ".join(difs)))

    # o segundo programa tem executor próprio, que junta feixe e sondas antes de julgar
    r = subprocess.run([sys.executable, str(RAIZ / "tools" / "analisar_decoder.py")],
                       capture_output=True, text=True)
    if r.returncode:
        print(r.stdout[-1500:], r.stderr[-1500:])
        return 1
    for arq in ("decoder_aurc.csv", "decoder_carga.csv"):
        novo = pd.read_csv(SAIDA / "decoder" / arq, dtype=str, keep_default_na=False)
        pub = pd.read_csv(PUB / "decoder" / arq, dtype=str, keep_default_na=False)
        difs = comparar(novo, pub)
        falhas += bool(difs)
        rel.append(("decl-10 a 13 + adendas", arq, len(novo),
                    "IDÊNTICA" if not difs else "; ".join(difs)))

    larg = max(len(a) for _, a, _, _ in rel)
    for did, arq, n, est in rel:
        print(f"{did:26s} {arq:{larg}s} {n:5d} linhas  {est}")
    print(f"\n{len(rel) - falhas} de {len(rel)} tabelas idênticas às publicadas")
    return 1 if falhas else 0


if __name__ == "__main__":
    sys.exit(main())
