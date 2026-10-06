"""Análise declarada do BC5CDR (decl-16 a decl-21): os MESMOS executores de GENIA/CoNLL, sem critério novo.

    python tools/analisar_bc5cdr.py encoder   # decl-18 e decl-19 (decl-17: ver nota abaixo)
    python tools/analisar_bc5cdr.py decoder   # decl-20 e decl-21 + adenda-01, via tools/analisar_decoder.py
    python tools/analisar_bc5cdr.py previsao  # tabela de insumo de P1/P2 (decl-16); NÃO julga

NOTA SOBRE A decl-17. A decl-17 é cópia da decl-04 (GLiNER large, 24 camadas) com só `declaration_id`
e `model` trocados para o `urchade/gliner_base`, que tem 12 camadas. O adaptador recusa a medição
(`layers` [0..23] não existe num modelo de 12 camadas). Não se improvisa: a declaração assinada não é
editada e este script não a mede. Ver o relatório.

SAÍDAS, em docs/tese/resultados/ (nada existente é sobrescrito; os nomes são novos):
    confirmatorio_bc5cdr_<did>_geometria.csv, confirmatorio_bc5cdr_<did>_tarefa.csv
    decoder/decoder_aurc_bc5cdr.csv, decoder/decoder_carga_bc5cdr.csv, decoder/NOTAS_bc5cdr.json
    confirmatorio_bc5cdr_previsao_insumos.csv
"""
from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(RAIZ / "tools"))
PUB = RAIZ / "docs" / "tese" / "resultados"
OURO = 9809   # entidades de ouro no teste do BC5CDR (soma sobre as 5.865 sentenças; ver exportar_bc5cdr.py)

ENCODER = [  # (declaração, modelo medido, pasta de medição)
    ("decl-18-bc5cdr-gliner-large", "gliner-large", RAIZ / "results"),
    ("decl-19-bc5cdr-ajustado", "gliner_base-ft-bc5cdr", RAIZ / "results"),
    # decl-26 substitui a decl-17 (inexequivel: layers 0-23 num modelo de 12 camadas); copia da decl-05
    ("decl-26-bc5cdr-gliner-base", "gliner-base", RAIZ / "results"),
]
DECODER = [
    ("qwen05b-ft-bc5cdr", "bc5cdr", "decl-20-bc5cdr-decoder-05b", "05b-bc5cdr"),
    ("qwen15b-ft-bc5cdr", "bc5cdr", "decl-21-bc5cdr-decoder-15b", "15b-bc5cdr"),
]
MEDIDOS_DEC = RAIZ / "saida" / "bc5cdr_medidos"
SAIDA_DEC = RAIZ / "saida" / "bc5cdr_decoder"


def encoder(selecao: set[str] | None = None) -> None:
    from src.selective.comparisons import run_declared_comparisons
    from src.selective.preregistration import load_preregistration
    from src.selective.task_impact import run_task_impact
    for did, modelo, base in ENCODER:
        if selecao and did not in selecao:
            continue
        d = base / modelo / "bc5cdr" / "test"
        if not (d / "entities.csv").exists():
            print(f"{did}: sem medição em {d}"); continue
        p = load_preregistration(RAIZ / "configs" / f"{did}.yaml")
        rg = run_declared_comparisons(d, p, "bc5cdr")
        rt = run_task_impact(d, p, "bc5cdr", n_gold=OURO)
        for sufixo, rel in (("geometria", rg), ("tarefa", rt)):
            df = pd.DataFrame(rel.linhas)
            if "modelo" not in df:
                df.insert(0, "modelo", modelo)
            if "corpus" not in df:
                df.insert(1, "corpus", "bc5cdr")
            alvo = PUB / f"confirmatorio_bc5cdr_{did}_{sufixo}.csv"
            df.to_csv(alvo, index=False)
            print(did, sufixo, len(df), alvo.name)
        (PUB / f"confirmatorio_bc5cdr_{did}_notas.txt").write_text(
            "\n".join(list(rg.notas) + list(rt.notas)) + "\n", encoding="utf-8")


def decoder() -> None:
    os.environ["SENTINEL_DADOS_MEDIDOS"] = str(MEDIDOS_DEC)
    os.environ["SENTINEL_SAIDA_ANALISE"] = str(SAIDA_DEC)
    import analisar_decoder as A
    A.DADOS, A.SAIDA = MEDIDOS_DEC, SAIDA_DEC
    A.OURO = {**A.OURO, "bc5cdr": OURO}
    A.PONTOS = [p for p in DECODER if (MEDIDOS_DEC / p[0] / p[1] / "test" / "entities.csv").exists()]
    A.main()
    (PUB / "decoder").mkdir(exist_ok=True)
    for a, b in (("decoder_aurc.csv", "decoder_aurc_bc5cdr.csv"),
                 ("decoder_carga.csv", "decoder_carga_bc5cdr.csv"),
                 ("NOTAS.json", "NOTAS_bc5cdr.json")):
        shutil.copy(SAIDA_DEC / a, PUB / "decoder" / b)


def previsao() -> None:
    """Insumos de P1/P2: AURC de acaso, massa e fração, na avaliação e na direção fixa. NÃO julga."""
    linhas = []
    fontes = [(PUB / f"confirmatorio_bc5cdr_{did}_geometria.csv", "encoder") for did, *_ in ENCODER]
    fontes.append((PUB / "decoder" / "decoder_aurc_bc5cdr.csv", "decoder"))
    for f, familia in fontes:
        if not f.exists():
            continue
        df = pd.read_csv(f)
        if "ponto" in df:     # decoder: uma tabela com os dois pontos; só a declaração de ponto, não a adenda
            df = df[df["origem"].astype(str).str.startswith("decl-")]
            df = df.assign(modelo=df["ponto"])
        for modelo, g in df.groupby("modelo"):
            g = g[(g["estrato"] == "todos") & (g["particao"] == "avaliação")]
            def v(m):
                s = g[g["metrica"] == m]["valor"]
                return float(s.iloc[0]) if len(s) else None
            ac, ma, fr = v("aurc_acaso"), v("aurc_span_mass"), v("aurc_geometric_fraction")
            linhas.append({
                "modelo": modelo, "familia": familia, "particao": "avaliação", "estrato": "todos",
                "aurc_acaso": ac, "aurc_massa": ma, "aurc_fracao": fr,
                "risco_removido_massa": None if None in (ac, ma) else round(1 - ma / ac, 4),
                "risco_removido_fracao": None if None in (ac, fr) else round(1 - fr / ac, 4),
                "dif_abs_massa_fracao": None if None in (ma, fr) else round(abs(ma - fr), 4),
                "nota": "insumo de P1 (sinais) e P2 (|dif|<=0,004, só bidirecionais); sem veredito"})
    out = pd.DataFrame(linhas)
    out.to_csv(PUB / "confirmatorio_bc5cdr_previsao_insumos.csv", index=False)
    print(out.to_string())


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    {"encoder": lambda: encoder(set(sys.argv[2:]) or None), "decoder": decoder, "previsao": previsao}[cmd]()
