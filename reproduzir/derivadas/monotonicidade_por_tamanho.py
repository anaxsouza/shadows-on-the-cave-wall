# Recuperado da linhagem do artefato que produziu a peça publicada; só os CAMINHOS
# foram trocados para o repositório. Rode por reproduzir/derivadas.py, que o executa
# em saida/derivadas/ e compara o que ele escreve com o publicado.
from pathlib import Path as _P
RAIZ = _P(__file__).resolve().parents[2]
import sys
import csv
import zlib

import numpy as np
import pandas as pd
import yaml
import time
from pathlib import Path

# Replicate the repository src path setup
sys.path.insert(0, str(RAIZ / "src"))

from selective.comparisons import sentence_lengths, _carregar_tabela, _escores_geometricos
from selective.conformal import crc_threshold

# Load config
decl_text = open(str(RAIZ) + "/configs/decl-03-tarefa.yaml").read()
decl = yaml.safe_load(decl_text)["selective"]

SCORES = ["model_confidence", "span_mass", "enrichment"]
ALPHA = decl["conformal_alpha"]
LOSS_BOUND = decl["loss_bound"]
SEED_BASE = decl["seed"]

class _MiniPrereg:
    def __init__(self, sink_policy):
        self.sink_policy = sink_policy

prereg_mini = _MiniPrereg(decl["sink_policy"])

# We need to load tables from disk paths since _carregar_tabela / sentence_lengths
# read from directory structures (attention subdir, entities.csv, etc.)
# The artifacts map entities.csv but the code uses full directory paths.
# We reconstruct the needed data by pointing to the validation dirs.

base = RAIZ

tables = {}
for corpus in ["genia", "conll2003"]:
    table_dir = base / f"dados_medidos/gliner-base/{corpus}/validation"
    lengths = sentence_lengths(table_dir / "attention")
    t = _carregar_tabela(table_dir, lengths)
    _escores_geometricos(t, prereg_mini)
    tables[corpus] = t

FRACOES_REPS = [
    (0.05, 200), (0.10, 200), (0.15, 150), (0.20, 150), (0.30, 100),
    (0.40, 100), (0.50, 80), (0.70, 50), (0.90, 30), (1.00, 1),
]

resultados = []
t0 = time.time()
for corpus, t in tables.items():
    sent_ids_all = np.unique(t["sentence_id"])
    n_sent_total = sent_ids_all.size
    sid_arr = t["sentence_id"]

    for escore_nome in SCORES:
        escores_full = t[escore_nome]
        perdas_full = t["loss"]
        # SEMENTE ESTÁVEL. O original usava hash((corpus, escore_nome)), e o hash de
        # texto do Python muda a cada processo (PYTHONHASHSEED): a tabela publicada em
        # 18/09 não pode ser refeita por ninguém, nem por este código. crc32 é o mesmo
        # em toda máquina e todo processo.
        rng = np.random.default_rng(SEED_BASE + zlib.crc32(f"{corpus}|{escore_nome}".encode()) % 10_000)
        for frac, n_reps in FRACOES_REPS:
            n_sent = max(5, int(round(frac * n_sent_total)))
            n_sent = min(n_sent, n_sent_total)
            reps = 1 if n_sent == n_sent_total else n_reps
            for rep in range(reps):
                if n_sent == n_sent_total:
                    escolhidas = sent_ids_all
                else:
                    escolhidas = rng.choice(sent_ids_all, size=n_sent, replace=False)
                m = np.isin(sid_arr, escolhidas)
                perdas_m, escores_m = perdas_full[m], escores_full[m]
                if np.unique(perdas_m).size < 2:
                    continue
                res = crc_threshold(perdas_m, escores_m, alpha=ALPHA, loss_bound=LOSS_BOUND)
                resultados.append(dict(
                    corpus=corpus, escore=escore_nome, fracao=frac, n_sentencas=n_sent,
                    n_entidades=int(m.sum()), rep=rep,
                    violacao=res.max_monotonicity_violation, monotono=res.monotone,
                    feasible=res.feasible,
                ))
        print(corpus, escore_nome, "feito,", f"{time.time()-t0:.1f}s decorridos")

print("total de linhas:", len(resultados))

df = pd.DataFrame(resultados)
agg = df.groupby(["corpus", "escore", "fracao"]).agg(
    n_sentencas_medio=("n_sentencas", "mean"),
    n_entidades_medio=("n_entidades", "mean"),
    n_reps=("violacao", "size"),
    violacao_media=("violacao", "mean"),
    violacao_mediana=("violacao", "median"),
    violacao_p90=("violacao", lambda s: s.quantile(0.90)),
    frac_monotona=("monotono", "mean"),
).reset_index()

pd.set_option("display.width", 160)
pd.set_option("display.max_rows", 100)

cols = ["corpus", "escore", "fracao", "n_sentencas_medio", "n_entidades_medio", "n_reps",
        "violacao_media", "violacao_mediana", "violacao_p90", "frac_monotona"]
agg_out = agg[cols].sort_values(["corpus", "escore", "fracao"]).copy()
agg_out.to_csv("monotonicidade_por_tamanho.csv", index=False)
print("monotonicidade_por_tamanho.csv", agg_out.shape)