# Verification record

Everything below ran in an environment built from `uv.lock` alone (Python 3.12, macOS
arm64), from this repository, without access to the development repository.

## 2026-10-05, revised submission (code v2.0-submission)

| level | what | result |
|---|---|---|
| 0 | `pytest tests/` | 401 pass, 2 skipped, 1 expected xfail |
| 0b | `tools/registro_publico.py --conferir` | identical to the deposited registry (version 3) |
| 1b | `reproduzir/artigo.py`: numbers, both stratum tables, null check, and the three tables added in the revision (prediction, error types, control windows) | **7 of 7 files byte-identical** |
| 2 | `tools/impressao_digital.py`: the 6 weights added in the revision (BERT x3, GLiNER and Qwen2.5 x2 on BC5CDR), downloaded from the training jobs before packing | 6 of 6 match `PESOS.json` |

`reproduzir/analise.py` (level 1) was not rerun for this release.

## 2026-09-28, before submission

| level | what | result |
|---|---|---|
| 0 | `pytest tests/` | 350 pass, 3 skipped, 1 expected xfail |
| 0b | `tools/registro_publico.py --conferir` | identical to the deposited registry |
| 1 | `reproduzir/analise.py`: 16 confirmatory tables (decl-02 to decl-13 + addenda) | **16 of 16 identical**, cell by cell; 26 min |
| 1b | `reproduzir/artigo.py`: numbers, both stratum tables and null check of the article | **4 of 4 files byte-identical** |
| — | the manuscript compiles from `artigo/` alone (generic version and, with the journal's class files, the CL version) | yes |
| 2 | `tools/empacotar_pesos.py`: 6 weights fingerprinted before packing for deposit | 6 of 6 match `PESOS.json` |

## 2026-09-24, first release candidate

| level | what | result |
|---|---|---|
| 0 | `pytest tests/` | 350 pass, 3 skipped, 1 expected xfail |
| 1 | `reproduzir/analise.py`: 16 confirmatory tables | **16 of 16 identical**, cell by cell |
| 1 | `reproduzir/derivadas.py`: 4 derived tables | 4 of 4 identical (one after a change of seed; see `docs/tese/resultados/monotonicidade_NOTA.md`) |
| 1 | decl-01 (pilot), through the executor | point estimates, weights and 4 verdicts identical; intervals differ in the 3rd decimal place (see REPRODUCE.md) |
| 2 | `tools/impressao_digital.py`: 6 weights | 6 of 6 match `PESOS.json` |
| 2 | `tools/exportar_corpus_teste.py`, from the public corpora | 2 of 2 files byte-identical to the manifest |
| 2 | `reproduzir/remedir.py gliner_base-ft-genia__sinais`, 30 sentences, same CPU | 0.0 difference in every column |
| 2 | `reproduzir/remedir.py qwen05b-ft-genia`, complete, CPU against the published GPU measurement | **predictions identical in all 5,572 rows** (span, label, correctness, indices, hallucinations); attention up to 2.3×10⁻⁶; hidden states up to 8.8×10⁻⁴; confidence up to 5.6×10⁻⁵; 2 h 2 min |

## What was NOT verified

- Complete remeasurement of the other five weights (the GLiNER weights were remeasured in
  decl-07/08, with predictions identical to decl-05/06 in 11,115 rows; the three remaining
  Qwen weights were not).
- Level 3: retraining.
- Platforms other than macOS arm64.
- Fitting the probes from the features (`features_sonda.npz`, 1.4 GB, not packaged);
  `sondas.csv` is read as data.
