# How to reproduce, and what was verified

Each level assumes the previous one. Next to each command is what it gave when the authors
ran it in a clean environment (see [VERIFICATION.md](VERIFICATION.md) for the dates). If
you obtain something else, that is a finding.

## Environment

Python 3.12 and the exact versions in `uv.lock`:

    uv sync --frozen            # or: pip install -r requirements-lock.txt

The verification environment was built from the lock file alone, with no inherited package.

## Level 0: the test suite

    pytest tests/

**Verified:** 350 pass, 3 are skipped and 1 is an expected xfail. Tests that remained in
the development repository are listed in `tests/FORA_DO_ESCOPO.md`, with the reason.

## Level 0b: the declarations are the deposited ones

    python tools/registro_publico.py --conferir

Compares the SHA-256 of every declaration, addendum and executable configuration with
`docs/tese/declaracoes/REGISTRO_PUBLICO.txt`, the file deposited on Zenodo before
submission (DOI in the article). **Verified:** identical.

## Level 1: every verdict, from the measured tables

    python reproduzir/analise.py      # the 16 confirmatory tables, ~30 min of CPU
    python reproduzir/derivadas.py    # derived tables and figures, ~5 min

`analise.py` runs the executors of the declarations on `dados_medidos/` and compares each
table, cell by cell, with `docs/tese/resultados/`.

**Verified:** the 16 confirmatory tables came out **identical**: 14 for the encoder
(decl-02 to decl-09) and 2 for the decoder (decl-10 to decl-13 with the addenda). Among
the derived tables, `consolidado_tres_modelos`, `monotonicidade_onde_ocorre`,
`monotonicidade_por_tamanho` (see `docs/tese/resultados/monotonicidade_NOTA.md`) and
`resultado_piloto` also came out identical.

**Known divergence: decl-01, the pilot.** Run:

    python main.py selective --test floor       --model gliner-base --dataset genia --output-dir dados_medidos
    python main.py selective --test added-value --model gliner-base --dataset genia --output-dir dados_medidos

(and the same with `--dataset conll2003`). The point estimates, the combination weights and
the four verdicts come out **exactly** as in `resultado_piloto.csv`. The intervals differ in
the third decimal place; for example, the GENIA floor gives [−0.0587, −0.0254] against the
published [−0.0575, −0.0267]. The cause: resampling became sentence-level after the pilot,
and the executor does not keep the old unit for version-1 declarations. No verdict changes.

`resultado_piloto.csv` was written by hand from the executor's output on 2026-09-02. The
script that generates it (`reproduzir/derivadas/resultado_piloto_tabela.py`) contains the
typed numbers, so its comparison is empty of content. The check that counts is the one
given by the commands above.

## Level 1b: every number in the text of the article

    python reproduzir/artigo.py

Regenerates `artigo/numeros.tex` (every number cited in the text), `artigo/tab_estratos.tex`, `artigo/tab_estratos_decoder.tex`
and `artigo/numeros_nulo.tex` (the numerical check of both nulls) and requires them to be
byte-identical to the published files; it also redraws the figures in `artigo/img/`.
**Verified:** 4 of 4 files identical.

## Level 2: the tables come from those weights

1. Download the weights (the two DOIs are in the article) and check each one:

       python tools/impressao_digital.py gliner_base-ft-genia qwen05b-ft-genia ...

   Every line must say `CONFERE` ("matches"). The fingerprints are in `PESOS.json`.

2. The decoder reads the test partition from a `jsonl` that is not distributed, because it
   contains corpus text. Rebuild it from the public corpora:

       python tools/exportar_corpus_teste.py

   **Verified:** both files came out byte-identical to the manifest (`f61f5c8a…` GENIA,
   `125d8654…` CoNLL), with an empty Hugging Face cache.

3. Measure one point and compare:

       python reproduzir/remedir.py gliner_base-ft-genia__sinais gliner_base-ft-genia
       python reproduzir/remedir.py qwen05b-ft-genia qwen05b-ft-genia

   Discrete columns (the predictions) must come out identical; continuous columns are
   reported by their largest difference.

   **Verified:**
   - GLiNER fine-tuned on GENIA, decl-07, on the same CPU where it was measured: a
     difference of 0.0 in every column.
   - Fine-tuned GLiNER, CPU against T4 (decl-07/08 against decl-05/06, same weights):
     predictions identical in all 11,115 rows, and confidence differing by at most 3×10⁻⁵.
   - Qwen 0.5B on GENIA, CPU against the GPU measurement: see `VERIFICATION.md`.

## Level 3: retraining (not verified here)

`tools/treinar_extrator.py` (GLiNER) and `tools/treinar_decoder.py` (Qwen) are the scripts
that produced the weights, with a fixed seed. On GPU, training is not bit-for-bit
deterministic: expect weights with a different fingerprint and a similar validation F1.
The decoder's training files were exported in a way that is not in this repository; their
manifest is in `docs/tese/resultados/procedencia/INSUMOS_decoder.json`, and the exporter is
an open debt.
