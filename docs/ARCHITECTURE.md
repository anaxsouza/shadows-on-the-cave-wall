# ARCHITECTURE.md — SENTINEL

**S**equential **E**ntity **N**eural **T**opology **I**nvestigation & **N**esting **E**valuation for **L**LMs

> Authoritative architecture reference. Every structural decision lives here.
> When code and this document disagree, this document wins — update the code to match.
> The thesis (`writing/shared/chapters/` for Ch01–03, `writing/thesis/chapters/` for Ch04–07) is the
> canonical source for experimental rationale and thresholds.

---

## 1. Project Identity

SENTINEL is the experimental framework developed for the DSc thesis:

> *"Doctor, I'm not a programmer, I don't know the model!": A Mechanistic Analysis of Language Model Architectures in Entity Recognition*

**Author:** Anaximandro Anderson Pereira Melo de Souza
**Advisor:** Prof. Rogerio Pinto Espindola, D.Sc.
**Institution:** COPPE/UFRJ
**Defense:** April 2027

### Acronym

| Letter | Word | Research Content |
|--------|------|-------------------|
| **S** | Sequential | H1's sequential processing bottleneck in autoregressive models |
| **E** | Entity | Named Entity Recognition as the primary experimental task |
| **N** | Neural | Transformer architectures as the subject of analysis |
| **T** | Topology | Mechanistic interpretability: circuits, attention maps, information flow |
| **I** | Investigation | Analytical methodology — causal interrogation, not just benchmarking |
| **N** | Nesting | Nested entity recognition (GENIA dataset, central to H1) |
| **E** | Evaluation | Comparative assessment across 6 models and 3 datasets |
| **L** | LLMs | The subject models under study |

### Research Question

One question. The thesis works around it, and every chapter, module and test
exists to answer it:

> Does span-attention mass add decision power over the confidence the model
> already provides, enabling per-entity abstention — and is the gain larger for
> nested entities than for flat ones?

### Two tests of it

| Test | What it decides | Criterion | Failure means |
|---|---|---|---|
| **floor** | Does span-attention mass rank errors at all? | Paired-bootstrap CI for ΔAURC vs random abstention excludes zero | The thesis ends; there is nothing to add to the model's confidence |
| **added-value** | Is the gain incremental over the model's own confidence, and larger where entities nest? | Paired-bootstrap CI for ΔAURC vs confidence alone excludes zero, and the nested gain exceeds the flat gain | The signal is redundant with what the model already reports |

These are tests of one question, not hypotheses. They are not numbered, they do
not combine by a counting rule, and there is no multiplicity correction because
the criteria are interval-based rather than p-value based (Section 8).

The previous design made seven simultaneous claims (H1.1–H1.4, H2.1–H2.3) with a
"3 of 4" convergence rule and a Bonferroni α = 0.0125. It was abandoned after
the qualifying examination; see `docs/tese/03_alternativas.md` for the decision
and `docs/tese/reorg/corte.md` for what it cost in code.

---

## 2. Stack

| Layer | Choice | Version |
|-------|--------|---------|
| Language | Python | 3.10+ |
| Deep Learning | PyTorch | >= 2.0.0 |
| Transformers | HuggingFace Transformers | pinned in pyproject.toml |
| Datasets | HuggingFace Datasets | pinned |
| Metrics | seqeval | pinned |
| Statistics | scipy, statsmodels | pinned |
| Visualization | matplotlib, seaborn | pinned |
| Console | Rich | pinned |
| Config | PyYAML | pinned |
| Package Manager | uv | uv.lock |

**Stack standard:** `~/.config/opencode/standards/python.md`

---

## 3. Module Structure

`src/` holds 78 Python files and 27,672 lines. Two movements produced that, and reporting
only their net would hide both: the 2026-09-02 cut removed 48 files and 17,871 lines from a
tree of 119 files and 44,158 (40.5% of the code), and `src/selective/` then added 6 files
and 1,360 lines — the contribution, which did not exist before. Net against the old tree:
−41 files, −16,486 lines (37.3%). The measurement is reproducible with
`python tools/reorg_c4_inventory.py`; what left is itemized in `docs/tese/reorg/corte.md`.

```
sentinel/                                     # Repository root
├── AGENTS.md                                 # Agent protocol for this repo
├── main.py                                   # Root entry point → src/cli/main.py
├── configs/config.yaml                       # Single source of truth (206 lines)
├── tests/                                    # 23 files, anti-drift suite
├── tools/reorg_c4_inventory.py               # Measures the repo against the question
├── docs/
│   ├── ARCHITECTURE.md                       # This file — authoritative architecture
│   ├── log.md                                # Changelog
│   └── tese/                                 # Post-qualification scope decision
│       ├── 01..05_*.md                       # Objections, literature, alternatives, C4
│       ├── PREREGISTRO.md                    # The six pre-registered items
│       └── reorg/                            # Inventory, cut dossier, test baseline
└── src/
├── cli/
│   ├── commands/
│   │   ├── __init__.py
│   │   └── selective.py                    # único comando: --test floor|added-value
│   ├── console/
│   ├── __init__.py
│   ├── __main__.py
│   └── main.py
├── config/
│   ├── __init__.py
│   └── manager.py
├── core/
│   ├── discovery/
│   ├── fine_tuning/
│   ├── interfaces/
│   │   ├── __init__.py
│   │   ├── batch.py
│   │   ├── dataset.py
│   │   └── processor.py
│   ├── loaders/                            # CoNLL-2003 (plano) e GENIA (aninhado)
│   │   ├── base/
│   │   │   ├── __init__.py
│   │   │   └── loader.py
│   │   ├── biomedical/
│   │   │   ├── __init__.py
│   │   │   └── genia.py
│   │   ├── conll/
│   │   │   ├── __init__.py
│   │   │   └── loader.py
│   │   ├── crossner/
│   │   └── __init__.py
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── batch_processor.py
│   │   └── ner_pipeline.py                 # encoder-only; falha alto em outra arquitetura
│   ├── processors/
│   │   ├── base/
│   │   │   ├── __init__.py
│   │   │   └── processor.py
│   │   ├── decoder/
│   │   ├── encoder/
│   │   │   ├── __init__.py
│   │   │   └── gliner.py
│   │   ├── encoder_decoder/
│   │   └── __init__.py
│   └── __init__.py
├── experiments/
│   ├── base/
│   │   ├── __init__.py
│   │   └── evaluation.py                   # a passagem que produz as predições
│   ├── hypotheses/
│   └── __init__.py
├── infrastructure/
│   ├── device/
│   │   ├── __init__.py
│   │   └── manager.py
│   ├── logging/
│   │   ├── __init__.py
│   │   ├── experiment_logger.py
│   │   └── formatters.py
│   ├── profiling/
│   │   ├── __init__.py
│   │   └── profiler.py
│   └── __init__.py
├── selective/                              # A CONTRIBUIÇÃO — predição seletiva
│   ├── __init__.py
│   ├── attention_mass.py                   # massa de atenção sobre o span (o instrumento)
│   ├── conformal.py                        # controle conformal de risco (a garantia)
│   ├── preregistration.py                  # os seis itens, sem valor padrão
│   ├── risk_coverage.py                    # curva risco-cobertura, AURC, ΔAURC pareado
│   └── runner.py                           # os dois testes: floor e added-value
├── shared/
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── analysis_utils.py
│   │   ├── metrics_calculator.py
│   │   └── xai_utils.py                    # AttentionAnalyzer — mapeia span→token
│   ├── data/
│   │   ├── __init__.py
│   │   ├── bio_extraction.py
│   │   ├── data_types.py
│   │   ├── data_utils.py
│   │   ├── dataset_profiler.py
│   │   ├── example_miner.py
│   │   └── nesting_utils.py
│   ├── experiments/
│   │   ├── __init__.py
│   │   └── scientific_constants.py
│   ├── metrics/
│   │   ├── __init__.py
│   │   ├── base_results_collector.py
│   │   ├── metrics_calculator.py
│   │   ├── multi_model_results_collector.py
│   │   └── single_model_results_collector.py
│   ├── model/
│   │   ├── __init__.py
│   │   ├── confidence_utils.py             # ConfidenceCalculator — agregação e ECE
│   │   ├── config_manager.py
│   │   ├── dynamic_prompt_engine.py
│   │   └── model_utils.py
│   ├── __init__.py
│   ├── local_cache.py
│   ├── naming_utils.py
│   └── reproducibility.py
├── utils/
├── visualization/
│   ├── __init__.py
│   ├── attention.py
│   ├── base.py
│   ├── config.py
│   ├── ner.py
│   └── research_utils.py
└── __init__.py
```

---

## 4. Architecture Patterns

### Layered Architecture
```
CLI (scripts/) → Pipeline → Processors → Model
```

### Strategy Pattern
- **Processors:** `DecoderOnlyProcessor`, `EncoderDecoderProcessor`, `EncoderOnlyGLiNERProcessor` — each implements the NER processing strategy for its architecture family
- **Loaders:** `CONLLLoader`, `GENIALoader`, `CrossNER*Loader` — each implements dataset loading for its dataset

### Factory Pattern
- Processor selection is done via `_create_processor()` in `NERPipeline` (architecture-type dispatch)
- `get_dataset_loader()` in `src/core/loaders/__init__.py`

### Adapter Pattern
- `src/core/fine_tuning/adapters.py` — architecture-specific model wrapping for fine-tuning
- 3 adapters: EncoderAdapter, DecoderAdapter, EncoderDecoderAdapter

### Facade Pattern
- No application facade — scripts call pipeline/processors directly

### SOLID Principles
- **S**ingle Responsibility: Each module has one reason to change
- **O**pen/Closed: Extend via new processors/loaders, not conditional chains
- **L**iskov: All processors implement the same NER interface
- **I**nterface Segregation: Focused interfaces in `src/core/interfaces/`
- **D**ependency Inversion: Pipeline depends on abstractions, injected via factories

---

## 5. Main Data Flow

```
config.yaml
    │
    ▼
ConfigManager.load()
    │
    ▼
DatasetFactory.get_loader(dataset_key)
    │
    ▼
BaseDatasetLoader.load_split(split="test", max_samples=N)
    │
    ▼
NERExample[]  (src/shared/data/data_types.py)
    │
    ▼
ProcessorFactory.get_processor(model_key)
    │
    ▼
processor.predict(examples) / processor.extract_attention(examples)
    │
    ▼
Results → ExperimentResultsCollector → JSON output (results/)
```

---

## 6. Model Registry

> **Environment note, measured 2026-09-15.** `gliner_large` loads only with
> `OMP_NUM_THREADS=1`. With 4 or 8 threads the process dies with SIGSEGV during
> model construction, even with `KMP_DUPLICATE_LIB_OK=TRUE` set — two OpenMP
> runtimes end up in one process (conda's `libomp` via numpy/scipy, and the one
> inside the torch wheel), and the larger model is the one that does not survive
> it. `gliner_base` happened to survive, which is why this only surfaced now.
> The model loads fine standalone at any thread count; the crash needs the CLI's
> import order. Single-threaded costs wall time on a long run and there is no
> alternative on this machine — it is an environment defect, not a repository
> one, so it is recorded here rather than worked around in code.

**One model per declaration, and the reason is the question itself.** The
question compares a model's attention against *that same model's* confidence, so
both must come from one set of weights. The invariant is therefore not "one model
in the registry" — it is that every declaration fixes exactly one model, which
`hash_version: 4` makes enforceable by putting `model` inside the measurement
hash.

| Key | Model ID (HuggingFace) | Encoder | Layers x Heads | Parameters | Declaration |
|-----|------------------------|---------|---------------:|-----------:|-------------|
| `gliner-base` | `urchade/gliner_base` | `deberta-v3-base` | 12 x 12 | 209M | `decl-01` … `decl-03` |
| `gliner-large` | `urchade/gliner_large` | `deberta-v3-large` | 24 x 16 | 445M | `decl-04-escala` |
| `gliner_base-ft-genia` | fine-tuned from `gliner_base` | `deberta-v3-base` | 12 x 12 | 197M | `decl-05-ajustado-genia` |
| `gliner_base-ft-conll2003` | fine-tuned from `gliner_base` | `deberta-v3-base` | 12 x 12 | 197M | `decl-06-ajustado-conll` |
| `gliner_base-ft-bc5cdr` | fine-tuned from `gliner_base` | `deberta-v3-base` | 12 x 12 | 197M | `decl-19-bc5cdr-ajustado` |

The registry grew to two entries on 2026-09-15, and the reason is an objection
rather than an ambition: the confirmatory result is negative, and the most
predictable attack on it is not about attention but about the *extractor*, which
errs 48.2% (GENIA) and 53.1% (CoNLL-2003). A second scale is the cheapest test of
that objection — same architecture, same methodology, only scale. Layer and head
counts were probed from the checkpoints' `config.json`, not assumed.

`decl-04` declares all 24 layers because what stays constant across scales is the
*rule* ("all layers"), not the index range. Declaring 0-11 on the larger model
would use half the network and let a depth effect masquerade as a scale effect.

Until 2026-09-02 this registry listed `bert-large` and `deberta-v3-base`. Neither
finds entities: both are bare encoders (`ner_specific: false`), and
`processors/encoder/gliner.py` hardcoded a *third* model to do the predicting,
ignoring the requested key entirely. Attention from one model explaining
confidence from another is not an imprecision — it is a meaningless measurement,
and a reviewer rejects it on first reading. Both were retired; the guard against
their return is `tests/test_config_model_consistency.py`.

**Why a span-scoring model and not a tagger.** GLiNER scores *spans* against
label descriptions instead of tagging tokens, which is what lets it predict
nested entities. BIO tagging cannot: nesting would need two labels on one token.
Since half the question is whether the gain is larger for nested entities, a
tagger answers only the flat half.

**The cheap robustness check that is deliberately not here.** A second GLiNER
size (`gliner_large`) varies scale without touching the architecture, and is the
least expensive generality check available. It stays out for now because adding
it changes runtime and the pre-registered layer range, which is a decision to
take explicitly rather than by accretion.

The decoder-only (qwen2.5-1.5b/3b) and encoder-decoder (t5-base, bart-base) arms
were removed on 2026-09-02. The architectural contrast they served left the
thesis, and it was that contrast — not the analysis — that made the work
GPU-dependent: the single question is answered with forward passes on CPU.

## 7. Dataset Registry

2 corpora, and the pair is the design rather than a limit on effort: the
contrast between flat and nested is half of the question.

| Key | Dataset ID (HuggingFace) | Domain | Entities | Role in the question |
|-----|--------------------------|--------|----------|----------------------|
| `conll2003` | `eriktks/conll2003` | News (general) | Flat PER/ORG/LOC/MISC | The flat side of the contrast |
| `genia` | `Aunderline/genia` | Biomedical | Nested (up to depth 3) | The nested side; where the gain should be larger |

The five CrossNER domains were removed on 2026-09-02. They served domain drift,
which is a different question from this one.

---

## 8. Experiment Specification

One question, two tests of it. This section replaces the seven sub-hypothesis
specifications (H1.1–H1.4, H2.1–H2.3) that occupied it until 2026-09-02, along
with the GPS framework and the overtraining controls that served them. Recover
any of it with `git show pre-c4-reorg:docs/ARCHITECTURE.md`; the reasoning behind
the cut is in `docs/tese/reorg/corte.md`.

### The question

> Does span-attention mass add decision power over the confidence the model
> already provides, enabling per-entity abstention — and is the gain larger for
> nested entities than for flat ones?

Both tests below are tests **of this question**. They are not hypotheses, they
are not numbered, and they do not combine into a verdict by a counting rule.
There is nothing here to count: a single question is either answered or not.

### No multiplicity correction, and why that is not a gap

The previous design needed α = 0.05/4 = 0.0125 because it made four
simultaneous claims and then asked how many had to hold (the "3 of 4" rule).
Neither the family nor the correction exists any more, and the reason is not
that the correction was dropped — it is that **the criteria are interval-based,
not p-value based**. Each test is decided by whether a bootstrap confidence
interval excludes zero at the pre-registered level, and the operating point is
decided by conformal risk control at the pre-registered α. There is no family of
p-values over which to correct.

Anyone reintroducing a Bonferroni α into this repository is reintroducing the
conjunction; `tests/test_threshold_contract.py` fails if they do.

### Test: floor

**Question decided:** does span-attention mass rank errors at all?

**Procedure.** Compute the risk–coverage curve of span-attention mass over
predicted entities, and compare its AURC against random abstention. The
comparison baseline is a constant score, whose AURC is exactly the base error
rate — random abstention does not change conditional selective risk at any
coverage, so its curve is a horizontal line.

**Success criterion.** The paired-bootstrap CI for ΔAURC (mass − random) excludes
zero, at the pre-registered level, in the negative direction (lower AURC is
better).

**Consequence of failure.** The thesis ends here. If the instrument does not rank
error, there is nothing to add to the model's own confidence, and the
added-value test is not run.

**Implementation.** `src/selective/risk_coverage.py`,
`src/selective/attention_mass.py`; entry point
`sentinel selective --test floor`.

### Test: added-value

**Question decided:** is the gain incremental over the confidence the model
already provides, and larger where entities nest?

**Procedure.** Combine the model's own per-entity confidence with span-attention
mass by the pre-registered combination rule, **fitted on the calibration
partition**, and compare the AURC of the combined score against the AURC of the
model's confidence alone on the evaluation partition. Report the same difference
separately for nested and flat entities.

**Success criterion.** The paired-bootstrap CI for ΔAURC (combined − confidence
alone) excludes zero in the negative direction, and the point estimate of the
gain is larger for nested entities than for flat ones.

**What the stratified intervals do NOT establish.** Two separate per-stratum
intervals are not a test of the difference between strata. Reading
non-overlapping intervals as a significant difference is the error this table
invites, and `SelectiveResult.render()` says so in the output itself.

**Implementation.** `src/selective/runner.py`; entry point
`sentinel selective --test added-value`.

### Operating point, with a guarantee

Conformal risk control (arXiv 2208.02814) selects the threshold with the
**largest coverage** whose empirical risk satisfies `R̂(λ) ≤ α − (B−α)/n` on the
calibration partition. Three properties of this guarantee must travel with any
sentence that cites it:

1. It is **marginal, not conditional**: it bounds the expected risk over draws
   of the calibration partition. It does not state that risk on this particular
   run fell below α.
2. It requires **exchangeability** between calibration and evaluation. Holds
   within one corpus under a random split; does **not** hold across corpora, so
   thresholds are never transferred from CoNLL-2003 to GENIA.
3. It requires the empirical risk to be **monotone** in the threshold — which is
   precisely what the floor test investigates, so it cannot be assumed.
   `crc_threshold` verifies it and reports `monotone=False` with the largest
   violation; when that happens the returned threshold does not carry the
   theorem's guarantee, and the thesis must say so rather than cite the theorem.

### Pre-registered parameters

Six items, declared in `configs/config.yaml` section `selective` and mirrored in
`docs/tese/PREREGISTRO.md`. **There is no default for any of them in code**:
`load_preregistration` raises and nothing runs if the section is absent. A
default in code is a choice someone may have made after seeing a result, which
is exactly objection R4.

| # | Item | Config key | Value |
|---|---|---|---|
| 1 | Per-entity confidence aggregation | `entity_confidence_aggregation` | `min` |
| 2 | Attention-sink convention | `sink_policy` | `drop_from_denominator` |
| 2 | Layer range | `layers` | 8–15 |
| 2 | Head range | `heads` | all |
| 3 | Combination rule | `combination_rule` | `logistic` |
| 3 | Calibration partition | `calibration_fraction` | 0.3 of **sentences** |
| 4 | Reported coverage levels | `coverage_levels` | 0.5, 0.7, 0.8, 0.9, 0.95 |
| 4 | Single operating point | `operating_point_coverage` | 0.8 |
| 5 | Added-value CI level | `added_value_ci_level` | 0.95 |
| 5 | Bootstrap resamples | `n_bootstrap_resamples` | 2000 |
| 6 | Conformal risk level | `conformal_alpha` | 0.1 |

The calibration partition is split **by sentence, not by entity**. Two entities
in one sentence share the same attention matrix; splitting them across
partitions leaks context, and that leak does not show up as an error — it shows
up as a good result.

### Inconclusive verdict

A test is inconclusive, rather than negative, when the bootstrap CI half-width
exceeds the absolute point estimate of ΔAURC — the sample cannot resolve the
sign of the effect. Report as inconclusive; do not report as absence of effect.

---

## 9. Configuration System

**File:** `configs/config.yaml` — single source of truth

**ExperimentConfigManager:** `src/config/manager.py`

### Structure

```yaml
general:          # Seed, device, log_level
models:           # 6 model entries with architecture, HF name, params
                  # (BERT-Large, DeBERTa-v3-base, Qwen2.5-1.5B/3B-Instruct, T5-Base, BART-Base
                  #  — matches thesis §3.4.4 L918 "six H1 model architectures")
datasets:         # 7 dataset entries with HF name, splits
experiments:      # Experiment-specific configs — ONLY 2 of 8 blocks present
                  # (base_ner, h1_attention_degradation). Every other threshold lives
                  # hardcoded in a module SUCCESS_THRESHOLDS dict, so editing config.yaml
                  # cannot change them. See the config-connectivity note below.
fine_tuning:      # Training hyperparameters, checkpoint collection, output paths
model_cache:      # Cache directories (base_models_dir, fine_tuned_dir)
analysis:         # Analysis thresholds (attention, entropy, attribution)
checkpoint_collection:  # Enabled, percentages, save_eval
eval:             # Evaluation config
```

### Config–code connectivity

> ⚠️ **The configuration layer is largely inert for experiment thresholds.** Only
> `base_ner` and `h1_attention_degradation` have `experiments:` blocks. Success thresholds for
> H1.1, H1.3, H1.4, H2.1, H2.2 and H2.3 are hardcoded in per-module `SUCCESS_THRESHOLDS`
> dicts and are not read from config. Known inert keys:
> - `analysis.position_weight: 0.1` — `calculate_entity_complexity` never reads config; the
>   effective value is the function default **0.5** (`xai_utils.py` L422)
> - `SUCCESS_THRESHOLDS["spearman_rho"]` in `h2_entropy_divergence.py` L82 — declared, never read
>
> Anti-drift coverage for this is in `tests/test_doc_code_alignment.py`.

## 10. Permanent Constraints

These must NEVER be violated:

### Import Locations (correct paths)

| Class | Location | Do NOT import from |
|-------|----------|--------------------|
| `ModelManager` | `src/shared/model/model_utils.py` | — |
| `EnhancedExperimentLogger` | `src/infrastructure/logging/experiment_logger.py` | `src/shared/logging/` (directory does not exist) |
| `NERExample`, `Entity` | `src/shared/data/data_types.py` | `src/utils/data/` (does not exist) |
| `StatisticalAnalyzer` | `src/shared/analysis/analysis_utils.py` | — |
| `AttentionAnalyzer` | `src/shared/analysis/xai_utils.py` | `src/utils/analysis/xai_utils` |
| `ConfidenceCalculator` | `src/shared/model/confidence_utils.py` | — |
| `span_attention_mass` | `src/selective/attention_mass.py` | — |
| `risk_coverage_curve`, `aurc`, `delta_aurc_paired_bootstrap` | `src/selective/risk_coverage.py` | — |
| `crc_threshold` | `src/selective/conformal.py` | — |
| `load_preregistration` | `src/selective/preregistration.py` | — |
| `SelectiveRunner` | `src/selective/runner.py` | `src/experiments/runner.py` (removed 2026-09-02) |
| `DeviceManager` | `src/infrastructure/device/manager.py` | — |
| `ConfigManager` | `src/config/manager.py` | — |
| `ReproducibilityManager` / `set_seed` | `src.shared.reproducibility` | `src/utils/reproducibility` |

### Modules removed on 2026-09-02 (never re-create these imports)

| Import | Served |
|---|---|
| `src.core.fine_tuning.*` | the training cycle |
| `src.shared.analysis.ifc_analyzer` (`IFCAnalyzer`) | H1.3 |
| `src.shared.analysis.ffn_analyzer` (`FFNAnalyzer`) | H1.4 |
| `src.shared.analysis.comparative_analyzer` | cross-architecture comparison |
| `src.experiments.hypotheses.*` | H1.1–H1.4, H2.1–H2.3 |
| `src.experiments.runner` (`ExperimentRunner`) | the conjunction's orchestrator |
| `src.experiments.gps` | the Generalised Performance Score |
| `src.utils.overtraining_controls` | overtraining controls |
| `src.core.processors.decoder.*`, `src.core.processors.encoder_decoder.*` | the architectural contrast |
| `src.core.loaders.crossner.*` | the five CrossNER domains |
| `src.core.discovery.*` | model/dataset discovery for the removed CLI |
| `src.infrastructure.device.resource_profiler` (`ResourceProfiler`) | fine-tuning resource planning |
| `src.shared.json_io` | a single legacy consumer |

Recover any of them with `git show pre-c4-reorg:<path>`; `tests/test_import_integrity.py`
fails if one comes back.

### Loader method signatures

| Correct | Wrong |
|---------|-------|
| `loader.load_split(split="test", max_samples=N)` | `loader.load(split="test", max_samples=N)` |
| `loader.load_raw_dataset(split="test")` | — |
| `loader.convert_to_examples(dataset)` | — |

### Architecture constraints

- `NERPipeline._create_processor` accepts `encoder-only-base` only, and raises with the
  reason on anything else. Do NOT add a branch: the architectural contrast left the thesis.
- `src/core/__init__.py` must NOT wrap its imports in `try/except ImportError`. It did
  until 2026-09-02, which turned a missing dependency into a logged warning and a silently
  degraded package — that is invisible to the anti-drift suite.
- `src/selective/attention_mass.py` and `risk_coverage.py` depend on numpy only. The
  instrument and the measure of the thesis must stay testable without loading a model.

### Removed files (Refactoring v5 — 2026-05-28)

The following files were removed during the thesis-alignment refactoring. Do NOT re-create them:

| File | Lines | Reason |
|------|-------|--------|
| `src/core/abc.py` | 272 | Orphaned ABCs — superseded by `core/interfaces/` |
| `src/core/environment.py` | 50 | Orphaned setup — never imported |
| `src/core/resources.py` | 95 | Orphaned ResourceManager — never imported |
| `src/visualization/statistical.py` | 737 | StatisticalVisualizer never instantiated by any experiment |
| `src/shared/analysis/visualizer_plots.py` | 173 | Visualizer class — dead chain with results_manager.py |
| `src/shared/analysis/results_manager.py` | 176 | ResultsManager — never imported by active code |
| `scripts/` (6 files) | ~2,100 | DEPRECATED — superseded by `src/cli/main.py` subcommands in PLANS v1-v4 |
| `src/cli/parser.py` | 208 | DEPRECATED — superseded by `src/cli/main.py` v2 CLI |
| `requirements.txt` | — | Superseded by `pyproject.toml` + `uv.lock` in Phase 10 |

### Data constraints

- `GENIALoader` uses HuggingFace dataset `Aunderline/genia`
- `CONLLLoader` uses `eriktks/conll2003`
- Calibration and evaluation partitions are split BY SENTENCE. Splitting by entity
  leaks attention context between partitions, and the leak surfaces as a good result
  rather than as an error.

---

## 11. Known Issues

Revised on 2026-09-02: every entry whose subject left the repository was removed
rather than marked resolved, because they were not fixed — the code they described
is gone. The audit trail stays in `git show pre-c4-reorg:docs/ARCHITECTURE.md`.

| # | Issue | Status |
|---|-------|--------|
| K1 | 116 broad `except Exception:` blocks in `src/`, several in data-loading paths, can swallow errors silently | OPEN. Same class of defect as the `try/except ImportError` removed from `src/core/__init__.py`: a failure that logs and continues is invisible to the anti-drift suite. |
| K2 | `trust_remote_code=True` on dataset loading (`src/core/loaders/base/loader.py`) | OPEN. Security risk; both remaining corpora are standard HuggingFace datasets and may not need it. |
| K3 | Model scale confound: bert-large (340M) vs deberta-v3-base (184M), no size-matched control | OPEN, and now smaller in consequence: the thesis compares each model's own signals against each other within the model, never absolute values across models. |
| K4 | `tests/verify_figure_naming.py` patches `src.visualization.base_visualizer`, an attribute the package does not have | OPEN, pre-existing. Recorded in `docs/tese/reorg/baseline_testes.txt` so it is not mistaken for a regression of the cut. |
| K5 | The per-entity table that `src/selective/runner.py` consumes had no producer | **CLOSED 2026-09-02.** `src/selective/measurement.py` emits `entities.csv` plus the full attention tensor (all layers, all heads, float16) in sharded `.npz`, with a pre-flight disk estimate that refuses the run before writing a byte if the projection exceeds the declared budget. The layer range comes from the pre-registration and is not an argument of `measure()`, so storing every layer does not enable re-choosing the range after seeing a result. What remains is the real-model adapter. |
| K6 | `NERVisualizer` defines **8 methods twice** in the same class (`_plot_general_performance_per_entity`, `_plot_performance_by_length`, `_plot_performance_by_position`, `_plot_sequence_length_vs_performance`, `_find_best_prediction_match`, `_record_frequency_effect`, `_analyze_token_boundary_precision`, `_is_compound_entity`). Each later definition silently shadows the earlier one, and the surviving `_plot_general_performance_per_entity` does not accept `metadata`, so it cannot apply the figure-naming convention | OPEN, predates the C4 reorganization. Deciding which of each pair survives is a code review of ~1,000 lines of plotting code. `tests/verify_figure_naming.py` declares it as `xfail(strict=True)`, so it fails loudly when fixed. |
| K8 | `tests/test_reproducibility.py` carrega `bert-large`, que saiu do roster em 2026-09-02 | OPEN. Os dois testes passam, mas verificam a reprodutibilidade de um modelo que a tese não usa; devem apontar para `gliner-base`. |
| K9 | A suíte passou de ~2 s para ~7 min quando `huggingface.co` foi liberado | OPEN. Os dois testes de reprodutibilidade respondem por 424 dos 426 s: antes eram pulados por falta de rede, agora baixam e executam pesos. Suíte de 7 min não é rodada a cada mudança, e guarda que não se roda não é guarda — precisam de marcador opt-in. |
| K7 | `src/visualization/attention.py` still plots the abandoned design (entropy × complexity correlation, `_plot_h1_*`) | OPEN. It survived the cut because it also plots confidence, which the single question uses. Needs the same treatment as the rest: keep what serves the question, cut the rest. |

## 12. Environment Setup

### Prerequisites

- Python 3.10–3.13
- GPU: NVIDIA (CUDA 11.8+) or Apple Silicon (MPS)
- Disk: 100GB+ free (models: ~20GB base + ~25GB fine-tuned)
- Internet: HuggingFace Hub access

### Installation

This project uses **uv** for reproducible dependency management (no `requirements.txt`).

```bash
# Install uv (if not installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create environment and install all dependencies
uv sync

# (Optional) Install optional groups for notebooks or XAI analysis
uv sync --group notebook --group xai

# Activate the virtual environment
source .venv/bin/activate

# (Optional) Install package in editable mode
pip install -e .
```

### Verification

```bash
# Python
python --version  # >= 3.10

# GPU
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, MPS: {torch.backends.mps.is_available()}')"

# Unified CLI (after pip install -e .)
sentinel list models
sentinel evaluate --model bert-large --dataset conll2003

# Unified CLI (without install, from project root)
uv run python main.py list models

# Config
uv run python -c "from src.config.manager import ExperimentConfigManager; c = ExperimentConfigManager(); print(len(c.get_models())); print(c.get('datasets').keys())"

# Seed reproducibility
uv run python -c "from src.shared.reproducibility import set_seed; import numpy as np; set_seed(42); a=np.random.rand(3); set_seed(42); b=np.random.rand(3); print('OK' if (a==b).all() else 'FAIL')"

# Datasets (smoke test)
uv run python -c "from datasets import load_dataset; d=load_dataset('eriktks/conll2003', split='train[:1%]'); print(len(d))"

# Output directories
mkdir -p results models/fine_tuned models/cache logs
```

### Common Issues

| Symptom | Fix |
|---------|-----|
| CUDA Out of Memory | `--training.batch_size 16` |
| MPS not available | Reinstall PyTorch: `uv pip install torch --upgrade --force-reinstall` |
| HF rate limit | `export HF_TOKEN=your_token` or `huggingface-cli login` |

---

## 13. Structural Change History

| Date | Change | Reason |
|------|--------|--------|
| 2026-05-27 | Phase 10: uv environment management + lazy import chain. See `docs/log.md` for full task breakdown. [UPDATES: pyproject.toml, ARCHITECTURE.md §14] | uv lockfile replaces ad-hoc pip; lazy import enables CLI without statsmodels/seaborn |
| 2026-05-27 | Phase 9: Root `main.py` entry point delegates to `src/cli/main.py`. Removed `main.py.broken`. Updated README.md Quick Start to use `python main.py`. | Restore single root entry point eliminated in Phase 3; make repo self-documenting for new contributors |
| 2026-05-26 | Docs refactoring: removed `code_plan.md`, `SENTINEL.md`, `EVOLUTION_HISTORY.md`, `preflight_checklist.md`; created `ARCHITECTURE.md`, `log.md`, `AGENTS.md` | Single source of truth aligned with thesis |
| 2026-05-27 | Phase 2: Consolidated experiment dispatch into `src/experiments/runner.py`; `scripts/run_experiments.py` is now a thin CLI wrapper | Decouple CLI from experiment logic — runner importable from Python API |
| 2026-05-27 | Unified CLI entry point: `src/cli/main.py` (6 subcommands: train/evaluate/experiment/list/checkpoints/recover), `src/cli/__main__.py`, `[project.scripts]` in pyproject.toml (`sentinel` command), `src/cli/commands/checkpoints.py`, `src/cli/commands/recover.py`. Removed broken `main.py`. All `scripts/*.py` and `src/cli/parser.py` marked DEPRECATED. | Eliminate 6 fragmented entry points, enable `pip install -e .` experience |
| 2026-05-26 | GCP infrastructure removed (`sentinel/gcp/`, `configs/gcp_config.yaml`, `scripts/gcp_train.py`, `docs/gcp_setup.md`) | Project runs on local GPU/MPS only |
| 2026-05-26 | Model list corrected: `bart-base` replaces `gliner-large` | Matches thesis (`ch04_proposed_work.tex`) and `config.yaml` |
| 2026-02-28 | `code_plan.md` v4.0 created | Comprehensive experimental specification |
| 2026-02-15 | Module structure finalized (`src/core/`, `src/shared/`, `src/infrastructure/`) | SOLID refactoring |
| 2026-01-20 | Repository initialized; SENTINEL acronym formalized | PhD thesis framework kickoff |
| 2026-05-26 | Phase 1: ConfigManager→ExperimentConfigManager, 3 class extractions from analysis_utils.py (MetricsCalculator, Visualizer, ResultsManager), dead code removal in model_utils.py (8 imports + 7 methods) | Split large files, eliminate dead code |
| 2026-05-27 | Phase 4: Fixed 6 experiment wiring bugs (B1–B6). B1: `run_h2_1` wired to `EntropyDivergenceAnalyzer.run()`. B2: `run_h2_2` constructor fixed + `analyzer.run(domain_examples)`. B3: IG `target=token_idx` semantic bug fixed — uses `label_id` as target. B4: `run_h2_3` removed `output_dir` from `H2IntegrationProtocol()`. B5: `run_controls` wired with stub eval callbacks. B6: hardcoded `architecture_type="encoder"` fallback uses `model_key` when metadata missing | Eliminate all 8 CRITICAL and 5 HIGH blocking bugs — experiments now wire end-to-end |
| 2026-05-27 | Phase 5: Replaced 8 bare `except:` blocks with specific exception types; removed deprecated `calculate_shannon_entropy()` (9216 warnings eliminated); replaced wrapper with `compute_attention_entropy()`; created `results/` directory | Code quality — eliminate bare excepts that catch SystemExit/KeyboardInterrupt; remove deprecated API 15 months past removal date |
| 2026-05-27 | Phase 6: Narrowed 14 broad `except Exception` blocks to specific exception types across trainer.py, versioning.py, base.py, ner.py, research_utils.py, attention.py. Left 24 as `Exception` where justified (outermost handlers, graceful degradation, external callbacks) | Code quality — eliminate silent error masking in broad catches |
| 2026-05-27 | Phase 7: Added 5 new test files (conftest.py, test_cli_commands.py, test_data_types.py, test_loader_data.py, test_analysis_metrics.py, test_results_manager.py). 66 new tests covering CLI dispatch, Entity/NERExample dataclasses, CONLL/GENIA loader conversion logic, MetricsCalculator F1/entropy/complexity, ResultsManager load/compare/analyze | Expand test coverage from 62→128 tests across previously untested modules |
| 2026-05-27 | Phase 8: Final documentation update — ARCHITECTURE.md §15 and log.md entries. Dead import scan (0 stale refs). CLI entry point verified. | Close out all structural cleanup phases |
| 2026-05-28 | Refactoring v5: Thesis-alignment dead code removal. Removed 6 orphan/side-effect files (1,503 lines): `core/abc.py`, `core/environment.py`, `core/resources.py`, `visualization/statistical.py`, `shared/analysis/visualizer_plots.py`, `shared/analysis/results_manager.py`. Cleaned `__init__.py` exports. Rewrote 3 anti-drift tests to target new CLI/runner. Added `gliner`, `sentencepiece`, `tiktoken` to pyproject.toml. Fixed skipped `test_4_ifc_gradient_computation`. Removed `test_results_manager.py`. Full test suite: 119/119 pass, 0 fail, 0 skip. | Eliminate ~2,481 LOC of dead code; align repository with thesis requirements; unblock encoder-only zero-shot evaluation |
| 2026-05-28 | Decoder pipeline optimization: Added Qwen2.5-1.5B/3B + Mistral-7B models (3 new decoder-only entries in registry, config, and fine-tuning). Fixed temperature/top_p wiring to model.generate() across all decoder strategies. Added use_cache=True to all generation calls. Removed GQA dead tensor allocation. Removed encoder-only-ner dead code path. Set padding_side=left for decoder tokenizers. Fixed processor.py hardcoded do_sample=False. Capped generation max_new_tokens per architecture type. [UPDATES: configs/config.yaml §models/§generation/§token_safety, src/shared/model/model_utils.py, src/core/processors/decoder/strategies.py, src/core/processors/decoder/processor.py, docs/ARCHITECTURE.md §6/§11/§12/§14/§15, writing/latex/chapters/ch04_proposed_work.tex] | Replace failing GPT-2/LLaMA decoders with instruct-tuned alternatives; fix critical inference bugs that prevent decoder evaluation and mechanistic analysis |
| 2026-09-02 | **Reorganization around a single question (C4).** Removed 48 code files and 17,871 lines (40.5% of `src/`): `experiments/hypotheses/` and `runner.py` (the H1.1–H1.4 / H2.1–H2.3 orchestrator), `core/fine_tuning/` and `resource_profiler.py`, the decoder and encoder-decoder processors, the five CrossNER loaders, `ifc_analyzer.py`, `ffn_analyzer.py`, `comparative_analyzer.py`, `gps.py`, `overtraining_controls.py`, `visualization/comparative.py`, `core/discovery/`, `shared/json_io.py`, and every CLI command except one. Cut the four import edges that dragged the abandoned design into the core: the re-exports in `core/loaders/__init__.py` and `shared/analysis/__init__.py`, the 3-way processor factory in `ner_pipeline.py`, and `comparative_analyzer → visualization/comparative`. Added `src/selective/` (6 files, 1,360 lines): span-attention mass, risk–coverage/AURC, conformal risk control, and the pre-registration as a data structure with no defaults in code. Removed 12 test files of the abandoned design and inverted `test_cli_commands.py`, `test_no_orphan_config.py` and `test_config_model_consistency.py` into negative assertions that fail if the conjunction returns. `config.yaml` 608 → 206 lines. `src/core/__init__.py` no longer swallows `ImportError`. [UPDATES: docs/ARCHITECTURE.md §1/§3/§5-§8/§10/§11, configs/config.yaml, AGENTS.md, docs/tese/PREREGISTRO.md, docs/tese/reorg/] | The qualifying committee's 162 objections concentrated on a design that made seven simultaneous claims; the thesis now works around one question, and "less is more" was the explicit instruction. Baseline before the cut: tag `pre-c4-reorg` plus `docs/tese/reorg/baseline_testes.txt` |
