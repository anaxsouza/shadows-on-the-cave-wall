# decl-06-ajustado-conll — o extrator competente, para fechar o Perímetro 2

**Estado:** ASSINADA em 2026-09-15. Os valores e critérios abaixo são COMPROMISSO:
nenhum deles muda depois de ver resultado. Analisar sob outros valores exige declaração
NOVA, nunca a edição desta.

| | |
|---|---|
| `declaration_id` | `decl-06-ajustado-conll` |
| Hash da declaração | `7726d98cdefd2ae6` |
| Hash de **medição** | `4658eceda04c48df` |
| Hash de análise | `f4ef835f18e9043c` |
| Modelo | `gliner_base-ft-conll2003` |
| Corpus | CoNLL-2003, partição de teste |
| Camadas × cabeças | 12 × todas |
| Herda medição de | — (nenhuma: exige medição própria) |

---

## Esta declaração não tem nenhum parâmetro livre

E isso é verificável, item por item:

- **Itens de análise**: copiados literalmente da `decl-04`, que os copiou da `decl-02` e
  da `decl-03`. Nenhum foi reescolhido.
- **Camadas**: todas as 12 do `gliner_base`, de que este modelo parte. A regra constante
  nas quatro declarações anteriores é *"todas as camadas"*, não um intervalo de índices.
- **Modelo**: determinado por uma regra de escolha de checkpoint declarada **antes** do
  treino, dentro de `tools/treinar_extrator.py` — melhor F1 estrito na validação, que
  nunca olha ΔAURC nem qualquer quantidade da hipótese.

Não houve o que escolher. A declaração existe porque o **modelo** mudou, e a versão 4 do
conjunto de itens põe o modelo dentro do hash de medição — o que torna impossível testar
uma tabela de outro modelo sob ela.

## O extrator, e por que ele fecha a objeção

A objeção que continuava de pé contra o resultado negativo não era sobre atenção: era
que os modelos medidos eram de prateleira e erravam 42% e 47%. Este é ajustado ao corpus.

| | valor |
|---|---|
| F1 estrito na validação, **antes** | 0.6726 (P 0.6512, R 0.6954) |
| F1 estrito na validação, **depois** | **0.9696** |
| ganho | +0.2970 |
| checkpoint escolhido | `checkpoint-4176` |
| épocas · lote · semente | 3 · 8 · 42 |
| hiperparâmetros | receita padrao do gliner, nao varrida |
| regra de checkpoint | melhor F1 estrito na validacao; nunca olha ΔAURC |

F1 por época, para que a regra seja auditável em vez de afirmada:

| checkpoint | F1 | P | R |
|---|---:|---:|---:|
| `checkpoint-1392` | 0.9557 | 0.9563 | 0.9552 |
| `checkpoint-2784` | 0.9667 | 0.9645 | 0.9690 |
| `checkpoint-4176` | 0.9696 | 0.9690 | 0.9701 |

## Uma declaração por corpus, e a razão é a invariante

Ajuste fino produz um modelo **por corpus**. A invariante do projeto é que cada
declaração fixe **um** modelo, porque a pergunta compara a atenção de um modelo com a
confiança *dos mesmos pesos*. Por isso são duas declarações irmãs,
`decl-05-ajustado-genia` e `decl-06-ajustado-conll`, idênticas exceto no modelo e no
corpus. Uma declaração cobrindo os dois violaria exatamente a invariante que a
`hash_version: 4` existe para tornar verificável.

## As sete comparações

| id | tipo | escore | contra | critério |
|---|---|---|---|---|
| `C1` | descriptive | — | — | — (descritiva) |
| `C2` | verdict | `span_mass` | `geometric_fraction` | `paired_delta_aurc_ci_excludes_zero` |
| `C3` | verdict | `model_confidence+enrichment` | `model_confidence` | `paired_delta_aurc_ci_excludes_zero` |
| `T1` | descriptive | — | — | — (descritiva) |
| `T2` | task_verdict | `span_mass` | `model_confidence` | `paired_review_load_ci_excludes_zero` |
| `T3` | task_verdict | `enrichment` | `model_confidence` | `paired_review_load_ci_excludes_zero` |
| `T4` | task_verdict | `model_confidence+enrichment` | `model_confidence` | `paired_review_load_ci_excludes_zero` |

---

## Assinatura

| | |
|---|---|
| Assinada em | 2026-09-15 |
| Commit da assinatura | `git log --diff-filter=A -1 -- docs/tese/declaracoes/decl-06-ajustado-conll.md` |
| Primeira medição sob ela | posterior a esse commit, por construção |

O hash **não muda** com a assinatura: a linha de estado é comentário nos dois arquivos,
não item declarado.

---

## A fonte, literal

```yaml
selective:
  declaration_id: decl-06-ajustado-conll
  hash_version: 4
  model: gliner_base-ft-conll2003
  sink_policy: drop_from_denominator
  layers:
  - 0
  - 1
  - 2
  - 3
  - 4
  - 5
  - 6
  - 7
  - 8
  - 9
  - 10
  - 11
  heads: null
  combination_rule: convex
  calibration_fraction: 0.3
  coverage_levels:
  - 0.5
  - 0.7
  - 0.8
  - 0.9
  - 0.95
  operating_point: derived_from_target_risk
  target_risk_grid:
  - 0.01
  - 0.02
  - 0.05
  - 0.1
  added_value_ci_level: 0.95
  n_bootstrap_resamples: 2000
  conformal_alpha: 0.05
  loss_bound: 1.0
  seed: 42
  comparison_scores:
  - span_size
  - geometric_fraction
  - enrichment
  - geometric_residual
  - sentence_length
  stratify_by:
  - span_size
  - is_nested
  span_size_bins:
  - - 1
    - 1
  - - 2
    - 2
  - - 3
    - 4
  - - 5
    - null
  task_quality_grid:
  - 0.7
  - 0.8
  - 0.9
  - 0.95
  task_metrics:
  - precision_delivered
  - recall_delivered
  - f1_delivered
  - review_load
  layer_profile: true
  comparisons:
  - id: C1
    kind: descriptive
    question: A massa de atenção acompanha o nulo k/|K|? Relata R² contra a fração
      geométrica e a mediana da razão observado/esperado, por corpus e por faixa.
  - id: C2
    kind: verdict
    question: A massa de atenção acrescenta sobre a geometria pura?
    score: span_mass
    against: geometric_fraction
    criterion: paired_delta_aurc_ci_excludes_zero
  - id: C3
    kind: verdict
    question: O instrumento desconfundido acrescenta sobre a confiança do modelo?
    score: model_confidence+enrichment
    against: model_confidence
    criterion: paired_delta_aurc_ci_excludes_zero
  - id: T1
    kind: descriptive
    question: Em cada meta de qualidade, qual a carga de revisão exigida por cada
      supervisor, e qual precisão, recall e F1 saem entre os entregues? Inclui a linha
      SEM supervisor (cobertura total) como referência, e declara "inalcançável" como
      resultado possível.
  - id: T2
    kind: task_verdict
    question: O supervisor baseado na massa de atenção exige MENOS revisão humana
      que a confiança do próprio modelo, para a mesma qualidade entregue?
    score: span_mass
    against: model_confidence
    criterion: paired_review_load_ci_excludes_zero
  - id: T3
    kind: task_verdict
    question: E o instrumento desconfundido pela razão — o enriquecimento — exige
      menos revisão que a confiança do próprio modelo?
    score: enrichment
    against: model_confidence
    criterion: paired_review_load_ci_excludes_zero
  - id: T4
    kind: task_verdict
    question: A combinação da confiança com o enriquecimento exige menos revisão que
      a confiança sozinha?
    score: model_confidence+enrichment
    against: model_confidence
    criterion: paired_review_load_ci_excludes_zero
```
