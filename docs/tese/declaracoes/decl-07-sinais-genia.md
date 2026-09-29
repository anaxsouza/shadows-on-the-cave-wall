# decl-07-sinais-genia — as três famílias de sinal contra a confiança do modelo

**GERADO DE `configs/decl-07-sinais-genia.yaml`.** Não editar à mão: um teste compara este documento com a
fonte executável e falha se divergirem.

**Estado:** ASSINADA em 2026-09-16, sob autorização explícita do autor concedida
ANTES da leitura deste texto. O registro é honesto sobre isso: ele pré-autorizou, e a
contrapartida é que esta declaração tem **zero parâmetro livre** — não havia o que
escolher, e é isso que a torna defensável mesmo sem leitura prévia.

**Hash da declaração:** `5eff5178cb46307f`
**Hash de medição:** `a068f4f0daf00399` — IDÊNTICO ao de `decl-05-ajustado-genia` (`a068f4f0daf00399`)
**Modelo:** `gliner_base-ft-genia` · **Corpus:** GENIA

---

## Por que o hash de medição é o mesmo, e por que isso importa

Os estados ocultos são medidos na **mesma faixa de camadas já declarada**. Não é
conveniência: se a família de estados ocultos tivesse faixa própria, ela seria um
parâmetro de medição novo, o hash mudaria, as tabelas já medidas deixariam de ser
comparáveis, e escolher a faixa depois de ver o resultado voltaria a ser uma linha de
código. Com o hash igual, a tabela é remedida sob a MESMA identidade e as colunas
antigas têm de reproduzir exatamente — o que é, por si, uma verificação forte.

## Zero parâmetro livre — o que foi copiado e o que foi derivado

| origem | o que veio de lá |
|---|---|
| `decl-05-ajustado-genia` | todo item de análise e de medição, literalmente |
| `src/selective/signals.py` | o conjunto de sinais e o ESTATUTO do nulo de cada um |
| `CRITERIO_DO_TIPO` | o critério, fixo por tipo de veredito |
| regra mecânica | um veredito por sinal, contra `model_confidence`, sem seleção |

A ausência de seleção é o que substitui aqui a correção de multiplicidade: não há
escolha de sinal favorito depois de ver o resultado porque **todos** entram.

## Os sinais, com o estatuto do nulo de cada um

| sinal | família | nulo | o que isso permite afirmar |
|---|---|---|---|
| `span_size` | geométrico | — | adversário já declarado nas anteriores |
| `geometric_fraction` | geométrico | — | adversário já declarado nas anteriores |
| `sentence_length` | geométrico | — | adversário já declarado nas anteriores |
| `row_entropy` | atencao | **exato** | afirmação sobre o MODELO: a esperança sai da álgebra do orçamento fixo |
| `row_max` | atencao | **exato** | afirmação sobre o MODELO: a esperança sai da álgebra do orçamento fixo |
| `emitted_mass` | atencao | **exato** | afirmação sobre o MODELO: a esperança sai da álgebra do orçamento fixo |
| `hidden_norm` | estados_ocultos | **empirico** | afirmação mais FRACA: forma suposta linear, coeficientes ajustados no mesmo dado |
| `hidden_dist_centroide` | estados_ocultos | **empirico** | afirmação mais FRACA: forma suposta linear, coeficientes ajustados no mesmo dado |
| `hidden_delta_camadas` | estados_ocultos | **empirico** | afirmação mais FRACA: forma suposta linear, coeficientes ajustados no mesmo dado |
| `logit_margin` | logits | **empirico** | afirmação mais FRACA: forma suposta linear, coeficientes ajustados no mesmo dado |
| `logit_entropy` | logits | **empirico** | afirmação mais FRACA: forma suposta linear, coeficientes ajustados no mesmo dado |
| `logit_max` | logits | **empirico** | afirmação mais FRACA: forma suposta linear, coeficientes ajustados no mesmo dado |

## A verificação de identidade embutida

`logit_max` é, por construção, o maior escore de rótulo do trecho — a **mesma**
quantidade que `model_confidence`. O veredito `S9` dele contra a confiança tem de dar
delta **exatamente zero**. Se não der, o encanamento está errado, e é melhor descobrir
isso por um sinal cujo valor esperado se conhece do que por um cujo valor esperado se
está medindo. A orientação dos dois é fixada por definição justamente para isso.

## A orientação do escore, e por que ela não é grau de liberdade

A curva entrega por escore decrescente. Para a confiança, alto significa entregar; para
a entropia dos escores de rótulo, alto significa o contrário. Testar as duas orientações
e relatar a melhor dobraria as chances ao acaso.

A orientação é decidida pelo **sinal da correlação entre o escore e o acerto na partição
de CALIBRAÇÃO**, nunca na de avaliação — a mesma disciplina que o peso convexo já usa.
É **invariante do protocolo** e não item por declaração: vale para todo sinal, em toda
declaração, como a separação por sentença.

## O que esta declaração NÃO pode afirmar

Se um sinal de nulo **empírico** bater a confiança, a afirmação é mais fraca que a de um
sinal de nulo **exato**, e a ressalva viaja com o número em toda tabela e todo texto: a
forma da relação é suposta linear e os coeficientes são ajustados no mesmo dado em que o
resíduo é avaliado. Um nulo exato não tem nenhuma das duas fraquezas.

## A fonte, congelada

```yaml
selective:
  declaration_id: decl-07-sinais-genia
  hash_version: 5
  model: gliner_base-ft-genia
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
  - sentence_length
  - row_entropy
  - row_max
  - emitted_mass
  - hidden_norm
  - hidden_dist_centroide
  - hidden_delta_camadas
  - logit_margin
  - logit_entropy
  - logit_max
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
  task_gap_grid:
  - 0.25
  - 0.5
  - 0.75
  - 0.9
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
  - id: S1
    kind: task_verdict
    question: A entropia das linhas de atenção do trecho — cujo nulo é EXATO, log|K|
      — exige menos revisão que a confiança do próprio modelo?
    score: row_entropy
    against: model_confidence
    criterion: paired_review_load_ci_excludes_zero
  - id: S2
    kind: task_verdict
    question: O máximo das linhas de atenção do trecho — nulo EXATO, 1/|K| — exige
      menos revisão que a confiança do próprio modelo?
    score: row_max
    against: model_confidence
    criterion: paired_review_load_ci_excludes_zero
  - id: S3
    kind: task_verdict
    question: A massa que o trecho EMITE para fora de si — nulo EXATO, complemento
      da recebida — exige menos revisão que a confiança?
    score: emitted_mass
    against: model_confidence
    criterion: paired_review_load_ci_excludes_zero
  - id: S4
    kind: task_verdict
    question: A norma dos estados ocultos do trecho — nulo EMPÍRICO, desconfundido
      por regressão — exige menos revisão que a confiança?
    score: hidden_norm
    against: model_confidence
    criterion: paired_review_load_ci_excludes_zero
  - id: S5
    kind: task_verdict
    question: A distância do trecho ao centroide da camada — nulo EMPÍRICO — exige
      menos revisão que a confiança?
    score: hidden_dist_centroide
    against: model_confidence
    criterion: paired_review_load_ci_excludes_zero
  - id: S6
    kind: task_verdict
    question: A variação do estado do trecho entre a primeira e a última camada declarada
      — nulo EMPÍRICO — exige menos revisão que a confiança?
    score: hidden_delta_camadas
    against: model_confidence
    criterion: paired_review_load_ci_excludes_zero
  - id: S7
    kind: task_verdict
    question: A margem entre o maior e o segundo maior escore de rótulo — nulo EMPÍRICO
      — exige menos revisão que a confiança?
    score: logit_margin
    against: model_confidence
    criterion: paired_review_load_ci_excludes_zero
  - id: S8
    kind: task_verdict
    question: A entropia normalizada dos escores de rótulo — nulo EMPÍRICO — exige
      menos revisão que a confiança?
    score: logit_entropy
    against: model_confidence
    criterion: paired_review_load_ci_excludes_zero
  - id: S9
    kind: task_verdict
    question: 'VERIFICAÇÃO DE IDENTIDADE: o maior escore de rótulo é, por construção,
      a própria confiança. O delta tem de ser EXATAMENTE zero; se não for, o encanamento
      está errado.'
    score: logit_max
    against: model_confidence
    criterion: paired_review_load_ci_excludes_zero
```
