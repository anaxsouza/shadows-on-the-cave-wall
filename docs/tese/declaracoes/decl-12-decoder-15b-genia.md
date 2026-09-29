# decl-12-decoder-15b-genia — os sinais internos do DECODER contra a confiança do modelo

**GERADO DE `configs/decl-12-decoder-15b-genia.yaml`.** Não editar à mão: um teste compara este documento com a
fonte executável e falha se divergirem.

**Estado:** ASSINADA em 2026-09-22 pelo autor, com autorização explícita
("assino as declarações, vamos avaliar os modelos decoder agora").

**O QUE FOI LIDO, dito com precisão porque a alternativa seria inflar a
procedência:** o autor leu a explicação em linguagem simples do estudo — o que
é atenção, por que o tamanho do trecho a contamina, e o que se mede — e não
declarou ter lido o texto integral desta declaração. A contrapartida que torna
isso defensável é a mesma das anteriores e é verificável: **zero parâmetro
livre**. Todo item de análise é copiado literalmente de `decl-07-sinais-genia.yaml`,
o conjunto de sinais vem do registro em `src/selective/signals.py` fixado em
teste, e as 36 comparações são geradas mecanicamente por
`tools/gerar_decl_decoder.py` sem seleção — não havia o que escolher aqui.

Gerada em 2026-09-18, quatro dias antes da assinatura. A medição não rodou
nesse intervalo. O treino dos pesos rodou, porque treinar não é medir.

Como as anteriores, esta tem **zero parâmetro livre**: todo item de análise é
copiado literalmente de `decl-07-sinais-genia.yaml` e as comparações são geradas
mecanicamente de `tools/gerar_decl_decoder.py`, sem seleção — logo não há onde
escolher o sinal favorito depois de ver o resultado.

**Hash da declaração:** `8cde6729246f9209`
**Hash de medição:** `4d8a138acc663343` — PRÓPRIO: este braço exige medição nova, e os itens de ANÁLISE vêm de `decl-07-sinais-genia`
**Modelo:** `qwen15b-ft-genia` · **Corpus:** GENIA

---

## Por que o hash de medição é PRÓPRIO, e por que isso importa

Ao contrário das declarações do braço encoder, esta NÃO herda o hash de medição
de nenhuma anterior, e não poderia: o modelo é outro, a arquitetura é outra, e a
faixa de camadas cobre todas as deste modelo. Um hash herdado afirmaria que a
tabela pode ser conferida contra colunas já medidas, e aqui não há colunas já
medidas — este braço exige medição nova, e é isso que o hash próprio declara.

O que É herdado, e apenas isso, são os itens de ANÁLISE: grades, critérios,
fração de calibração, número de reamostragens, semente. Eles vêm literalmente da
`decl-07`, e é essa herança que torna os dois braços comparáveis sem que a
comparação dependa de uma escolha feita depois de ver resultado.

## Zero parâmetro livre — o que foi copiado e o que foi derivado

| origem | o que veio de lá |
|---|---|
| `decl-07-sinais-genia` | todo item de ANÁLISE, literalmente. Medição NÃO: é própria |
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
| `row_entropy_causal` | atencao | **exato** | afirmação sobre o MODELO: a esperança sai da álgebra do orçamento fixo |
| `row_max_causal` | atencao | **exato** | afirmação sobre o MODELO: a esperança sai da álgebra do orçamento fixo |
| `hidden_norm` | estados_ocultos | **empirico** | afirmação mais FRACA: forma suposta linear, coeficientes ajustados no mesmo dado |
| `hidden_dist_centroide` | estados_ocultos | **empirico** | afirmação mais FRACA: forma suposta linear, coeficientes ajustados no mesmo dado |
| `hidden_delta_camadas` | estados_ocultos | **empirico** | afirmação mais FRACA: forma suposta linear, coeficientes ajustados no mesmo dado |
| `sonda_ocultos` | estados_ocultos | **empirico** | afirmação mais FRACA: forma suposta linear, coeficientes ajustados no mesmo dado |
| `sonda_atencao_cabecas` | atencao | **empirico** | afirmação mais FRACA: forma suposta linear, coeficientes ajustados no mesmo dado |

## A verificação de identidade embutida

**Este braço NÃO TEM a verificação que o encoder tem, e a ausência é declarada.**
No encoder, `logit_max` é por construção a mesma quantidade que
`model_confidence`, e o veredito dele contra a confiança tem de dar delta
exatamente zero — um teste de encanamento cujo valor esperado se conhece. Num
extrator generativo o rótulo é TEXTO que o modelo escreve, então não existe vetor
fixo de escores de rótulo, a família de logits não existe, e com ela se perde o
teste. Dizer isso é melhor que omitir: este braço mede sem aquela rede.

O que há em lugar dela, e é mais fraco por ser aditivo e não estrutural: a média
das features de atenção POR CABEÇA tem de reproduzir o escalar agregado declarado.
Medido em 18/09/2026 no 0,5B ajustado — correlação 1,0000 e erro absoluto médio
0,00000 — e verificado também que os escalares saem bit-a-bit idênticos com e sem
a extração de features, o que é a garantia de que a exploração não move o veredito.

## A orientação do escore, e por que ela não é grau de liberdade

A curva entrega por escore decrescente. Para a confiança, alto significa entregar; para
a entropia das linhas causais, alto significa o contrário. Testar as duas orientações
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
  declaration_id: decl-12-decoder-15b-genia
  hash_version: 5
  model: qwen15b-ft-genia
  sink_policy: keep
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
  - 12
  - 13
  - 14
  - 15
  - 16
  - 17
  - 18
  - 19
  - 20
  - 21
  - 22
  - 23
  - 24
  - 25
  - 26
  - 27
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
  - row_entropy_causal
  - row_max_causal
  - hidden_norm
  - hidden_dist_centroide
  - hidden_delta_camadas
  - sonda_ocultos
  - sonda_atencao_cabecas
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
  - id: D1
    kind: descriptive
    question: A massa de atenção causal acompanha qual nulo? Relata R² contra `expected_causal`,
      que depende de tamanho, comprimento E POSIÇÃO, e contra `expected_bidir`, que
      depende só dos dois primeiros — na MESMA linha, para que "a posição acrescenta
      dimensão ao confundidor?" seja medida e não argumentada. É a pergunta própria
      deste braço.
  - id: D2
    kind: descriptive
    question: Em cada meta de qualidade, qual a carga de revisão exigida por cada
      supervisor, e qual precisão, recall e F1 saem entre os entregues? Inclui a linha
      SEM supervisor como referência e declara "inalcançável" como resultado possível.
  - id: C2
    kind: verdict
    question: A massa de atenção causal acrescenta sobre a geometria pura?
    score: span_mass
    against: geometric_fraction
    criterion: paired_delta_aurc_ci_excludes_zero
  - id: C3
    kind: verdict
    question: O instrumento desconfundido acrescenta sobre a confiança do modelo?
    score: model_confidence+enrichment
    against: model_confidence
    criterion: paired_delta_aurc_ci_excludes_zero
  - id: S1
    kind: verdict
    question: '`span_mass` acrescenta sobre `model_confidence` em AURC?'
    score: span_mass
    against: model_confidence
    criterion: paired_delta_aurc_ci_excludes_zero
  - id: S2
    kind: task_verdict
    question: '`span_mass` acrescenta sobre `model_confidence` em carga de revisão?'
    score: span_mass
    against: model_confidence
    criterion: paired_review_load_ci_excludes_zero
  - id: S3
    kind: verdict
    question: '`span_mass` acrescenta sobre `aggseq` em AURC?'
    score: span_mass
    against: aggseq
    criterion: paired_delta_aurc_ci_excludes_zero
  - id: S4
    kind: task_verdict
    question: '`span_mass` acrescenta sobre `aggseq` em carga de revisão?'
    score: span_mass
    against: aggseq
    criterion: paired_review_load_ci_excludes_zero
  - id: S5
    kind: verdict
    question: '`row_entropy_causal` acrescenta sobre `model_confidence` em AURC?'
    score: row_entropy_causal
    against: model_confidence
    criterion: paired_delta_aurc_ci_excludes_zero
  - id: S6
    kind: task_verdict
    question: '`row_entropy_causal` acrescenta sobre `model_confidence` em carga de
      revisão?'
    score: row_entropy_causal
    against: model_confidence
    criterion: paired_review_load_ci_excludes_zero
  - id: S7
    kind: verdict
    question: '`row_entropy_causal` acrescenta sobre `aggseq` em AURC?'
    score: row_entropy_causal
    against: aggseq
    criterion: paired_delta_aurc_ci_excludes_zero
  - id: S8
    kind: task_verdict
    question: '`row_entropy_causal` acrescenta sobre `aggseq` em carga de revisão?'
    score: row_entropy_causal
    against: aggseq
    criterion: paired_review_load_ci_excludes_zero
  - id: S9
    kind: verdict
    question: '`row_max_causal` acrescenta sobre `model_confidence` em AURC?'
    score: row_max_causal
    against: model_confidence
    criterion: paired_delta_aurc_ci_excludes_zero
  - id: S10
    kind: task_verdict
    question: '`row_max_causal` acrescenta sobre `model_confidence` em carga de revisão?'
    score: row_max_causal
    against: model_confidence
    criterion: paired_review_load_ci_excludes_zero
  - id: S11
    kind: verdict
    question: '`row_max_causal` acrescenta sobre `aggseq` em AURC?'
    score: row_max_causal
    against: aggseq
    criterion: paired_delta_aurc_ci_excludes_zero
  - id: S12
    kind: task_verdict
    question: '`row_max_causal` acrescenta sobre `aggseq` em carga de revisão?'
    score: row_max_causal
    against: aggseq
    criterion: paired_review_load_ci_excludes_zero
  - id: S13
    kind: verdict
    question: '`hidden_norm` acrescenta sobre `model_confidence` em AURC?'
    score: hidden_norm
    against: model_confidence
    criterion: paired_delta_aurc_ci_excludes_zero
  - id: S14
    kind: task_verdict
    question: '`hidden_norm` acrescenta sobre `model_confidence` em carga de revisão?'
    score: hidden_norm
    against: model_confidence
    criterion: paired_review_load_ci_excludes_zero
  - id: S15
    kind: verdict
    question: '`hidden_norm` acrescenta sobre `aggseq` em AURC?'
    score: hidden_norm
    against: aggseq
    criterion: paired_delta_aurc_ci_excludes_zero
  - id: S16
    kind: task_verdict
    question: '`hidden_norm` acrescenta sobre `aggseq` em carga de revisão?'
    score: hidden_norm
    against: aggseq
    criterion: paired_review_load_ci_excludes_zero
  - id: S17
    kind: verdict
    question: '`hidden_dist_centroide` acrescenta sobre `model_confidence` em AURC?'
    score: hidden_dist_centroide
    against: model_confidence
    criterion: paired_delta_aurc_ci_excludes_zero
  - id: S18
    kind: task_verdict
    question: '`hidden_dist_centroide` acrescenta sobre `model_confidence` em carga
      de revisão?'
    score: hidden_dist_centroide
    against: model_confidence
    criterion: paired_review_load_ci_excludes_zero
  - id: S19
    kind: verdict
    question: '`hidden_dist_centroide` acrescenta sobre `aggseq` em AURC?'
    score: hidden_dist_centroide
    against: aggseq
    criterion: paired_delta_aurc_ci_excludes_zero
  - id: S20
    kind: task_verdict
    question: '`hidden_dist_centroide` acrescenta sobre `aggseq` em carga de revisão?'
    score: hidden_dist_centroide
    against: aggseq
    criterion: paired_review_load_ci_excludes_zero
  - id: S21
    kind: verdict
    question: '`hidden_delta_camadas` acrescenta sobre `model_confidence` em AURC?'
    score: hidden_delta_camadas
    against: model_confidence
    criterion: paired_delta_aurc_ci_excludes_zero
  - id: S22
    kind: task_verdict
    question: '`hidden_delta_camadas` acrescenta sobre `model_confidence` em carga
      de revisão?'
    score: hidden_delta_camadas
    against: model_confidence
    criterion: paired_review_load_ci_excludes_zero
  - id: S23
    kind: verdict
    question: '`hidden_delta_camadas` acrescenta sobre `aggseq` em AURC?'
    score: hidden_delta_camadas
    against: aggseq
    criterion: paired_delta_aurc_ci_excludes_zero
  - id: S24
    kind: task_verdict
    question: '`hidden_delta_camadas` acrescenta sobre `aggseq` em carga de revisão?'
    score: hidden_delta_camadas
    against: aggseq
    criterion: paired_review_load_ci_excludes_zero
  - id: E25
    kind: verdict
    question: '`sonda_ocultos` acrescenta sobre `model_confidence` em AURC? Compartimento
      exploratório: a sonda tem pesos ajustados na calibração e ABANDONA o nulo exato
      da família, logo não decide veredito sozinha.'
    score: sonda_ocultos
    against: model_confidence
    criterion: paired_delta_aurc_ci_excludes_zero
  - id: E26
    kind: task_verdict
    question: '`sonda_ocultos` acrescenta sobre `model_confidence` em carga de revisão?
      Compartimento exploratório: a sonda tem pesos ajustados na calibração e ABANDONA
      o nulo exato da família, logo não decide veredito sozinha.'
    score: sonda_ocultos
    against: model_confidence
    criterion: paired_review_load_ci_excludes_zero
  - id: E27
    kind: verdict
    question: '`sonda_ocultos` acrescenta sobre `aggseq` em AURC? Compartimento exploratório:
      a sonda tem pesos ajustados na calibração e ABANDONA o nulo exato da família,
      logo não decide veredito sozinha.'
    score: sonda_ocultos
    against: aggseq
    criterion: paired_delta_aurc_ci_excludes_zero
  - id: E28
    kind: task_verdict
    question: '`sonda_ocultos` acrescenta sobre `aggseq` em carga de revisão? Compartimento
      exploratório: a sonda tem pesos ajustados na calibração e ABANDONA o nulo exato
      da família, logo não decide veredito sozinha.'
    score: sonda_ocultos
    against: aggseq
    criterion: paired_review_load_ci_excludes_zero
  - id: E29
    kind: verdict
    question: '`sonda_atencao_cabecas` acrescenta sobre `model_confidence` em AURC?
      Compartimento exploratório: a sonda tem pesos ajustados na calibração e ABANDONA
      o nulo exato da família, logo não decide veredito sozinha.'
    score: sonda_atencao_cabecas
    against: model_confidence
    criterion: paired_delta_aurc_ci_excludes_zero
  - id: E30
    kind: task_verdict
    question: '`sonda_atencao_cabecas` acrescenta sobre `model_confidence` em carga
      de revisão? Compartimento exploratório: a sonda tem pesos ajustados na calibração
      e ABANDONA o nulo exato da família, logo não decide veredito sozinha.'
    score: sonda_atencao_cabecas
    against: model_confidence
    criterion: paired_review_load_ci_excludes_zero
  - id: E31
    kind: verdict
    question: '`sonda_atencao_cabecas` acrescenta sobre `aggseq` em AURC? Compartimento
      exploratório: a sonda tem pesos ajustados na calibração e ABANDONA o nulo exato
      da família, logo não decide veredito sozinha.'
    score: sonda_atencao_cabecas
    against: aggseq
    criterion: paired_delta_aurc_ci_excludes_zero
  - id: E32
    kind: task_verdict
    question: '`sonda_atencao_cabecas` acrescenta sobre `aggseq` em carga de revisão?
      Compartimento exploratório: a sonda tem pesos ajustados na calibração e ABANDONA
      o nulo exato da família, logo não decide veredito sozinha.'
    score: sonda_atencao_cabecas
    against: aggseq
    criterion: paired_review_load_ci_excludes_zero
```
