# decl-04-escala — o mesmo desenho, uma escala acima

**Estado:** ASSINADA em 2026-09-15. Os valores e critérios abaixo são COMPROMISSO: nenhum deles
muda depois de ver resultado. Analisar sob outros valores exige uma declaração NOVA — outro
`declaration_id`, outro hash —, nunca a edição desta.

| | |
|---|---|
| `declaration_id` | `decl-04-escala` |
| Hash da declaração | `58a3fa48845bb673` |
| Hash de **medição** | `5454128f1df4da6f` |
| Hash de análise | `f4ef835f18e9043c` |
| Versão do conjunto de itens | 4 |
| Modelo | `urchade/gliner_large` |
| Herda medição de | — (nenhuma: exige medição própria) |

---

## Por que ela existe

O resultado confirmatório é negativo e forte, e a objeção mais previsível contra ele não é
sobre a atenção: é sobre o **extrator**. O `gliner_base` erra 48,2% no GENIA e 53,1% no
CoNLL-2003, e com precisão de partida em 0,52 e 0,47 um revisor pode alegar que a faixa
dinâmica dos dois sinais estava comprimida e que o nulo é do arranjo, não da atenção.

A segunda escala é a verificação mais barata dessa objeção: **mesma** arquitetura, **mesma**
metodologia, só escala. Se o nulo sobrevive a um modelo com o dobro da profundidade, "o
arranjo é fraco" perde força. Se **não** sobrevive, isso é achado e não contratempo — e é por
isso que a declaração vem antes da medição.

## Esta declaração NÃO herda medição, e isso é o ponto

O hash de medição dela é `5454128f1df4da6f`; o das três declarações assinadas é
`8a798025fb755a32`. São diferentes porque a versão 4 do conjunto de itens inclui o
**modelo** entre os itens de medição, e trocar de modelo muda os números *dentro* da tabela.

Até 15/09/2026 o modelo não entrava no hash e o `MEDIDA.json` não registrava qual modelo
produziu a tabela — o único traço era o nome do diretório. As duas escalas teriam hashes de
medição idênticos, e o guarda de procedência aceitaria testar uma tabela do `base` sob esta
declaração sem reclamar. O buraco foi fechado **antes** de existir um segundo modelo.

## A faixa de camadas, que é o único ponto onde há escolha

A `decl-01` declara as camadas 0 a 11 do `gliner_base`, isto é, **todas** as 12. O
`gliner_large` tem 24, e aqui declaram-se as 24 (0 a 23).
O que se mantém constante é a **regra** — "todas as camadas" —, não o intervalo de índices.
Declarar 0 a 11 no modelo grande usaria metade da rede, e isso seria regra nova em vez da
mesma regra: um efeito de profundidade entraria disfarçado de efeito de escala.

Sondado do `config.json` dos checkpoints em 15/09/2026, não suposto:

| modelo | encoder | camadas | cabeças | hidden |
|---|---|---:|---:|---:|
| `urchade/gliner_base` | `microsoft/deberta-v3-base` | 12 | 12 | 768 |
| `urchade/gliner_large` | `microsoft/deberta-v3-large` | **24** | **16** | 1024 |

Os dois usam o mesmo tokenizador sentencepiece (vocab 128.100), então os comprimentos de
subpalavra não mudam entre escalas e a projeção de disco sai da soma dos T² já medida:
**6,29 GB** para as quatro medições contra 2,36 GB do `base`, com 541 GB livres. Não há teto.

## O que é copiado, e por quê

Todos os itens de **análise** vêm literalmente da `decl-02` e da `decl-03` — copiados e não
reescolhidos. A pergunta desta declaração é o efeito da **escala**; reescolher qualquer item
de análise misturaria os dois efeitos e nenhuma comparação entre escalas seria possível.

- combinação `convex`, calibração 30% das sentenças
- coberturas de relato: 50% · 70% · 80% · 90% · 95%
- grade de riscos alvo: 1% · 2% · 5% · 10%
- grade de metas de qualidade: 70% · 80% · 90% · 95%
- IC de 95% com 2000 reamostragens de SENTENÇAS
- alpha conformal 0.05, perda limitada por 1, semente 42
- escores de comparação: `span_size`, `geometric_fraction`, `enrichment`, `geometric_residual`, `sentence_length`
- estratificação: `span_size`, `is_nested`; faixas de span: k=1 · k=2 · k=3-4 · k>=5
- métricas da tarefa: `precision_delivered`, `recall_delivered`, `f1_delivered`, `review_load`

## As sete comparações, e a razão de C1-C3 voltarem

A `decl-03` deixou C1, C2 e C3 de fora dizendo que já estavam decididas. Ali estava certo: era
o mesmo modelo. Aqui é **outro** modelo, e nada sobre ele foi decidido — repetir as sete é o
que torna a comparação entre escalas uma comparação, e não duas medições sobre perguntas
diferentes.

| id | tipo | escore | contra | critério |
|---|---|---|---|---|
| `C1` | descriptive | — | — | — (descritiva) |
| `C2` | verdict | `span_mass` | `geometric_fraction` | `paired_delta_aurc_ci_excludes_zero` |
| `C3` | verdict | `model_confidence+enrichment` | `model_confidence` | `paired_delta_aurc_ci_excludes_zero` |
| `T1` | descriptive | — | — | — (descritiva) |
| `T2` | task_verdict | `span_mass` | `model_confidence` | `paired_review_load_ci_excludes_zero` |
| `T3` | task_verdict | `enrichment` | `model_confidence` | `paired_review_load_ci_excludes_zero` |
| `T4` | task_verdict | `model_confidence+enrichment` | `model_confidence` | `paired_review_load_ci_excludes_zero` |

As perguntas, na íntegra:

- **C1** — A massa de atenção acompanha o nulo k/|K|? Relata R² contra a fração geométrica e a mediana da razão observado/esperado, por corpus e por faixa.
- **C2** — A massa de atenção acrescenta sobre a geometria pura?
- **C3** — O instrumento desconfundido acrescenta sobre a confiança do modelo?
- **T1** — Em cada meta de qualidade, qual a carga de revisão exigida por cada supervisor, e qual precisão, recall e F1 saem entre os entregues? Inclui a linha SEM supervisor (cobertura total) como referência, e declara "inalcançável" como resultado possível.
- **T2** — O supervisor baseado na massa de atenção exige MENOS revisão humana que a confiança do próprio modelo, para a mesma qualidade entregue?
- **T3** — E o instrumento desconfundido pela razão — o enriquecimento — exige menos revisão que a confiança do próprio modelo?
- **T4** — A combinação da confiança com o enriquecimento exige menos revisão que a confiança sozinha?

---

## Assinatura

| | |
|---|---|
| Assinada em | 2026-09-15 |
| Commit da assinatura | o commit que introduz esta alteração — `git log --diff-filter=M -1 -- docs/tese/declaracoes/decl-04-escala.md` |
| Primeira medição sob ela | posterior a este commit, por construção |

Assinar é apagar a linha de estado do topo deste documento e da fonte, e commitar **antes** de
rodar qualquer medição ou análise sob esta declaração. É o commit que estabelece a ordem.

O hash **não muda** com a assinatura, e isso é por desenho: a linha de estado é comentário nos
dois arquivos, não item declarado. Se assinar mudasse o hash, a própria assinatura invalidaria
a prova de ordem que ela existe para estabelecer.

---

## A fonte, literal

```yaml
selective:
  declaration_id: decl-04-escala
  hash_version: 4
  model: urchade/gliner_large
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
