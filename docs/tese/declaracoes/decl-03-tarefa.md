# decl-03-tarefa — o impacto na TAREFA, e não na curva

**Estado:** ASSINADA em 2026-09-15. Os valores e critérios abaixo são COMPROMISSO: nenhum deles
muda depois de ver resultado. Analisar sob outros valores exige uma declaração NOVA — outro
`declaration_id`, outro hash —, nunca a edição desta.

**Identificador:** `decl-03-tarefa`  
**Hash da declaração:** `5f30a2ff3e493c19`  
**Hash de medição:** `8a798025fb755a32` — idêntico ao da `decl-01` e da `decl-02`  
**Versão do conjunto de itens:** 3  
**Gerado de:** `configs/decl-03-tarefa.yaml` (este documento não é digitado à mão; um teste compara os dois)

---

## Por que ela existe

A `decl-02` decidiu em ΔAURC, que é área sob uma curva. Ninguém em NER decide nada com
uma área: decide com *que qualidade sai do sistema* e *quanto trabalho humano isso custa*.
Esta declaração traduz a mesma medição para essas duas unidades, e traz critério próprio —
porque o resultado na tarefa é a afirmação central do paper, e tabela sem critério declarado
é ilustração do resultado, não o resultado.

## O que ela não muda

Nada da medição e nada da ordenação. Sumidouro, camadas, cabeças, partição por sentença,
semente e regra de combinação são os da `decl-02`. É por isso que o hash de medição é
idêntico (`8a798025fb755a32`) e as tabelas já medidas servem sem remedir — e a herança
é **declarada**, não inferida:

- herda a medição de `decl-01-gliner-base`, hash `348e90cce8ca9874`

## A restrição estrutural que molda tudo aqui

O supervisor só **remove** predições; ele nunca cria. As entidades anotadas que o extrator
não apontou são invisíveis para ele, então o recall tem teto fixo — o recall do extrator — e
abster-se só o baixa. Qualquer formulação em que *o supervisor melhora o F1* é, por
construção, insustentável.

Por isso a unidade declarada é **carga de revisão a qualidade fixa**, e não ganho de F1: é a
pergunta que o arranjo pode responder.

## Item novo 1 — a grade de metas de qualidade

**Valor:** precisão entre os entregues em 70% · 80% · 90% · 95%.

Valores convencionais e redondos, escolhidos sem olhar o teste: são patamares que a prática
usa, não pontos lidos de uma curva.

É **grade** e não meta única pela mesma razão do risco alvo. Com erro base de 48% (GENIA) e
53% (CoNLL), a precisão a cobertura total é 0,52 e 0,47. Uma meta única fixada no escuro
poderia cair onde nenhum supervisor chega, e a declaração não decidiria nada. Com a grade,
*inalcançável em qualquer cobertura* é uma linha da tabela e um resultado legítimo.

E a grade **não custa grau de liberdade**: o veredito é a comparação entre supervisores em
cada meta, então não há meta a escolher depois de ver número.

## Item novo 2 — as quatro métricas, declaradas juntas

| métrica | definição |
|---|---|
| `precision_delivered` | 1 − erro médio entre os entregues |
| `recall_delivered` | entregues corretos ÷ total de entidades anotadas |
| `f1_delivered` | harmônica das duas acima |
| `review_load` | fração das predições mandadas para revisão humana |

Juntas de propósito: o supervisor faz precisão subir e recall cair **pela mesma ação**, e
relatar só uma seria relatar a que subiu. O casamento com o ouro é **estrito** — mesma
fronteira e mesmo rótulo —, herdado da medição.

## As comparações

### T1 — descritivo, sem veredito

Em cada meta de qualidade, qual a carga de revisão exigida por cada supervisor, e qual precisão, recall e F1 saem entre os entregues? Inclui a linha SEM supervisor (cobertura total) como referência, e declara "inalcançável" como resultado possível.

### T2 — task_verdict

O supervisor baseado na massa de atenção exige MENOS revisão humana que a confiança do próprio modelo, para a mesma qualidade entregue?

- **escore:** `span_mass`
- **contra:** `model_confidence`
- **refuta-se por:** `paired_review_load_ci_excludes_zero`

### T3 — task_verdict

E o instrumento desconfundido pela razão — o enriquecimento — exige menos revisão que a confiança do próprio modelo?

- **escore:** `enrichment`
- **contra:** `model_confidence`
- **refuta-se por:** `paired_review_load_ci_excludes_zero`

### T4 — task_verdict

A combinação da confiança com o enriquecimento exige menos revisão que a confiança sozinha? É a versão na tarefa da pergunta que C3 respondeu em AURC, e serve para mostrar se a unidade muda a conclusão.

- **escore:** `model_confidence+enrichment`
- **contra:** `model_confidence`
- **refuta-se por:** `paired_review_load_ci_excludes_zero`

`C1`, `C2` e `C3` da `decl-02` **não** são refeitas aqui: já foram decididas, e redecidir a
mesma pergunta sob outra declaração seria dois vereditos sobre um fato.

## O critério, sem jargão

Para cada meta da grade, mede-se a carga de revisão que cada supervisor exige para atingir
aquela precisão entre os entregues. A diferença entre dois supervisores vem com intervalo de
95% por reamostragem de **sentenças** (2000
reamostragens) — sentenças e não entidades, porque entidades da mesma sentença compartilham a
matriz de atenção e contá-las como independentes estreitaria o intervalo.

**Refuta-se a afirmação da tese** se, em alguma meta declarada, o intervalo da diferença
excluir zero em favor do supervisor de atenção — ou seja, ele exigir menos revisão.

**Inconclusivo** quando o intervalo contém zero: não é equivalência provada, é afirmação
sobre o que este experimento resolve.

**Inalcançável** é resultado declarado: se nenhuma cobertura atinge a meta, a linha diz isso
em vez de a execução ser perdida.

## Assinatura

| | |
|---|---|
| Assinada em | 2026-09-15 |
| Commit da assinatura | o commit que introduz esta alteração — `git log --diff-filter=M -1 -- docs/tese/declaracoes/decl-03-tarefa.md` |
| Primeira análise sob ela | posterior a este commit, por construção |

O commit é identificado pela **consulta** e não pelo hash, pela mesma razão da `decl-01` e da
`decl-02`: o hash do commit não pode constar do arquivo que o próprio commit introduz.

O hash da declaração **não muda** com a assinatura, e isso é por desenho: a linha de estado é
comentário nos dois arquivos, não item declarado. Verificado: `5f30a2ff3e493c19` antes e depois.
Se assinar mudasse o hash, a própria assinatura invalidaria a prova de ordem que ela existe para
estabelecer.

---

## A fonte, literal

```yaml
selective:
  declaration_id: decl-03-tarefa
  hash_version: 3
  inherits_measurement_from:
    declaration_id: decl-01-gliner-base
    declaration_hash: 348e90cce8ca9874
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
  comparisons:
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
      a confiança sozinha? É a versão na tarefa da pergunta que C3 respondeu em AURC,
      e serve para mostrar se a unidade muda a conclusão.
    score: model_confidence+enrichment
    against: model_confidence
    criterion: paired_review_load_ci_excludes_zero
```
