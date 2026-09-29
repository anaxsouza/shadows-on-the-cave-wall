# decl-02-geometria — declaração de ANÁLISE

**Estado:** ASSINADA em 2026-09-10. Os valores e critérios abaixo são COMPROMISSO: nenhum deles
muda depois de ver resultado. Analisar sob outros valores exige uma declaração NOVA — outro
`declaration_id`, outro hash —, nunca a edição desta.

| | |
|---|---|
| `declaration_id` | `decl-02-geometria` |
| Hash da declaração | `4d91c3607206b9ab` |
| Hash de **medição** | `8a798025fb755a32` |
| Hash de **análise** | `99901191a300a18b` |
| Versão do conjunto hasheado | 2 |
| Medição herdada de | `decl-01-gliner-base` (`348e90cce8ca9874`) |
| Fonte | `configs/decl-02-geometria.yaml` |

**Gerado da fonte executável, não digitado.** Um teste da suíte compara este documento com
`configs/decl-02-geometria.yaml` e falha se divergirem.

---

## O que esta declaração muda, e o que ela deliberadamente não muda

Ela **não muda nada da medição**: o hash de medição é `8a798025fb755a32`, idêntico ao de
`decl-01-gliner-base`. Sumidouro, camadas e cabeças são os mesmos, e por isso as
tabelas já medidas continuam servindo — o guarda de procedência compara o hash de medição, não
o da declaração inteira. Remedir seria gastar horas para obter exatamente os mesmos números.

Ela muda o que se faz **depois** da tabela: as comparações, as bases contra as quais o sinal
tem de se provar, e a estratificação obrigatória do relato.

### Por que isto não é inserir grau de liberdade

Uma base de comparação só pode **enfraquecer** a afirmação de quem a declara. Ela não dá mais
chances de encontrar efeito; dá um nulo mais difícil de bater. É o oposto de trocar de modelo
ou de ajustar hiperparâmetros, que abrem escolhas capazes de favorecer o resultado.

---

## As três comparações

### C1 — descritiva, sem veredito

A massa de atenção acompanha o nulo k/|K|? Relata R² contra a fração geométrica e a mediana da razão observado/esperado, por corpus e por faixa.

### C2 — veredito

A massa de atenção acrescenta sobre a geometria pura?

- **escore:** `span_mass`
- **contra:** `geometric_fraction`
- **refuta-se por:** `paired_delta_aurc_ci_excludes_zero`

### C3 — veredito

O instrumento desconfundido acrescenta sobre a confiança do modelo?

- **escore:** `model_confidence+enrichment`
- **contra:** `model_confidence`
- **refuta-se por:** `paired_delta_aurc_ci_excludes_zero`

`C1` não tem veredito **de propósito**. Caracterizar o nulo é parte do resultado sem ser teste;
transformá-la em veredito gastaria um teste pré-registrado para confirmar o que a validação já
mostrou.

`C3` usa o instrumento **desconfundido** e não a massa crua. Testar a massa crua contra a
confiança deixaria aberta a saída "vocês testaram a versão confundida do instrumento".

### Como o veredito é calculado

Diferença de AURC **pareada**, com IC 95% por 2000 reamostragens de **sentenças** —
entidades da mesma sentença compartilham a matriz de atenção, e reamostrar entidades infla o
tamanho efetivo da amostra. **IC contendo zero significa que não acrescenta.** Sem exceção
aberta depois de ver o número.

---

## Escores de comparação

| escore | usa modelo? | o que é |
|---|---|---|
| `span_size` | não | k, o número de tokens do span |
| `geometric_fraction` | não | k/\|K\|, a massa **esperada** sob permutabilidade das chaves |
| `enrichment` | sim | massa dividida pelo esperado — desconfundição por **razão** |
| `geometric_residual` | sim | massa menos o ajuste em k/\|K\| — desconfundição por **regressão** |
| `sentence_length` | não | T, que é o que decide o sinal da massa |

Três deles não usam modelo nenhum: são contas sobre a anotação. Se a massa de atenção não bater
uma conta que qualquer pessoa faz sem GPU, ela não é instrumento.

As duas desconfundições coexistem de propósito, porque erram em direções opostas: a razão
sobrecorrige (a correlação com k/T vira −0,43 e −0,51) e a regressão subcorrige. Declarar as
duas impede escolher depois a que der o número desejado.

---

## Estratificação, e por que as bordas estão declaradas

`stratify_by: ['span_size', 'is_nested']` · faixas: **k=1 · k=2 · k=3–4 · k≥5**

Sem as bordas, `stratify_by` diria *que* estratificar e não *como* — e o corte muda a leitura.
Medido na validação do GENIA, a inversão de sinal do enriquecimento entre entidades certas e
erradas vale para:

| esquema de corte | GENIA | CoNLL-2003 |
|---|---|---|
| `1 \| 2 \| 3–4 \| 5+` (o declarado) | 3 de 4 estratos | 3 de 4 |
| `1 \| 2 \| 3+` | 4 de 4 | 2 de 3 |
| `1–2 \| 3+` | 1 de 2 | 1 de 2 |
| quartis de k | 4 de 4 | 2 de 3 |

Escolher a borda depois de ver esta tabela seria escolher a conclusão. **O veredito não depende
dela**: os critérios de `C2` e `C3` são diferenças de AURC sobre a amostra inteira, que não vê
estrato nenhum. As faixas governam a tabela **descritiva**, e o relato traz os quatro esquemas
acima como análise de sensibilidade — para que ninguém precise confiar em que não escolhemos.

O motivo de estratificar é medido: no GENIA a associação entre enriquecimento e erro é
**positiva** no agregado (+0,0212, Mann-Whitney p = 3,9·10⁻⁴) e **negativa** em três das quatro
faixas declaradas — −0,0117 (k=1, n=970), −0,0164 (k=2, n=1.204), −0,0114 (k=3–4, n=1.896) —,
sendo +0,0010 na quarta (k≥5, n=862), praticamente nula. É paradoxo de Simpson, causado por
spans de 1 token terem ao mesmo tempo erro alto (0,71) e enriquecimento alto.

---

## Procedência

Saghir (2026), `arXiv 2605.00269`, estabelece que sinais de confiança de caixa branca em modelos
de linguagem são estruturalmente confundidos por comprimento (|r| ≥ 0,61) e colapsam para
0,491–0,527 de AUROC (acaso 0,5) sob avaliação com comprimento pareado, com uma base trivial de
contagem de tokens alcançando 0,874 e 0,919. Os três controles que ele recomenda — base trivial
de comprimento, residualização contra comprimento e relato da correlação escore-comprimento —
são adotados aqui. **Não são invenção nossa, e é isso que os torna difíceis de recusar.**

O nulo `k/|K|` é derivado, não emprestado: sai da linearidade da esperança sobre matriz
linha-estocástica, e foi verificado por permutação em 716 entidades (desvio +0,00013 e +0,00017,
contra desvios observados 327× e 335× maiores).

Os itens herdados de `decl-01` mantêm a procedência registrada em `docs/tese/PREREGISTRO.md`.

---

## A declaração, como está na fonte

```yaml
declaration_id: decl-02-geometria
hash_version: 2
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
layer_profile: true
combination_rule: convex
calibration_fraction: 0.3
coverage_levels:
- 0.5
- 0.7
- 0.8
- 0.9
- 0.95
operating_point: derived_from_target_risk
added_value_ci_level: 0.95
n_bootstrap_resamples: 2000
target_risk_grid:
- 0.01
- 0.02
- 0.05
- 0.1
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
comparisons:
- id: C1
  kind: descriptive
  question: A massa de atenção acompanha o nulo k/|K|? Relata R² contra a fração geométrica
    e a mediana da razão observado/esperado, por corpus e por faixa.
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
```

---

## Assinatura

| | |
|---|---|
| Assinada em | 2026-09-10 |
| Commit da assinatura | o commit que introduz esta alteração — `git log --diff-filter=M -1 -- docs/tese/declaracoes/decl-02-geometria.md` |
| Primeira análise sob ela | posterior a este commit, por construção |

O hash da declaração **não muda** com a assinatura, e isso é por desenho: a linha de estado é
comentário nos dois arquivos, não item declarado. Verificado: `4d91c3607206b9ab` antes e depois.
Se assinar mudasse o hash, a própria assinatura invalidaria a prova de ordem que ela existe para
estabelecer.

A data deste commit é anterior à da primeira análise sob esta declaração. É o commit que
estabelece a ordem, e nenhuma frase escrita depois a recupera.

Mudar qualquer item agora não é proibido — é rastreável, e deixa de ser esta declaração.
