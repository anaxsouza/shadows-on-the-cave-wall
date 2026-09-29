# O impacto na tarefa — leitura do confirmatório sob `decl-03-tarefa`

**Declaração:** `decl-03-tarefa`, hash `5f30a2ff3e493c19`, assinada em 15/09/2026 (commit
`cb3b5e6`) · **Medição:** `8a798025fb755a32`, herdada de `decl-01`
**Dado:** `confirmatorio_decl03_tarefa.csv` (82 linhas) e `confirmatorio_decl03_notas.txt`
**Partição:** avaliação do split de teste — 4.038 entidades no GENIA, 5.701 no CoNLL-2003

---

## ERRATA — uma afirmação da própria declaração foi refutada pelo dado

A `decl-03` justifica a escolha da unidade com esta frase, que está **errada**:

> "Qualquer formulação em que *o supervisor melhora o F1* é, por construção, insustentável."

O dado mostra o contrário no CoNLL-2003: sem supervisor o F1 entre os entregues é **0,4712**;
com o supervisor da confiança do modelo na meta de 70% de precisão ele sobe para **0,5133**.
A precisão vai de 0,469 a 0,700 e o recall cai de 0,4734 a 0,4053 — o ganho de precisão supera
a perda de recall, e o F1 **sobe**. No GENIA a mesma conta dá o contrário (0,4392 para 0,3387),
porque ali o recall de partida já é baixo.

**O que estava certo e o que estava errado.** Certo: o recall tem teto fixo e abster-se só o
baixa. Errado: concluir daí que o F1 não pode subir. F1 é uma troca entre as duas, e quando a
precisão de partida é baixa (0,47), levá-la a 0,70 paga a perda de recall.

**Consequências, e o que NÃO muda.** A frase é prosa de justificativa, não item declarado: não
está no hash, não define métrica nem critério, e nenhum veredito de T2, T3 ou T4 depende dela.
Os arquivos assinados **não** são editados — editar declaração assinada é o que não se faz. Esta
errata é o registro, e a frase não pode ser repetida no paper.

A escolha da unidade continua justificada pelo resto do argumento: o teto de recall é real, e
carga de revisão a qualidade fixa é a unidade em que um praticante decide. Mas ela deixa de ser
"a única unidade possível" e passa a ser "a unidade escolhida", o que é mais honesto.

---

## T1 — o retrato, com a linha sem supervisor

Carga de revisão = fração das predições mandadas para revisão humana. Cobertura = o que é
entregue. Recall tem como denominador o total de entidades **anotadas** do split inteiro.

### GENIA (sem supervisor: precisão 0,5191 · recall 0,3807 · F1 0,4392)

| supervisor | meta 70% | meta 80% | meta 90% | meta 95% |
|---|---|---|---|---|
| `model_confidence` | revisa 56,5% (F1 0,339) | revisa 84,9% (F1 0,160) | revisa 98,2% (F1 0,024) | **inalcançável** |
| `span_mass` | **inalcançável** | **inalcançável** | **inalcançável** | **inalcançável** |
| `enrichment` | revisa 99,7% (F1 0,004) | revisa 99,7% | revisa 100,0% | revisa 100,0% |
| `geometric_residual` | revisa 99,7% | **inalcançável** | **inalcançável** | **inalcançável** |
| `geometric_fraction` | **inalcançável** | **inalcançável** | **inalcançável** | **inalcançável** |
| `span_size` | **inalcançável** | **inalcançável** | **inalcançável** | **inalcançável** |
| `sentence_length` | **inalcançável** | **inalcançável** | **inalcançável** | **inalcançável** |

### CoNLL-2003 (sem supervisor: precisão 0,4690 · recall 0,4734 · F1 0,4712)

| supervisor | meta 70% | meta 80% | meta 90% | meta 95% |
|---|---|---|---|---|
| `model_confidence` | revisa 42,6% (F1 0,513) | revisa 61,9% (F1 0,445) | revisa 76,2% (F1 0,348) | revisa 87,3% (F1 0,216) |
| `span_mass` | **inalcançável** | **inalcançável** | **inalcançável** | **inalcançável** |
| `enrichment` | revisa 97,5% (F1 0,034) | revisa 99,7% | revisa 99,7% | revisa 99,8% |
| `sentence_length` | revisa 90,5% (F1 0,122) | revisa 95,5% | revisa 96,9% | revisa 97,6% |
| `geometric_residual` | **inalcançável** | **inalcançável** | **inalcançável** | **inalcançável** |
| `geometric_fraction` | **inalcançável** | **inalcançável** | **inalcançável** | **inalcançável** |
| `span_size` | **inalcançável** | **inalcançável** | **inalcançável** | **inalcançável** |

---

## Os vereditos

### T2 — `span_mass` contra `model_confidence`

**Inalcançável para `span_mass` em todas as quatro metas, nos dois corpora.** Não há carga de
revisão a comparar porque não existe cobertura nenhuma em que ordenar pela massa de atenção
atinja 70% de precisão entre os entregues. É o resultado mais forte possível na direção do
nulo: não "pior por pouco", e sim **não atinge o piso declarado em cobertura alguma**.

Na meta de 95% do GENIA a `model_confidence` também é inalcançável, e isso vai na tabela.

### T3 — `enrichment` contra `model_confidence`

Em **todas** as metas atingíveis, nos dois corpora, o instrumento desconfundido exige **mais**
revisão, com IC de 95% excluindo zero:

| corpus | meta | revisão com `enrichment` | com `model_confidence` | Δ | IC 95% |
|---|---|---|---|---|---|
| GENIA | 70% | 99,65% | 56,49% | +0,4316 | [+0,3567; +0,4980] |
| GENIA | 80% | 99,70% | 84,89% | +0,1481 | [+0,0906; +0,2342] |
| GENIA | 90% | 99,98% | 98,17% | +0,0181 | [+0,0005; +0,0711] |
| CoNLL | 70% | 97,54% | 42,64% | +0,5490 | [+0,5080; +0,5915] |
| CoNLL | 80% | 99,70% | 61,88% | +0,3782 | [+0,3350; +0,4156] |
| CoNLL | 90% | 99,74% | 76,23% | +0,2350 | [+0,2078; +0,2620] |
| CoNLL | 95% | 99,82% | 87,28% | +0,1254 | [+0,0982; +0,1769] |

### T4 — `model_confidence+enrichment` contra `model_confidence`

**Não distingue, em todas as metas, nos dois corpora.** No GENIA o Δ é exatamente 0,0000 com IC
[0,0000; 0,0000], e a razão é aritmética e não acidente: o peso convexo ajustado na calibração
colapsou para **1,00 na confiança e 0,00 no enriquecimento** — a combinação *descartou* o
instrumento, então os dois escores são literalmente o mesmo vetor. No CoNLL o peso ficou em
0,99/0,01 e os Δ são da ordem de 0,003 a 0,014, com IC contendo zero.

É a distinção que decide a leitura: **descartou**, não empatou.

---

## Por que a unidade mudou o tamanho do resultado, e não o sinal

Em ΔAURC as diferenças eram minúsculas — `C2` deu −0,0039 no GENIA. Em carga de revisão elas
são de **0,43 a 0,55**. O sinal é o mesmo; a magnitude não.

A razão é que AURC é média sobre **toda** a curva, inclusive a região de cobertura alta, onde
qualquer ordenação entrega quase tudo e portanto se parecem. Os pontos de operação declarados
ficam na região de **precisão alta**, onde a ordenação tem de ser genuinamente boa — e é lá que
o sinal de atenção falha por completo.

Isso responde de forma substantiva a quem disser que o trabalho mediu uma quantidade abstrata:
na unidade em que se decide, a diferença não é pequena — é a diferença entre revisar 43% e
revisar tudo.

---

## O que estes números NÃO fecham

A objeção do Perímetro 2 continua aberta e a tabela até a torna mais visível: para chegar a 70%
de precisão o melhor supervisor exige revisar **42,6%** (CoNLL) e **56,5%** (GENIA) das
predições. Isso é propriedade do **extrator**, que erra metade. Um revisor pode dizer que com um
extrator competente a faixa dinâmica dos sinais seria outra. Nenhuma linha desta tabela responde
isso — só medição nova com extrator competente responde.
