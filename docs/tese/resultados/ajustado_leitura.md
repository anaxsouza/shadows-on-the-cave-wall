# O extrator competente — leitura do confirmatório sob `decl-05` e `decl-06`

**Declarações:** `decl-05-ajustado-genia` (`74a976d10b39b0cb`) e `decl-06-ajustado-conll`
(`7726d98cdefd2ae6`), assinadas em 15/09/2026 no commit `6bd0fce`, **antes** de qualquer medição
**Medições:** `a068f4f0daf00399` e `4658eceda04c48df` — próprias, não herdadas
**Modelos:** `gliner_base` ajustado a cada corpus na Tesla T4 (`tools/treinar_extrator.py`)
**Dado:** `confirmatorio_ajustado_geometria.csv` (118 linhas) e `confirmatorio_ajustado_tarefa.csv` (82)
**Partição:** teste — 1.854 sentenças no GENIA, 3.453 no CoNLL-2003

---

## O que esta medição foi feita para fechar

A objeção que continuava de pé contra o resultado negativo não era sobre atenção: era que os
extratores medidos eram de prateleira e erravam perto de metade. Um revisor podia dizer que a
faixa dinâmica dos sinais estava comprimida e que o nulo era do arranjo.

## O extrator ficou competente

F1 estrito na validação, durante o ajuste (`ajuste_genia.json`, `ajuste_conll2003.json`):

| corpus | F1 antes | F1 depois | checkpoint escolhido |
|---|---:|---:|---|
| GENIA | 0,5597 | **0,7678** | `checkpoint-3378` |
| CoNLL-2003 | 0,6726 | **0,9696** | `checkpoint-4176` |

E a taxa de erro base na partição de teste, que é a horizontal do acaso de toda curva
risco-cobertura, ao longo dos três modelos medidos:

| corpus | `gliner-base` | `gliner-large` | **ajustado** |
|---|---:|---:|---:|
| GENIA | 0,4818 | 0,4224 | **0,2458** |
| CoNLL-2003 | 0,5310 | 0,4667 | **0,1028** |

O erro cai para **um quarto** no GENIA e para **um décimo** no CoNLL. A objeção "o extrator de
vocês erra metade" não tem mais objeto em nenhum dos dois. O GENIA continua errando cerca de um
quarto — competente, não resolvido, e o texto deve dizer isso por corpus.

## O nulo replica, e o único sinal positivo DESAPARECE

### C1 — terceiro modelo, mesma geometria

| corpus | R² massa × fração geométrica | mediana observado/esperado |
|---|---:|---:|
| GENIA | **0,9648** | 0,5339 |
| CoNLL-2003 | **0,9222** | 0,4693 |

Três modelos agora — `gliner_base`, `gliner_large` e o ajustado —, dois corpora, e a massa de
atenção sobre o span segue explicada em **92% a 96%** pela fração de chaves que o span ocupa. A
dependência geométrica não é artefato de modelo pequeno nem de modelo não treinado no corpus. É
consequência do orçamento fixo de atenção, e o orçamento é fixo em qualquer modelo.

### C2 — o achado que não sobreviveu ao extrator competente

| corpus | `gliner-base` | `gliner-large` | **ajustado** |
|---|---|---|---|
| GENIA | Δ −0,0039 [−0,0071; −0,0009] **acrescenta** | Δ −0,0036 [−0,0067; −0,0006] **acrescenta** | Δ −0,0014 [−0,0042; +0,0014] **não acrescenta** |
| CoNLL | não acrescenta | não acrescenta | Δ −0,0024 [−0,0060; +0,0010] não acrescenta |

**Este é o resultado mais informativo da medição.** O único lugar em que a massa de atenção
mostrava conteúdo próprio — acrescentar sobre a geometria pura, no GENIA, replicado nas duas
escalas — **desaparece** quando o extrator é competente. O intervalo passa a conter zero.

A leitura honesta: aquele efeito de terceira casa decimal era propriedade do **extrator fraco**,
não da atenção. Com um extrator que erra um quarto em vez de metade, a massa de atenção não
acrescenta nada nem sobre uma conta que dispensa modelo.

### C3 — a combinação continua sem acrescentar, e no CoNLL o peso colapsa de novo

| corpus | Δ | IC | veredito |
|---|---|---|---|
| GENIA | +0,0020 | [−0,0008; +0,0051] | não acrescenta |
| CoNLL-2003 | **+0,0000** | [+0,0000; +0,0000] | não acrescenta |

O zero exato do CoNLL é o colapso do peso convexo: a calibração deu peso 1,00 à confiança do
modelo e 0,00 ao enriquecimento — **descartou** o instrumento. Um revisor não pode alegar que o
ganho se diluiu; o procedimento teve a chance de usar o instrumento e recusou.

## Na tarefa, e uma observação sobre as metas

Com extrator competente, as metas de 70% e 90% ficaram **abaixo ou junto** da precisão que o
extrator já entrega sem abster-se de nada (0,7582 no GENIA, 0,8997 no CoNLL). Nessas metas a
carga de revisão é zero ou quase, e a comparação entre supervisores não distingue nada — não
porque os supervisores sejam equivalentes, mas porque **não há o que supervisionar**. É um efeito
da grade ter sido declarada quando o extrator era pior, e está registrado como está.

As metas informativas passaram a ser as altas, e ali o veredito é claro:

| comparação | corpus | meta | carga do supervisor | carga da confiança | veredito |
|---|---|---|---:|---:|---|
| T2 `span_mass` | GENIA | 90% | 0,9997 | 0,4583 | **exige mais revisão** (Δ +0,5414) |
| T2 `span_mass` | CoNLL | 95% | 0,9998 | 0,1429 | **exige mais revisão** (Δ +0,8568) |
| T3 `enrichment` | GENIA | 90% | 0,9970 | 0,4583 | **exige mais revisão** (Δ +0,5387) |
| T3 `enrichment` | CoNLL | 95% | 0,9752 | 0,1429 | **exige mais revisão** (Δ +0,8323) |
| T4 combinação | ambos | todas | — | — | não distingue |

Para entregar com 95% de precisão no CoNLL, ordenar pela confiança do modelo exige revisar
**14%** das predições; ordenar pela massa de atenção exige revisar **99,98%** — isto é, revisar
tudo. A diferença não é de terceira casa decimal: é a diferença entre um revisor humano viável e
um que confere o trabalho inteiro.

## O que esta medição fecha

A objeção do extrator, na forma em que podia ser feita. A afirmação do paper passa a ser
sustentada em **três modelos** — dois de prateleira em escalas diferentes e um ajustado ao corpus,
com erro de um quarto e de um décimo — e em nenhum deles a atenção acrescenta poder de decisão
sobre a confiança que o próprio modelo reporta. E o único sinal positivo que existia não
sobreviveu à melhora do extrator, o que é evidência a favor da afirmação e não contra.

## O que ela não fecha

A grade de metas ficou parcialmente vazia para o extrator bom, pelo motivo acima. Declarar uma
grade que acompanhe a precisão de base do extrator seria uma declaração nova, e a atual fica
como está — a limitação é de relato, não de veredito, porque os vereditos saem das metas altas,
que continuam informativas.
