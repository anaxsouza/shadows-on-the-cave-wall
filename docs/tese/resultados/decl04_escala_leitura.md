# A segunda escala — leitura do confirmatório sob `decl-04-escala`

**Declaração:** `decl-04-escala`, hash `58a3fa48845bb673`, assinada em 15/09/2026 (commit `eaf6264`)
**Medição:** `5454128f1df4da6f` — **própria**, não herdada: o modelo entra no hash na versão 4
**Modelo:** `urchade/gliner_large` (`deberta-v3-large`, 24 camadas × 16 cabeças, 445,5 M)
**Dado:** `confirmatorio_decl04_geometria.csv` (118 linhas) e `confirmatorio_decl04_tarefa.csv` (82)
**Partição:** teste — 1.854 sentenças no GENIA, 3.453 no CoNLL-2003

---

## Por que esta medição existe

Para fechar a única objeção que continuava aberta, e ela não é sobre atenção: é sobre o
**extrator**. O `gliner_base` erra 48,2% e 53,1%, e um revisor pode alegar que a faixa dinâmica
dos sinais estava comprimida e que o nulo é do arranjo. A segunda escala é a verificação mais
barata disso — mesma arquitetura, mesma metodologia, só escala.

## O extrator melhorou, e melhorou nas três dimensões que sustentavam a objeção

| | `gliner-base` | `gliner-large` |
|---|---:|---:|
| erro base, GENIA | 0,4818 | **0,4224** |
| erro base, CoNLL-2003 | 0,5310 | **0,4667** |
| preditas, GENIA (ouro 5.506) | 5.799 | 5.367 |
| preditas, CoNLL (ouro 5.648) | 8.104 | **7.321** |
| revisão exigida pela confiança a 70%, GENIA | 0,565 | **0,451** |
| revisão exigida pela confiança a 70%, CoNLL | 0,426 | **0,323** |

Erro 5,9 e 6,4 pontos percentuais menor, sobrepredição menor (era parte do que sustentava a
objeção), e a carga de revisão da confiança caindo 11 e 10 pontos. O arranjo ficou
**substancialmente melhor** — e é por isso que o resultado abaixo pesa mais do que pesaria antes.

## O nulo replica

### C1 — a massa continua sendo a geometria

| | R² massa × fração geométrica | mediana observado/esperado |
|---|---:|---:|
| GENIA, base | 0,9589 | 0,5655 |
| GENIA, **large** | **0,9610** | **0,4674** |
| CoNLL, base | 0,9319 | 0,5110 |
| CoNLL, **large** | **0,9603** | **0,4564** |

O R² **não caiu** — subiu, e no CoNLL subiu de 0,93 para 0,96. Com o dobro da profundidade, a
massa de atenção sobre o span continua explicada em 96% pela fração de chaves que o span ocupa.
A dependência geométrica não é artefato de modelo pequeno.

A razão observado/esperado caiu de ~0,55 para ~0,46 nos dois corpora: o modelo grande concentra
**menos** atenção nos tokens da entidade do que o pequeno, relativamente à parte que lhes cabe.
O desvio do nulo é estável em direção e magnitude em quatro medições (dois modelos × dois
corpora), o que o torna propriedade da atenção e não de um checkpoint.

### C2 — a massa acrescenta sobre a geometria pura, e isso replica

| | base | large |
|---|---|---|
| GENIA | Δ −0,0039 IC [−0,0071; −0,0009] **acrescenta** | Δ −0,0036 IC [−0,0067; −0,0006] **acrescenta** |
| CoNLL | Δ +0,0010 IC [−0,0021; +0,0039] não acrescenta | Δ −0,0002 IC [−0,0029; +0,0028] não acrescenta |

Veredito e magnitude **replicam nas duas escalas**: no GENIA a massa acrescenta sobre a geometria
pura, no CoNLL não. É o único lugar em que a atenção mostra conteúdo próprio, e a magnitude diz
quanto: ΔAURC de −0,0036, com AURC de 0,3829 contra 0,3864 da geometria pura.

**Detectável e operacionalmente nulo.** Na tarefa (T2) a mesma massa é **inalcançável** no GENIA:
não existe cobertura em que ordenar por ela atinja 70% de precisão entre os entregues. Um efeito
que aparece na terceira casa decimal do ΔAURC e desaparece na unidade em que se decide.

### C3 — na escala nova a combinação fica PIOR, e isso é mais forte que o nulo

| | base | large |
|---|---|---|
| GENIA | Δ +0,0000 IC [0; 0] não distingue | Δ **+0,0071** IC [+0,0010; +0,0133] **pior que a base** |
| CoNLL | Δ −0,0004 IC [−0,0015; +0,0008] não distingue | Δ −0,0013 IC [−0,0049; +0,0022] não distingue |

No `base` o peso convexo colapsou para 1,00/0,00 — a calibração **descartou** o enriquecimento, e
o Δ era exatamente zero. No `large` o peso ficou em **0,96/0,04**: a calibração manteve 4% do
enriquecimento, e o resultado foi **pior** que a confiança sozinha, com IC excluindo zero.

Isso é mais informativo que o colapso. Colapso diz "o sinal não acrescenta". Peso pequeno com
resultado pior diz que o sinal **carrega informação enganosa** que sobreviveu à calibração o
suficiente para degradar o veredito.

## Na unidade em que se decide, a distância não diminuiu

T1 na meta de 70% de precisão entre os entregues, sob o `large`:

| supervisor | GENIA | CoNLL-2003 |
|---|---|---|
| `model_confidence` | rev **0,451**, cob 0,549, F1 0,3814 | rev **0,323**, cob 0,677, F1 0,5312 |
| `span_mass` | inalcançável | rev 0,9992, cob 0,001, F1 0,0011 |
| `enrichment` | rev 0,9981, cob 0,002, F1 0,0018 | rev 0,9196, cob 0,080, F1 0,0948 |
| `geometric_residual` | inalcançável | rev 0,9992, cob 0,001, F1 0,0011 |
| `span_size` | inalcançável | inalcançável |
| (sem supervisor) | rev 0,000, cob 1,000, F1 0,4695 | rev 0,000, cob 1,000, F1 0,5051 |

As linhas em que a massa "atinge" a meta atingem-na entregando **0,1% das predições** — duas a
quatro entidades. Alcançável no papel, inútil na prática, e a coluna de cobertura é o que torna
isso visível: sem ela, "rev 0,9992" pareceria apenas caro em vez de vazio.

E os vereditos de tarefa mantêm a direção com IC excluindo zero: **T3** exige mais revisão em
toda meta atingível nas duas escalas (GENIA a 70%: 0,9981 contra 0,4510, Δ +0,5471); **T4** não
distingue em nenhuma; **T2** é inalcançável em todas no GENIA e, no CoNLL a 70%, passa a ser
alcançável no `large` mas exigindo 99,9% de revisão contra 32,3% da confiança.

## O que esta medição fecha, e o que ela não fecha

**Fecha** a objeção do extrator na forma em que ela podia ser feita: o extrator melhorou em erro,
em sobrepredição e em carga de revisão da base, e **nenhum** sinal de atenção passou a ser útil.
A afirmação deixa de ser "num extrator fraco a atenção não acrescenta" e passa a ser "em duas
escalas, com o arranjo melhorando, a atenção não acrescenta".

**Não fecha** a forma forte da objeção: `gliner_large` continua errando 42% e 47%, e um revisor
pode exigir um extrator ajustado ao corpus. Isso é o Perímetro 2, exige GPU, e nenhuma linha
desta tabela responde.

**Uma coisa que ela muda de estatuto:** a inversão de sinal no CoNLL, medida no `base`, era o
achado positivo candidato. No `large` o `span_mass` dá AURC 0,4789 contra acaso 0,4677 — ainda
pior que o acaso, mas a margem caiu de 0,0319 para 0,0112. A inversão **enfraquece com a escala**,
o que a torna menos promissora como fenômeno a explicar mecanisticamente do que parecia.

## Um defeito de método apanhado por esta execução

A `decl-04` é a primeira declaração com comparações de geometria e de tarefa no **mesmo** arquivo.
Isso expôs que `comparisons.py` tratava tudo que não fosse descritivo como veredito de ΔAURC —
inclusive as `task_verdict`, cujo critério declarado é carga de revisão. A primeira execução
imprimiu `T2 [todos] ΔAURC +0.1048 -> PIOR que a base`: uma quantidade que ninguém declarou,
com etiqueta de veredito. Se tivesse entrado no CSV, entraria no paper como pré-registrada.

Corrigido com roteamento por tipo e omissão **declarada em ressalva**, mais dois testes: um varre
todas as declarações exigindo que cada comparação seja reivindicada por exatamente um runner e
que o critério case com o tipo; o outro roda o runner de geometria sobre a tabela real e exige que
as de tarefa fiquem fora com ressalva nomeada.
