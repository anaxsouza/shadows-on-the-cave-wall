# A violação de monotonicidade é intrínseca ao escore ou artefato do tamanho da calibração?

**ESTATUTO: exploratório, partição de validação, sem declaração.** Este documento não é
veredito. Responde a §8.3 de `FORMALIZACAO.md` com o dado já medido, e o resultado deve
entrar na tese como discussão do limite 4 (§7, item 4) — não como afirmação confirmatória.
Nenhuma declaração nova foi criada; alpha, `loss_bound` e `sink_policy` foram REUTILIZADOS
de `decl-03-tarefa` (0,05 / 1,0 / `drop_from_denominator`) só para reproduzir o mesmo
procedimento de `crc_threshold`, sem assinar compromisso sobre o resultado.

## Método

Para cada corpus (GENIA, CoNLL-2003) e para `model_confidence`, `span_mass` e `enrichment`,
a partição de **calibração** foi variada por subamostragem SEM reposição de **sentenças**
(nunca de entidades — entidades da mesma sentença partilham a matriz de atenção, `_separar_por_sentenca`
em `comparisons.py`), em frações de 5% a 100% do total de sentenças da partição de
**validação** de cada corpus (GENIA: 1.582 sentenças / 4.932 entidades; CoNLL-2003: 3.115
sentenças / 8.469 entidades). Para cada tamanho, `crc_threshold` (`conformal.py`) foi chamado
200 vezes (menos repetições nos tamanhos maiores, por custo computacional; 1 vez no tamanho
100%, que é determinístico) e `max_monotonicity_violation` foi registrada.

A partição de **teste** não foi tocada — o experimento inteiro roda dentro de
`results/gliner-base/{genia,conll2003}/validation/`.

## CORREÇÃO da sessão principal (15/09/2026) — leia antes do resto

Este documento foi produzido por uma frente delegada e **os números dele conferem**: os seis
expoentes log-log e todos os extremos foram recalculados do próprio CSV e reproduzem
exatamente, e `frac_monotona` é 0,0 nas 60 linhas. Duas afirmações interpretativas, porém, não
se sustentam, e a segunda inverte a conclusão.

**Primeira: a frase-resumo generaliza além da tabela.** Ela diz que `span_mass` e `enrichment`
"platôam nos dois corpora, expoentes entre −0,20 e +0,10". O `span_mass` do CoNLL-2003 tem
expoente **−0,49** e encolhe 3,8× — a mesma assinatura de ruído do ``model_confidence``. O único
escore que platôa nos DOIS corpora é o `enrichment` (−0,20 e +0,10); o `span_mass` é misto.
A tabela de números do relatório traz o −0,49 correto; é o resumo que agrupou errado.

**Segunda, e é a que decide a pergunta: a varredura testou o tamanho da calibração, mas a
quantidade medida é dominada por outra coisa.** A violação aparece em frações pequenas e exatas
— 1/36, 1/30, 1/15, 1/12, 1/7, 1/4 — e isso é assinatura de quantização, não de magnitude.
`max_monotonicity_violation` é o maior RECUO do risco entre pontos consecutivos da varredura de
candidatos, e a varredura começa no escore mais alto: no primeiro ponto o conjunto entregue tem
**3 entidades**, e ali o risco só pode valer 0, 1/3, 2/3 ou 1. Uma única entidade mudando de
lado produz um "recuo" de 1/3.

Medindo ONDE a violação máxima ocorre, na validação inteira:

| corpus | escore | violação total | cobertura no ponto | entidades entregues ali | violação com n ≥ 30 | com n ≥ 100 |
|---|---|---:|---:|---:|---:|---:|
| GENIA | `model_confidence` | 0,0833 | 0,06% | **3** | 0,0034 | 0,0017 |
| GENIA | `span_mass` | 0,0833 | 0,06% | **3** | 0,0092 | 0,0033 |
| GENIA | `enrichment` | 0,1429 | 0,12% | **6** | 0,0218 | 0,0061 |
| CoNLL-2003 | `model_confidence` | 0,0007 | **47,2%** | **3.998** | 0,0007 | 0,0007 |
| CoNLL-2003 | `span_mass` | 0,0659 | 0,15% | **13** | 0,0242 | 0,0119 |
| CoNLL-2003 | `enrichment` | 0,1667 | 0,04% | **3** | 0,0080 | 0,0025 |

Em cinco dos seis casos a violação máxima vive no extremo de cobertura quase nula, com 3 a 13
entidades entregues. Restringindo à região onde o conjunto entregue tem ao menos 100 entidades,
a violação cai por fatores de **5,5× a 65,7×** nesses cinco casos — o maior encolhimento é o
`enrichment` do CoNLL, de 0,1667 para 0,0025 (65,7×), e o menor é o `span_mass` do CoNLL, de
0,0659 para 0,0119 (5,5×). Os fatores por caso estão em `monotonicidade_onde_ocorre.csv`,
coluna `razao_total_sobre_n100`.

**Por que isso explica o platô e desfaz a conclusão.** O platô não apareceu porque a violação é
intrínseca ao escore: apareceu porque o topo da ordenação tem ~3 entidades **independentemente
do tamanho da calibração**. Aumentar a calibração não move o piso de granularidade daquele
extremo, então nenhum expoente log-log poderia detectar o que a varredura foi procurar. A
varredura mediu o tamanho da calibração; a quantidade dominante era a granularidade do risco na
cobertura mínima.

**A resposta honesta a §8.3, portanto, é: não determinável com esta medida de violação** — e
há uma exceção informativa, o `model_confidence` no CoNLL-2003, cuja violação máxima ocorre em
cobertura de 47% com 3.998 entidades entregues e **não** encolhe sob restrição. Essa é a única
violação que vive na região onde a garantia seria de fato usada, e ela é minúscula (0,0007).

**O que NÃO foi feito, e por quê.** Não alterei `conformal.py`. A medida de violação como está
implementada reporta o extremo da curva, que não é onde a garantia é usada — reportar também a
violação restringida a uma região utilizável é conserto defensável, mas o resultado
"inconclusivo" do braço conformal foi relatado sob declaração assinada, e trocar a medida
mudaria esse relato retroativamente. É item para declaração nova, e a decisão é do autor.

---

## Achado 1 — a violação nunca desaparece na faixa disponível

Em **nenhuma** das 6.366 reamostragens — nenhum corpus, nenhum escore, nenhum tamanho de
calibração testado, incluindo a partição de validação INTEIRA (a maior amostra que existe em
disco sem tocar o teste) — o limiar resultante foi monótono (`frac_monotona = 0,0` em toda
linha da varredura). Isso por si só já diz algo: dentro do que está medido, aumentar a
calibração não elimina a violação para NENHUM dos três escores.

## Achado 2 — mas a MAGNITUDE se comporta de forma diferente por escore

A tabela abaixo resume o encolhimento da violação (mediana) do menor para o maior tamanho de
calibração testado, e o expoente de um ajuste log-log (`log(violação) ~ expoente · log(n) `).
Um expoente perto de `-0,5` é a assinatura esperada de RUÍDO DE AMOSTRA PEQUENA — a mesma taxa
`1/√n` da variância de uma média amostral. Um expoente perto de `0` é ausência de tendência
com o tamanho — a violação PLATÔ, não artefato de tamanho.

| corpus | escore | n min | n max | violação mediana (n min) | violação mediana (n max) | razão de encolhimento | expoente log-log |
|---|---|---:|---:|---:|---:|---:|---:|
| conll2003 | enrichment | 423 | 8469 | 0.1583 | 0.1667 | 0.95× | +0.10 |
| conll2003 | model_confidence | 427 | 8469 | 0.0026 | 0.0007 | 3.65× | -0.48 |
| conll2003 | span_mass | 425 | 8469 | 0.2500 | 0.0659 | 3.79× | -0.49 |
| genia | enrichment | 246 | 4932 | 0.2500 | 0.1429 | 1.75× | -0.20 |
| genia | model_confidence | 246 | 4932 | 0.0333 | 0.0833 | 0.40× | +0.47 |
| genia | span_mass | 247 | 4932 | 0.1500 | 0.0833 | 1.80× | -0.20 |

## Leitura

**`model_confidence` em CoNLL-2003 tem a assinatura de artefato de tamanho.** A violação cai
quase 10× (de 0,0073 para 0,0007) enquanto a calibração cresce ~20×, e o expoente do ajuste
log-log (-0,48) é quase exatamente o -0,5 esperado de ruído de amostra pequena. Este é o único
par escore/corpus em que os dados sustentam com clareza a hipótese de artefato.

**`model_confidence` em GENIA não mostra o mesmo padrão.** A violação NÃO encolhe de forma
sistemática — oscila entre 0,025 e 0,167 conforme o tamanho, e no maior tamanho disponível
(4.932 entidades) está em 0,083, MAIOR que no menor tamanho testado (246 entidades, 0,033). O
expoente log-log é positivo (+0,47), o oposto do esperado sob a hipótese de artefato. Não dá
para separar ruído de reamostragem específico deste corpus (GENIA é ~3× menor que CoNLL) de um
efeito real — a leitura aqui é **inconclusiva**, não "artefato confirmado".

**`span_mass` e `enrichment` — os escores que sustentam a contribuição de C4 — platôam nos
dois corpora.** Os expoentes log-log ficam entre -0,20 e +0,10, longe da assinatura -0,5 de
ruído amostral, e a violação estabiliza numa faixa substancial (7%–25%) mesmo na maior
calibração disponível. Em CoNLL, `enrichment` nem chega a encolher (razão 0,95×, isto é,
violação praticamente igual em n=423 e n=8.469). Esta é a leitura mais forte deste
levantamento: para os DOIS escores geométricos, nos DOIS corpora, o comportamento da violação
com o tamanho é qualitativamente diferente do que se vê em `model_confidence` em CoNLL (o
único caso com assinatura limpa de artefato) — o que aponta para violação **intrínseca ao
escore**, dentro do que os dados em disco permitem discriminar.

**O limite do que esta análise pode dizer.** A maior calibração testável é a partição de
validação inteira (1.582 sentenças em GENIA, 3.115 em CoNLL) — um teto que não pode ser
ultrapassado sem tocar a partição de teste, reservada para os testes confirmatórios C1-C3/T1-T4.
Não é possível, com o dado em disco, descartar que uma calibração uma ordem de grandeza maior
(mais dado do que existe hoje para estes corpora) eventualmente encolhesse a violação de
`span_mass`/`enrichment` também. O que os dados permitem afirmar é comparativo: dentro da faixa
alcançável, a violação de `model_confidence` em CoNLL tem a assinatura estatística de artefato
de tamanho e a de `span_mass`/`enrichment` não tem — nos dois corpora.

**Nota de procedência.** No log da execução confirmatória de `decl-03-tarefa`
(`confirmatorio_decl03_notas.txt`, partição de TESTE), o escore combinado usado no teste de
valor adicionado (`model_confidence+enrichment`) colapsou para peso 0,97–1,00 em
`model_confidence` nos dois corpora — ou seja, o limiar conformal do teste de valor adicionado
já citado em `FORMALIZACAO.md` §7 é, na prática, dominado por `model_confidence`, não pelos
escores geométricos. Isso não foi remedido aqui (mediria a mesma coisa que `model_confidence`
sozinho, a menos do peso residual de 0%–3% em `enrichment`) — é citado só para registrar que a
peça mais bem-comportada (`model_confidence`) é a que domina o limiar já reportado, e a peça
com violação mais robusta (`span_mass`/`enrichment`) é a que aparece isolada no teste do piso
(`floor`, que usa `span_mass` diretamente como escore de calibração) e nas comparações
descritivas C1-C4.

## Decisão que fica para o autor

Esta caracterização é exploratória e não fecha o limite 4 como veredito — só como discussão
informada. Dois caminhos, com custo diferente:

1. **Escrever como discussão/limitação, sem nova declaração.** Custo: nenhum trabalho adicional;
   a tese registra que a violação foi CARACTERIZADA (não resolvida) e que a evidência aponta
   para origem diferente por escore — artefato plausível em `model_confidence`/CoNLL,
   inconclusivo em `model_confidence`/GENIA, provavelmente intrínseca em `span_mass`/`enrichment`
   nos dois corpora. O limite 4 permanece "não determinado" no sentido confirmatório, mas deixa
   de ser uma lacuna muda.
2. **Pré-registrar um teste confirmatório novo** (critério explícito para "artefato" vs
   "intrínseco", ex.: expoente log-log acima/abaixo de um corte declarado a priori) para decidir
   com força de veredito. Custo: exige declaração nova (`decl-0X`) e dado que ESTA análise já
   olhou toda a partição de validação disponível — o teste confirmatório precisaria rodar sobre
   dado ainda não visto, o que aqui só existe na partição de teste, reservada aos testes C1-C4/
   T1-T4. Fechar esta pergunta com força de veredito comete a partição de teste a mais uma
   pergunta, o que não estava no escopo original dela.

Não escolhi entre os dois — é decisão de escopo e ambição da tese, não de implementação.
