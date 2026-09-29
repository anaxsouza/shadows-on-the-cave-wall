# Formalização — o que está estabelecido, o que não está, e o que vem depois

**Data:** 15/09/2026 · **Estado:** ativo. Substrato do paper e da tese.
**Regra deste documento:** cada afirmação vem com o **estatuto** dela. Confirmatório significa
medido na partição de teste sob declaração assinada antes. Exploratório significa medido na
validação, e nenhuma quantidade exploratória decide veredito. Não verificado significa que não
foi medido — e está dito, não omitido.

---

## 0. O sistema medido, em uma linha

Um extrator de entidades baseado em **trechos** (`urchade/gliner_base`, 209 M de parâmetros, 12
camadas × 12 cabeças, atenção bidirecional, não autorregressivo), avaliado em dois corpora de
teste: GENIA (biomédico, com aninhamento) e CoNLL-2003 (notícias, plano).

**Um modelo só, e a razão é a pergunta.** Ela compara a atenção do modelo com a confiança *do
mesmo modelo*. Até 02/09/2026 o registro listava dois encoders crus que não encontram entidade,
enquanto a predição vinha, fixa, de um terceiro modelo — a atenção sairia de um e a confiança de
outro. Isso não é imprecisão, é medida sem sentido.

---

## 1. O problema

Um extrator de entidades **não se cala**: para todo texto ele produz predições, e uma fração
delas está errada. No arranjo medido essa fração é **48,2%** (GENIA) e **53,1%** (CoNLL-2003).

Em uso real isso exige um segundo decisor — um **supervisor** — que não extrai nada e decide
apenas o que é entregue e o que vai para revisão humana. É o arranjo da **predição seletiva**.

O supervisor precisa de um escore por predição. O extrator já fornece um: a confiança que ele
próprio reporta. A pergunta é se existe, **dentro do modelo**, um sinal que decida melhor —
especificamente **onde o modelo estava olhando** quando predisse.

### 1.1 A pergunta única

> A massa de atenção sobre os tokens do trecho acrescenta poder de decisão sobre a confiança que
> o próprio modelo reporta, em predição seletiva para reconhecimento de entidades?

### 1.2 O que o supervisor pode e não pode fazer

Ele **só remove** predições; nunca cria. As entidades anotadas que o extrator nunca apontou são
invisíveis para ele. Consequência exata: o **recall tem teto fixo**, igual ao recall do extrator,
e abster-se só o baixa.

**Consequência que NÃO se segue, e que foi refutada pelo dado:** que o F1 não possa subir. F1 é
troca entre precisão e recall; quando a precisão de partida é baixa, levá-la para cima paga a
perda de recall. Medido no CoNLL: F1 entre os entregues vai de 0,4712 (sem supervisor) para
0,5133 com o supervisor da confiança na meta de 70% de precisão. No GENIA dá o contrário
(0,4392 → 0,3387), porque lá o recall de partida já é baixo. Ver a errata em
`resultados/decl03_tarefa_leitura.md`.

---

## 2. Definições

Sejam `A` a matriz de atenção de uma sentença (média sobre as camadas e cabeças declaradas), `Q`
o conjunto de consultas, `K` o de chaves permitidas, `S ⊂ K` os tokens do trecho predito,
`k = |S|` e `T` o número de tokens.

**Massa de atenção sobre o trecho.**

```
massa(S) = Σ_{q∈Q} Σ_{j∈S} A[q,j]  /  Σ_{q∈Q} Σ_{j∈K} A[q,j]
```

Razão adimensional em [0, 1]. `K` exclui o token sumidouro pela política declarada
(`drop_from_denominator`), e essa escolha não é detalhe: ela muda o resultado por um fator
grande, e por isso é argumento obrigatório e não valor padrão.

**Curva risco-cobertura.** Ordena-se por escore decrescente (escore alto = entregar) e, para
cada cobertura, mede-se o risco entre os entregues. **AURC** é a área sob essa curva; menor é
melhor. Pontos são reportados apenas em fronteira de grupo de empate — empate não tem ordem
interna, e cortar dentro dele reportaria cobertura que o supervisor não realiza.

**Ponto de operação.** Declara-se a **qualidade alvo** e maximiza-se a cobertura sujeita a ela
(convenção de Geifman & El-Yaniv, 2017: declara-se risco, não cobertura). A **carga de revisão**
é a fração não entregue. Busca-se o **maior** prefixo que atinge a meta, e não o primeiro:
precisão não é monótona na cobertura, e parar no primeiro atribuiria ao supervisor uma carga que
ele não precisa pagar.

**Casamento com a anotação:** estrito — mesma fronteira e mesmo rótulo. Trecho com fronteira
errada é entrega errada; chamá-lo de acerto parcial infla o desempenho na direção desejada.

---

## 3. A régua: o valor esperado exato (contribuição teórica)

### 3.1 O nulo

Cada linha de `A` soma 1 — o orçamento de atenção é fixo. Sob o nulo em que a identidade do
trecho é **permutável** com as demais chaves, sortear quais `k` das `|K|` chaves formam o trecho
dá, por linearidade da esperança:

```
E[massa(S)] = k / |K|,     com |K| = T − 1 sob a política de sumidouro declarada
```

**Não é aproximação.** É consequência da estocasticidade das linhas. A distinção importa: com
previsão aproximada, parte do desvio observado seria erro da fórmula, e não haveria como
atribuí-lo ao modelo.

### 3.2 Verificação, não afirmação

Permutando quais colunas contam como trecho, em 716 entidades dos dois corpora com 200
permutações cada: desvio médio entre a massa permutada e `k/(T−1)` de **+0,00013** e **+0,00017**
— ruído de Monte Carlo. Os desvios **observados** são −0,043 e −0,058: **327× e 335×** maiores.
A derivação está certa e o afastamento do nulo é real. *(Estatuto: verificação numérica,
reproduzida em teste sobre matriz aleatória gama-normalizada para não depender da forma da
distribuição.)*

### 3.3 Os dois instrumentos desconfundidos

- **Enriquecimento** = massa observada ÷ esperada. Vale 1,0 quando o trecho recebe exatamente a
  parte dele. Corrige por razão e **sobrecorrige**.
- **Resíduo geométrico** = massa menos o ajuste linear em `k/|K|`. Corrige por regressão e
  **subcorrige**.

Os dois coexistem por decisão declarada: erram em direções opostas, e relatar ambos impede
escolher depois o que der o número desejado.

### 3.4 Por que trocar de fórmula não escapa

O orçamento é fixo, logo **qualquer agregado dos pesos** herda dependência do comprimento. É o
que explica — em vez de apenas constatar — por que nenhuma variante da fórmula escapa do
confundidor. Medido: **2.496 leituras por corpus** (4.992 no total), variando direção, política
de sumidouro, faixa de camadas e conjunto de cabeças; nenhuma escapa. *(Estatuto: exploratório,
validação.)*

### 3.5 Posicionamento na literatura

Saghir (2026, arXiv 2605.00269) estabelece dependência do comprimento para sinais de caixa
branca, variando o **comprimento da entrada inteira**, e relata o sinal colapsando para o acaso.
Aqui a sentença é **fixa** e varia a **fração dela que o trecho ocupa** — caso para o qual existe
valor esperado exato **por entidade**, que não havia sido escrito. E onde eles relatam colapso,
mediu-se **inversão** num dos corpora: pior que o acaso, não apenas neutro.

---

## 4. O método

**Pré-registro em duas peças.** `PROTOCOLO.md` diz quais itens existem e qual o espaço de cada
um (reutilizável, agnóstico de modelo). A **declaração** diz quais valores estão em vigor, é
imutável, e tem hash do próprio conteúdo.

**Hash partido em dois.** `measurement_hash` cobre o que produz a tabela (sumidouro, camadas,
cabeças); `declaration_hash` cobre tudo. Uma declaração que muda só a análise reusa a medição
sem remedir, e a herança é **declarada**, com id e hash da origem, não inferida.

**Conjunto de itens versionado.** Acrescentar item ao hash mudaria o hash de declarações já
assinadas, e assinatura que deixa de ser reproduzível não prova mais ordem nenhuma.

**As três declarações assinadas:**

| id | hash | assinada | o que declara |
|---|---|---|---|
| `decl-01-gliner-base` | `348e90cce8ca9874` | 02/09/2026 | a medição: sumidouro, camadas, cabeças, partição, semente |
| `decl-02-geometria` | `4d91c3607206b9ab` | 10/09/2026 | os adversários geométricos e os critérios em AURC |
| `decl-03-tarefa` | `5f30a2ff3e493c19` | 15/09/2026 | a grade de metas de qualidade e os critérios em carga de revisão |

As três compartilham `measurement_hash` `8a798025fb755a32`: nenhuma remediu nada.

**Separação por sentença, não por entidade.** Calibração e avaliação se separam por sentença
porque entidades da mesma sentença compartilham a matriz de atenção. Separar por entidade vaza
contexto de um lado para o outro — vazamento que não aparece como erro, aparece como resultado
bom. O intervalo de confiança segue a mesma unidade: reamostragem de **conglomerado**.

**Nota de método, registrada por ter sido um erro meu:** "reamostragem por conglomerado alarga o
intervalo" é teorema para estatística **suave** sob correlação positiva. Carga de revisão é
**função degrau** — o maior prefixo que atinge a meta — e no dado com efeito de sentença o
intervalo por entidade saiu mais largo. A justificativa para reamostrar sentenças é o **desenho
amostral**, nunca a largura que sai.

---

## 5. O que está estabelecido (confirmatório)

### 5.1 A massa de atenção é, quase toda, geometria

| | R² massa × fração geométrica | mediana observado/esperado |
|---|---:|---:|
| GENIA | **0,9589** | 0,5655 |
| CoNLL-2003 | **0,9319** | 0,5110 |

93–96% da variação da massa é explicada por dois números que não dependem do modelo: o tamanho
do trecho e o da sentença. E o trecho recebe cerca de **metade** da parte que o nulo prevê,
estável nos dois corpora — propriedade real do modelo, não artefato.

### 5.2 Em AURC, a diferença existe e é minúscula

| comparação | GENIA | CoNLL-2003 |
|---|---|---|
| `C2` massa × fração geométrica pura | ΔAURC −0,0039, IC [−0,0071; −0,0009] → **acrescenta** | +0,0010, IC [−0,0021; +0,0039] → **não acrescenta** |
| `C3` confiança+enriquecimento × confiança | **não acrescenta** (peso colapsou 1,00/0,00) | **não acrescenta** |

### 5.3 Na tarefa, a diferença é enorme

| veredito | resultado |
|---|---|
| `T2` massa × confiança do modelo | **inalcançável para a massa nas quatro metas, nos dois corpora** — não existe cobertura em que ordenar pela atenção atinja 70% de precisão entre os entregues |
| `T3` enriquecimento × confiança | exige **mais** revisão em toda meta atingível, IC excluindo zero. GENIA 70%: 99,65% contra 56,49% (Δ +0,4316). CoNLL 70%: 97,54% contra 42,64% (Δ +0,5490) |
| `T4` combinação × confiança sozinha | **não distingue** em todas as metas. GENIA: Δ exatamente 0,0000, IC [0,0000; 0,0000], porque o peso colapsou para 1,00/0,00 — os dois escores são o mesmo vetor |

### 5.4 A unidade muda a magnitude, não o sinal

Em ΔAURC as diferenças são de ordem 0,004; em carga de revisão, de 0,43 a 0,55. AURC é média
sobre a curva inteira, inclusive a região de cobertura alta onde qualquer ordenação entrega quase
tudo. Os pontos de operação declarados ficam na região de **precisão alta**, onde a ordenação tem
de ser genuinamente boa — e é lá que o sinal falha por completo.

### 5.5 O custo do melhor supervisor disponível

Para atingir 70% de precisão entre os entregues, o melhor supervisor — a confiança do próprio
modelo — exige revisar **42,6%** (CoNLL) e **56,5%** (GENIA) das predições. Isso é propriedade do
**extrator**, que erra metade.

---

## 6. Medido, mas não veredito (exploratório)

- **Invariância da conclusão à leitura.** 2.496 leituras por corpus, na validação. **1.961 das
  4.992** têm ΔAURC negativo contra a confiança, e a melhor isolada dá −0,0178 — 4,5× o efeito
  confirmatório. Exatamente por isso a varredura **não pode** ser afirmação: o máximo sobre
  milhares de leituras é estatística enviesada. Ela mostra que a conclusão não depende da leitura
  escolhida, e qualquer leitura que pareça boa aqui é candidata a declaração futura.
- **A inversão de sinal no CoNLL**, explicada pela decomposição `k/T`. No CoNLL sentenças curtas
  erram mais (taxa de erro 0,606 contra 0,429; `corr(erro, T) = −0,147`). A massa é ~`k/T`, logo
  massa alta ≈ sentença curta: entregar por massa decrescente é entregar as sentenças curtas
  primeiro, que é a ordenação `1/T` — e essa ordenação dá AURC **0,5721 contra acaso 0,5169**,
  isto é, **pior que o acaso**. A massa herda isso, e é daí que vem a inversão. (A ordenação
  oposta, `T` maior primeiro, dá 0,4414 e bate o acaso — mas é uma contagem de tokens, não um
  sinal do modelo.) No GENIA o erro é plano nos quartis de `T` e a inversão não aparece. Vira
  **previsão testável** em corpus novo: onde sentenças curtas errarem mais, a massa será
  anti-preditiva.
- **Paradoxo de Simpson no enriquecimento** (GENIA): agregado e estratos por tamanho de trecho
  dão sinais opostos, porque trechos de 1 token têm simultaneamente erro alto e enriquecimento
  alto. Sem estratificar, a **direção** do efeito não é interpretável. É também a razão pela qual
  as bordas das faixas são item declarado.
- **A varredura fica na validação por decisão registrada** (`PROTOCOLO.md` §5): repeti-la no
  teste não acrescentaria nada confirmatório e custaria 2.496 olhares na partição confirmatória.

---

## 7. Os limites — o que NÃO foi medido

1. **Um extrator só, e ele erra metade.** É a objeção mais previsível e nenhuma linha das tabelas
   a responde. Com precisão de partida em 0,47–0,52, um revisor pode alegar que a faixa dinâmica
   dos dois sinais estava comprimida e que o nulo é do arranjo.
2. **Uma escala só.** `gliner_large` não foi medido; é a verificação de robustez mais barata
   disponível e não muda a arquitetura, só a escala.
3. **Uma arquitetura só, e bidirecional.** Nada aqui foi medido em modelo autorregressivo. Ver §8.1.
4. **A garantia conformal está inconclusiva** pelo critério declarado, por violação da hipótese
   de monotonicidade. Não foi determinado se a violação é intrínseca ao escore ou artefato do
   tamanho da partição de calibração — e essa é pergunta respondível com o dado já no disco.
5. **Rótulo trocado × trecho espúrio não foram separados.** O casamento é estrito, então as duas
   formas de erro entram na mesma coluna. Separá-las exige alterar o adaptador e remedir.
6. **Dois corpora, um idioma.**

---

## 8. As perguntas abertas, ordenadas por custo

### 8.1 O nulo com máscara causal — a pergunta do modelo autorregressivo

**A hipótese levantada:** a diferença pode ser maior em decoders, pelo gargalo sequencial dos
modelos autogerativos.

**O que é independente de arquitetura.** O confundidor não desaparece: `E[massa] = k/|K|` vem da
soma unitária das linhas, que vale para qualquer atenção softmax — encoder, decoder ou
encoder-decoder.

**O que muda, e é derivável.** Com máscara causal, a consulta na posição `i` só vê chaves até
`i`, então o número de chaves permitidas **depende da consulta**. Para um trecho em
`[a, a+k−1]`, somando sobre as `T` consultas:

```
E[massa] = (1/T) · [ Σ_{i=a}^{a+k−1} (i−a+1)/(i+1)  +  k · (H_T − H_{a+k}) ]
```

com `H_n` o n-ésimo número harmônico. Aproximadamente `(k/T)·ln(T/(a+k))`. O nulo passa a
depender da **posição** `a`: trecho no começo da sentença é visto por todas as consultas
posteriores; trecho no fim é visto por poucas.

**Para que lado isso cai — e é o contrário do palpite.** A máscara causal **acrescenta uma
dimensão ao confundidor** (posição, além de tamanho e comprimento), e posição em NER correlaciona
com estrutura sintática e com tipo de entidade. Além disso, num modelo autorregressivo a
confiança do próprio modelo é a verossimilhança da sequência, que agrega muitas decisões por
token e é **base mais forte** de bater. Nos dois eixos a comparação fica **mais difícil** para a
atenção, não mais fácil.

**Quanto isso pesa, medido com a fórmula.** Em `T = 60`: um trecho de **1 token em `a = 0`** tem
massa esperada **0,0780**; um de **4 tokens em `a = 45`** tem **0,0169**. O de um token recebe
4,6× o de quatro. No caso bidirecional é o oposto e exato: quatro tokens recebem **quatro vezes**
um token, porque lá só o tamanho conta. Num decoder, **a posição domina o tamanho e na direção
contrária.**

*(Estatuto: **verificada**. `expected_mass_causal` está em `src/selective/geometry.py`, e o teste
permuta os pesos entre as chaves permitidas de cada linha sobre matriz causal linha-estocástica
de densidade gama — não uniforme, de propósito. Desvios abaixo de 2,7 erros-padrão da média em
seis configurações de `(T, a, k)`. Nenhuma declaração vigente a usa: o modelo medido é
bidirecional. Ela existe para esta pergunta e para deixar registrado que a régua generaliza — com
outra fórmula, não com a mesma.)*

**Custo:** extrator generativo, GPU, declaração nova, e o nulo re-derivado — mais a reabertura do
roster de modelos, que a reorganização de 02/09 fechou.

### 8.2 Extrator competente, mesma metodologia
Fecha o limite 1, que é a objeção que decide se o negativo é decisivo ou refutável em revisão.
Exige GPU e declaração nova. Nenhum outro resultado a responde.

### 8.3 A monotonicidade: intrínseca ou artefato?
Respondível com o dado já medido, sem GPU e sem declaração nova se ficar como exploratório.
É o mais barato dos três e fecha o limite 4.

### 8.4 Segunda escala do mesmo modelo
Mais barata que 8.1 e 8.2 e responde metade do limite 1: se o nulo sobrevive a um modelo maior
da mesma família, "o arranjo é fraco" perde força.

---

## 9. O que o trabalho vale, e o que não vale

**Vale:** um valor esperado exato que não havia sido escrito para esta medida, verificável por
permutação e reutilizável por quem construir instrumentos de atenção; a medida de um desvio real
e estável (metade da parte prevista, dois corpora); uma previsão testável em corpus novo; e um
negativo **decisivo** em vez de refutável — pré-registro assinado antes, peso colapsado em vez de
diluído, geometria controlada explicitamente e intervalo pela unidade certa.

**Não vale:** nada aqui prova que atenção é inútil em geral. O extrator erra metade, e isso está
declarado como limite, não escondido.
