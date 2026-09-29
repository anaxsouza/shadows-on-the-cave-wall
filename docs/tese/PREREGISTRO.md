# PREREGISTRO — declaração `decl-01-gliner-base`

**Estado:** ASSINADA em 2026-09-02. Os valores abaixo deixaram de ser proposta e passaram a ser
compromisso: nenhum deles muda depois de ver resultado. Medir sob outros valores exige uma
declaração NOVA — outro `declaration_id`, outro hash —, nunca a edição desta.
**Instância do protocolo:** `docs/tese/PROTOCOLO.md` — ele fixa *quais* itens existem, o
espaço de cada um e os critérios; esta declaração fixa *quais valores* estão em vigor.
**Hash do conteúdo declarado:** `348e90cce8ca9874`
**Fonte executável:** `configs/config.yaml`, seção `selective`. Este documento e essa seção
são a mesma declaração em dois formatos; divergência entre eles quebra
`tests/test_threshold_contract.py`.

## Os valores em vigor

| Item | Valor | Onde se justifica |
|---|---|---|
| `sink_policy` | `drop_from_denominator` | §5 |
| `layers` | todas as 12 do `gliner_base` | §2 |
| `heads` | todas | §2 |
| `combination_rule` | `convex` | §6 |
| `calibration_fraction` | 0,30 das **sentenças** | §3 |
| `coverage_levels` | 0,50 / 0,70 / 0,80 / 0,90 / 0,95 | curva reportada |
| `operating_point` | `derived_from_target_risk` | PROTOCOLO §2 |
| `target_risk_grid` | 0,01 / 0,02 / 0,05 / 0,10 (relato) | PROTOCOLO §2 |
| `conformal_alpha` = r* da **garantia** | 0,05 | §6 |
| `added_value_ci_level` | 0,95 | §4 |
| `n_bootstrap_resamples` | 2.000 | §4 |
| `loss_bound` / `seed` | 1,0 / 42 | perda 0/1 |

**O item que saiu.** `entity_confidence_aggregation` foi removido em 02/09/2026: o GLiNER
entrega **um** escore por trecho, então não há o que agregar dentro da entidade, e nenhuma
conta consumia o item — era resquício do desenho com etiquetador BIO. Item de pré-registro
que não entra em conta nenhuma dá impressão de rigor sem o rigor. A volta dele quebra
`tests/test_threshold_contract.py`.

---

## Para que serve, e o que ele não pode fazer

A banca fez 162 apontamentos, e a família `R4` — sete deles — é sobre cortes numéricos sem
sistemática: limiares que aparecem no texto sem que nada explique por que aquele número. A
resposta não é escolher números melhores. É declarar cada um deles **antes** de ver o
resultado, e deixar rastro se algum mudar depois.

O que este documento pode garantir é exatamente isso: rastro. Ele não impede a mudança —
nada impede. Mudar um item exige editar dois arquivos versionados, o que aparece no
`git log` com data e autor. É pouco, e é tudo o que um pré-registro faz.

O que o **código** garante é mais forte: `src/selective/preregistration.py` não tem valor
padrão para nenhum dos seis itens. Sem a seção `selective` no config, `load_preregistration`
levanta erro e nada roda. Um padrão em código seria uma escolha que alguém pode ter feito
depois de ver um resultado.

---

## Item 1 — Regra de agregação de confiança por entidade

**Valor:** `min` (o menor valor entre os tokens da entidade)
**Chave:** `entity_confidence_aggregation`

Uma entidade tem vários tokens, cada um com sua probabilidade. Reduzir isso a um número por
entidade é uma escolha, e as candidatas dão ordenações diferentes: o mínimo, a média, a média
geométrica, ou a probabilidade do primeiro token.

`min` é a escolha conservadora e é a que corresponde ao uso: uma entidade cuja fronteira é
duvidosa **é** duvidosa, mesmo que os tokens internos estejam saturados. A média dilui
exatamente o sinal que interessa — um token fraco entre cinco fortes desaparece na média e
domina o mínimo. Como a decisão é abster-se ou entregar a entidade inteira, o elo mais fraco
é a quantidade relevante.

---

## Item 2 — Convenção de sumidouro e faixa de camadas e cabeças

**Valores:** `sink_policy: drop_from_denominator`, `layers: [8..15]`, `heads: null` (todas)
**Chaves:** `sink_policy`, `layers`, `heads`

**O sumidouro.** Em transformadores treinados, o primeiro token ([CLS] ou BOS) recebe uma
fração desproporcional da atenção de todas as consultas sem carregar conteúdo proporcional.
Mantê-lo no denominador esmaga a massa dos tokens de conteúdo num intervalo estreito perto de
zero, e comprimir escala destrói poder de ordenação. Excluí-lo torna a razão "entre os tokens
de conteúdo, quanto vai para o span", que é a pergunta que interessa.

O efeito é grande, e por isso a convenção é item de pré-registro e não detalhe de
implementação: no exemplo verificado em `tests/test_selective_conformal.py`, a mesma entidade
do mesmo modelo tem massa 0,30 com o sumidouro no denominador e 0,75 sem ele — **2,5 vezes**.

**A faixa de camadas.** Terço médio de um modelo de 24 camadas. As camadas iniciais carregam
sobretudo posição e as finais especializam-se na tarefa; a estrutura de span vive no meio.
Todas as cabeças, porque selecionar cabeça por desempenho observado seria escolher depois de
ver o resultado — precisamente o que este documento existe para impedir.

---

## Item 3 — Regra de combinação e partição de calibração

**Valores:** `combination_rule: logistic`, `calibration_fraction: 0.3`
**Chaves:** `combination_rule`, `calibration_fraction`

A regressão logística sobre os dois sinais (confiança do modelo, massa de atenção) é
ajustada **na partição de calibração** e aplicada na de avaliação. Ajustar na calibração é o
que torna a combinação honesta: coeficientes escolhidos olhando a avaliação fariam o ganho
medido incluir o próprio ajuste.

**A partição é por SENTENÇA, não por entidade**, e 30% é fração de sentenças. Duas entidades
da mesma sentença compartilham a matriz de atenção e o contexto: separá-las entre calibração e
avaliação deixa informação passar de uma para a outra. Esse vazamento não aparece como erro —
aparece como resultado bom, que é o motivo de estar declarado aqui em vez de comentado no
código.

---

## Item 4 — Níveis de cobertura e o ponto operacional único

**Valores:** `coverage_levels: [0.5, 0.7, 0.8, 0.9, 0.95]`, `operating_point_coverage: 0.8`
**Chaves:** `coverage_levels`, `operating_point_coverage`

A curva risco-cobertura é reportada inteira: ela É a varredura de todos os limiares, e é essa
varredura que responde à objeção `R4`. Os cinco níveis acima são os pontos tabelados no texto.

O **ponto operacional é único** e declarado antes: cobertura de 80%. Reportar o melhor ponto
depois de ver a curva seria escolher o corte pelo resultado, exatamente a prática criticada.
O 80% não é ótimo de nada — é uma escolha de operação, entregar quatro de cada cinco entidades
e abster-se da quinta, feita antes de existir número para olhar.

---

## Item 5 — Critério de valor adicionado e número de reamostragens

**Valores:** `added_value_ci_level: 0.95`, `n_bootstrap_resamples: 2000`
**Chaves:** `added_value_ci_level`, `n_bootstrap_resamples`

O critério é o intervalo de confiança de 95% para ΔAURC **excluir zero**, na direção negativa
(AURC menor é melhor). Sem p-valor, e portanto sem família de testes e sem correção de
multiplicidade — ver `docs/ARCHITECTURE.md` §8.

A reamostragem é **pareada**: cada uma das 2.000 sorteia entidades e recalcula as duas AURC no
mesmo sorteio, porque os dois escores são medidos nas mesmas entidades. Sortear
independentemente trataria como independentes duas quantidades que compartilham a amostra
inteira, e daria intervalo largo demais — erro na direção de parecer conservador, que é pior
do que errar de forma visível.

**Refutação declarada:** o intervalo conter zero refuta o valor adicionado. Um intervalo cuja
semilargura excede o próprio estimador em módulo é **inconclusivo**, não negativo: a amostra
não resolve o sinal do efeito, e reportar isso como ausência de efeito seria afirmar mais do
que a medida permite.

---

## Item 6 — Riscos alvo: a grade de relato e o nível da garantia

**Valores:** `target_risk_grid: [0.01, 0.02, 0.05, 0.1]` (relato) e
`conformal_alpha: 0.05` (garantia), com `loss_bound: 1.0`, perda 0/1
**Chaves:** `target_risk_grid`, `conformal_alpha`, `loss_bound`

O risco alvo r\* é a fração de entidades **entregues** que se admite errada. Duas coisas
distintas usam esse número, e separá-las é o que resolve uma assimetria do desenho.

**A grade, para relato.** Geifman & El-Yaniv (2017) não declaram cobertura: a Tabela 1 deles
é indexada por r\* em {0,01 … 0,06} e reporta a cobertura obtida em cada um. Adotamos a mesma
forma, e ela não custa grau de liberdade nenhum: os critérios dos dois testes são intervalos
sobre AURC — sobre a curva inteira — e não olham para r\*. O que a grade resolve é concreto:
a taxa de erro base do modelo não se conhece antes de medir, então não se sabe de antemão se
um r\* apertado é alcançável. Com a grade, "0,01 é inalcançável em qualquer cobertura" sai
como linha da tabela, que é um resultado, em vez de execução perdida.

**O nível da garantia, único.** O *conformal risk control* (arXiv 2208.02814) admite um só
nível, e ele é **α = 0,05** — o meio da faixa com precedente (0,01–0,06 em Geifman &
El-Yaniv; 0,1 em Angelopoulos & Bates; 0,1 e 0,05 em Singer et al.), correspondendo às 95%
de cobertura que Singer et al. reportam. Ele tem de estar **dentro** da grade, e o
carregador recusa se não estiver: senão o único número que sustenta a garantia seria o único
que não se lê na tabela de relato.

O procedimento escolhe o limiar de **maior cobertura** cujo risco empírico na calibração
satisfaz `R̂(λ) ≤ α − (B−α)/n`. A correção `(B−α)/n` é o preço de estimar risco em amostra
finita, e o exigido não é α: com n = 50 e α = 0,1 o exigido é **0,082** — fixado nesse par
de valores em `tests/test_selective_conformal.py`, como verificação da fórmula, para que
ninguém leia o α declarado como o valor exigido na calibração. Sob o α declarado (0,05) e o
mesmo n, o exigido seria 0,031.

**Três ressalvas que têm de viajar com qualquer frase que cite o teorema:**

1. A garantia é **marginal, não condicional**. Ela limita o risco esperado sobre sorteios da
   partição de calibração. Não afirma que nesta execução o risco ficou abaixo de α. A tese
   pode escrever "risco esperado controlado em α"; não pode escrever "risco garantido abaixo
   de α neste conjunto".
2. Exige **permutabilidade** entre calibração e avaliação. Vale dentro de um corpus com
   partição aleatória; **não vale** entre corpora, e por isso nenhum limiar é transferido de
   CoNLL-2003 para GENIA.
3. Exige risco **monótono** no limiar — que é justamente o que o teste do piso investiga, logo
   não pode ser suposto. `crc_threshold` verifica e reporta `monotone=False` com a violação
   máxima; quando isso acontece, o limiar devolvido **não carrega** a garantia do teorema, e o
   texto tem de dizê-lo em vez de citar o teorema.

---

## Semente

`seed: 42`. Governa a partição por sentença e a reamostragem. Duas execuções com a mesma
semente sobre a mesma tabela dão o mesmo resultado; está fixado em
`tests/test_selective_runner.py`.

---

## Histórico de alteração

| Data | Item | De | Para | Motivo |
|------|------|----|------|--------|
| 2026-09-02 | — | — | — | Primeira redação, a partir do §3 do documento 04. Nenhum resultado observado até aqui: `src/selective/` foi escrito e testado sobre tabela sintética, e a tabela por entidade medida ainda não existe. |

---

## Confirmatório e exploratório: a separação que protege o item 2

Alterado em 02/09/2026, e a alteração tem duas causas.

**A primeira é forçada.** A faixa declarada era 8–15, escolhida quando o modelo era o
`bert-large`, de 24 camadas. O modelo da tese passou a ser o `gliner_base`, que tem **12**
(índices 0 a 11): as camadas 12 a 15 não existem. `src/selective/gliner_adapter.py` recusa
a rodar em vez de truncar, porque truncar mudaria em silêncio quais camadas produziram a
medida — o número sairia, pareceria certo, e seria de outras camadas.

**A segunda é uma escolha, e ela evita um problema.** A faixa declarada passa a ser
**todas as 12 camadas**. Declarar uma sub-faixa exigiria defender por que aquela, e a
banca já apontou cortes numéricos sem sistemática (família `R4`, sete apontamentos).
Declarar todas remove a escolha arbitrária: não há faixa privilegiada a justificar e não
há nada escolhido depois de ver resultado.

**O risco disso, e por que ele deixa de ser cego.** Misturar camadas iniciais — que
carregam mais posição e forma de palavra que entidade — pode diluir o sinal. Se diluir, o
teste do piso falha por causa da faixa e não por causa do instrumento, e sem mais
informação seria impossível distinguir as duas coisas. Por isso a medição emite também um
**perfil por camada**, em `layer_profile.csv`:

| | Confirmatório | Exploratório |
|---|---|---|
| O que é | a faixa declarada em `selective.layers` | uma linha por entidade **por camada** |
| Para que serve | decidir o veredito dos dois testes | descrever onde o sinal mora na profundidade |
| Pode escolher a faixa? | é a faixa | **não** |
| Pode decidir veredito? | sim | **não** |

**Por que a separação não é formalidade.** Medir 12 camadas e depois anunciar a melhor
daria 12 chances ao acaso: alguma pareceria boa por sorte, e o efeito reportado incluiria
a própria escolha. É a objeção `R4` de volta, agora multiplicada. Três coisas mantêm a
separação de pé, e as três são testadas:

1. o perfil sai em **arquivo separado** — se as massas por camada estivessem em
   `entities.csv`, a pescaria estaria a uma coluna de distância;
2. `src/selective/runner.py` **não lê** `layer_profile`, e um teste falha se passar a ler;
3. `layer_profile` está declarado como **não pré-registrado** em
   `tests/test_threshold_contract.py`, junto com a razão.

**O que fazer se o perfil mostrar concentração.** Ele vira hipótese pré-registrada de um
estudo seguinte, ou entra no artigo do instrumento como descrição exploratória rotulada
como tal. O que ele **não** pode fazer é redefinir a faixa desta execução depois do fato.

**Verificado rodando** em 02/09/2026 sobre três frases e sete entidades previstas: a faixa
inválida foi recusada, as 12 camadas foram aceitas, e o perfil saiu com 84 linhas
(7 entidades × 12 camadas). Três frases não são evidência de nada sobre o fenômeno — é
verificação de encanamento.


---

# Procedência de cada número

Escrito em 02/09/2026 a pedido do autor: **um pré-registro com números arbitrários troca
"escolher depois" por "escolher antes", e a objeção `R4` continua de pé — com data.** O que
a fecha é cada valor ter origem verificável. Três origens possíveis, e a coluna diz qual:

- **L** — a literatura usa este valor, e a fonte está citada.
- **M** — derivado de conta reproduzível, e a conta está aqui.
- **C** — convenção de relato, sem base externa. Declarada como tal, não disfarçada.

| # | Item | Valor | Origem | Fonte ou conta |
|---|---|---|---|---|
| 1 | agregação da confiança na entidade | *inerte* | — | não se aplica ao modelo escolhido (ver §1) |
| 2 | convenção de sumidouro | `drop_from_denominator` | **L** | Papamichalis & Ruane 2026 |
| 2 | faixa de camadas | todas as 12 | **M** | evita as 12 chances ao acaso (ver §2) |
| 3 | partição de calibração | 30% das **sentenças** | **M** | folga de amostra finita (ver §3) |
| 3 | regra de combinação | `logistic` | **C** | sem base; ver §6, pendência |
| 4 | coberturas reportadas | 50/70/80/90/95% | **L** | a curva inteira é a norma |
| 4 | ponto operacional | 80% | **C** | sem base; ver §6, pendência |
| 5 | nível do IC / reamostragens | 95% / 2.000 | **L+M** | 1.000 na literatura, 2.000 pela conta (ver §4) |
| 6a | grade de riscos alvo (relato) | 0,01 / 0,02 / 0,05 / 0,10 | **L** | Geifman & El-Yaniv 2017, Tabela 1 |
| 6b | alpha da garantia | 0,05 | **L** | Angelopoulos & Bates 2021; Singer et al. 2026 |

![Justificativa dos itens 3 e 5](img/prereg_justificativa.png)

*Painel (a): a cobertura realizada de um procedimento conforme não é exatamente 1−α — ela
flutua conforme o tamanho da calibração, e a faixa cinza é o intervalo de 5% a 95% dessa
flutuação. Painel (b): o extremo de um intervalo por bootstrap oscila só por causa do
sorteio, e essa oscilação cai como 1/√B.*

---

## §1 O item 1 está inerte, e isso se diz

`entity_confidence_aggregation: min` é gravado nos metadados de procedência e **nenhuma
conta o consome** — verificado por busca no código. Não é esquecimento: o GLiNER entrega
**um** escore por trecho, então não existe nada para agregar dentro da entidade. O item é
resquício do desenho com etiquetador BIO, em que havia um softmax por token.

A literatura usa `min` em outro nível: em Zheng et al. (2026) a confiança da *sentença* é o
mínimo entre as entidades dela, e dentro do trecho eles usam a **média** da máxima softmax
da primeira subpalavra. Nossa unidade de análise é a entidade, então nenhum dos dois se
aplica. **Pendência: o item deve sair do pré-registro** — decisão do autor, porque remover
item de pré-registro é mudança de pré-registro.

## §2 Faixa de camadas: por que todas

Ver a seção anterior deste documento. Em resumo: declarar uma sub-faixa exigiria defender
por que aquela; declarar todas remove a escolha. O perfil por camada existe, é exploratório
e não decide veredito — medir 12 e anunciar a melhor daria 12 chances ao acaso.

## §3 Partição de calibração: 30% das sentenças

**A conta que a literatura dá.** Singer et al. (2026), eq. 8: o limiar é o
⌈(1−α)(1+|I_cal|)⌉-ésimo maior escore de não-conformidade na calibração. Duas consequências
diretas:

1. **Existência.** O índice tem de cair dentro da amostra, o que exige um mínimo de
   calibração: **n ≥ 9** para α = 0,1 e **n ≥ 19** para α = 0,05.
2. **Folga.** A cobertura em amostra finita fica em [1−α, 1−α + 1/(n+1)]. Manter o excesso
   abaixo de 0,5 ponto percentual exige **n ≥ 199**.

**A unidade permutável é a sentença, não a entidade** — e isto não é detalhe. Duas entidades
da mesma sentença compartilham a matriz de atenção e o contexto: tratá-las como unidades
independentes infla o tamanho efetivo da amostra e a garantia conformal deixa de valer no
nível declarado. Por isso a partição é por sentença, e o *n* que entra nas contas acima é o
número de **sentenças** de calibração.

**O que 30% entrega nos dois corpora.** Com 3.453 sentenças de teste no CoNLL-2003 e 1.854
no GENIA:

| Corpus | sentenças | calibração (30%) | excesso máximo | cobertura realizada, 5%–95% |
|---|---:|---:|---:|---|
| CoNLL-2003 | 3.453 | 1.035 | 0,10 p.p. | 0,885 – 0,915 (±1,5 p.p.) |
| GENIA | 1.854 | 556 | 0,18 p.p. | 0,880 – 0,921 (±2,1 p.p.) |

Contra o mínimo de 199 para folga de meio ponto, o CoNLL-2003 fica **5,2×** acima e o GENIA
**2,8×**. O corpus que limita é o GENIA — a margem dele é pouco mais de um terço da do
CoNLL — e mesmo lá a cobertura realizada cabe em ±2 pontos. É isso que justifica 30%: não é
um número redondo, é uma fração que deixa o corpus mais apertado ainda com quase o triplo do
que a própria fórmula exige.

**O que a conta NÃO cobre.** A garantia pressupõe permutabilidade entre calibração e teste.
Ela vale dentro de um corpus e **não** vale sob deriva de domínio — é exatamente a ressalva
que o próprio tutorial de Angelopoulos & Bates (2021) faz, e ela tem de viajar com qualquer
citação da garantia.

## §4 Reamostragens: 2.000, e a conta que separa de 1.000

**A literatura mais próxima usa 1.000.** Zheng et al. (2026) reportam intervalos de 95% por
bootstrap com 1.000 reamostragens sobre as instâncias de teste.

**A conta.** O extremo de um intervalo por bootstrap é ele mesmo uma estimativa e oscila só
por causa do sorteio. Simulando sobre uma diferença pareada realista (1.200 entidades,
efeito de 0,02, dispersão de 0,12), com 120 repetições por valor de B:

| B | erro do extremo | como fração da largura do IC |
|---:|---:|---:|
| 250 | 0,00055 | 4,1% |
| 500 | 0,00039 | 3,0% |
| **1.000** (literatura) | 0,00027 | **2,2%** |
| **2.000** (declarado) | 0,00019 | **1,4%** |
| 5.000 | 0,00012 | 0,9% |

Dobrar B corta o ruído por 1/√2, como a teoria prevê. A escolha de 2.000 põe o ruído do
sorteio **abaixo de 1,5% da largura do intervalo** — ou seja, os extremos ficam estáveis na
terceira casa decimal, que é a casa em que a comparação de AURC vai ser reportada. Com
1.000 o ruído seria 2,2%, o que já se aproxima da terceira casa. Ir a 5.000 ganharia meio
ponto percentual ao custo de 2,5× o tempo, e não muda nenhuma conclusão.

## §5 Convenção de sumidouro: a literatura mostra que a escolha decide

Papamichalis & Ruane (2026) tratam cada linha da matriz de atenção como dado composicional
e mostram que a escolha entre manter o token de sumidouro ou descartá-lo e renormalizar é
"uma escolha que os artigos raramente reportam" — e que ela **reverte de 17% a 47% dos
vereditos** em dez modelos pré-treinados de cinco famílias.

Isso justifica **declarar** a convenção; não diz qual adotar. A adoção de
`drop_from_denominator` tem argumento próprio: o sumidouro não é conteúdo da sentença, é um
lugar onde o modelo deposita massa que não vai a token nenhum de interesse. Mantê-lo no
denominador faz a massa sobre o trecho parecer menor por uma razão que não tem a ver com o
trecho — e o efeito medido neste repositório é de fator ~2,5.

## §6 O que NÃO tem base, e fica declarado como tal

**Ponto operacional em 80% de cobertura.** Não achei base na literatura. Zheng et al. (2026)
reportam a curva inteira mais a área (AURC) e citam pontos específicos no texto — 60% num
dos resultados. A curva inteira é a norma; um ponto operacional único é conveniência de
relato, para haver um número citável. Ele **não** decide nada: os critérios dos dois testes
são intervalos sobre a área, não sobre um ponto.

**Regra de combinação `logistic`.** Também sem base direta. O precedente mais próximo é
Zheng et al. (2026), que ajustam uma combinação convexa em validação — e cuja combinação
**colapsou para um único sinal**: a confiança de tipo não contribuiu nada (AUROC 0,49–0,54,
mal acima do acaso). O achado é diretamente relevante e sugere reportar se a combinação
acrescenta algo sobre o melhor sinal isolado, em vez de assumir que acrescenta.

Os dois são **pendências do autor**, não minhas: mudar item de pré-registro é decisão de
método.
