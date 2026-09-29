# PROTOCOLO — o que precisa ser declarado, e por quê

**Estado:** protocolo, reutilizável. Não é declaração.
**Declarações que o instanciam:** `decl-01-gliner-base` (ver `PREREGISTRO.md`).

---

## Por que protocolo e declaração são documentos separados

Pedido do autor em 02/09/2026: que o pré-registro fosse **parametrizável**, para que a
mesma metodologia sirva a outros modelos e outras faixas no futuro.

O pedido tem uma tensão dentro dele, e ignorá-la desfaria o pré-registro. A função de um
pré-registro é **remover** graus de liberdade. Um arquivo onde se troca a faixa de camadas e
o modelo é um gabarito, e num gabarito a pergunta "quais valores estavam em vigor quando
este resultado saiu?" deixa de ter resposta — que é exatamente a pergunta que a família de
objeções `R4` faz.

A separação resolve as duas coisas ao mesmo tempo:

| | Protocolo (este arquivo) | Declaração (`PREREGISTRO.md`) |
|---|---|---|
| O que fixa | **quais** itens têm de ser declarados, o espaço de cada um, e os critérios dos testes | **quais valores** foram declarados, para um modelo e um corpus |
| Varia com o modelo? | não | sim — uma declaração por modelo |
| Muda depois de ver resultado? | só por revisão explícita e datada | **nunca** |
| Identidade | nome do arquivo | `declaration_id` + hash do conteúdo |

Um protocolo, N declarações. Cada execução se amarra a exatamente uma, pelo hash.

---

## 1. Os itens que toda declaração tem de preencher

O espaço de cada item é restrito ao que tem **precedente citável**. A coluna "base" diz de
onde vem a *forma* do item — o valor é escolha da declaração, dentro do espaço.

| Item | Espaço permitido | Base da forma |
|---|---|---|
| `sink_policy` | `keep`, `drop_from_denominator`, `drop_from_queries_and_denominator` | Papamichalis & Ruane (2026): manter ou descartar o sumidouro é escolha que **reverte 17–47% dos vereditos** em dez modelos de cinco famílias, e que os artigos raramente reportam |
| `layers` | qualquer subconjunto não vazio das camadas **existentes no modelo** | forçado pelo modelo; a declaração é recusada se citar camada inexistente |
| `heads` | subconjunto, ou ausente para todas | idem |
| `combination_rule` | `convex`, `rank_average`, `product` | Zheng et al. (2026) ajustam combinação **convexa** em partição separada. `logistic` não está no espaço: ninguém na linha citada usa |
| `calibration_fraction` | fração de **sentenças** que satisfaça o mínimo da §3 da declaração | Singer et al. (2026), eq. 8 |
| `coverage_levels` | níveis reportados na curva | Geifman & El-Yaniv (2017) tratam a curva risco-cobertura como o perfil de desempenho inteiro; Zheng et al. (2026) reportam curva mais AURC |
| `operating_point` | **apenas** `derived_from_target_risk` | ver §2 — é a inversão que a literatura canônica impõe |
| `target_risk_grid` | riscos alvo **relatados**, cada um em (0, 1] | Geifman & El-Yaniv (2017), Tabela 1: indexada por r* em {0,01 … 0,06}, reportando a cobertura obtida em cada um. Não custa grau de liberdade — os critérios dos dois testes são intervalos sobre AURC e não olham para r* |
| `conformal_alpha` = r* da **garantia** | 0,01–0,10, e **dentro** de `target_risk_grid` | 0,01–0,06 em Geifman & El-Yaniv (2017); 0,1 em Angelopoulos & Bates (2021); 0,1 e 0,05 em Singer et al. (2026) |
| `added_value_ci_level` | nível do IC | 95% em Zheng et al. (2026) |
| `n_bootstrap_resamples` | >= 1.000 | 1.000 em Zheng et al. (2026); acima disso por erro de Monte Carlo (§4 da declaração) |
| `loss_bound`, `seed` | — | reprodutibilidade |

**Não é item do protocolo:** `layer_profile` (saída exploratória) e `declaration_id`
(identidade). Ambos estão nomeados como não pré-registrados em
`tests/test_threshold_contract.py`, com a razão ao lado.

---

## 2. O ponto operacional é derivado, e a inversão vem da literatura

**Como estava:** declarava-se uma cobertura (80%) e lia-se o risco nela.

**Como a literatura canônica faz:** Geifman & El-Yaniv (2017) partem de um classificador,
uma amostra, um parâmetro de confiança e um **risco alvo** r*, e procuram a função de
seleção que **maximiza a cobertura** sujeita a esse risco. A Tabela 1 deles é indexada por
r* em {0,01 … 0,06} e reporta a cobertura obtida. O controle conformal de risco
(Angelopoulos et al., 2022) tem a mesma forma: declara-se o nível de risco, o limiar é
consequência.

**Consequência para o desenho:** declarar cobertura fixa era inconsistente com o próprio
método usado para a garantia. E, corrigida a direção, os dois parâmetros colapsam num só —
o `conformal_alpha` **é** o risco alvo r*. O ponto operacional passa a ser "a cobertura
obtida ao risco alvo", lido da curva e não declarado.

Um parâmetro a menos para justificar, e a redução vem da literatura, não de gosto.

---

## 3. O que NÃO varia entre declarações

Estas quatro decisões são do protocolo. Uma declaração que as mudasse não seria outra
instância deste protocolo — seria outro protocolo, e o documento tem de dizer isso.

1. **A unidade permutável é a SENTENÇA.** Duas entidades da mesma sentença compartilham a
   matriz de atenção e o contexto; tratá-las como independentes infla o tamanho efetivo da
   amostra e a garantia conformal deixa de valer no nível declarado.
2. **O casamento com a anotação é ESTRITO** — mesma fronteira e mesmo rótulo. A unidade da
   curva é o que se entrega: chamar fronteira errada de acerto parcial infla o desempenho na
   direção do resultado desejado.
3. **O estrato aninhado vem da ANOTAÇÃO**, não da predição. Se dependesse do previsto, o
   estrato mudaria com a qualidade do modelo e as execuções deixariam de ser comparáveis.
4. **Confirmatório e exploratório são separados por mecanismo**, não por promessa: o perfil
   por camada sai em arquivo próprio, o runner não o lê, e há teste para os dois.
5. **A ORIENTAÇÃO de cada escore é decidida na CALIBRAÇÃO** (acrescentado em 16/09/2026, com
   a entrada das três famílias de sinal). A curva risco-cobertura entrega por escore
   decrescente. Para a confiança do modelo, alto significa entregar; para a entropia dos
   escores de rótulo, alto significa o contrário — o modelo está indeciso, e aquela entidade
   deveria ir para revisão. Um sinal na orientação errada sai pior que o acaso por razão
   trivial, e relatar isso como achado seria erro grosseiro.

   Testar as duas orientações e reportar a melhor **dobraria as chances ao acaso**, e é
   exatamente o grau de liberdade que este protocolo existe para fechar. Então a orientação
   sai do sinal da correlação entre o escore e o acerto na partição de **calibração**, nunca
   na de avaliação — a mesma disciplina que o peso convexo já usava. Nada é escolhido depois
   de ver o veredito, e nada é escolhido pelo operador.

   Duas exceções são fixadas **por definição** e não por dado: `model_confidence` e
   `logit_max` são a mesma quantidade e são confiança, logo alto significa entregar. Fixá-las
   é o que faz a verificação de identidade valer — se a orientação de `logit_max` viesse do
   dado, ela poderia sair invertida em relação à confiança e o delta deixaria de ser
   exatamente zero por uma razão que não é defeito de encanamento.

---

## 3b. Os dois ESTATUTOS de nulo, e por que o estatuto viaja com o número

Acrescentado em 16/09/2026. As famílias de sinal **não têm o mesmo tipo de nulo**, e misturá-las
sem dizer qual é qual transformaria um resultado limpo em confusão.

**Nulo EXATO — família de atenção.** Cada linha da matriz soma 1, por construção. Desse
orçamento fixo sai um valor esperado exato para qualquer agregado dos pesos, derivável e
verificável por permutação, sem ajustar nada ao dado: a massa tem esperança `k/|K|`, a entropia
da linha `log|K|`, o máximo `1/|K|`. É a mesma álgebra três vezes, e é por isso que a literatura
observa dependência de comprimento em toda variante que tenta — não existe variante que escape
de um orçamento fixo.

**Nulo EMPÍRICO — estados ocultos e logits.** Não há orçamento: a norma de um vetor de estado
não é restrita a somar coisa alguma, e o escore de um trecho não compete com os dos outros
trechos por uma massa constante. Sem orçamento não há esperança derivável, então a
desconfundição é por **regressão** nas covariáveis geométricas — e isso é mais fraco em dois
sentidos que ficam declarados em cada linha de resultado: a forma da relação é **suposta**
linear, e os coeficientes são **ajustados no mesmo dado** em que o resíduo é avaliado.

**A consequência, e a razão de o estatuto viajar com o número:** um nulo exato refutado é uma
afirmação sobre o modelo; um nulo empírico refutado pode ser uma afirmação sobre a forma da
regressão. Um revisor tem direito a saber qual dos dois está lendo, em cada linha, em cada
tabela e em cada frase do texto.

O estatuto de cada sinal é declarado **uma vez**, no registro de `src/selective/signals.py`, e
fixado em teste — a construção do registro **recusa** um sinal de nulo exato que traga covariável
e um de nulo empírico que não traga. As declarações não o repetem, porque duas versões da mesma
verdade divergiriam no primeiro sinal novo.

---

## 4. Os critérios dos dois testes

Independentes de modelo e de corpus.

**Piso — a massa de atenção calibrada bate a abstenção aleatória?**
Refutado se o IC de 95% da diferença de AURC contra a abstenção aleatória **contém zero ou
favorece o acaso**. Inconclusivo se a hipótese de monotonicidade do controle conformal for
violada além do limite declarado — e a violação é *verificada*, não suposta.

**Valor adicionado — o ganho é incremental sobre a confiança do modelo, e maior no aninhado?**
Refutado se o IC de 95% do ΔAURC pareado (combinação contra confiança isolada) contém zero.
O peso da combinação convexa é **reportado**: peso em 0 ou 1 é a combinação colapsando para
um sinal isolado, que foi o que aconteceu com Zheng et al. (2026), e reportar isso é parte
do resultado — não nota de rodapé.

---

## 5. Como fazer uma declaração nova

Para outro modelo, outra arquitetura ou outra faixa:

1. Copiar a seção `selective` de `configs/config.yaml`, mudar o `declaration_id` e os
   valores, mantendo cada um dentro do espaço da §1.
2. Rodar qualquer comando: o adaptador **recusa** faixa de camadas que o modelo não tenha, e
   o carregador de declaração recusa item ausente ou fora do espaço.
3. Congelar: acrescentar a declaração a `docs/tese/declaracoes/` com o hash impresso pelo
   `--dry-run`, e commitar **antes** de medir. É o commit que estabelece a ordem.
4. Medir. O hash vai para `MEDIDA.json` ao lado da tabela.

O hash cobre os itens declarados e **não** cobre `source` nem `declaration_id`: mover o
arquivo ou renomear a declaração não muda o que foi declarado. Mudar qualquer item muda o
hash, e é isso que impede uma tabela medida sob uma declaração de ser testada sob outra —
sem o hash, o resultado sairia sem erro, parecendo válido, com metade dos parâmetros de cada
uma.

---

## Referências

- Angelopoulos, A. N. & Bates, S. (2021). *A Gentle Introduction to Conformal Prediction
  and Distribution-Free Uncertainty Quantification.* arXiv:2107.07511.
- Angelopoulos, A. N., Bates, S., Fisch, A. & Lei, L. (2022). *Conformal Risk Control.*
  arXiv:2208.02814.
- Geifman, Y. & El-Yaniv, R. (2017). *Selective Classification for Deep Neural Networks.*
  arXiv:1705.08500.
- Papamichalis, M. & Ruane, R. (2026). *Which Question Is Your Attention Metric Answering?
  Attention Rows as Compositional Data.* arXiv:2608.14712.
- Singer, M., Sengupta, S. & Pazdernik, K. (2026). *Uncertainty Quantification for Named
  Entity Recognition via Full-Sequence and Subsequence Conformal Prediction.*
  arXiv:2601.16999.
- Zheng, Z., Li, B., Yao, J. & Long, J. (2026). *Reliable Financial Named Entity Recognition
  Under Domain Shift: Confidence Estimation and Selective Prediction.* arXiv:2608.19558.


---

# A cadeia de análise, e os controles que a literatura exige

**Acrescentado em 03/09/2026**, depois de uma revisão de literatura dirigida à pergunta que
o achado de comprimento levantou: *alguém já notou que sinais de atenção são confundidos por
comprimento?* A resposta é **sim**, e isso reorganiza a contribuição da tese em vez de
enfraquecê-la. Diagrama: `docs/tese/resultados/protocolo.png`.

## 1. A cadeia, em quatro etapas

| Etapa | O que faz | Onde |
|---|---|---|
| 1 · medir | o modelo prevê e fornece a atenção, uma vez por corpus e split | `selective --measure` |
| 2 · escores | quatro ordenações candidatas por entidade predita | `entities.csv` |
| 3 · ordenar | curva risco-cobertura; a área (AURC) é o número que compara | `risk_coverage.py` |
| 4 · comparar | três comparações pareadas, com IC declarado | `runner.py` |

**Os quatro escores.** Dois vêm do modelo — a confiança que ele declara e a massa de atenção
sobre o span. **Dois não usam modelo nenhum**: `k`, o número de tokens do span, e `k/T`, a
fração do comprimento da sentença que o span ocupa. Os dois últimos são bases de comparação
triviais, e a razão de existirem está na §3.

**As três comparações, e só três.**

- **C1 · piso** — a massa ordena melhor que o sorteio? *Estado: GENIA sim; CoNLL-2003 não, com
  sinal invertido (a massa é pior que o acaso).*
- **C2 · a tese** — confiança + massa ordena melhor que a confiança sozinha? *Estado: refutado
  nos dois corpora, IC contendo zero.*
- **C3 · o controle** — o que a massa acrescenta, contar tokens já acrescentava? *Estado:
  exploratório, e o resultado é que sim.*

**O critério de refutação** é o mesmo para as três: diferença de AURC pareada, IC 95% por
2.000 reamostragens, **reamostrando sentenças e não entidades** — entidades da mesma sentença
compartilham a matriz de atenção, e tratá-las como independentes infla o tamanho efetivo da
amostra. IC contendo zero significa que não acrescenta, sem exceção aberta depois de ver o
número.

## 2. O que a literatura já exige, e não é nossa descoberta

Saghir (2026), `arXiv 2605.00269`, mostra que sinais de confiança de caixa branca em modelos de
linguagem — incluindo entropia de atenção — são **estruturalmente confundidos por comprimento
de sequência** (|r| ≥ 0,61) e **colapsam para 0,491–0,527 de AUROC** (acaso = 0,5) sob
avaliação com comprimento pareado. Uma base trivial de contagem de tokens, sem modelo nenhum,
alcança 0,874 e 0,919 nas tarefas deles — comparável aos métodos elaborados.

A teoria que eles dão: a atenção opera sobre um simplex cujo tamanho depende do comprimento,
de modo que **qualquer** agregado de pesos de atenção herda dependência Θ(log T). A frase que
importa para nós é a deles: *o confundidor afeta o sinal subjacente, não implementações
específicas.*

**Isso prevê o resultado do nosso Perímetro 1.** Varremos 2.496 leituras de atenção — quatro
direções, três políticas de sumidouro, dezesseis conjuntos de camadas, treze de cabeças — e
nenhuma escapa. Não era acidente da nossa escolha de leitura; era consequência estrutural, e
agora há teoria publicada para ela.

**Três controles passam a ser obrigatórios**, e são adotados por serem prática estabelecida e
não invenção nossa:

1. base de comparação trivial de comprimento (`k`), reportada ao lado do sinal elaborado;
2. residualização do escore contra o comprimento antes de calcular a AURC;
3. relato explícito da correlação entre o escore e o comprimento.

## 3. Onde está a contribuição desta tese, com precisão

**O confundidor deles e o nosso são geometricamente diferentes.** Em Saghir (2026) o que varia
é o comprimento da **entrada**, e o efeito é Θ(log T) sobre um escore de sequência. Aqui a
sentença é a **mesma** para todas as entidades dela, e o que varia é a fração `k/T` que o span
ocupa dentro de um simplex de tamanho fixo.

Medido na validação, sobre 4.932 entidades do GENIA e 8.469 do CoNLL:

| | GENIA | CoNLL-2003 |
|---|---:|---:|
| correlação massa × `k/T` | **+0,981** | **+0,980** |
| variância da massa explicada por `k/T` | **96,2%** | **96,0%** |
| coeficiente do ajuste massa = a·(`k/T`) | 0,517 | 0,455 |

A massa de atenção sobre o span é, em 96% da sua variância, **a fração geométrica que o span
ocupa na sentença**. O coeficiente ~0,5 é o único conteúdo sistematicamente não geométrico:
spans recebem cerca de metade do que receberiam sob atenção uniforme.

E a consequência para a decisão, também na validação:

| AURC | GENIA | CoNLL-2003 |
|---|---:|---:|
| acaso | 0,4941 | 0,5169 |
| `k/T` sozinho, sem modelo | 0,4239 | 0,5300 |
| massa de atenção | 0,4196 | 0,5341 |
| massa sem a geometria (resíduo) | 0,4705 | 0,5532 |
| confiança do modelo | 0,3301 | 0,2425 |
| **Δ de confiança + massa** | **−0,0067** | +0,0000 |
| **Δ de confiança + `k/T`** | **−0,0068** | +0,0000 |
| **Δ de confiança + resíduo** | −0,0005 | +0,0000 |

Tudo o que a massa de atenção contribui, a fração geométrica contribui igual. O resíduo — a
parte que é de fato atenção e não geometria — contribui nada.

**O que não está em Saghir (2026):** eles relatam colapso **para** o acaso. No CoNLL-2003 a
massa é **pior** que o acaso, e o resíduo é pior ainda. Colapso e inversão são fenômenos
diferentes, e a inversão em texto plano é achado nosso.

**A resposta de uma frase à objeção "isto é 2605.00269 aplicado a NER":** o confundidor deles é
o comprimento da entrada num escore de sequência; o nosso é a fração ocupada dentro de um
contexto fixo, num escore por entidade, com pré-registro assinado antes da medição e com um
fenômeno que eles não observam — inversão, não colapso.

## 4. O que falta para C3 ser confirmatório

Uma declaração nova, `decl-02`, declarando antes de medir no teste: `k` e `k/T` como escores de
comparação, o resíduo da massa após remover `k/T`, e o critério — se o IC da diferença de AURC
entre massa e `k/T` contiver zero, a massa não acrescenta sobre geometria.

**Uma base de comparação que só pode enfraquecer a própria afirmação não é grau de liberdade.**
Ela não dá mais chances de encontrar efeito; dá um nulo mais difícil de bater. É o oposto de
trocar de modelo ou de ajustar hiperparâmetros de treino, que abririam escolhas capazes de
favorecer o resultado.


---

## 5. A varredura de leituras fica na validação, e a razão é a multiplicidade

**Decidido em 03/09/2026.** As 2.496 leituras do Perímetro 1 são medidas na validação e **não**
são repetidas no teste. A razão não é custo — recalcular do tensor guardado leva minutos.

Repetir no teste não acrescentaria nada confirmatório e custaria a única coisa que o teste tem.
Nada é selecionado na varredura: o que se reporta é a distribuição inteira dos ΔAURC, e uma
distribuição não vira veredito. Já a exposição é real — 2.496 olhares sobre a partição
confirmatória, cada um deles uma oportunidade de reagir ao que se vê. Um leitor que visse a
varredura rodada no teste teria razão em perguntar quantas dessas leituras foram olhadas antes
de a declarada ser escolhida, e a resposta honesta ("nenhuma, ela foi declarada antes") deixa de
ser verificável no momento em que as outras 2.495 também foram calculadas ali.

A divisão que fica, e que o paper declara:

| | onde | estatuto |
|---|---|---|
| a leitura DECLARADA | teste | confirmatório |
| as outras 2.495 leituras | validação | robustez, exploratório |
| a distribuição dos ΔAURC | validação | robustez, exploratório |

Consequência que viaja com o relato: a afirmação de invariância é uma afirmação sobre a
**validação**, e o paper diz isso onde a apresenta. Ela sustenta "o resultado não é artefato da
leitura escolhida"; não sustenta um número confirmatório para leitura nenhuma além da declarada.
