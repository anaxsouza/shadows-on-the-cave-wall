# Onde mora a informação de decisão — leitura do confirmatório sob `decl-09`

**Declaração:** `decl-09-sinais-large`, hash `bc162a9a0cb9e256`, assinada em 16/09/2026
(commit `143c4e2`), **antes** de qualquer coluna de sinal existir no disco ou no histórico.
**Medição:** `5454128f1df4da6f` — a mesma de `decl-04`, porque nenhum parâmetro de medição novo
entrou.
**Modelo:** `urchade/gliner_large` · **Dado:** `confirmatorio_sinais_tarefa.csv` (418 linhas)

---

## A pergunta

O nulo da massa de atenção está fechado em três arranjos. A pergunta seguinte, e a que a banca
faria, é: **então onde mora a informação de decisão, se não na atenção?**

Nove sinais entraram, em três famílias, cada um com um veredito contra a confiança que o próprio
modelo reporta. Nenhuma seleção: todos os sinais do registro entram, e é a ausência de seleção
que substitui aqui a correção de multiplicidade.

## O encanamento foi verificado antes do resultado

`logit_max` é, por construção, a **mesma quantidade** que `model_confidence`. A declaração
previu que o veredito `S9` teria de dar delta exatamente zero. Deu: **0,0000 nas dezesseis
células**, nos dois corpora. Um sinal cujo valor esperado se conhece de antemão é a maneira mais
barata de descobrir que o encanamento está errado, e ele não está.

## O resultado

De 72 células declaradas (9 sinais × 2 corpora × 4 frações do vão), 49 têm intervalo computável
e 23 são inalcançáveis — o sinal não atinge a meta de qualidade em cobertura nenhuma, o que já é
resultado.

| | células | favoráveis | desfavoráveis |
|---|---:|---:|---:|
| atenção (nulo **exato**) | 16 | **0** | 15 |
| estados ocultos (nulo empírico) | 11 | **0** | 9 |
| logits (nulo empírico) | 22 | 2 | 6 |
| **total** | **49** | **2** | **30** |

**Nenhum sinal de nulo exato acrescenta nada.** A família de atenção inteira — entropia da linha,
máximo da linha, massa emitida — perde ou não atinge a meta em todas as 16 células. Isso é a
previsão da álgebra confirmada: as três são agregados do mesmo orçamento fixo, e nenhuma escapa.

**Os estados ocultos também não.** Zero favoráveis em 11 células. A `hidden_delta_camadas` é
inalcançável nas oito.

## Os dois favoráveis, e por que eles NÃO são um achado

Os dois estão no mesmo sinal, `logit_margin`, no mesmo corpus, em frações adjacentes do vão:

| corpus | vão | margem | confiança | Δ | IC 95% |
|---|---:|---:|---:|---:|---|
| CoNLL-2003 | 50% | 0,4489 | 0,4848 | −0,0359 | [−0,0721; −0,0148] |
| CoNLL-2003 | 75% | 0,6799 | 0,7343 | −0,0543 | [−0,0957; −0,0028] |

Três razões para não os chamar de achado, e as três são medidas e não opinião:

1. **Estão dentro do acaso.** A 95%, com 49 intervalos, esperam-se ≈1,2 exclusões de zero em
   cada direção só por sorteio. Dois favoráveis é o que o acaso produz. Trinta desfavoráveis é
   **24 vezes** o acaso, e essa assimetria é o resultado.
2. **São pequenos.** O |Δ| mediano favorável é 0,0451; o desfavorável é 0,2648, seis vezes maior,
   chegando a 0,7727.
3. **Não replicam, e não são independentes entre si.** No GENIA a mesma margem não tem **nenhuma**
   célula favorável: em três das quatro frações ela não distingue, e na de 25% ela distingue na
   direção **contrária** (Δ +0,0535, IC [+0,0029; +0,0967] — exige mais revisão). E as duas
   células favoráveis do CoNLL são frações adjacentes do mesmo corpus e do mesmo sinal, logo
   compartilham quase todo o dado.

Há ainda um padrão que desfaz a leitura otimista: no vão de 25% a `logit_margin` exige **mais**
revisão que a confiança, nos **dois** corpora (Δ +0,0535 e +0,0269, IC excluindo zero). Ela não é
um sinal melhor — é um sinal que troca de lado conforme o ponto de operação.

**A regra declarada antes vale aqui:** um sinal que aparece favorável sob varredura não pode ser
afirmado. `logit_margin` no CoNLL em vãos médios vira candidata a declaração confirmatória
futura, com hash próprio e medição em partição que ainda não foi usada — nunca afirmação desta.

## O que este resultado permite afirmar

A afirmação da tese **se alarga**, e alargar é fortalecer:

> Nem a atenção, nem os estados ocultos, nem os escores de saída de um extrator de entidades
> acrescentam poder de decisão sobre a confiança que o próprio modelo reporta, em predição
> seletiva.

E a parte da atenção é mais forte que as outras duas, por uma razão que tem de viajar com a
frase: o nulo da atenção é **exato** — sai da álgebra do orçamento fixo, sem ajustar nada ao
dado. O dos estados ocultos e dos logits é **empírico**, desconfundido por regressão, com a forma
suposta linear e os coeficientes ajustados no mesmo dado. Um nulo exato refutado é afirmação
sobre o modelo; um empírico refutado pode ser afirmação sobre a forma da regressão.

## O que ele NÃO fecha

O extrator aqui é o `gliner_large` de prateleira, que erra 42% e 47%. As declarações `decl-07` e
`decl-08`, para os extratores **ajustados** (erro 25% e 10%), foram assinadas em `d70ed66` e
**não** foram medidas: a máquina com a T4 caiu depois da assinatura, e os checkpoints ajustados
existem só lá — 1,38 GB, acima do limite de transferência. Medi-las é o próximo passo, e o
arranjo delas é mais forte que este.

A limitação fica declarada e não escondida: este é o arranjo mais fraco dos três disponíveis.
