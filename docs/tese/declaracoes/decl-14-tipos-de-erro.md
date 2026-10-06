# decl-14-tipos-de-erro — C4: a atenção separa um tipo de erro?

**Estado:** ASSINADA em 2026-10-01. Assinada sob pré-autorização expressa do autor, sem leitura do texto final ("certo, aprovo tudo, dispare todos os testes e processos", 01/10/2026). A contrapartida, registrada aqui e não omitida: nenhum parâmetro desta declaração foi escolhido nesta data; cada um é herdado de declaração assinada ou é a regra sem parâmetro livre aprovada pelo autor em SUGESTOES_C1_C4.md.

**Origem:** revisão de J. V. Miranda e Silva (30/09/2026) e a segunda explicação alternativa da §5 do
artigo: um erro pode ter fronteira errada, rótulo errado ou nenhum par anotado, e misturar os três pode
esconder um sinal específico de um deles.

**Pontos**: os dez do artigo, sob as declarações de origem — `gliner-base` e `gliner-large` em GENIA e
CoNLL-2003 (decl-02, decl-04), `gliner_base-ft-genia` (decl-05), `gliner_base-ft-conll2003` (decl-06), e os
quatro Qwen2.5 (decl-10 a decl-13). Partição de teste.

## O dado novo

As tabelas por entidade já medidas não guardam o trecho previsto nem o rótulo. A remedição grava, para cada
previsão, `(início, fim, rótulo)` em caracteres. O ouro é o do carregador de corpus, o mesmo da medição.

## Taxonomia (por precedência; cada erro recebe exatamente um tipo)

1. **rótulo** — existe entidade anotada com as mesmas fronteiras e rótulo diferente;
2. **fronteira** — não vale (1), e o trecho previsto se sobrepõe (≥ 1 caractere) a alguma entidade anotada;
3. **sem par** — não se sobrepõe a nenhuma entidade anotada.

Acerto continua sendo o casamento estrito (mesmas fronteiras e mesmo rótulo), como em todo o artigo.

## Comparações declaradas

Para cada ponto e cada tipo *t*, o conjunto analisado é **todos os acertos + os erros do tipo *t***. Nele:

- **E1 (veredito)** — massa de atenção contra fração geométrica (a comparação C2 da origem; nos decoders,
  a fração causal);
- **E2 (veredito)** — confiança + atenção desconfundida contra confiança (a comparação C3 da origem);
- **E3 (descritivo)** — o número de erros de cada tipo e a fração do risco do acaso removida pela massa de
  atenção e pela confiança, em cada conjunto.

Total: 10 pontos × 3 tipos × 2 vereditos = 60 intervalos, todos relatados, com o número de erros de cada
conjunto ao lado (um tipo raro dá intervalo largo, e isso aparece no próprio intervalo).

## O que contaria como achado

Que a atenção ajude num tipo e não nos outros: intervalos favoráveis de E1 ou E2 concentrados num tipo,
acima da taxa de acaso, em mais de um ponto. Intervalos favoráveis espalhados na taxa do acaso não contam.

## Regras comuns (herdadas, não reescolhidas)

- **Partição, calibração e incerteza**: as do ponto de origem — calibração em 30% das sentenças com
  `seed: 42`, todos os valores relatados na avaliação, 2.000 réplicas de *bootstrap* por sentença,
  intervalo de 95%. Um intervalo é **favorável** se inteiro abaixo de zero, **adverso** se inteiro acima,
  **degenerado** se os dois limites são zero; a taxa de acaso é 0,025 × o número de intervalos **não
  degenerados**.
- **Direção dos escores**: a de cada declaração de origem (fixa nos encoders; fixada na calibração nos
  decoders). Os resultados dos decoders são relatados também na direção fixa, como no artigo revisado.
- **Guarda de remedição**: a passada que produz os dados novos refaz as previsões do extrator. Antes de
  qualquer análise, a coluna `loss` remedida tem de sair **idêntica, linha a linha**, à da tabela já
  medida sob a declaração de origem; havendo uma linha diferente, a análise não roda e o fato é relatado.
