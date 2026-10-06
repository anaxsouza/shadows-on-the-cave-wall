# decl-15-janelas-de-controle — C1: o modelo destaca as entidades?

**Estado:** ASSINADA em 2026-10-01. Assinada sob pré-autorização expressa do autor, sem leitura do texto final ("certo, aprovo tudo, dispare todos os testes e processos", 01/10/2026). A contrapartida, registrada aqui e não omitida: nenhum parâmetro desta declaração foi escolhido nesta data; cada um é herdado de declaração assinada ou é a regra sem parâmetro livre aprovada pelo autor em SUGESTOES_C1_C4.md.

**Origem:** revisão de J. V. Miranda e Silva (30/09/2026): comparar a atenção da entidade com a de trechos do
mesmo tamanho na mesma frase. Hoje o artigo compara com o modelo indiferente, uma conta exata; este teste
compara com o próprio modelo, em trechos que não são entidades.

**Pontos**: os dez do artigo, os mesmos da decl-14, partição de teste.

## Janelas de controle (sem sorteio)

Para cada entidade prevista com *k* tokens na frase, as janelas são **todas** as sequências contíguas de *k*
tokens da frase (tokens de conteúdo da frase; no GLiNER, fora do prompt de rótulos; no Qwen, dentro do trecho
da frase no prompt) que **não se sobrepõem** a nenhuma entidade anotada nem a nenhuma entidade prevista da
mesma frase. Não há semente nem número de janelas a escolher. Entidade sem janela possível fica de fora e é
contada.

## Estatística

Para a entidade *e* e cada janela *w*: enriquecimento = massa recebida / massa esperada pelo modelo
indiferente (Eq. 3 nos encoders, Eq. 4 nos decoders, com a posição da própria janela). A razão da entidade é
`r(e) = enriquecimento(e) / média dos enriquecimentos das suas janelas`.

- **J1 (veredito)** — mediana de `log r(e)` sobre as entidades. "O modelo destaca entidades" se o intervalo
  de 95% ficar inteiro acima de zero; "o modelo as atenua" se inteiro abaixo; nenhum dos dois se contiver zero.
- **J2 (descritivo)** — a mesma mediana separada em acertos e erros, e a fração de entidades com `r(e) > 1`.
- **J3 (veredito)** — a diferença entre as medianas de `log r` de acertos e de erros, com intervalo de 95%.
  É a ponte com a pergunta do artigo: destacar entidades não é o mesmo que saber se estão certas.

Excluído deste teste, por exigir um desenho de perturbação com parâmetros novos: injetar erros em entidades.

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
