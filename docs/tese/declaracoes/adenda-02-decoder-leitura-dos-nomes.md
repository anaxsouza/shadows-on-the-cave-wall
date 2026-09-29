# Adenda 02 às declarações decl-10 a decl-13: como os nomes do encoder se leem no decoder

**Data:** 2026-09-23.
**Decidida por:** o autor, em três perguntas feitas ANTES de qualquer execução da
análise declarada.
**Estado da análise no momento desta adenda:** NÃO executada. Nenhuma AURC,
nenhuma carga de revisão e nenhum veredito de decl-10 a decl-13 foi calculado.
Continua valendo o que a adenda 01 disse ter sido visto: a AUC exploratória das
sondas e de `model_confidence` nos dois pontos de 0,5B.

## Por que existe

As declarações assinadas reaproveitam nomes do braço encoder (`span_mass`,
`geometric_fraction`, `enrichment`, `geometric_residual`, `sentence_length`), e o
executor da análise (`src/selective/comparisons.py`, `task_impact.py`) foi escrito
para o encoder. No decoder dois desses nomes admitem mais de uma leitura, e as
declarações não fixam qual. Deixar a escolha para depois do resultado seria um grau
de liberdade. Esta adenda a fixa antes.

## Decisões do autor

**1. `span_mass` é `causal_mass_prompt`.** É a massa RECEBIDA pelo trecho, somada
sobre todas as consultas da atenção do prompt: a mesma definição do encoder, e a
única das três leituras medidas que tem nulo exato calculado. As outras duas
(`causal_mass_autofoco`, `causal_mass_geracao`) não entram em comparação nenhuma.
Dado visto antes da decisão, e declarado por isso: R² da massa contra
`expected_causal` de 0,853 e 0,850 (0,5B, GENIA e CoNLL), contra 0,53 a 0,56 do
autofoco e 0,18 a 0,19 da leitura na geração. É descritivo, não é veredito.

**2. O nulo de C2 e C3 é o CAUSAL.** `geometric_fraction` é `expected_causal`
(tamanho, comprimento e posição sob máscara causal), e não `expected_mass(k, T)`,
que é o nulo bidirecional que o executor calcula para o encoder. Em consequência:
`enrichment = span_mass / expected_causal`, e `geometric_residual` é o resíduo da
regressão de `span_mass` sobre `expected_causal`. É a mesma lógica da regra 4 de
`O_QUE_MUDA` das declarações, aplicada aqui aos escores geométricos. D1 relata o R²
contra os dois nulos, como a própria D1 declara.

**3. As menções inventadas saem antes de comparar.** Uma linha com
`ancorada == 0` é um nome que o modelo escreveu e que não está no texto. Todas são
erro, nenhuma tem posição, e qualquer sistema as descartaria sem custo. Saem para
TODOS os escores igualmente, e a contagem vai no relatório: 24 (0,5B GENIA),
80 (0,5B CoNLL), 22 (1,5B GENIA) e 49 (1,5B CoNLL), contadas em 23/09/2026.

## Resolvido por mim, reportado aqui

**4. Junção com o AggSeq por `(sentence_id, mencao, rotulo)`**, a chave que
`medir_aggseq.py` já declarava. A tabela principal não gravava `mencao` nem
`rotulo` (defeito meu de desenho). As duas colunas foram acrescentadas NO FIM, e os
quatro pontos SERÃO remedidos na GPU (jobs `sentinel-medir-chave-*`, EM CURSO quando
esta adenda foi escrita — nenhum deles havia terminado). A medição é
determinística (verificado: zero células diferentes entre duas corridas na GPU), e
toda coluna anterior tem de sair idêntica. Isso é conferido antes da análise, e uma
diferença a interrompe. Um par guloso ausente de todos os feixes recebe
`aggseq = 0`: é a massa de feixe que o contém, por definição. A contagem desses
pares vai no relatório. Se uma chave casar com mais de uma linha, a análise PARA.

**5. `sentence_length` é `n_tokens`**, o comprimento da frase em tokens do modelo,
que é o `T` que o medidor usa nos nulos.

**6. `n_gold`**, o denominador do recall, é contado do test exportado:
5.506 (GENIA) e 5.648 (CoNLL). São os mesmos valores do braço encoder.

**7. O executor é estendido, não reescrito.** Os nomes do decoder entram na lista
de colunas opcionais do carregador, e o nulo causal é usado quando a tabela traz
`expected_causal`. Tabelas do encoder não têm essa coluna, e o caminho delas não
muda. Verificado pela suíte de testes antes da análise.

## O que NÃO muda

O conjunto de comparações, as grades, os critérios, o bootstrap e o nível de
confiança seguem exatamente como assinados. Os hashes de decl-10 a decl-13 não
mudam.
