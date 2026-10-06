# decl-22-bert — C3: um classificador por token (BERT + softmax)

**Estado:** ASSINADA em 2026-10-01. Assinada sob pré-autorização expressa do autor, sem leitura do texto final ("certo, aprovo tudo, dispare todos os testes e processos", 01/10/2026). A contrapartida, registrada aqui e não omitida: nenhum parâmetro desta declaração foi escolhido nesta data; cada um é herdado de declaração assinada ou é a regra sem parâmetro livre aprovada pelo autor em SUGESTOES_C1_C4.md.

**Origem:** revisão de J. V. Miranda e Silva: "por que não usar um modelo de classificação padrão (BERT/RoBERTa
+ softmax)?". O artigo testa um extrator que pontua trechos (GLiNER) e um que escreve as entidades (Qwen2.5);
este é a terceira família, a que dá um rótulo a cada palavra.

## Receita (fixa; nenhuma busca de hiperparâmetro)

- Modelo de partida: `google-bert/bert-base-cased` (12 camadas, 12 cabeças).
- Esquema de rótulos BIO por palavra; cada palavra é rotulada no **primeiro sub-token**, os demais ficam fora
  da perda (-100).
- Taxa de aprendizado 5e-5, lote 32, 3 épocas, comprimento máximo 256 sub-tokens, decaimento linear sem
  aquecimento, `weight_decay` 0,01, AdamW, `seed: 42`; dentro da grade recomendada por Devlin et al. (2019).
- Checkpoint: o da **última** época, sem seleção.
- Decodificação: trechos são as sequências máximas B-X I-X…; um I-X sem B-X/I-X antes abre trecho (convenção
  do conlleval). No GENIA, que tem entidades aninhadas, o treino usa as etiquetas planas do carregador; as
  previsões são planas e casam com o ouro aninhado pelo critério estrito de sempre.

## Confiança do trecho

**Média geométrica**, sobre as palavras do trecho, da probabilidade máxima do softmax no primeiro sub-token —
a mesma definição já declarada para a confiança dos decoders.

## Atenção

Idêntica à do GLiNER: média de todas as camadas e cabeças antes das razões; o `[CLS]` (posição 0) sai do
denominador; os tokens do trecho são achados pelos deslocamentos de caractere. Referência: Eq. 3.

## Pontos

decl-23 (GENIA), decl-24 (CoNLL-2003), decl-25 (BC5CDR, também sob a decl-16). Cada um é cópia da decl-05
com só o modelo trocado; valem as comparações C1, C2, C3 e T1–T4 da decl-05.
