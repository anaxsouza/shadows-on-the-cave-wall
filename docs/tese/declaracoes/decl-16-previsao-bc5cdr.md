# decl-16-previsao-bc5cdr — C2: prever o sinal num corpus nunca olhado

**Estado:** ASSINADA em 2026-10-01, **antes** de qualquer previsão de qualquer extrator no BC5CDR. Assinada sob pré-autorização expressa do autor, sem leitura do texto final ("certo, aprovo tudo, dispare todos os testes e processos", 01/10/2026). A contrapartida, registrada aqui e não omitida: nenhum parâmetro desta declaração foi escolhido nesta data; cada um é herdado de declaração assinada ou é a regra sem parâmetro livre aprovada pelo autor em SUGESTOES_C1_C4.md.

## O corpus

BC5CDR (Li et al., 2016), versão em sentenças de `tner/bc5cdr`, revisão
`f68cdc7db924369241e7868656f583072acd4e90`, com a separação oficial: treino 5.227, validação 5.329, teste 5.864
sentenças; rótulos `Chemical` e `Disease`. SHA-256 dos arquivos baixados:

- `test.json`  20a04588b1a67c65203df698709a4384e15dac1597aa1cfb52365e7d0d6708b5
- `train.json` 62fe247960ff1b4270cb71500fb99748aab371c2dce0ef5b048fc44cb0f82ec8
- `valid.json` d74d511b8708c645c51e98bfd3f191d54ec83a601b3c5dde84a25ad4363f7d8e
- `label.json` d1d6998c78bc510b526538212c3b8eee97f1dd45ded428fc1965139c368ed59f

Descrições de rótulo dadas aos modelos, pela regra já usada nos outros corpora (o nome do tipo em
minúsculas): `chemical → Chemical`, `disease → Disease`. O texto do corpus não é redistribuído.

## Os extratores (cada um com sua declaração de ponto)

| ponto | modelo | molde | treino |
|---|---|---|---|
| decl-17 | GLiNER base | decl-04 | nenhum |
| decl-18 | GLiNER large | decl-04 | nenhum |
| decl-19 | GLiNER base ajustado | decl-05 | `tools/treinar_extrator.py`, regra de checkpoint dele (melhor F1 estrito na validação) |
| decl-20 | Qwen2.5-0.5B ajustado | decl-10 | `tools/treinar_decoder.py`, a mesma receita e a mesma regra de checkpoint de GENIA e CoNLL |
| decl-21 | Qwen2.5-1.5B ajustado | decl-12 | idem |
| decl-25 | bert-base-cased ajustado | decl-05 | a receita da decl-22 |

## A previsão (regra mecânica, sem parâmetro livre)

A fração geométrica não usa atenção: sai das previsões e dos dois comprimentos. Para cada extrator, na
direção fixa (valores maiores = mais provável acerto) e na partição de avaliação:

- **P1 (sinal)** — o sinal do risco removido pela massa de atenção, `1 − AURC/AURC_acaso`, será o sinal do
  risco removido pela fração geométrica (a causal nos decoders).
- **P2 (equivalência, só bidirecionais: GLiNER e BERT)** — |AURC(massa) − AURC(fração)| ≤ **0,004**, a maior
  diferença observada nos bidirecionais do artigo (`\encMassGeoMaxGap`), herdada e não escolhida.

Uma previsão é **confirmada** se P1 (e P2, onde se aplica) valer. Relatam-se as seis, uma a uma; não há
agregação que esconda uma falha. Uma falha é resultado, não desvio, e é relatada como tal.

Também correm, em cada ponto, as comparações C1, C2, C3 e T1–T4 do molde, sem mudança.

## O que contaria como replicação

As seis previsões confirmadas. Cinco de seis: replicação com uma exceção nomeada. Menos: o mecanismo, como
enunciado, não generaliza para este corpus, e o artigo o diz.
