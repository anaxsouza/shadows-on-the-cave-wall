# BERT (decl-22 a decl-25): classificador por token

bert-base-cased + softmax, receita da decl-22 sem desvio (lr 5e-5, lote 32, 3 épocas, máx. 256, decaimento linear sem aquecimento, wd 0,01, AdamW, seed 42, última época, fp32). Um job SageMaker ml.g4dn.xlarge por corpus.

## Resultados por corpus

| corpus | F1 estrito teste | F1 estrito validação | trechos previstos | ouro | erro | n avaliação (70%) | AURC acaso | AURC confiança | AURC massa | AURC fração |
|---|---|---|---|---|---|---|---|---|---|---|
| genia | 0.7182 | 0.7354 | 5366 | 5506 | 0.2725 | 3788 | 0.2777 | 0.1203 | 0.2507 | 0.2375 |
| conll2003 | 0.9053 | 0.9495 | 5866 | 5648 | 0.1115 | 4038 | 0.1149 | 0.0255 | 0.1581 | 0.1528 |
| bc5cdr | 0.8477 | 0.8613 | 10345 | 9809 | 0.1743 | 7194 | 0.1736 | 0.0421 | 0.1522 | 0.15 |

## Comparações declaradas

| corpus | C1 R² (massa vs fração) | mediana obs/esp | C2 ΔAURC (massa − fração) [IC95] | C2 veredito | C3 ΔAURC | C3 veredito |
|---|---|---|---|---|---|---|
| genia | 0.9626 | 0.5657 | 0.0131 [0.0099, 0.0172] | PIOR que a base | 0.0 | NÃO ACRESCENTA (IC contém zero) |
| conll2003 | 0.9813 | 0.4578 | 0.0053 [0.0038, 0.0067] | PIOR que a base | 0.0 | NÃO ACRESCENTA (IC contém zero) |
| bc5cdr | 0.9832 | 0.5572 | 0.0022 [0.0011, 0.0033] | PIOR que a base | 0.0 | NÃO ACRESCENTA (IC contém zero) |

T2–T4 (carga de revisão, por meta 0,70/0,80/0,90/0,95): ver `docs/tese/resultados/confirmatorio_bert_tarefa.csv`. Em todos os corpora T4 não distingue da confiança (IC contém zero), e T2/T3 só distinguem no sentido de a massa/enriquecimento exigirem MAIS revisão que a confiança do modelo, ou ficam inalcançáveis nas metas altas.

Leitura: nos três corpora a massa de atenção segue a fração geométrica (R² 0,96–0,98) e é PIOR que ela como supervisor (C2 IC acima de zero); a combinação confiança+enriquecimento colapsa para peso 1,00 na confiança (C3: ΔAURC = 0). A confiança do próprio classificador domina (AURC 0,12 / 0,026 / 0,042).

## Procedência

| corpus | job | faturado (s) | custo US$ | impressão digital dos pesos (sha256-bytes-dos-pesos-v1) |
|---|---|---|---|---|
| genia | sentinel-bert-genia-20261001-134530 | 846 | 0.294 | `ca32dd2892944ffecb04f769c2827381f07131504c641097a14886da97f00926` |
| conll2003 | sentinel-bert-conll2003-20261001-141714 | 556 | 0.193 | `6f925d398ba25a61a5a20276d8df232765ea2dabb6a9f072d4f7487dd5715c3a` |
| bc5cdr | sentinel-bert-bc5cdr-20261001-143006 | 480 | 0.167 | `27c98c2390f87178c4835d396ee887eb500187321080ac04fc85e8f8822c9041` |

Custo total de treino: US$ 0.654 (teto da tarefa: US$ 8). Peso de partida google-bert/bert-base-cased, revisão HF cd5ef92a9fb2f889e972770a36d4ed042daf221e, impressão digital `c1b0bbb77f455bec3cb3c064c6524a38b8ac3c21433d3128a8ed9edff69b4fc3`.

## Decisões e ressalvas (declaradas)

1. **GENIA, etiquetas planas reconstruídas.** O carregador não roda sem rede; as etiquetas planas foram reconstruídas dos `genia_train.jsonl` pela regra do `_create_bio_labels` (sobrescrita na ordem do arquivo). Presume que a ordem das entidades no jsonl é a do carregador; não pude conferir.
2. **weight_decay** não se aplica a bias e LayerNorm (convenção do Trainer); laço de treino próprio em PyTorch.
3. **T (n_tokens)** = sub-tokens incluindo [CLS] e [SEP]; o denominador da fração geométrica é T−1 (só [CLS] sai).
4. **BC5CDR** lido de `dados_bc5cdr/*.json` (tner, BIO plano); validação = `valid.json`. Os dados dos três corpora foram copiados para o canal privado privado no S3 (não publicado).
5. **Testes:** os testes novos (14) passam. Na suíte completa há 12 falhas que NÃO vêm deste trabalho: 5 de `test_config_model_consistency`/`test_doc_code_alignment` (config já alterado pela tarefa BC5CDR) e 8 de `test_threshold_contract::test_o_documento_nao_diverge_da_fonte` (decl-17..21 e decl-23..25), cujo `doc.split('```yaml')[1]` não encontra bloco yaml nos documentos assinados, que não posso editar.
6. A medição usa uma sentença por vez em CPU; o F1 estrito medido coincide com o da avaliação em lote no contêiner (0,7182 / 0,9053 / 0,8477).

## Código criado

`tools/treinar_bert.py`, `src/selective/bert_adapter.py`, `tools/medir_bert.py`, `tools/submeter_bert.py`, `tests/test_bert_adapter.py`, `tests/test_bert_medicao.py`.
Tabelas: `saida/bert/<modelo>/<corpus>/test/{entities.csv,sentence_lengths.csv,MEDIDA.json}`; pesos em `saida/bert_pesos/`; análises em `docs/tese/resultados/confirmatorio_bert_{geometria,tarefa}.csv` e `_notas.txt`.
