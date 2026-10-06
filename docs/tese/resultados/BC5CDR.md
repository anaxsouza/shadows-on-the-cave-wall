# BC5CDR — medição dos extratores sob decl-16 a decl-21 (02/10/2026)

Nenhuma declaração assinada foi editada. Esta nota relata o que foi medido, sem julgar P1/P2 (decl-16): as
tabelas trazem os insumos, e o veredito é do autor.

## Pendência que impede um extrator: decl-17

A decl-17 (GLiNER base sem ajuste) é a cópia da decl-04 (GLiNER large) com só `declaration_id` e `model`
trocados; ficou com `layers: [0..23]`, e o `urchade/gliner_base` tem 12 camadas. O adaptador recusa
(corretamente: truncar mudaria em silêncio quais camadas produzem a medida). **Não medida.** Ela não pode ser
executada como escrita; corrigi-la é nova declaração (camadas 0–11, como a decl-02), decisão do autor.
Consequência: P1/P2 têm 4 extratores medidos (de 5 previstos na decl-16, sem contar o BERT da decl-25).

## Corpus

Os quatro arquivos conferem com o SHA-256 da decl-16 (o carregador falha alto se divergir). Contagens reais:
**5.228 / 5.330 / 5.865** sentenças (treino/validação/teste), não 5.227 / 5.329 / 5.864: a prosa da declaração
contou quebras de linha (`wc -l`) e o arquivo não termina em newline. Entidades de ouro no teste: 9.809.

## Por extrator (partição de teste, 5.865 sentenças)

| ponto | modelo | n entidades preditas | taxa de erro base | F1 estrito de validação | impressão digital dos pesos (sha256 v1) |
|---|---|---|---|---|---|
| decl-17 | GLiNER base | — não medida (ver acima) | — | — | — |
| decl-18 | GLiNER large | 11.064 | 0,3443 | — | — |
| decl-19 | gliner_base-ft-bc5cdr | 10.478 | 0,1629 | 0,6700 → 0,8718 (checkpoint-1962) | `cf39f4d1efe0282a12ecb03f103456f6b97bcf2e5ae7b269e28a138f3b87b080` |
| decl-20 | qwen05b-ft-bc5cdr | 9.888 (123 não ancoradas fora) | 0,1347 | 0,0000 → 0,8922 (checkpoint-981) | `a54941ea137c03b96e12972c915edcf89a6d389edfdd2aca0b9cafe1310b3fd5` |
| decl-21 | qwen15b-ft-bc5cdr | 10.226 (60 não ancoradas fora) | 0,1379 | 0,0000 → 0,9303 (checkpoint-981) | `f2027393ad6538ce7d980e97b6f5170fe2a7990a9bbb43982d9bf451c5c98437` |

Os hashes de medição gravados em MEDIDA.json coincidem com os das declarações (decl-18 `5454128f1df4da6f`,
decl-19 `503cd1b0b98cacef`, decl-20 `afe6a84294888252`, decl-21 `84b51948c3c3c0fe`).
F1 de validação: 400 sentenças (GLiNER) e 300 (Qwen), a regra de cada receita; casamento por texto no Qwen,
que não é o F1 por posição da tabela de entidades (diferença declarada em `treinar_decoder.py`).

## AURC na avaliação, direção fixa (insumo de P1/P2; `confirmatorio_bc5cdr_previsao_insumos.csv`)

| modelo | AURC acaso | AURC massa | AURC fração | 1−AURC/acaso (massa) | 1−AURC/acaso (fração) | \|massa−fração\| |
|---|---|---|---|---|---|---|
| gliner-large | 0,3448 | 0,3737 | 0,3695 | −0,0838 | −0,0716 | 0,0042 |
| gliner_base-ft-bc5cdr | 0,1668 | 0,1832 | 0,1825 | −0,0983 | −0,0941 | 0,0007 |
| qwen05b-ft-bc5cdr (causal) | 0,1301 | 0,1367 | 0,1316 | −0,0507 | −0,0115 | 0,0051 |
| qwen15b-ft-bc5cdr (causal) | 0,1369 | 0,1406 | 0,1321 | −0,0270 | +0,0351 | 0,0085 |

A coluna `dif` é entregue para P2 (limite 0,004, só bidirecionais) sem veredito. C2 (massa contra fração, ΔAURC
pareado, IC 95%) e C3, por extrator, estão nas tabelas completas.

## Tabelas (docs/tese/resultados/)

- `confirmatorio_bc5cdr_decl-18-bc5cdr-gliner-large_{geometria,tarefa}.csv` (59 e 41 linhas), `..._notas.txt`
- `confirmatorio_bc5cdr_decl-19-bc5cdr-ajustado_{geometria,tarefa}.csv` (59 e 41 linhas), `..._notas.txt`
- `decoder/decoder_aurc_bc5cdr.csv` (468 linhas), `decoder/decoder_carga_bc5cdr.csv` (708), `decoder/NOTAS_bc5cdr.json`
  (decl-20, decl-21 e adenda-01, pelo mesmo `tools/analisar_decoder.py`)
- `confirmatorio_bc5cdr_previsao_insumos.csv`; `ajuste_bc5cdr.json`, `ajuste_decoder_qwen05b_bc5cdr.json`,
  `ajuste_decoder_qwen15b_bc5cdr.json` (procedência dos ajustes)

## Jobs SageMaker e custo (total US$ 15,88; teto US$ 20)

| job | instância | s faturados | US$ |
|---|---|---|---|
| sentinel-bc5cdr-gliner-1001140452 | g4dn.xlarge | 671 | 0,233 |
| sentinel-bc5cdr-dec-05b-1001135608 | g5.2xlarge | 1.985 | 1,420 |
| sentinel-bc5cdr-dec-15b-1001152957 | g5.2xlarge | 4.896 | 3,502 |
| sentinel-bc5cdr-medir-05b-1001172308 (falhou) | g5.2xlarge | 209 | 0,150 |
| sentinel-bc5cdr-medir-05b-1001175730 | g5.2xlarge | 5.954 | 4,259 |
| sentinel-bc5cdr-medir-15b-1001193730 | g5.2xlarge | 8.835 | 6,320 |
| sentinel-bc5cdr-medir-15b-1001220615 (duplicata por engano, parada) | g5.2xlarge | 0 | 0 |

Medição e AggSeq de cada Qwen rodaram no MESMO job (`tools/medir_bc5cdr_job.py`), uma passagem depois da outra.
O primeiro job de medição morreu em 209 s: `medir_decoder.py` (da tarefa Remedição) passou a importar
`src.selective.janelas`, ausente do tarball do submissor; corrigido em `tools/submeter_bc5cdr.py`, e a fumaça
passou a ser feita com o tarball real em diretório isolado.

## Desvios e ressalvas

- decl-17 não medida (acima).
- A fonte de `medir_decoder.py` / `medir_aggseq.py` no job é a de 01/10/2026 (sha256 em `FONTE_SHA256.json` ao
  lado de cada medição bruta); `medir_decoder.py` recusa `--corpus bc5cdr` por `choices`, e foi invocado pelo
  ambiente (SENTINEL_CORPORA).
- Os Qwen do BC5CDR foram treinados com bf16 + Adam de 8 bits na g5.2xlarge, o regime de GENIA/CoNLL; o GLiNER
  recebeu `urchade/gliner_base` baixado do hub dentro do job (os pesos de partida do GLiNER não têm impressão digital).
- A guarda de remedição não se aplica: nenhum ponto já medido foi refeito.
- Medições brutas dos decoders (com `features_sonda.npz`) estão em `saida/bc5cdr_decoder_brutos/` e não em `results/`,
  porque `tests/test_threshold_contract.py` exige `declaration_id` em todo MEDIDA.json sob `results/`, e o
  MEDIDA do decoder não o grava.
