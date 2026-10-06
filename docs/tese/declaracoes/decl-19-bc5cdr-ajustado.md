# decl-19-bc5cdr-ajustado

**Estado:** ASSINADA em 2026-10-01, sob a declaração-guarda `decl-16-previsao-bc5cdr`. Assinada sob pré-autorização expressa do autor, sem leitura do texto final ("certo, aprovo tudo, dispare todos os testes e processos", 01/10/2026). A contrapartida, registrada aqui e não omitida: nenhum parâmetro desta declaração foi escolhido nesta data; cada um é herdado de declaração assinada ou é a regra sem parâmetro livre aprovada pelo autor em SUGESTOES_C1_C4.md.

| | |
|---|---|
| `declaration_id` | `decl-19-bc5cdr-ajustado` |
| Hash da declaração | `6b6b05ab90df00a3` |
| Hash de **medição** | `503cd1b0b98cacef` |
| Hash de análise | `f4ef835f18e9043c` (idêntico ao de `decl-05-ajustado-genia`) |
| Modelo | `gliner_base-ft-bc5cdr` |
| Corpus | BC5CDR, partição de teste |
| Molde | `configs/decl-05-ajustado-genia.yaml`, copiado com só `declaration_id` e `model` trocados |

GLiNER base ajustado ao BC5CDR com a receita de tools/treinar_extrator.py. Todos os itens de análise, as comparações (C1, C2, C3, T1–T4) e os critérios são os do
molde, palavra por palavra; o gerador `tools/gerar_decl_revisao.py` recusa a cópia se qualquer
outra linha diferir e confere que o hash de análise é o mesmo do molde. O que esta declaração
acrescenta ao molde está em `decl-16-previsao-bc5cdr.md`.
