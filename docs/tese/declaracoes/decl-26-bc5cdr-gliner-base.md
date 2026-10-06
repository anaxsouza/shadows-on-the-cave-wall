# decl-26-bc5cdr-gliner-base

**Estado:** ASSINADA em 2026-10-02, sob a declaração-guarda `decl-16-previsao-bc5cdr`. Assinada sob pré-autorização expressa do autor, sem leitura do texto final ("certo, aprovo tudo, dispare todos os testes e processos", 01/10/2026). A contrapartida, registrada aqui e não omitida: nenhum parâmetro desta declaração foi escolhido nesta data; cada um é herdado de declaração assinada ou é a regra sem parâmetro livre aprovada pelo autor em SUGESTOES_C1_C4.md.

| | |
|---|---|
| `declaration_id` | `decl-26-bc5cdr-gliner-base` |
| Hash da declaração | `59ab28f110d1c092` |
| Hash de **medição** | `fcf84d0cc9391e66` |
| Hash de análise | `f4ef835f18e9043c` (idêntico ao de `decl-05-ajustado-genia`) |
| Modelo | `urchade/gliner_base` |
| Corpus | BC5CDR, partição de teste |
| Molde | `configs/decl-05-ajustado-genia.yaml`, copiado com só `declaration_id` e `model` trocados |

GLiNER base sem ajuste. CORRIGE a decl-17, que foi gerada por engano a partir da decl-04 (GLiNER large) e por isso declara as camadas 0 a 23 para um modelo de 12 camadas; o adaptador a recusa e ela nunca foi medida. Esta declaração usa como molde a decl-05, a configuração assinada do mesmo modelo de partida (gliner_base, camadas 0 a 11), com só o modelo trocado. É assinada e depositada antes de qualquer medição do GLiNER base no BC5CDR; a decl-17 fica no registro como declaração inexequível, não é apagada. Todos os itens de análise, as comparações (C1, C2, C3, T1–T4) e os critérios são os do
molde, palavra por palavra; o gerador `tools/gerar_decl_revisao.py` recusa a cópia se qualquer
outra linha diferir e confere que o hash de análise é o mesmo do molde. O que esta declaração
acrescenta ao molde está em `decl-16-previsao-bc5cdr.md`.
