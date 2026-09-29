"""O vocabulário de rótulos do braço decoder, em UM lugar só.

POR QUE ESTE ARQUIVO EXISTE

Num extrator generativo o rótulo é TEXTO que o modelo escreve. Isso faz do
vocabulário de rótulos parte do contrato entre treino e medição, e não detalhe
de apresentação: um modelo ajustado para escrever `cell type` avaliado contra
ouro que diz `CELL_TYPE` erra 100% daquela classe, sem erro nenhum aparecer.

O DEFEITO QUE ISTO CORRIGE, medido em 16/09/2026 antes de qualquer medição

O export de treino traz as descrições; o carregador de corpus traz as siglas.

    GENIA   treino  ['DNA', 'RNA', 'cell line', 'cell type', 'protein']
            loader  ['CELL_LINE', 'CELL_TYPE', 'DNA', 'PROTEIN', 'RNA']
                    -> 3 das 5 classes divergem

    CoNLL   treino  ['location', 'miscellaneous', 'organization', 'person']
            loader  ['LOC', 'MISC', 'ORG', 'PER']
                    -> as 4 divergem; NENHUMA coincide

`tools/medir_decoder.py` dava as siglas do carregador ao modelo no prompt E
comparava contra elas. No CoNLL isso levaria a taxa de erro base para perto de
1,0, e toda quantidade condicionada a acerto — que é o objeto do braço — viraria
ruído. A medição de 6 sentenças do GENIA que expôs isto devolveu erro base
0,5000, consistente com 3 de 5 classes impossíveis de acertar.

CUIDADO AO CITAR ESTE PAR DE NÚMEROS: o 0,5000 é de 6 sentenças e o 0,2069
medido depois da correção é de 30. NÃO são antes/depois da mesma amostra, e
apresentá-los como tal — que foi o que eu fiz na prosa e numa linha de memória
de 16/09/2026 — descreve um desenho experimental que não houve. A direção e a
ordem de grandeza da correção continuam sustentadas; a comparação controlada
não foi feita.

`treinar_decoder.py` JÁ trazia o mapa certo, em `ROTULOS`, e não o usava: a
linha de seleção era `sorted(ROTULOS[corpus].values()) if False else
sorted({...do jsonl})`. O `if False` é um interruptor de depuração que ficou, e
com ele o mapa virou documentação de algo que o código não fazia. O treino em si
estava CORRETO — lê o jsonl, que é a verdade do que ele viu — mas a única
definição do vocabulário estava num arquivo que não a aplicava.

A REGRA: falhar alto

`descricao()` levanta em rótulo não mapeado em vez de devolvê-lo intacto. A
alternativa silenciosa — repassar o desconhecido — é como este defeito
sobreviveria a um corpus novo: o rótulo passaria, o modelo não o reconheceria, e
a conta fecharia com número plausível. O princípio é o mesmo que decidiu a
questão do BART neste repositório: falha alta é preferível a falha silenciosa.
"""
from __future__ import annotations

import functools
import os
from pathlib import Path

import yaml

# A FONTE É ÚNICA E JÁ EXISTIA: `configs/config.yaml`, seção `gliner_labels`,
# que mapeia descrição -> rótulo do corpus e é o que o braço encoder usa (ver
# src/selective/gliner_adapter.py:41 e src/cli/commands/selective.py:135, que
# traz o princípio escrito: "duplicá-los criaria duas fontes para uma entrada do
# modelo, que é exatamente o tipo de divergência que não aparece como erro").
#
# Este arquivo NÃO redeclara a tabela. A primeira versão dele redeclarava, e era
# uma TERCEIRA cópia — o mesmo defeito que o texto acima condena, cometido no
# ato de corrigi-lo. Aqui só se inverte o mapa e se torna a busca tolerante a
# caixa e separador.
REGISTRO = Path(os.environ.get("SENTINEL_REGISTRO", "configs/config.yaml"))


@functools.lru_cache(maxsize=None)
def _mapa(corpus: str) -> dict[str, str]:
    """corpus -> {rótulo do carregador: descrição}, lido do registro."""
    caminhos = [REGISTRO, Path(__file__).resolve().parent / REGISTRO.name,
                Path(__file__).resolve().parent.parent / "configs" / "config.yaml"]
    for c in caminhos:
        if c.is_file():
            cfg = yaml.safe_load(c.read_text(encoding="utf-8"))
            break
    else:
        raise FileNotFoundError(
            f"registro de rótulos não encontrado (tentados: {[str(c) for c in caminhos]}). "
            f"As descrições de rótulo são ENTRADA do modelo; não há padrão em código "
            f"para elas, e inventar um aqui recriaria a divergência que este arquivo "
            f"existe para fechar.")
    por_descricao = (cfg.get("gliner_labels") or {}).get(corpus)
    if not por_descricao:
        raise KeyError(f"{REGISTRO} não declara gliner_labels para {corpus!r}")
    # Invertido: o registro guarda descrição -> corpus; aqui precisamos do
    # caminho oposto, porque é o carregador que entrega o rótulo do corpus.
    return {_normaliza(v): k for k, v in por_descricao.items()}


def _normaliza(rotulo: str) -> str:
    """Caixa e separador não distinguem rótulo: `cell_line`, `CELL LINE` e
    `Cell-Line` são o mesmo rótulo. A tolerância é aqui e não nas chaves, para
    que a tabela acima tenha uma entrada por classe e não quatro grafias."""
    return str(rotulo).strip().upper().replace(" ", "_").replace("-", "_")


def descricao(corpus: str, rotulo: str) -> str:
    """A descrição que o modelo sabe escrever, para um rótulo do carregador."""
    mapa = _mapa(corpus)
    chave = _normaliza(rotulo)
    if chave not in mapa:
        raise KeyError(
            f"rótulo {rotulo!r} do corpus {corpus!r} não tem descrição em "
            f"gliner_labels. Declarados: {sorted(mapa)}. Repassar o rótulo "
            f"intacto faria o modelo ser avaliado num vocabulário que não viu "
            f"no treino, e a conta fecharia com número plausível e errado.")
    return mapa[chave]


def descricoes(corpus: str) -> list[str]:
    """Todas as descrições do corpus, ordenadas — a lista que vai no prompt."""
    return sorted(_mapa(corpus).values())
