"""O adaptador do decoder, sem modelo, sem rede e sem GPU.

O que se fixa aqui são as quatro coisas que dão errado em silêncio quando se
liga um decoder generativo a uma tabela por entidade:

1. A recusa de `sdpa`. Com a implementação padrão o pedido de atenção volta
   VAZIO sem erro, e a medição produziria nada. O construtor tem de recusar.
2. A ancoragem da menção no texto. Um decoder pode ESCREVER uma string que não
   está no texto — e isso é resultado (alucinação), não erro a esconder.
3. Duas menções idênticas na mesma sentença têm de ancorar em posições
   diferentes, e não na mesma.
4. O deslocamento do texto dentro do prompt. O prompt começa com `Text: `, então
   o caractere 0 da sentença não é o caractere 0 do prompt — e esquecer isso é o
   erro de indexação mais provável do módulo, do tipo que sai como número
   plausível.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.selective.decoder_adapter import (
    SEP,
    DecoderAdapter,
    DecoderAdapterError,
    ancorar,
    indices_de_token,
    parse_saida,
    prompt_de,
)


# ---------------------------------------------------------------------------
# A recusa que impede uma medição vazia e silenciosa
# ---------------------------------------------------------------------------

class _Config:
    def __init__(self, impl):
        self._attn_implementation = impl


class _Modelo:
    def __init__(self, impl="eager"):
        self.config = _Config(impl)
        self.device = "cpu"


def test_sdpa_e_recusado_no_construtor():
    with pytest.raises(DecoderAdapterError, match="eager"):
        DecoderAdapter(_Modelo("sdpa"), tokenizador=None, rotulos=["protein"])


def test_flash_attention_tambem_e_recusado():
    with pytest.raises(DecoderAdapterError, match="eager"):
        DecoderAdapter(_Modelo("flash_attention_2"), tokenizador=None, rotulos=["protein"])


def test_eager_e_aceito():
    a = DecoderAdapter(_Modelo("eager"), tokenizador=None, rotulos=["b", "a"])
    assert a.rotulos == ["a", "b"], "os rótulos têm de vir ordenados, como no treino"


# ---------------------------------------------------------------------------
# O prompt é o MESMO do treino
# ---------------------------------------------------------------------------

def test_prompt_e_deterministico_e_ordena_os_rotulos():
    p1 = prompt_de("IL-2 activates T cells.", ["protein", "cell type"])
    p2 = prompt_de("IL-2 activates T cells.", ["cell type", "protein"])
    assert p1 == p2, "a ordem dos rótulos na entrada não pode mudar o prompt"
    assert p1.startswith("Text: IL-2 activates T cells.\n")
    assert "Entity types: cell type, protein" in p1


def test_o_texto_da_sentenca_NAO_comeca_no_caractere_zero_do_prompt():
    """O deslocamento existe e é positivo — é o que `indices_de_token` corrige."""
    texto = "IL-2 activates T cells."
    p = prompt_de(texto, ["protein"])
    assert p.index(texto) == len("Text: ") > 0


# ---------------------------------------------------------------------------
# A saída gerada: o que é menção e o que não é
# ---------------------------------------------------------------------------

def test_parse_preserva_a_ORDEM_da_geracao():
    """A ordem liga cada menção às posições que a escreveram; um conjunto perderia."""
    saida = f"T cells{SEP}cell type\nIL-2{SEP}protein\nNF-kappa B{SEP}protein"
    assert parse_saida(saida) == [("T cells", "cell type"), ("IL-2", "protein"),
                                  ("NF-kappa B", "protein")]


def test_parse_descarta_linha_sem_separador_e_o_none():
    assert parse_saida("none") == []
    assert parse_saida("lixo sem separador\nIL-2 ## protein") == [("IL-2", "protein")]
    assert parse_saida("") == []


def test_parse_aceita_repeticao_da_mesma_mencao():
    saida = f"IL-2{SEP}protein\nIL-2{SEP}DNA"
    assert parse_saida(saida) == [("IL-2", "protein"), ("IL-2", "DNA")]


# ---------------------------------------------------------------------------
# Ancoragem: onde a menção está no texto, e o caso em que ela NÃO está
# ---------------------------------------------------------------------------

def test_mencao_ausente_do_texto_NAO_ancora():
    """Alucinação é resultado, não erro a esconder por casamento aproximado."""
    assert ancorar("IL-2 activates T cells.", "interleukin-2", []) is None


def test_mencao_presente_ancora_no_intervalo_certo():
    texto = "IL-2 activates T cells."
    par = ancorar(texto, "T cells", [])
    assert par is not None and texto[par[0]:par[1]] == "T cells"


def test_duas_mencoes_IDENTICAS_ancoram_em_posicoes_diferentes():
    texto = "IL-2 and IL-2 again"
    p1 = ancorar(texto, "IL-2", [])
    p2 = ancorar(texto, "IL-2", [p1])
    assert p1 != p2, "a segunda ocorrência não pode ancorar na primeira"
    assert texto[p2[0]:p2[1]] == "IL-2"


def test_terceira_ocorrencia_quando_so_existem_duas_nao_ancora():
    texto = "IL-2 and IL-2"
    p1 = ancorar(texto, "IL-2", [])
    p2 = ancorar(texto, "IL-2", [p1])
    assert ancorar(texto, "IL-2", [p1, p2]) is None


# ---------------------------------------------------------------------------
# O mapeamento caractere -> token, com o deslocamento do prompt
# ---------------------------------------------------------------------------

def _offsets_de(prompt: str, palavras: list[str]) -> list[tuple[int, int]]:
    """Offsets simulados: um token por palavra, na ordem em que aparecem."""
    fora, cursor = [], 0
    for w in palavras:
        i = prompt.index(w, cursor)
        fora.append((i, i + len(w)))
        cursor = i + len(w)
    return fora


def test_indices_de_token_usa_o_DESLOCAMENTO_do_prompt():
    texto = "IL-2 activates T cells"
    p = prompt_de(texto, ["protein"])
    desloc = p.index(texto)
    offs = _offsets_de(p, ["Text", "IL-2", "activates", "T", "cells", "Entity"])
    # "T cells" começa no caractere 15 do TEXTO
    assert texto[15:22] == "T cells"
    idx = indices_de_token(offs, 15, 22, desloc)
    assert idx == [3, 4], f"esperava os tokens de 'T' e 'cells', recebi {idx}"


def test_sem_o_deslocamento_o_mapeamento_apontaria_para_o_lugar_ERRADO():
    """A prova de que o deslocamento é load-bearing e não cerimônia."""
    texto = "IL-2 activates T cells"
    p = prompt_de(texto, ["protein"])
    offs = _offsets_de(p, ["Text", "IL-2", "activates", "T", "cells", "Entity"])
    certo = indices_de_token(offs, 15, 22, p.index(texto))
    errado = indices_de_token(offs, 15, 22, 0)
    assert certo != errado, "com e sem deslocamento tem de dar diferente"
    # O que torna este defeito perigoso não é ele dar vazio — é dar NÃO VAZIO.
    # Sem o deslocamento o intervalo cai em tokens DESLOCADOS (aqui 'activates'
    # e 'T' em vez de 'T' e 'cells'), e a medição sairia como número plausível
    # sobre os tokens errados, não como erro.
    assert errado, "o resultado errado é não vazio, e é isso que o torna perigoso"
    assert set(errado) != set(certo)


def test_intervalo_fora_do_texto_devolve_lista_vazia():
    p = prompt_de("IL-2", ["protein"])
    offs = _offsets_de(p, ["Text", "IL-2"])
    assert indices_de_token(offs, 500, 510, p.index("IL-2")) == []


# =============================================================================
# O invariante que a exceção do guarda NÃO pode afrouxar
# =============================================================================
# `tests/test_doc_code_alignment.py` permite que os adaptadores importem torch.
# A permissão vale porque o que a regra protege é outra coisa: que a ANÁLISE
# siga numpy puro. Estes dois testes fixam isso de dentro, para a exceção não
# virar porta aberta.


def test_as_funcoes_PURAS_do_adaptador_nao_precisam_de_torch():
    """Estas quatro são onde vive o erro de indexação, e têm de ser testáveis sem pesos.

    O `import torch` do módulo é LAZY, dentro de `__call__`. Isso não é estilo:
    é o que permite que os quinze testes deste arquivo rodem em milissegundos,
    sem GPU e sem baixar 1 GB. Se alguém mover o import para o topo, este teste
    falha.
    """
    import ast
    from pathlib import Path

    fonte = Path(__file__).resolve().parents[1] / "src" / "selective" / "decoder_adapter.py"
    arvore = ast.parse(fonte.read_text(encoding="utf-8"))
    topo = set()
    for no in arvore.body:                      # SÓ o nível de módulo
        if isinstance(no, ast.Import):
            topo |= {a.name.split(".")[0] for a in no.names}
        elif isinstance(no, ast.ImportFrom) and no.module:
            topo.add(no.module.split(".")[0])
    assert "torch" not in topo, (
        "torch subiu para o nível de módulo: os testes das funções puras passariam a "
        "exigir a pilha inteira")


def test_a_lista_de_arquivos_que_tocam_o_modelo_nao_cresceu():
    """A exceção do guarda é uma lista curta, e a curtidão é o ponto.

    Se um terceiro arquivo de `src/selective/` passar a importar torch, esta
    asserção falha antes de o guarda maior precisar decidir se afrouxa.
    """
    import ast
    from pathlib import Path

    sel = Path(__file__).resolve().parents[1] / "src" / "selective"
    tocam = set()
    for arq in sel.glob("*.py"):
        arvore = ast.parse(arq.read_text(encoding="utf-8"))
        imp = set()
        for no in ast.walk(arvore):
            if isinstance(no, ast.Import):
                imp |= {a.name.split(".")[0] for a in no.names}
            elif isinstance(no, ast.ImportFrom) and no.module:
                imp.add(no.module.split(".")[0])
        if imp & {"torch", "transformers", "datasets", "gliner"}:
            tocam.add(arq.name)
    assert tocam == {"gliner_adapter.py", "decoder_adapter.py"}, (
        f"a lista de arquivos que tocam o modelo mudou: {sorted(tocam)}. "
        f"Um adaptador novo é legítimo; um módulo de ANÁLISE importando torch não é."
    )


# =============================================================================
# A contagem de descarte: descarte silencioso NÃO é ausência
# =============================================================================
# O teste de fumaça com o modelo base mostrou o caso real: ele gerou `IL-2`,
# `T cell` e `Human blood` — três menções plausíveis, todas sem o separador — e
# o parser as descartou sem deixar rastro. Num extrator ajustado isso deve ser
# raro, mas "deve ser raro" é exatamente a suposição que precisa de número: se o
# extrator perder o formato em parte das sentenças, a taxa de erro medida sai
# OTIMISTA, porque o que foi descartado não entra no denominador.


def test_conta_as_linhas_descartadas_por_formato():
    from src.selective.decoder_adapter import parse_saida_contando

    # O caso REAL do teste de fumaça, verbatim.
    pares, n = parse_saida_contando("IL-2\nT cell\nHuman blood")
    assert pares == [] and n == 3, f"esperava 3 descartes, recebi {n}"


def test_mistura_de_bem_e_mal_formatado_conta_so_o_mal():
    from src.selective.decoder_adapter import SEP, parse_saida_contando

    pares, n = parse_saida_contando(f"IL-2{SEP}protein\nT cell")
    assert pares == [("IL-2", "protein")]
    assert n == 1


def test_none_e_linha_vazia_NAO_contam_como_descarte():
    """`none` é a resposta legítima para sentença sem entidade, não um defeito."""
    from src.selective.decoder_adapter import parse_saida_contando

    for texto in ("none", "", "\n\n", "  \n none \n "):
        pares, n = parse_saida_contando(texto)
        assert pares == [] and n == 0, f"{texto!r} deu {n} descartes"


def test_saida_bem_formatada_nao_descarta_nada():
    from src.selective.decoder_adapter import SEP, parse_saida_contando

    pares, n = parse_saida_contando(f"IL-2{SEP}protein\nT cells{SEP}cell type")
    assert len(pares) == 2 and n == 0


def test_separador_presente_mas_lado_vazio_conta_como_descarte():
    from src.selective.decoder_adapter import SEP, parse_saida_contando

    pares, n = parse_saida_contando(f"{SEP}protein\nIL-2{SEP}")
    assert pares == [] and n == 2, "separador sozinho não é menção válida"
