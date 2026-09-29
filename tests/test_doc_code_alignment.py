"""Alinhamento entre o que a tese afirma e o que o código faz.

Este é o guarda de deriva mais importante da suíte, e o que ele fiscaliza mudou
inteiramente em 02/09/2026. Antes, ele afirmava o desenho da conjunção: a regra
3-de-4, α = 0,05/4 = 0,0125 por Bonferroni com k = 4, a entropia em log₂ como
instrumento, o coeficiente de fluxo de informação, a rede feed-forward, o GPS, o
caminho Procrustes, a grade de taxa de aprendizado e os marcos de checkpoint.
Treze desses testes eram `xfail(strict=True)`: divergências conhecidas entre a
tese e o código, mantidas visíveis em vez de escondidas.

Agora ele fiscaliza a pergunta única, e a divisão é a mesma: o que a tese afirma,
o que o código faz, e as divergências que ainda existem, declaradas como `xfail`
em vez de omitidas. O original está em
`git show pre-c4-reorg:tests/test_doc_code_alignment.py`.
"""

from __future__ import annotations

import ast
import re
import inspect
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
ARCH = REPO / "docs" / "ARCHITECTURE.md"
PREREG = REPO / "docs" / "tese" / "PREREGISTRO.md"
SRC = REPO / "src"
DOC04 = REPO / "docs" / "tese" / "04_c4_fundamentacao.md"


def arch() -> str:
    return ARCH.read_text(encoding="utf-8")


def secao8() -> str:
    t = arch()
    return t[t.index("## 8."):t.index("## 9.")]


# ---------------------------------------------------------------------------
# 1. A tese afirma uma pergunta, e os documentos têm de dizer isso
# ---------------------------------------------------------------------------

def test_a_arquitetura_declara_uma_pergunta_e_dois_testes():
    t = arch()
    assert "One question" in t or "one question" in t
    s = secao8()
    assert "### Test: floor" in s
    assert "### Test: added-value" in s


def test_a_regra_de_convergencia_da_conjuncao_nao_esta_mais_especificada():
    """A §8 pode CITAR a regra 3-de-4 para dizer que ela saiu; não especificá-la."""
    s = secao8()
    for morta in ("at least 3 of 4", "3 of 4 sub-hypotheses", "both H2.1 and H2.2"):
        assert morta not in s, f"§8 voltou a especificar a conjunção: '{morta}'"


def test_a_arquitetura_explica_por_que_nao_ha_correcao_de_multiplicidade():
    """Ausência de correção sem explicação é indistinguível de esquecimento.

    Um leitor da banca que conheceu o desenho anterior vai procurar o alfa. A §8
    tem de dizer que os critérios são intervalos e não p-valores, e é por isso
    que não há família sobre a qual corrigir.
    """
    s = secao8()
    assert "No multiplicity correction" in s
    assert "interval-based" in s
    assert "p-value" in s


def test_o_criterio_de_valor_adicionado_esta_declarado_nos_dois_documentos():
    for arquivo, texto in (("ARCHITECTURE.md §8", secao8()), ("PREREGISTRO.md", PREREG.read_text(encoding="utf-8"))):
        assert "excludes zero" in texto or "excluir zero" in texto, (
            f"{arquivo} não declara o critério de valor adicionado"
        )


# ---------------------------------------------------------------------------
# 2. O instrumento: massa como proporção, e a convenção de sumidouro explícita
# ---------------------------------------------------------------------------

def test_a_massa_e_proporcao_e_nao_media_de_submatriz():
    """A medida antiga não media nada, e o código não pode voltar a ela.

    `_extract_entity_attention` reportava `given`, a média das linhas do span.
    Como toda linha de atenção soma 1 por construção, essa média é exatamente
    1/T para qualquer modelo, texto ou entidade.
    """
    from src.selective.attention_mass import span_attention_mass

    fonte = inspect.getsource(span_attention_mass)
    assert "denominador" in fonte and "numerador" in fonte, (
        "span_attention_mass deixou de calcular uma razão"
    )
    # a propriedade, e não só o texto: massa é invariante a escala da matriz
    a = np.array([[[[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.2, 0.2, 0.6]]]])
    m1 = span_attention_mass(a, [1], sink_policy="keep").mass
    m2 = span_attention_mass(a * 3.0, [1], sink_policy="keep").mass
    assert m1 == pytest.approx(m2), "a massa deixou de ser adimensional"


def test_a_convencao_de_sumidouro_e_obrigatoria_e_nao_tem_padrao():
    """Uma convenção que muda o resultado 2,5x não pode ser implícita.

    Item 2 do pré-registro. Um valor padrão em código seria uma escolha que
    alguém pode ter feito depois de ver um resultado.
    """
    from src.selective.attention_mass import span_attention_mass

    sig = inspect.signature(span_attention_mass)
    p = sig.parameters["sink_policy"]
    assert p.default is inspect.Parameter.empty, (
        "sink_policy ganhou valor padrão; ver docs/tese/PREREGISTRO.md item 2"
    )
    assert p.kind is inspect.Parameter.KEYWORD_ONLY


def test_o_documento_declara_o_tamanho_do_efeito_do_sumidouro():
    """O número que justifica o item 2 tem de estar escrito, não subentendido."""
    t = PREREG.read_text(encoding="utf-8")
    assert "0,30" in t and "0,75" in t and "2,5" in t, (
        "PREREGISTRO.md item 2 deixou de mostrar o efeito medido da convenção"
    )




# ---------------------------------------------------------------------------
# 3. A medida: denominador condicional, reamostragem pareada, correção finita
# ---------------------------------------------------------------------------

def test_o_risco_e_condicional_as_entidades_retidas():
    """É daí que vem a força do piso.

    Com denominador global, abster-se reduziria o risco automaticamente e a curva
    do acaso desceria — não haveria o que bater. Condicional, a curva do acaso é
    uma horizontal na taxa de erro base.
    """
    from src.selective.risk_coverage import risk_coverage_curve

    perdas = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
    c = risk_coverage_curve(perdas, [0.5] * 6)
    assert list(c.risk) == [pytest.approx(0.5)], (
        "escore constante deixou de dar a taxa de erro base em toda a faixa"
    )


def test_a_reamostragem_do_valor_adicionado_e_pareada():
    """Os dois escores são medidos nas MESMAS entidades.

    Sortear independentemente trataria como independentes duas quantidades que
    compartilham a amostra inteira, e daria intervalo largo demais — erro na
    direção de parecer conservador, que é pior do que errar de forma visível.
    """
    from src.selective.risk_coverage import delta_aurc_paired_bootstrap

    fonte = inspect.getsource(delta_aurc_paired_bootstrap)
    arvore = ast.parse(inspect.getsource(delta_aurc_paired_bootstrap))
    sorteios = [
        n for n in ast.walk(arvore)
        if isinstance(n, ast.Call) and getattr(n.func, "attr", "") in ("integers", "choice")
    ]
    assert len(sorteios) == 1, (
        f"a reamostragem faz {len(sorteios)} sorteios; pareada faz um só, reusado "
        f"pelos dois escores"
    )
    assert "pareada" in fonte or "paired" in fonte


def test_o_limiar_conformal_usa_a_correcao_de_amostra_finita():
    """O número exigido na calibração não é alpha.

    Com n = 50 e alpha = 0,1, a desigualdade do teorema exige risco empírico
    <= 0,1 - (1 - 0,1)/50 = 0,082. Ler o alpha do pré-registro como o valor
    exigido é o erro que este teste impede.
    """
    from src.selective.conformal import crc_threshold

    rng = np.random.default_rng(0)
    escores = rng.uniform(0, 1, 50)
    perdas = (escores < 0.5).astype(float)
    r = crc_threshold(perdas, escores, alpha=0.1)
    assert r.bound == pytest.approx(0.1 - (1.0 - 0.1) / 50)
    assert r.bound < 0.1


def test_a_hipotese_de_monotonicidade_e_verificada_e_nao_suposta():
    """É o que o teste do piso investiga, logo não pode ser premissa."""
    from src.selective.conformal import CRCThreshold, crc_threshold

    assert "monotone" in CRCThreshold.__dataclass_fields__
    assert "max_monotonicity_violation" in CRCThreshold.__dataclass_fields__
    escores = [1.0, 0.9, 0.8, 0.3, 0.2, 0.1]
    ideal = crc_threshold([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], escores, alpha=0.4)
    anti = crc_threshold([1.0, 1.0, 1.0, 0.0, 0.0, 0.0], escores, alpha=0.4)
    assert ideal.monotone and not anti.monotone, (
        "o diagnóstico de monotonicidade inverteu de direção — já esteve invertido"
    )


def test_as_tres_ressalvas_da_garantia_estao_nos_documentos():
    """Garantia citada sem ressalva é garantia afirmada além do que o teorema dá."""
    s = secao8()
    p = PREREG.read_text(encoding="utf-8")
    assert "marginal, not conditional" in s and "marginal, não condicional" in p
    assert "exchangeability" in s and "permutabilidade" in p
    assert "monotone" in s and "monótono" in p


# ---------------------------------------------------------------------------
# 4. A partição por sentença, que é onde o vazamento não apareceria como erro
# ---------------------------------------------------------------------------

def test_a_particao_e_por_sentenca_no_codigo_e_nos_documentos():
    from src.selective.runner import SelectiveRunner

    fonte = inspect.getsource(SelectiveRunner._split_by_sentence)
    assert "sentence" in fonte or "sentenca" in fonte or "sentenças" in fonte
    assert "BY SENTENCE" in arch() or "by sentence" in arch()
    assert "por SENTENÇA" in PREREG.read_text(encoding="utf-8")


def test_nenhuma_sentenca_aparece_nas_duas_particoes(tmp_path: Path):
    """A propriedade, medida — não a intenção declarada no comentário."""
    import csv

    from src.selective.runner import SelectiveRunner

    cfg = tmp_path / "c.yaml"
    cfg.write_text(
        "selective:\n"
        "  declaration_id: decl-teste\n  sink_policy: keep\n  layers: [8]\n"
        "  heads: null\n  combination_rule: rank_average\n  calibration_fraction: 0.5\n"
        "  coverage_levels: [1.0]\n  operating_point: derived_from_target_risk\n"
        "  added_value_ci_level: 0.9\n  n_bootstrap_resamples: 200\n"
        "  target_risk_grid: [0.3]\n  conformal_alpha: 0.3\n  seed: 7\n",
        encoding="utf-8",
    )
    tab = tmp_path / "r" / "m" / "genia" / "test" / "entities.csv"
    tab.parent.mkdir(parents=True, exist_ok=True)
    with tab.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["sentence_id", "loss", "model_confidence", "span_mass", "is_nested"])
        for s in range(20):
            for _ in range(3):
                w.writerow([f"s{s}", s % 2, 0.9, 0.5, 0])
    runner = SelectiveRunner("m", "genia", config_path=str(cfg), output_dir=str(tmp_path / "r"))
    tabela = runner._load_table()
    cal, aval = runner._split_by_sentence(tabela)
    assert not (set(cal["sentence_id"]) & set(aval["sentence_id"])), (
        "há sentença nas duas partições: o contexto de atenção vaza da calibração "
        "para a avaliação, e o vazamento aparece como resultado bom"
    )


# ---------------------------------------------------------------------------
# 5. O que a pergunta única NÃO usa
# ---------------------------------------------------------------------------

def test_o_modulo_da_contribuicao_nao_depende_do_instrumento_abandonado():
    """A entropia era o instrumento da tese antiga (objeção O3 do documento 01).

    `src/selective/` depende de numpy e nada mais, e é isso que o mantém
    testável em milissegundos sem carregar modelo.
    """
    # Os ADAPTADORES tocam o modelo, e são nomeados aqui um por um. A exceção é o
    # que dá sentido à regra: se `risk_coverage.py` passar a importar torch, este
    # teste falha, e é justamente isso que se quer. Sem nomear a exceção, a regra
    # inteira teria de cair.
    #
    # `decoder_adapter.py` entrou em 16/09/2026 com o braço do decoder. A regra
    # NÃO foi afrouxada para acomodá-lo: o que ela protege é que a análise
    # (`risk_coverage`, `comparisons`, `task_impact`, `geometry`, `signals`) siga
    # numpy puro e testável em milissegundos, e o critério é a FUNÇÃO do arquivo
    # — adaptador —, não a contagem. O teste irmão abaixo fixa que a lista não
    # cresce por descuido.
    TOCA_O_MODELO = {"gliner_adapter.py", "decoder_adapter.py"}
    tocam = set()
    for arquivo in (SRC / "selective").glob("*.py"):
        texto = arquivo.read_text(encoding="utf-8")
        arvore = ast.parse(texto)
        importados = set()
        for no in ast.walk(arvore):
            if isinstance(no, ast.ImportFrom) and no.module:
                importados.add(no.module.split(".")[0])
            elif isinstance(no, ast.Import):
                importados |= {a.name.split(".")[0] for a in no.names}
        proibidos = importados & {"torch", "transformers", "datasets", "gliner"}
        if proibidos:
            tocam.add(arquivo.name)
        assert not proibidos or arquivo.name in TOCA_O_MODELO, (
            f"{arquivo.name} importa {proibidos}. Só {sorted(TOCA_O_MODELO)} pode tocar o "
            f"modelo; o resto de src/selective/ depende de numpy e nada mais, e é isso que "
            f"mantém a análise testável em milissegundos."
        )
        # O guarda ficou GENÉRICO demais e passou a proibir a palavra em vez do
        # objeto. A entropia do desenho abandonado era uma construção específica:
        # entropia da distribuição de atenção como PROXY DE CONFIANÇA, em log2,
        # com a regra 3-de-4 sobre um roster de seis modelos. A entropia que
        # `signals.py` traz é outra coisa — membro da família de atenção, em
        # nats, com nulo EXATO `log|K|` derivado do orçamento fixo, e que serve
        # justamente para mostrar que a família inteira herda a dependência de
        # comprimento. Proibir a palavra obrigaria a renomear a função para
        # driblar o teste, que é o pior desfecho possível: o guarda perderia o
        # sentido e o código ficaria com nome ruim.
        #
        # Então o que se proíbe passa a ser o que MARCAVA o desenho antigo, e a
        # lista é mais específica e mais dura do que a anterior.
        # '3-de-4' NÃO entra na lista, e a razão é um falso positivo real: o
        # `preregistration.py` diz "a inversão de sinal vai de 3-de-4 estratos a
        # 4-de-4", que é conteúdo do desenho VIGENTE. A regra abandonada era
        # inseparável do alfa de Bonferroni, então são `bonferroni` e `0.0125`
        # que a marcam sem ambiguidade — e um termo que produz falso positivo
        # acaba desativado por alguém, o que é pior que não tê-lo.
        #
        # E o casamento é por PALAVRA, não por substring. `gps` como substring
        # casa dentro de `logps` — o acumulador de log-probabilidades do
        # adaptador do decoder —, que é falso positivo do mesmo tipo que tirou
        # '3-de-4' da lista. Renomear a variável para driblar o teste seria o
        # desfecho ruim: o guarda perderia o sentido e o código ficaria com nome
        # pior. Fronteira de palavra resolve sem afrouxar nada, porque o GPS do
        # desenho abandonado sempre apareceu como termo isolado.
        for morto in ("IFC", "FFN", "procrustes", "gps", "log2", "log_2",
                      "bonferroni", "0.0125"):
            padrao = re.compile(rf"(?<![0-9A-Za-z_]){re.escape(morto)}(?![0-9A-Za-z_])",
                                re.IGNORECASE)
            assert not padrao.search(texto), (
                f"{arquivo.name} menciona '{morto}' como palavra, do desenho abandonado"
            )
        # E a entropia, onde aparecer, tem de ser a de nulo exato: em nats e
        # ancorada em log|K|. Se alguém reintroduzir a versão de proxy de
        # confiança em log2, o teste acima a pega pelo `log2`; este pega a
        # ausência da âncora.
        # Exigido de quem DEFINE, não de quem reexporta: o `__init__.py` cita o
        # nome na lista de exportação e não tem onde ancorar nada.
        if "def row_entropy" in texto or "def expected_row_entropy" in texto:
            assert "log|K|" in texto or "log(K)" in texto or "np.log(K)" in texto, (
                f"{arquivo.name} usa entropia sem ancorá-la no nulo exato log|K|; "
                f"entropia sem o nulo é o instrumento abandonado de volta"
            )
    assert tocam <= TOCA_O_MODELO, (
        f"arquivos novos tocando o modelo: {sorted(tocam - TOCA_O_MODELO)}"
    )


def test_o_roteiro_da_tese_e_a_arquitetura_concordam_no_numero_de_modelos():
    import yaml

    cfg = yaml.safe_load((REPO / "configs" / "config.yaml").read_text(encoding="utf-8"))

    # A invariante NÃO é a contagem de modelos no registro — era isso que este
    # teste afirmava, e estava certo pela razão errada. O que a pergunta exige é
    # que atenção e confiança venham dos MESMOS pesos, o que se garante fixando
    # UM modelo por DECLARAÇÃO. O registro pode ter N escalas; o que não pode é
    # uma declaração ambígua sobre qual delas produziu a tabela.
    for nome, dados in cfg["models"].items():
        assert dados.get("ner_specific") is True, (
            f"{nome} não encontra entidade: encoder cru no registro seria uma opção em que a "
            f"atenção vem de um modelo e a confiança de outro (ARCHITECTURE.md §6)"
        )
        assert dados.get("is_autoregressive") is False, f"{nome} não é bidirecional"

    declaracoes = sorted((REPO / "configs").glob("decl-*.yaml"))
    assert declaracoes, "nenhuma declaração encontrada"
    for caminho in declaracoes:
        d = yaml.safe_load(caminho.read_text(encoding="utf-8"))["selective"]
        if d.get("hash_version", 1) >= 4:
            assert isinstance(d.get("model"), str) and d["model"], (
                f"{caminho.name} é da versão 4 e não fixa UM modelo"
            )
    assert "One model per declaration, and the reason is the question itself" in arch()


# ---------------------------------------------------------------------------
# 6. Divergências conhecidas, declaradas em vez de omitidas
# ---------------------------------------------------------------------------

def test_existe_produtor_da_tabela_de_entidades():
    """K5 fechado em 02/09/2026: src/selective/measurement.py é o produtor.

    Este teste era xfail(strict=True) justamente para falhar no dia em que o
    produtor existisse, obrigando a fechar K5 no ARCHITECTURE.md em vez de
    deixar a pendência descrever um estado que passou. Foi o que aconteceu.
    """
    fontes = [p.read_text(encoding="utf-8") for p in SRC.rglob("*.py")]
    assert any("entities.csv" in f and "writer" in f for f in fontes)


@pytest.mark.xfail(
    strict=True,
    reason="K1 do §11: 116 blocos `except Exception` amplos em src/, vários em caminho "
           "de carga de dados. Mesma classe de defeito que o try/except ImportError "
           "removido de core/__init__.py.",
)
def test_nao_ha_except_exception_amplo_no_caminho_de_dados():
    loader = (SRC / "core" / "loaders" / "base" / "loader.py").read_text(encoding="utf-8")
    assert "except Exception" not in loader
