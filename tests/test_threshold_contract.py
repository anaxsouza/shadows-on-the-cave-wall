"""Contrato do pré-registro: config.yaml, PREREGISTRO.md e ARCHITECTURE.md §8.

Este arquivo verificava os limiares das sete afirmações (H1.1-H1.4, H2.1-H2.3)
contra o §8 do ARCHITECTURE.md, e o centro dele era o alfa de Bonferroni
α = 0,05/4 = 0,0125 sobre a família de quatro sub-hipóteses. Não há mais família:
uma pergunta, dois testes dela, e os critérios são intervalos e não p-valores.

O que ele fiscaliza agora é a única coisa que pode ser fiscalizada
mecanicamente num pré-registro: que a declaração esteja nos três lugares com o
MESMO valor. O documento é o que a banca lê, a seção do config é o que o código
executa, e o §8 é o que descreve o desenho; divergência entre eles é deriva
silenciosa, que é o modo de falha que esta suíte existe para pegar.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

RAIZ = Path(__file__).parent.parent
CONFIG = RAIZ / "configs" / "config.yaml"
PREREGISTRO = RAIZ / "docs" / "tese" / "PREREGISTRO.md"
ARQUITETURA = RAIZ / "docs" / "ARCHITECTURE.md"

# Os seis itens, pela chave que os nomeia no config. A lista é fechada: item novo
# tem de ser acrescentado aqui de propósito, e não aparecer sem que ninguém veja.
ITENS = (
    "sink_policy",
    "layers",
    "heads",
    "combination_rule",
    "calibration_fraction",
    "coverage_levels",
    # Não é uma cobertura: é a declaração de que o ponto operacional é DERIVADO
    # do risco alvo. Geifman & El-Yaniv (2017) fixam r* e maximizam a cobertura
    # sujeita a ele, e o controle conformal de risco declara alpha com o mesmo
    # papel — declarar cobertura fixa inverteria a parametrização do método.
    "operating_point",
    "added_value_ci_level",
    "n_bootstrap_resamples",
    # Grade de RELATO: cobertura obtida em cada risco alvo, forma da Tabela 1 de
    # Geifman & El-Yaniv (2017). Não custa grau de liberdade porque os critérios
    # dos dois testes são intervalos sobre AURC e não olham para r*.
    "target_risk_grid",
    # alpha É o risco alvo DA GARANTIA r*. Eram dois itens declarados; a literatura mostra
    # que é um, e `Preregistration.target_risk` expõe isso pelo nome.
    "conformal_alpha",
)

# O item que SAIU em 02/09/2026, nomeado para que a volta seja deliberada:
# `entity_confidence_aggregation` era inerte — o GLiNER entrega um escore por
# TRECHO, então não há o que agregar dentro da entidade, e nenhuma conta o
# consumia. Era resquício do desenho com etiquetador BIO.
ITENS_APOSENTADOS = ("entity_confidence_aggregation", "operating_point_coverage")

# Chaves da seção `selective` que NÃO são itens pré-registrados, declaradas aqui
# uma a uma. A lista existe para que acrescentar um interruptor à seção seja um
# ato deliberado e não um acidente: o guarda acima falha em qualquer chave que
# não esteja em ITENS nem aqui.
#
# `layer_profile` liga a saída EXPLORATÓRIA por camada. Ele não é item
# pré-registrado porque não decide nada: descreve onde o sinal mora ao longo da
# profundidade. Medir N camadas e depois anunciar a melhor daria N chances ao
# acaso, e é isso que a separação entre `layers` (confirmatório) e o perfil
# (exploratório) impede.
NAO_PREREGISTRADOS = ("layer_profile", "declaration_id")

# Âncoras numéricas do desenho abandonado. Se qualquer uma reaparecer nos
# documentos ou no config, a conjunção está voltando.
ALFAS_APOSENTADOS = (r"0\.0125", r"\b0\.017\b", r"[Bb]onferroni")
REGRAS_APOSENTADAS = (r"3\s*(of|de)\s*4", r"2\s*(of|de)\s*3", r"\bH1\.\d\b", r"\bH2\.\d\b")


def selective() -> dict:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))["selective"]


def test_os_seis_itens_estao_declarados_no_config():
    s = selective()
    faltando = [k for k in ITENS if k not in s]
    assert not faltando, (
        f"itens de pré-registro ausentes em configs/config.yaml (seção selective): {faltando}"
    )


def test_nenhum_item_extra_entrou_sem_declaracao():
    """Chave nova na seção `selective` é decisão científica, não configuração."""
    extras = sorted(set(selective()) - set(ITENS) - {"loss_bound", "seed"})
    extras = [e for e in extras if e not in NAO_PREREGISTRADOS]
    assert not extras, (
        f"chaves não declaradas na seção selective: {extras}. Acrescente-as a ITENS neste "
        f"teste e a docs/tese/PREREGISTRO.md, ou remova-as."
    )


@pytest.mark.parametrize("chave", ITENS)
def test_cada_item_aparece_no_documento_de_preregistro(chave):
    texto = PREREGISTRO.read_text(encoding="utf-8")
    assert f"`{chave}`" in texto, (
        f"o item '{chave}' está no config mas não é nomeado em docs/tese/PREREGISTRO.md. "
        f"O documento é o que a banca lê: item que só existe no config não está pré-registrado."
    )


@pytest.mark.parametrize(
    "chave,esperado",
    [
        ("conformal_alpha", "0.05"),
        ("calibration_fraction", "0.3"),
        ("added_value_ci_level", "0.95"),
        ("n_bootstrap_resamples", "2000"),
    ],
)
def test_valor_numerico_do_config_aparece_no_documento(chave, esperado):
    """O valor, e não só o nome do item, tem de estar escrito no documento."""
    s = selective()
    assert str(s[chave]) == esperado, (
        f"config mudou: {chave} = {s[chave]}, teste espera {esperado}. Se a mudança é "
        f"deliberada, atualize PREREGISTRO.md, o histórico de alteração dele e este teste."
    )
    texto = PREREGISTRO.read_text(encoding="utf-8")
    assert esperado in texto, f"o valor {esperado} de '{chave}' não aparece em PREREGISTRO.md"


def test_a_secao_8_da_arquitetura_descreve_os_dois_testes_e_nao_hipoteses():
    texto = ARQUITETURA.read_text(encoding="utf-8")
    inicio = texto.index("## 8.")
    fim = texto.index("## 9.")
    secao = texto[inicio:fim]
    for obrigatorio in ("floor", "added-value", "conformal", "risk–coverage", "pre-registered"):
        assert obrigatorio in secao, f"§8 não menciona '{obrigatorio}'"
    for morto in REGRAS_APOSENTADAS:
        achados = [m.group(0) for m in re.finditer(morto, secao)]
        # a seção pode CITAR o desenho abandonado para dizer que ele saiu; o que
        # não pode é especificá-lo. A citação vive num parágrafo próprio.
        assert len(achados) <= 2, (
            f"§8 ainda especifica o desenho abandonado: {morto} aparece {len(achados)} vezes"
        )


def test_alfa_de_bonferroni_nao_esta_no_config_nem_no_preregistro():
    """Asserção negativa: o alfa da conjunção não volta pela porta do config.

    α = 0,0125 = 0,05/4 só faz sentido com quatro afirmações simultâneas. Sua
    presença aqui significaria que a família voltou.
    """
    for arquivo in (CONFIG, PREREGISTRO):
        texto = arquivo.read_text(encoding="utf-8")
        for padrao in ALFAS_APOSENTADOS:
            achados = re.findall(padrao, texto)
            assert not achados, (
                f"{arquivo.name} contém '{padrao}' ({len(achados)}x): o alfa de Bonferroni "
                f"pertence ao desenho de sete afirmações, que a tese abandonou."
            )


def test_o_documento_declara_a_condicao_de_refutacao_e_a_de_inconclusivo():
    """Pré-registro sem critério de refutação declarado não é pré-registro."""
    texto = PREREGISTRO.read_text(encoding="utf-8")
    assert "Refutação declarada" in texto
    assert "inconclusivo" in texto.lower()
    assert "marginal, não condicional" in texto, (
        "a ressalva sobre a garantia conformal ser marginal tem de estar no documento: "
        "é o que separa 'risco esperado controlado' de 'risco garantido neste conjunto'"
    )


def test_o_codigo_nao_tem_valor_padrao_para_item_nenhum():
    """A recusa é a garantia. Sem a seção declarada, nada roda."""
    from src.selective.preregistration import PreregistrationError, load_preregistration

    vazio = RAIZ / "tests" / "_config_sem_selective.yaml"
    vazio.write_text("models: {}\n", encoding="utf-8")
    try:
        with pytest.raises(PreregistrationError, match="não tem a seção 'selective'"):
            load_preregistration(vazio)
    finally:
        vazio.unlink()


def test_o_perfil_por_camada_nao_decide_veredito():
    """A separação confirmatório/exploratório, verificada no código.

    O perfil descreve onde o sinal mora; o veredito sai da faixa declarada em
    `layers`. Se o runner passasse a ler `layer_profile`, escolher a melhor
    camada depois de ver o resultado seria uma linha de código — e seria a
    objeção R4 da banca de volta, agora com N chances em vez de uma.
    """
    runner = (RAIZ / "src" / "selective" / "runner.py").read_text(encoding="utf-8")
    assert "layer_profile" not in runner, (
        "runner.py passou a ler layer_profile; o perfil é exploratório e não pode "
        "alimentar o veredito"
    )

    # E o perfil sai em ARQUIVO SEPARADO: se as massas por camada estivessem em
    # entities.csv, a mesma pescaria estaria a uma coluna de distância.
    from src.selective.measurement import COLUNAS, COLUNAS_PERFIL

    assert "layer" not in COLUNAS
    assert COLUNAS_PERFIL == ("sentence_id", "entity_idx", "layer", "span_mass")


def test_o_preregistro_declara_a_faixa_que_o_modelo_tem():
    """A faixa não pode citar camada que o modelo não possui.

    A faixa 8-15 vigorou até 02/09/2026 e foi escolhida para um modelo de 24
    camadas; o gliner_base tem 12. Truncar mudaria em silêncio quais camadas
    produziram a medida, então o adaptador recusa — e este teste guarda a
    coerência do documento com o modelo declarado no ARCHITECTURE.md §6.
    """
    camadas = selective()["layers"]
    assert max(camadas) <= 11, (
        f"a faixa declarada vai até a camada {max(camadas)}, mas o gliner_base tem 12 "
        f"(índices 0 a 11). Ver ARCHITECTURE.md §6."
    )
    assert min(camadas) >= 0


def test_o_item_inerte_nao_voltou():
    """`entity_confidence_aggregation` era declarado e não fazia nada.

    O GLiNER entrega um escore por trecho: não há o que agregar dentro da
    entidade, e nenhuma conta consumia o item. Item de pré-registro que não
    entra em conta nenhuma dá a impressão de rigor sem o rigor.
    """
    s = selective()
    voltaram = [i for i in ITENS_APOSENTADOS if i in s]
    assert not voltaram, (
        f"itens aposentados voltaram ao config: {voltaram}. Ver docs/tese/PROTOCOLO.md."
    )


def test_o_risco_alvo_e_o_alpha_e_nao_dois_numeros():
    """Um parâmetro, não dois — e a razão é da literatura, não de gosto."""
    from src.selective.preregistration import load_preregistration

    p = load_preregistration(CONFIG)
    assert p.target_risk == p.conformal_alpha
    assert p.target_risk in p.target_risk_grid, (
        f"o nível da garantia {p.target_risk} tem de aparecer na grade de relato "
        f"{p.target_risk_grid}"
    )
    assert 0.01 <= p.target_risk <= 0.10, (
        f"risco alvo {p.target_risk} fora da faixa com precedente citável: "
        f"0,01-0,06 em Geifman & El-Yaniv (2017), 0,1 em Angelopoulos & Bates (2021) "
        f"e em Singer et al. (2026)"
    )


def test_a_declaracao_tem_identidade_estavel():
    """O hash identifica o CONTEÚDO declarado, não o arquivo.

    Mover o config de lugar ou renomear a declaração não muda o que foi
    declarado; mudar qualquer item muda. É isso que permite amarrar uma tabela
    medida à declaração que a produziu.
    """
    from dataclasses import replace

    from src.selective.preregistration import load_preregistration

    p = load_preregistration(CONFIG)
    assert len(p.declaration_hash) == 16
    assert replace(p, source="outro/lugar.yaml").declaration_hash == p.declaration_hash
    assert replace(p, declaration_id="outro-nome").declaration_hash == p.declaration_hash
    assert replace(p, conformal_alpha=0.02).declaration_hash != p.declaration_hash
    assert replace(p, layers=(0, 1)).declaration_hash != p.declaration_hash
    assert replace(p, target_risk_grid=(0.05,)).declaration_hash != p.declaration_hash


def test_toda_tabela_medida_veio_de_declaracao_assinada():
    """Medição sob declaração não assinada é resultado que não conta.

    Este guarda existe porque "assinado" seria só uma linha de texto sem ele. A
    ordem — assinar, depois medir — é a única coisa que um pré-registro garante,
    e é ela que responde à família de objeções R4. Um resultado medido sob
    declaração ainda em PROPOSTA não é inválido por si: é EXPLORATÓRIO, e
    apresentá-lo como confirmatório é que seria o erro.

    Passa trivialmente quando não há medição nenhuma no disco.
    """
    import json

    resultados = RAIZ / "results"
    if not resultados.is_dir():
        pytest.skip("nenhuma medição no disco")

    pendentes = []
    for proc in resultados.rglob("MEDIDA.json"):
        try:
            dados = json.loads(proc.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pendentes.append(f"{proc}: JSON inválido")
            continue
        # Bloco aninhado `preregistro`, que é como measurement.py escreve. Ler a
        # chave errada aqui faria este guarda acusar "não grava declaration_id"
        # em toda medição válida — ruído que treina a pessoa a ignorá-lo.
        bloco = dados.get("preregistro") or dados.get("preregistration") or dados
        ident = bloco.get("declaration_id")
        if not ident:
            pendentes.append(f"{proc}: não grava declaration_id")
            continue
        doc = RAIZ / "docs" / "tese" / "declaracoes" / f"{ident}.md"
        if not doc.is_file():
            pendentes.append(f"{proc}: declaração '{ident}' não está congelada em docs/tese/declaracoes/")
        elif "**Estado:** ASSINADA" not in doc.read_text(encoding="utf-8"):
            pendentes.append(f"{proc}: declaração '{ident}' não está assinada")

    assert not pendentes, (
        "medições produzidas sob declaração não assinada ou não congelada:\n  "
        + "\n  ".join(pendentes)
    )


# =============================================================================
# decl-02: o que a declaração de ANÁLISE tem de recusar
# =============================================================================
# Estes testes existem porque cada recusa aqui fecha uma porta por onde um grau
# de liberdade voltaria depois de ver o resultado.

DECL2 = RAIZ / "configs" / "decl-02-geometria.yaml"

_MINIMA = """
selective:
  declaration_id: 'd'
  hash_version: 2
  sink_policy: 'drop_from_denominator'
  layers: [0, 1]
  heads: null
  combination_rule: 'convex'
  calibration_fraction: 0.3
  coverage_levels: [0.5, 1.0]
  operating_point: 'derived_from_target_risk'
  added_value_ci_level: 0.95
  n_bootstrap_resamples: 100
  target_risk_grid: [0.1]
  conformal_alpha: 0.1
  loss_bound: 1.0
  seed: 1
  comparison_scores: [span_size, geometric_fraction]
  stratify_by: [span_size]
  span_size_bins: [[1, 1], [2, null]]
  comparisons:
    - id: 'C2'
      kind: 'verdict'
      question: 'q'
      score: 'span_mass'
      against: 'geometric_fraction'
      criterion: 'paired_delta_aurc_ci_excludes_zero'
"""


def _carregar(tmp_path, texto):
    from src.selective.preregistration import load_preregistration
    alvo = tmp_path / "d.yaml"
    alvo.write_text(texto, encoding="utf-8")
    return load_preregistration(str(alvo))


def test_a_declaracao_minima_carrega(tmp_path):
    p = _carregar(tmp_path, _MINIMA)
    assert p.span_size_bins == ((1, 1), (2, None))
    assert [c.id for c in p.comparisons] == ["C2"]


def test_faixa_com_buraco_e_recusada(tmp_path):
    """Buraco nas faixas faz entidades sumirem do relato sem ninguém notar."""
    from src.selective.preregistration import PreregistrationError
    with pytest.raises(PreregistrationError, match="buraco"):
        _carregar(tmp_path, _MINIMA.replace("[[1, 1], [2, null]]", "[[1, 1], [3, null]]"))


def test_ultima_faixa_fechada_e_recusada(tmp_path):
    """Span maior que a última borda ficaria fora do relato."""
    from src.selective.preregistration import PreregistrationError
    with pytest.raises(PreregistrationError, match="aberta"):
        _carregar(tmp_path, _MINIMA.replace("[[1, 1], [2, null]]", "[[1, 1], [2, 9]]"))


def test_descritiva_com_criterio_e_recusada(tmp_path):
    """Veredito escondido numa comparação declarada como descritiva."""
    from src.selective.preregistration import PreregistrationError
    ruim = _MINIMA + """    - id: 'C1'
      kind: 'descriptive'
      question: 'q'
      criterion: 'paired_delta_aurc_ci_excludes_zero'
"""
    with pytest.raises(PreregistrationError, match="descritiva"):
        _carregar(tmp_path, ruim)


def test_escore_desconhecido_e_recusado(tmp_path):
    from src.selective.preregistration import PreregistrationError
    with pytest.raises(PreregistrationError, match="desconhecido"):
        _carregar(tmp_path, _MINIMA.replace("score: 'span_mass'", "score: 'intuicao'"))


def test_declaracao_so_descritiva_e_recusada(tmp_path):
    """Uma declaração que não refuta nada não é declaração."""
    from src.selective.preregistration import PreregistrationError
    so_desc = _MINIMA[: _MINIMA.index("  comparisons:")] + """  comparisons:
    - id: 'C1'
      kind: 'descriptive'
      question: 'q'
"""
    with pytest.raises(PreregistrationError, match="não refuta"):
        _carregar(tmp_path, so_desc)


def test_decl01_reproduz_o_hash_que_assinou(tmp_path):
    """O teste mais importante deste arquivo.

    decl-01 está ASSINADA, e a assinatura só prova que o critério veio antes do
    número enquanto o hash for recalculável. Acrescentar itens ao conjunto
    hasheado quebraria isso — é por isso que o conjunto é versionado.
    """
    from src.selective.preregistration import load_preregistration
    d1 = load_preregistration(str(CONFIG))
    assert d1.hash_version == 1
    assert d1.declaration_hash == "348e90cce8ca9874", (
        "decl-01 deixou de reproduzir o hash assinado em 02/09/2026"
    )


def test_decl02_herda_a_medicao_de_decl01(tmp_path):
    """O que permite reusar as tabelas sem remedir por burocracia."""
    from src.selective.preregistration import load_preregistration
    d1 = load_preregistration(str(CONFIG))
    d2 = load_preregistration(str(DECL2))
    assert d2.measurement_hash == d1.measurement_hash
    assert d2.declaration_hash != d1.declaration_hash
    assert d2.inherits_measurement_from == (d1.declaration_id, d1.declaration_hash)


def _todas_as_declaracoes():
    """Toda `configs/decl-*.yaml`, descoberta e não listada.

    Estava escrito para a `decl-02` e só para ela. Guarda amarrado a um arquivo
    deixa de guardar no instante em que aparece outro — e apareceu: a `decl-03` e
    a `decl-04` passaram meses fora da conferência sem nada acusar.
    """
    return sorted((RAIZ / "configs").glob("decl-*.yaml"))


def test_ha_declaracoes_a_conferir():
    """Sem isto, o teste abaixo passaria por vacuidade se o glob quebrasse."""
    assert len(_todas_as_declaracoes()) >= 3, _todas_as_declaracoes()


@pytest.mark.parametrize("fonte_path", _todas_as_declaracoes(), ids=lambda p: p.stem)
def test_o_documento_nao_diverge_da_fonte(fonte_path):
    """Documento gerado da fonte; digitado divergiria sem nada acusar."""
    from src.selective.preregistration import load_preregistration

    p = load_preregistration(str(fonte_path))
    doc_path = RAIZ / "docs" / "tese" / "declaracoes" / f"{p.declaration_id}.md"
    assert doc_path.is_file(), (
        f"{fonte_path.name} não tem documento congelado em docs/tese/declaracoes/: "
        f"a fonte é executável e o documento é o que a banca lê, e os dois têm de existir"
    )
    doc = doc_path.read_text(encoding="utf-8")
    fonte = yaml.safe_load(fonte_path.read_text(encoding="utf-8"))["selective"]

    # AS DECLARAÇÕES DA REVISÃO (decl-17 a decl-25, 01/10/2026) não embutem o bloco
    # YAML: o gerador `tools/gerar_decl_revisao.py` escreveu documentos que citam a
    # fonte em vez de copiá-la, e eles foram depositados assim (registro v2,
    # 10.5281/zenodo.23086955). Editá-los agora quebraria o depósito. O guarda de
    # divergência, para elas, passa a ser o próprio registro: o SHA-256 do documento
    # e o da fonte estão listados lado a lado no REGISTRO_PUBLICO.txt depositado, e o
    # documento tem de nomear a fonte e registrar os dois hashes de identidade.
    if "```yaml" not in doc:
        import hashlib
        reg = (RAIZ / "docs" / "tese" / "declaracoes" / "REGISTRO_PUBLICO.txt").read_text(encoding="utf-8")
        sha_fonte = hashlib.sha256(fonte_path.read_bytes()).hexdigest()
        sha_doc = hashlib.sha256(doc_path.read_bytes()).hexdigest()
        assert f"{sha_fonte}  configs/{fonte_path.name}" in reg, f"{fonte_path.name} fora do registro"
        assert f"{sha_doc}  docs/tese/declaracoes/{doc_path.name}" in reg, f"{doc_path.name} fora do registro"
        assert "tools/gerar_decl_revisao.py" in doc
        for h in (p.declaration_hash, p.measurement_hash):
            assert h in doc, f"hash {h} não aparece em {doc_path.name}"
        return

    # O invólucro `selective:` aparece em alguns documentos e não em outros —
    # decisão de formatação do gerador, não do que foi declarado. O guarda
    # compara os VALORES; exigir o invólucro faria um teste de conteúdo falhar
    # por diferença de formato.
    embutido = yaml.safe_load(doc.split("```yaml")[1].split("```")[0])
    if isinstance(embutido, dict) and set(embutido) == {"selective"}:
        embutido = embutido["selective"]
    assert embutido == fonte, (
        f"{doc_path.name} divergiu de {fonte_path.name}: regenere da fonte"
    )
    # Os dois hashes de IDENTIDADE, e não os três. O `analysis_hash` é derivado
    # (é o conjunto inteiro menos os itens de medição) e o documento da `decl-03`
    # não o registra. Exigi-lo obrigaria a editar um documento ASSINADO para
    # acrescentar um valor que não muda nada do que foi declarado — e editar
    # declaração assinada é precisamente o que não se faz, mesmo quando a edição
    # parece inócua. O bloco YAML acima já garante que o documento não divergiu
    # da fonte; estes dois garantem que ele registra de qual declaração fala.
    for h in (p.declaration_hash, p.measurement_hash):
        assert h in doc, f"hash {h} não aparece em {doc_path.name}"


# =============================================================================
# decl-03: o que a declaração de TAREFA tem de recusar
# =============================================================================
# O par tipo × critério e a grade de metas são as duas portas por onde um grau de
# liberdade voltaria aqui: julgar carga de revisão por critério de AURC, ou
# escolher a meta depois de ver a carga.

DECL3 = RAIZ / "configs/decl-03-tarefa.yaml"


def _decl3_com(tmp_path, substituicoes: dict[str, str]):
    """Copia a decl-03 trocando trechos, para exercitar uma recusa por vez."""
    t = DECL3.read_text(encoding="utf-8")
    for antigo, novo in substituicoes.items():
        assert antigo in t, f"âncora ausente na decl-03: {antigo!r}"
        t = t.replace(antigo, novo, 1)
    alvo = Path(tmp_path) / "d3.yaml"
    alvo.parent.mkdir(parents=True, exist_ok=True)
    alvo.write_text(t, encoding="utf-8")
    return str(alvo)


def test_decl03_carrega_e_herda_a_medicao_das_assinadas():
    from src.selective.preregistration import load_preregistration

    d1 = load_preregistration(str(CONFIG))
    d3 = load_preregistration(str(DECL3))
    assert d3.hash_version == 3
    assert d3.measurement_hash == d1.measurement_hash, (
        "a decl-03 não muda nada da medição; hash de medição diferente significa que muda"
    )
    assert d3.declaration_hash != d1.declaration_hash


def test_as_quatro_metricas_sao_declaradas_juntas(tmp_path):
    """Relatar só precisão seria relatar a que o supervisor faz subir."""
    from src.selective.preregistration import PreregistrationError, load_preregistration

    caminho = _decl3_com(tmp_path, {
        "    - precision_delivered\n    - recall_delivered\n": "    - precision_delivered\n"})
    with pytest.raises(PreregistrationError, match="omite"):
        load_preregistration(caminho)


def test_grade_de_metas_fora_de_ordem_e_recusada(tmp_path):
    from src.selective.preregistration import PreregistrationError, load_preregistration

    caminho = _decl3_com(tmp_path, {
        "task_quality_grid: [0.70, 0.80, 0.90, 0.95]": "task_quality_grid: [0.90, 0.70, 0.80]"})
    with pytest.raises(PreregistrationError, match="fora de ordem"):
        load_preregistration(caminho)


def test_veredito_de_tarefa_com_criterio_de_aurc_e_recusado(tmp_path):
    """A porta principal: julgar carga de revisão com o critério da curva."""
    from src.selective.preregistration import PreregistrationError, load_preregistration

    caminho = _decl3_com(tmp_path, {
        "      criterion: 'paired_review_load_ci_excludes_zero'":
        "      criterion: 'paired_delta_aurc_ci_excludes_zero'"})
    with pytest.raises(PreregistrationError, match="critério desse tipo"):
        load_preregistration(caminho)


def test_veredito_de_tarefa_exige_a_versao_3_do_hash(tmp_path):
    """Sem a grade no hash, a meta poderia ser escolhida depois de ver o número."""
    from src.selective.preregistration import PreregistrationError, load_preregistration

    caminho = _decl3_com(tmp_path, {"  hash_version: 3": "  hash_version: 2"})
    with pytest.raises(PreregistrationError, match="hash_version 3"):
        load_preregistration(caminho)


def test_itens_da_tarefa_em_declaracao_anterior_sao_recusados(tmp_path):
    """Item fora do hash da versão declarada não é declaração, é comentário."""
    from src.selective.preregistration import PreregistrationError, load_preregistration

    cfg = tmp_path / "d2_com_grade.yaml"
    cfg.write_text(
        DECL2.read_text(encoding="utf-8") + "\n  task_quality_grid: [0.9]\n", encoding="utf-8")
    with pytest.raises(PreregistrationError, match="hash_version 3"):
        load_preregistration(str(cfg))


# =============================================================================
# Versão 4 do conjunto de itens: o MODELO entra no hash de medição
# =============================================================================


def test_model_em_declaracao_de_versao_anterior_e_recusado(tmp_path):
    """Item fora do hash da versão declarada não é declaração, é comentário."""
    from src.selective.preregistration import PreregistrationError, load_preregistration

    caminho = _decl3_com(tmp_path, {"  hash_version: 3": "  hash_version: 3\n  model: 'x/y'"})
    with pytest.raises(PreregistrationError, match="hash_version 4"):
        load_preregistration(caminho)


def test_versao_4_exige_o_modelo(tmp_path):
    from src.selective.preregistration import PreregistrationError, load_preregistration

    caminho = _decl3_com(tmp_path, {"  hash_version: 3": "  hash_version: 4"})
    with pytest.raises(PreregistrationError, match="model"):
        load_preregistration(caminho)


def test_o_modelo_muda_o_hash_de_MEDICAO_e_nao_so_o_da_declaracao(tmp_path):
    """É o ponto inteiro da versão 4: duas escalas não podem compartilhar medição."""
    from src.selective.preregistration import load_preregistration

    a = load_preregistration(_decl3_com(
        tmp_path / "a", {"  hash_version: 3": "  hash_version: 4\n  model: 'urchade/gliner_base'"}))
    b = load_preregistration(_decl3_com(
        tmp_path / "b", {"  hash_version: 3": "  hash_version: 4\n  model: 'urchade/gliner_large'"}))
    assert a.measurement_hash != b.measurement_hash, (
        "duas escalas com o mesmo hash de medição: o guarda aceitaria cruzar as tabelas"
    )
    assert a.declaration_hash != b.declaration_hash


def test_as_tres_assinadas_seguem_reproduzindo_apos_a_versao_4():
    """A razão do versionamento: assinatura que deixa de reproduzir não prova ordem."""
    from src.selective.preregistration import load_preregistration

    for caminho, esperado in ((CONFIG, "348e90cce8ca9874"), (DECL2, "4d91c3607206b9ab"),
                              (DECL3, "5f30a2ff3e493c19")):
        d = load_preregistration(str(caminho))
        assert d.declaration_hash == esperado, f"{caminho.name} deixou de reproduzir o assinado"
        assert d.model is None, "versão anterior à 4 não declara modelo"
