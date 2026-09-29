"""Os dois testes de ponta a ponta, sobre tabela sintética.

Sintética e não medida, e a distinção importa: aqui se verifica que o
encanamento calcula o que diz calcular — separação por sentença, ajuste da
combinação na calibração, veredito pelo intervalo — e não que a massa de
atenção funcione. Isso é o piloto, e é medição.

A tabela é construída com um sinal PLANTADO: `span_mass` correlacionada com
acerto e `model_confidence` deliberadamente inútil. Se o runner não detectar
ganho aqui, o defeito é do runner.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from src.selective.preregistration import PreregistrationError
from src.selective.runner import EntityTableError, SelectiveRunner

CONFIG = """
selective:
  declaration_id: decl-teste
  sink_policy: drop_from_denominator
  layers: [8, 9, 10, 11]
  heads: null
  combination_rule: convex
  calibration_fraction: 0.3
  coverage_levels: [0.5, 0.8, 1.0]
  operating_point: derived_from_target_risk
  added_value_ci_level: 0.95
  n_bootstrap_resamples: 300
  target_risk_grid: [0.2]
  conformal_alpha: 0.2
  loss_bound: 1.0
  seed: 11
"""


def _escrever_tabela(destino: Path, n_sentencas: int = 60, seed: int = 3) -> None:
    """Duas entidades por sentença; massa informativa, confiança do modelo inútil."""
    rng = np.random.default_rng(seed)
    destino.parent.mkdir(parents=True, exist_ok=True)
    with destino.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["sentence_id", "loss", "model_confidence", "span_mass", "is_nested"])
        for s in range(n_sentencas):
            for k in range(2):
                errada = rng.random() < 0.35
                massa = rng.normal(0.25 if errada else 0.55, 0.08)
                w.writerow([
                    f"s{s}",
                    1 if errada else 0,
                    round(float(rng.uniform(0.80, 0.99)), 4),   # sem relação com o erro
                    round(float(np.clip(massa, 0.01, 0.99)), 4),
                    int(k == 1),
                ])


@pytest.fixture()
def ambiente(tmp_path: Path) -> tuple[str, str]:
    cfg = tmp_path / "config.yaml"
    cfg.write_text(CONFIG, encoding="utf-8")
    saida = tmp_path / "results"
    _escrever_tabela(saida / "modelo-x" / "genia" / "test" / "entities.csv")
    return str(cfg), str(saida)


def _runner(ambiente: tuple[str, str]) -> SelectiveRunner:
    cfg, saida = ambiente
    return SelectiveRunner("modelo-x", "genia", config_path=cfg, output_dir=saida)


def test_o_teste_do_piso_detecta_sinal_plantado_na_massa(ambiente):
    r = _runner(ambiente).run("floor")
    assert r.test == "floor"
    assert r.n_entities == 120
    assert r.n_sentences == 60
    assert r.n_calibration_entities + r.n_test_entities == r.n_entities
    # a massa foi plantada informativa, então tem de bater a abstenção aleatória
    assert r.delta.delta < 0
    assert not r.delta.contains_zero
    assert r.delta.aurc_a == pytest.approx(r.base_risk, abs=1e-9)
    assert "massa de atenção ordena melhor" in r.delta.render()


def test_o_teste_de_valor_adicionado_estratifica_aninhado_e_plano(ambiente):
    r = _runner(ambiente).run("added-value")
    assert r.test == "added-value"
    assert r.nested_delta is not None and r.flat_delta is not None
    assert r.nested_delta.n_entities + r.flat_delta.n_entities == r.n_test_entities
    # a confiança do modelo foi plantada inútil; combinar com a massa tem de ganhar
    assert r.delta.delta < 0
    texto = r.render()
    assert "NÃO constituem teste da diferença entre estratos" in texto


def test_a_separacao_e_por_sentenca_e_nenhuma_sentenca_cai_nos_dois_lados(ambiente):
    runner = _runner(ambiente)
    tabela = runner._load_table()
    cal, aval = runner._split_by_sentence(tabela)
    assert set(cal["sentence_id"]).isdisjoint(set(aval["sentence_id"]))
    # a fração declarada é de sentenças, e 30% de 60 são 18
    assert np.unique(cal["sentence_id"]).size == 18


def test_a_separacao_e_reproduzivel_pela_semente(ambiente):
    cfg, saida = ambiente
    a = SelectiveRunner("modelo-x", "genia", config_path=cfg, output_dir=saida)
    b = SelectiveRunner("modelo-x", "genia", config_path=cfg, output_dir=saida)
    ca, _ = a._split_by_sentence(a._load_table())
    cb, _ = b._split_by_sentence(b._load_table())
    assert set(ca["sentence_id"]) == set(cb["sentence_id"])


def test_o_relatorio_serializa_com_o_preregistro_inteiro(ambiente):
    d = _runner(ambiente).run("floor").to_dict()
    assert d["preregistration"]["conformal_alpha"] == 0.2
    assert d["preregistration"]["calibration_fraction"] == 0.3
    assert d["threshold"]["alpha"] == 0.2
    assert d["delta"]["n_resamples"] <= 300


def test_tabela_ausente_ou_incompleta_diz_o_que_falta(tmp_path: Path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(CONFIG, encoding="utf-8")
    runner = SelectiveRunner("m", "genia", config_path=str(cfg), output_dir=str(tmp_path / "r"))
    with pytest.raises(EntityTableError, match="tabela de entidades ausente"):
        runner.run("floor")

    caminho = tmp_path / "r" / "m" / "genia" / "test" / "entities.csv"
    caminho.parent.mkdir(parents=True, exist_ok=True)
    caminho.write_text("sentence_id,loss\ns0,0\n", encoding="utf-8")
    with pytest.raises(EntityTableError, match="não tem as colunas"):
        runner.run("floor")


def test_config_sem_a_secao_selective_impede_a_execucao(tmp_path: Path):
    """A recusa é a garantia: sem os seis itens declarados, nada roda."""
    cfg = tmp_path / "vazio.yaml"
    cfg.write_text("models: {}\n", encoding="utf-8")
    with pytest.raises(PreregistrationError, match="não tem a seção 'selective'"):
        SelectiveRunner("m", "genia", config_path=str(cfg))


def test_ponto_operacional_declarado_como_cobertura_e_recusado(tmp_path: Path):
    """Declarar cobertura fixa inverteria a parametrização do próprio método.

    Geifman & El-Yaniv (2017) declaram o risco alvo r* e maximizam a cobertura
    sujeita a ele; o controle conformal de risco declara alpha com o mesmo papel.
    Uma cobertura declarada aqui seria o método ao contrário, e a recusa é o que
    impede a declaração antiga de voltar em silêncio.
    """
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        CONFIG.replace("operating_point: derived_from_target_risk", "operating_point: 0.8"),
        encoding="utf-8",
    )
    with pytest.raises(PreregistrationError, match="derived_from_target_risk"):
        SelectiveRunner("m", "genia", config_path=str(cfg))

def test_teste_desconhecido_e_recusado(ambiente):
    with pytest.raises(ValueError, match="teste desconhecido"):
        _runner(ambiente).run("H1.1")


class TestProcedenciaDaTabela:
    """Medida e teste têm de vir de uma declaração ADMISSÍVEL.

    Admissível é a própria declaração, ou aquela de quem esta HERDA a medição — e
    a herança é DECLARADA, não inferida pelo guarda. É o que deixa uma declaração
    só de análise reusar horas de medição sem afrouxar a conferência.
    """

    def _medir(self, ambiente, *, declaration_hash=None, measurement_hash=None,
               declaration_id="decl-antiga") -> None:
        import json
        _, saida = ambiente
        alvo = Path(saida) / "modelo-x" / "genia" / "test" / "MEDIDA.json"
        alvo.parent.mkdir(parents=True, exist_ok=True)
        bloco = {"declaration_id": declaration_id}
        if declaration_hash is not None:
            bloco["declaration_hash"] = declaration_hash
        if measurement_hash is not None:
            bloco["measurement_hash"] = measurement_hash
        alvo.write_text(json.dumps({"preregistro": bloco}), encoding="utf-8")

    def _decl2(self, tmp_path, origem, *, com_hash=True):
        """Uma declaração de ANÁLISE que herda a medição de `origem`."""
        texto = CONFIG.replace("declaration_id: decl-teste", "declaration_id: decl-teste-2") + (
            "  hash_version: 2\n"
            "  comparison_scores: [span_size, geometric_fraction]\n"
            "  stratify_by: [span_size]\n"
            "  span_size_bins: [[1, 1], [2, null]]\n"
            "  comparisons:\n"
            "    - id: C2\n"
            "      kind: verdict\n"
            "      question: q\n"
            "      score: span_mass\n"
            "      against: geometric_fraction\n"
            "      criterion: paired_delta_aurc_ci_excludes_zero\n"
            "  inherits_measurement_from:\n"
            "    declaration_id: decl-teste\n"
        )
        if com_hash:
            texto += f"    declaration_hash: '{origem.declaration_hash}'\n"
        alvo = tmp_path / ("decl2.yaml" if com_hash else "ruim.yaml")
        alvo.write_text(texto, encoding="utf-8")
        return alvo

    def test_hash_diferente_e_recusado(self, ambiente):
        """Trocar um item e rodar sobre a tabela antiga produziria resultado sem
        erro nenhum, com metade dos parâmetros de cada declaração."""
        self._medir(ambiente, declaration_hash="0000000000000000")
        with pytest.raises(EntityTableError, match="não herda|exige remedir"):
            _runner(ambiente).run("floor")

    def test_hash_igual_passa(self, ambiente):
        r = _runner(ambiente)
        self._medir(ambiente, declaration_hash=r.prereg.declaration_hash)
        assert r.run("floor") is not None

    def test_parametro_de_medicao_diferente_e_recusado(self, ambiente):
        """measurement_hash divergente é RECUSA e não ressalva: sumidouro,
        camadas ou cabeças diferentes produzem outros números na tabela."""
        r = _runner(ambiente)
        self._medir(ambiente, declaration_hash=r.prereg.declaration_hash,
                    measurement_hash="ffffffffffffffff")
        with pytest.raises(EntityTableError, match="parâmetros de medição"):
            r.run("floor")

    def test_medicao_herdada_passa_com_ressalva(self, ambiente, tmp_path):
        """O caso que a partição do hash existe para permitir."""
        origem = _runner(ambiente).prereg
        r = SelectiveRunner("modelo-x", "genia",
                            config_path=str(self._decl2(tmp_path, origem)),
                            output_dir=ambiente[1])
        assert r.prereg.measurement_hash == origem.measurement_hash, "a medição é a mesma"
        assert r.prereg.declaration_hash != origem.declaration_hash, "as identidades diferem"
        self._medir(ambiente, declaration_hash=origem.declaration_hash,
                    measurement_hash=origem.measurement_hash, declaration_id="decl-teste")
        res = r.run("floor")
        assert any("herdada" in n for n in res.notes), res.notes

    def test_heranca_sem_hash_e_recusada(self, ambiente, tmp_path):
        """Herança sem hash não é verificável, e herança não verificável é
        adivinhação — que é o que o guarda existe para impedir."""
        origem = _runner(ambiente).prereg
        with pytest.raises(PreregistrationError, match="declaration_hash"):
            SelectiveRunner("modelo-x", "genia",
                            config_path=str(self._decl2(tmp_path, origem, com_hash=False)),
                            output_dir=ambiente[1])

    def test_itens_de_analise_exigem_versao_2(self, tmp_path):
        """Item declarado fora do hash não é declaração, é comentário."""
        alvo = tmp_path / "v1.yaml"
        alvo.write_text(CONFIG + "  comparison_scores: [span_size]\n", encoding="utf-8")
        with pytest.raises(PreregistrationError, match="hash_version 2"):
            SelectiveRunner("modelo-x", "genia", config_path=str(alvo),
                            output_dir=str(tmp_path))

    def test_procedencia_ausente_e_ressalva_e_nao_recusa(self, ambiente):
        """Tabela anterior ao registro de procedência ainda serve — declarando isso."""
        res = _runner(ambiente).run("floor")
        assert any("procedência NÃO verificada" in n for n in res.notes)


class TestProcedenciaDoModelo:
    """QUAL modelo produziu a tabela, conferido separadamente do hash.

    Separado porque as versões 1 a 3 do conjunto de itens não cobrem o modelo:
    para elas o bloco `modelo` do MEDIDA.json é o único registro, e exigir
    casamento seria exigir de uma declaração algo que ela não declara. A partir
    da versão 4 o modelo entra no hash de medição e o casamento é obrigatório.
    """

    @staticmethod
    def _gravar_modelo(ambiente, modelo_id):
        import json

        # Ao lado da TABELA, que é onde o guarda procura — não na raiz da saída.
        proc = Path(ambiente[1]) / "modelo-x" / "genia" / "test" / "MEDIDA.json"
        proc.parent.mkdir(parents=True, exist_ok=True)
        dados = json.loads(proc.read_text(encoding="utf-8")) if proc.is_file() else {}
        if modelo_id is None:
            dados.pop("modelo", None)
        else:
            dados["modelo"] = {"id": modelo_id, "camadas": 2, "cabecas": 1}
        proc.write_text(json.dumps(dados, ensure_ascii=False), encoding="utf-8")

    def test_versao_sem_modelo_declarado_registra_ressalva_e_nao_recusa(self, ambiente):
        """A declaração não fixa modelo: registrar é o máximo que se pode fazer."""
        self._gravar_modelo(ambiente, "urchade/gliner_base")
        r = _runner(ambiente)
        res = r.run("floor")
        assert any("NÃO declara modelo" in n for n in res.notes), (
            "a tabela traz o modelo e a declaração não o cobre: isso tem de aparecer como "
            "ressalva, não passar em silêncio"
        )

    def test_declaracao_que_fixa_modelo_recusa_tabela_de_OUTRO_modelo(self, ambiente):
        """A recusa que a versão 4 existe para produzir.

        A declaração é construída com `dataclasses.replace` em vez de um config
        sintético da versão 4: o que se testa aqui é o GUARDA, e montar um config
        completo da versão 4 testaria o carregador junto, escondendo qual dos
        dois falhou.
        """
        import dataclasses

        self._gravar_modelo(ambiente, "urchade/gliner_base")
        r = _runner(ambiente)
        object.__setattr__(
            r, "prereg",
            dataclasses.replace(r.prereg, model="urchade/gliner_large", hash_version=4))
        with pytest.raises(EntityTableError, match="produzida por"):
            r.run("floor")

    def test_declaracao_que_fixa_modelo_recusa_tabela_SEM_modelo_gravado(self, ambiente):
        """Tabela anterior ao registro de modelo não serve a quem fixa modelo."""
        import dataclasses

        self._gravar_modelo(ambiente, None)
        r = _runner(ambiente)
        object.__setattr__(
            r, "prereg",
            dataclasses.replace(r.prereg, model="urchade/gliner_large", hash_version=4))
        with pytest.raises(EntityTableError, match="não grava qual modelo"):
            r.run("floor")

    def test_modelo_igual_ao_declarado_passa(self, ambiente):
        import dataclasses

        self._gravar_modelo(ambiente, "urchade/gliner_large")
        r = _runner(ambiente)
        object.__setattr__(
            r, "prereg",
            dataclasses.replace(r.prereg, model="urchade/gliner_large", hash_version=4))
        assert r.run("floor") is not None
