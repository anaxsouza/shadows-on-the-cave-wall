"""A superfície de linha de comando, e o que ela tem de PROIBIR.

Este arquivo testava sete famílias de parser (train, evaluate, experiment
--hypothesis H1.x, list, checkpoints, recover, gc). A CLI tem um comando, então
o arquivo encolheu — mas ganhou a metade que antes não existia: as asserções
NEGATIVAS.

Um teste que só verifica que o comando novo funciona não impede o antigo de
voltar. Metade daqui existe para que a volta da conjunção quebre a suíte: sem
`--hypothesis`, sem `--threshold`, dois testes e não sete, dois corpora e não
sete. Se alguém reintroduzir qualquer um deles, é aqui que aparece.
"""

from __future__ import annotations

import argparse

import pytest

from src.cli.commands.selective import TESTES, build_selective_parser


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="sentinel selective")
    build_selective_parser(p)
    return p


def _parse(argv: list[str]) -> argparse.Namespace:
    return _parser().parse_args(argv)


BASE = ["--test", "floor", "--model", "bert-large", "--dataset", "genia"]


class TestOComandoUnico:
    def test_argumentos_minimos(self):
        a = _parse(BASE)
        assert a.test == "floor"
        assert a.model == "bert-large"
        assert a.dataset == "genia"
        assert a.split == "test"
        assert a.max_samples == -1
        assert a.config == "configs/config.yaml"

    def test_os_dois_testes_sao_aceitos_e_nada_mais(self):
        assert set(TESTES) == {"floor", "added-value"}
        assert _parse(["--test", "added-value", "--model", "m", "--dataset", "conll2003"]).test == (
            "added-value"
        )
        with pytest.raises(SystemExit):
            _parse(["--test", "ceiling", "--model", "m", "--dataset", "genia"])

    def test_os_dois_corpora_sao_aceitos_e_nada_mais(self):
        for corpus in ("conll2003", "genia"):
            assert _parse(["--test", "floor", "--model", "m", "--dataset", corpus]).dataset == corpus
        for morto in ("crossner_politics", "crossner", "ontonotes"):
            with pytest.raises(SystemExit):
                _parse(["--test", "floor", "--model", "m", "--dataset", morto])

    def test_modelo_e_corpus_sao_obrigatorios(self):
        with pytest.raises(SystemExit):
            _parse(["--test", "floor", "--dataset", "genia"])
        with pytest.raises(SystemExit):
            _parse(["--test", "floor", "--model", "m"])
        with pytest.raises(SystemExit):
            _parse(["--model", "m", "--dataset", "genia"])

    def test_dry_run_e_max_samples(self):
        a = _parse(BASE + ["--dry-run", "--max-samples", "50"])
        assert a.dry_run is True
        assert a.max_samples == 50


class TestOQueAConjuncaoDeixouDePoderVoltar:
    """Asserções negativas: a suíte quebra se o desenho antigo reaparecer."""

    def test_nao_existe_hypothesis(self):
        """--hypothesis H1.1..H2.3 era a conjunção na linha de comando."""
        with pytest.raises(SystemExit):
            _parse(["--hypothesis", "H1.1", "--model", "m", "--dataset", "genia"])
        assert "hypothesis" not in _parser().format_help()

    def test_nao_existe_threshold(self):
        """A ausência é a resposta à objeção R4, e não um esquecimento.

        A curva risco-cobertura é a varredura de TODOS os limiares, e o ponto
        operacional é calibrado em partição separada com o nível pré-registrado.
        Um --threshold aqui seria exatamente o grau de liberdade criticado.
        """
        with pytest.raises(SystemExit):
            _parse(BASE + ["--threshold", "0.5"])
        ajuda = _parser().format_help()
        assert "threshold" not in ajuda
        assert "limiar" not in ajuda

    def test_nao_existem_argumentos_de_treino(self):
        for morto in ("--training.lr", "--training.epochs", "--resume", "--controls"):
            with pytest.raises(SystemExit):
                _parse(BASE + [morto, "1"])

    def test_a_cli_nao_expoe_os_comandos_do_ciclo_de_fine_tuning(self):
        from src.cli import commands

        assert not hasattr(commands, "train_model")
        assert not hasattr(commands, "evaluate_checkpoints")
        assert not hasattr(commands, "gc_models")
        assert commands.__all__ == ["build_selective_parser", "run_selective"]


class TestEtapaDeMedicao:
    """A medição é etapa do comando único, e as duas não se misturam."""

    def test_medir_e_testar_sao_mutuamente_exclusivos(self):
        """Pedir os dois é erro, e é erro de propósito.

        Se a medição pudesse acompanhar um teste, um comando que se espera
        rápido (estatística sobre uma tabela, segundos) viraria uma execução de
        horas sobre o corpus inteiro sem ninguém ter pedido.
        """
        with pytest.raises(SystemExit):
            _parse(["--measure", "--test", "floor", "--model", "gliner-base",
                    "--dataset", "genia"])

    def test_um_dos_dois_e_obrigatorio(self):
        with pytest.raises(SystemExit):
            _parse(["--model", "gliner-base", "--dataset", "genia"])

    def test_medir_dispensa_o_teste(self):
        a = _parse(["--measure", "--model", "gliner-base", "--dataset", "genia"])
        assert a.measure is True and a.test is None

    def test_a_medicao_nao_reintroduziu_limiar(self):
        """A ausência de --threshold é a resposta à objeção R4, e vale para a etapa nova."""
        with pytest.raises(SystemExit):
            _parse(["--measure", "--model", "gliner-base", "--dataset", "genia",
                    "--threshold", "0.5"])

    def test_so_a_medicao_carrega_o_modelo(self):
        """Nenhum caminho de teste importa gliner nem torch.

        O runner consome uma tabela já medida; se ele passasse a carregar
        modelo, os testes estatísticos deixariam de ser reexecutáveis em
        segundos sobre resultado já obtido.
        """
        import ast
        from pathlib import Path

        runner = Path(__file__).resolve().parents[1] / "src" / "selective" / "runner.py"
        arvore = ast.parse(runner.read_text(encoding="utf-8"))
        importados = set()
        for no in ast.walk(arvore):
            if isinstance(no, ast.ImportFrom) and no.module:
                importados.add(no.module.split(".")[0])
            elif isinstance(no, ast.Import):
                importados |= {n.name.split(".")[0] for n in no.names}
        assert not importados & {"gliner", "torch", "transformers", "datasets"}, (
            f"runner.py importa {importados & {'gliner','torch','transformers','datasets'}}"
        )
