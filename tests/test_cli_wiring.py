"""O comando não pode contornar o config, e o pré-registro não pode ter atalho.

Antes: `train.py`, `evaluate.py` e `list.py` tinham de importar `ConfigManager`,
para que nenhum comando lesse hiperparâmetro fora do `config.yaml`. Os três
saíram, e a exigência mudou de forma: existe um comando, e o que ele não pode
contornar é a seção `selective` do config — que é o pré-registro.
"""

from __future__ import annotations

import ast
from pathlib import Path

RAIZ = Path(__file__).parent.parent
SELECTIVE_CLI = RAIZ / "src" / "cli" / "commands" / "selective.py"
RUNNER = RAIZ / "src" / "selective" / "runner.py"
COMANDOS = RAIZ / "src" / "cli" / "commands"


def arvore(p: Path) -> ast.Module:
    return ast.parse(p.read_text(encoding="utf-8"))


def chamadas(p: Path) -> set[str]:
    nomes = set()
    for no in ast.walk(arvore(p)):
        if isinstance(no, ast.Call):
            f = no.func
            nomes.add(f.id if isinstance(f, ast.Name) else getattr(f, "attr", ""))
    return nomes


def flags(p: Path) -> set[str]:
    """Flags declaradas via add_argument, pelo nome do primeiro argumento."""
    achadas = set()
    for no in ast.walk(arvore(p)):
        if isinstance(no, ast.Call) and getattr(no.func, "attr", "") == "add_argument":
            for arg in no.args:
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    achadas.add(arg.value)
    return achadas


def test_existe_exatamente_um_comando():
    arquivos = sorted(
        p.name for p in COMANDOS.glob("*.py") if p.name != "__init__.py"
    )
    assert arquivos == ["selective.py"], (
        f"a CLI deveria ter um comando; encontrados: {arquivos}. Sete comandos "
        f"existiam para o ciclo de fine-tuning e a conjunção de sete afirmações."
    )


def test_o_comando_aceita_config_e_o_repassa():
    assert "--config" in flags(SELECTIVE_CLI)
    fonte = SELECTIVE_CLI.read_text(encoding="utf-8")
    assert "config_path=args.config" in fonte, (
        "o comando declara --config mas não o repassa ao runner"
    )


def test_o_runner_carrega_o_preregistro_no_construtor():
    """Carregar no __init__, e não no ponto de uso, é o que faz a recusa valer.

    Se o pré-registro fosse lido tarde, uma execução poderia carregar modelo,
    varrer o corpus e só então descobrir que a declaração falta. A recusa tem de
    acontecer antes de qualquer número existir.
    """
    init = next(
        n for n in ast.walk(arvore(RUNNER))
        if isinstance(n, ast.FunctionDef) and n.name == "__init__"
    )
    chamadas_init = {
        (c.func.id if isinstance(c.func, ast.Name) else getattr(c.func, "attr", ""))
        for c in ast.walk(init) if isinstance(c, ast.Call)
    }
    assert "load_preregistration" in chamadas_init, (
        "SelectiveRunner.__init__ não chama load_preregistration"
    )


def test_o_comando_tem_dry_run():
    assert "--dry-run" in flags(SELECTIVE_CLI)
    assert "plan" in chamadas(SELECTIVE_CLI), (
        "--dry-run tem de mostrar o plano, incluindo o pré-registro em vigor"
    )


def test_o_comando_nao_declara_limiar_nem_hipotese():
    """Asserção negativa, e é a razão de este arquivo continuar existindo."""
    declaradas = flags(SELECTIVE_CLI)
    for morta in ("--threshold", "--hypothesis", "--controls", "--resume"):
        assert morta not in declaradas, (
            f"{morta} voltou à CLI. A curva risco-cobertura é a varredura de todos os "
            f"limiares (objeção R4), e a conjunção de hipóteses saiu da tese."
        )
