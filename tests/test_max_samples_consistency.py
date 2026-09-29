"""A convenção de max_samples: -1 significa tudo, e a amostra truncada se declara.

Este teste comparava o padrão de `max_samples` entre sete comandos, para impedir
que um usasse -1, outro None e outro 0 para "todas as amostras". Com um comando
não há mais o que comparar, então ele passou a fixar as duas coisas que a
convenção realmente tem de garantir: que -1 é o padrão, e que uma execução
limitada não se apresente como execução completa.

A segunda é a que importa cientificamente. Um intervalo de confiança calculado
sobre 20 sentenças que se apresenta como intervalo do corpus é reporte de
sucesso indevido, e é silencioso.
"""

from __future__ import annotations

import ast
from pathlib import Path

RAIZ = Path(__file__).parent.parent
SELECTIVE_CLI = RAIZ / "src" / "cli" / "commands" / "selective.py"
RUNNER = RAIZ / "src" / "selective" / "runner.py"


def test_o_padrao_de_max_samples_e_menos_um():
    arvore = ast.parse(SELECTIVE_CLI.read_text(encoding="utf-8"))
    padrao = None
    for no in ast.walk(arvore):
        if isinstance(no, ast.Call) and getattr(no.func, "attr", "") == "add_argument":
            if any(isinstance(a, ast.Constant) and a.value == "--max-samples" for a in no.args):
                for kw in no.keywords:
                    if kw.arg == "default":
                        padrao = ast.literal_eval(kw.value)
    assert padrao == -1, f"padrão de --max-samples é {padrao!r}; a convenção é -1 = todas"


def test_o_runner_registra_ressalva_quando_a_amostra_e_limitada():
    """A ressalva tem de estar no código do runner, não na boa memória de quem roda."""
    fonte = RUNNER.read_text(encoding="utf-8")
    assert "max_samples > 0" in fonte, "o runner não distingue execução limitada"
    assert "não o corpus inteiro" in fonte, (
        "o runner não declara que os intervalos de uma execução limitada não valem "
        "para o corpus"
    )


def test_o_truncamento_e_por_sentenca_e_nao_por_entidade():
    """Truncar por entidade partiria sentenças ao meio.

    Metade das entidades de uma sentença com a outra metade fora muda o
    denominador da massa de atenção sem que nada avise, porque a matriz de
    atenção é da sentença inteira.
    """
    arvore = ast.parse(RUNNER.read_text(encoding="utf-8"))
    fn = next(
        n for n in ast.walk(arvore)
        if isinstance(n, ast.FunctionDef) and n.name == "_load_table"
    )
    corpo = ast.unparse(fn)
    assert "sentence_id" in corpo and "max_samples" in corpo, (
        "_load_table não trunca por sentence_id"
    )