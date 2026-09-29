"""Comandos da CLI. Um só: `selective`.

Havia sete (list, train, evaluate, experiment, checkpoints, recover, gc), que
serviam ao ciclo de fine-tuning e à conjunção de sete afirmações. Recuperáveis em
`git show pre-c4-reorg:src/cli/commands/<arquivo>`.
"""

from .selective import build_selective_parser, run_selective

__all__ = [
    "build_selective_parser",
    "run_selective",
]
