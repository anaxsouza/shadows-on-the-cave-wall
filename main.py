#!/usr/bin/env python3
"""
SENTINEL — Sequential Entity Neural Topology Investigation & Nesting Evaluation for LLMs.

Root entry point. Delegates to the unified CLI.

A tese responde uma pergunta, e a CLI tem um comando com dois testes dela:

    python main.py selective --test floor       --model bert-large --dataset genia
    python main.py selective --test added-value --model bert-large --dataset genia

Os comandos antigos (train, evaluate, experiment --hypothesis H1.1..H2.3, list,
checkpoints, recover, gc) saíram com o desenho da conjunção — ver
docs/tese/reorg/corte.md e a tag pre-c4-reorg.

After ``pip install -e .``, the ``sentinel`` command is equivalent.
"""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.cli.main import main

if __name__ == "__main__":
    sys.exit(main())
