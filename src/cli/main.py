#!/usr/bin/env python3
"""SENTINEL — CLI. Um comando, porque a tese responde uma pergunta.

    A massa de atenção sobre o span adiciona poder de decisão sobre a confiança
    que o próprio modelo já fornece, permitindo abstenção por entidade — e o
    ganho é maior em entidades aninhadas que em planas?

Dois testes dessa pergunta, e são testes e não hipóteses numeradas:

    sentinel selective --test floor        --model bert-large --dataset genia
    sentinel selective --test added-value  --model bert-large --dataset genia

`floor` é o piso: a massa de atenção, calibrada, ordena entidades melhor que a
abstenção aleatória? Sem isso não há o que somar ao softmax. `added-value` é o
teto: o ganho é incremental sobre a confiança do próprio modelo, e maior no
corpus aninhado (GENIA) que no plano (CoNLL-2003)?

O que saiu daqui, e por quê. Esta CLI expunha sete afirmações
(`--hypothesis H1.1 .. H2.3`) mais o ciclo de fine-tuning (`train`,
`checkpoints`, `gc`, `recover`) e a descoberta de modelos (`list`, `evaluate`).
A tese deixou de ser uma conjunção de sete afirmações paralelas, e o código
seguiu: ver `docs/tese/reorg/corte.md`. Para recuperar qualquer comando antigo,
`git show pre-c4-reorg:src/cli/commands/<arquivo>`.
"""

import argparse
import logging
import os
import sys
import warnings

from src.cli.commands.selective import build_selective_parser, run_selective


def _silence_noisy_libs() -> None:
    for nome in ("transformers", "datasets", "urllib3", "filelock", "huggingface_hub"):
        logging.getLogger(nome).setLevel(logging.ERROR)


def _setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    _silence_noisy_libs()


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="sentinel",
        description="SENTINEL — predição seletiva de NER com massa de atenção sobre o span",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")

    sub = parser.add_subparsers(dest="command", required=True, help="Subcomando")
    build_selective_parser(
        sub.add_parser(
            "selective",
            help="Roda um dos dois testes da pergunta única (floor ou added-value)",
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )
    )

    args = parser.parse_args()
    _setup_logging(args.verbose)

    os.environ["HF_DATASETS_DISABLE_PROGRESS_BARS"] = "1"
    os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    warnings.filterwarnings("ignore", message=".*torch\\.jit\\.script.*")
    warnings.filterwarnings("ignore", message=".*Memory Efficient attention.*")

    try:
        return run_selective(args)
    except KeyboardInterrupt:
        print("\nInterrompido", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
