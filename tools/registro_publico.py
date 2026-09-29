"""The file deposited on Zenodo to give the declarations a PUBLIC date.

    python tools/registro_publico.py            # writes docs/tese/declaracoes/REGISTRO_PUBLICO.txt
    python tools/registro_publico.py --conferir # checks the current files against it

WHAT THE DEPOSIT PROVES, AND WHAT IT DOES NOT. The declarations were signed in commits of
a private repository, and a local commit date can be rewritten, so no file can prove
today when they were written. What a deposit proves is that, from the deposit's date on,
every declaration, addendum and executable configuration is byte for byte the one listed
here. A reader of the paper can then hash the released files and compare.

For each declaration it lists the SHA-256 of the signed document (`.md`), of the
configuration the executor reads (`.yaml`), and the two short hashes the executor itself
computes (`declaration_hash`, `measurement_hash`), which the documents quote and which
`load_preregistration` checks before running anything.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
from src.selective.preregistration import load_preregistration  # noqa: E402

DOCS = RAIZ / "docs" / "tese" / "declaracoes"
SAIDA = DOCS / "REGISTRO_PUBLICO.txt"


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def linhas() -> list[str]:
    out = []
    for md in sorted(DOCS.glob("decl-*.md")):
        did = md.stem
        cfg = RAIZ / "configs" / f"{did}.yaml"
        out.append(f"{sha(md)}  docs/tese/declaracoes/{md.name}")
        if cfg.exists():
            p = load_preregistration(cfg)
            out.append(f"{sha(cfg)}  configs/{cfg.name}  "
                       f"declaration_hash={p.declaration_hash}  measurement_hash={p.measurement_hash}")
    for ad in sorted(DOCS.glob("adenda-*.md")):
        out.append(f"{sha(ad)}  docs/tese/declaracoes/{ad.name}")
    for extra in (RAIZ / "docs" / "tese" / "PREREGISTRO.md",):
        if extra.exists():
            out.append(f"{sha(extra)}  {extra.relative_to(RAIZ)}")
    return out


def main() -> int:
    atual = linhas()
    if "--conferir" in sys.argv:
        dep = [l for l in SAIDA.read_text(encoding="utf-8").splitlines() if l and not l.startswith("#")]
        dif = sorted(set(dep) ^ set(atual))
        print("IDÊNTICO ao registro depositado" if not dif else "DIFERE:\n" + "\n".join(dif))
        return 1 if dif else 0
    cab = ["# Public registry of the pre-registered declarations for the article",
           "# 'Attention Is Not Not Geometry'.",
           "# SHA-256 of each file; check with: python tools/registro_publico.py --conferir",
           "# The date that counts is the date of the Zenodo deposit, not that of any file listed here.", ""]
    SAIDA.write_text("\n".join(cab + atual) + "\n", encoding="utf-8")
    print(f"{len(atual)} linhas -> {SAIDA.relative_to(RAIZ)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
