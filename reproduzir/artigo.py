"""NÍVEL 1b DE REPRODUÇÃO: refaz cada número, tabela e figura do artigo.

    python reproduzir/artigo.py

O artigo não tem nenhum resultado digitado: todo número do texto é uma macro de
`artigo/numeros.tex`, as tabelas por estrato são `artigo/tab_estratos.tex` e `artigo/tab_estratos_decoder.tex` e a conferência
do nulo é `artigo/numeros_nulo.tex`. Este script guarda os três como estão, regenera-os a
partir de `docs/tese/resultados/` (as tabelas que `reproduzir/analise.py` confere) e do
código de `src/selective/`, e exige que saiam IDÊNTICOS, byte a byte. Redesenha também as
figuras de `artigo/img/`, que não são comparadas byte a byte (o PDF carrega data de criação).

Termina com código 1 se qualquer arquivo diferir.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
ART = RAIZ / "artigo"
GERADOS = ("numeros.tex", "tab_estratos.tex", "tab_estratos_decoder.tex", "numeros_nulo.tex")
PASSOS = (("números e tabela por estrato", "numeros.py"),
          ("conferência numérica dos nulos", "verificar_nulo.py"),
          ("figuras", "figuras.py"))


def main() -> int:
    antes = {f: (ART / f).read_bytes() for f in GERADOS}
    env = {**os.environ, "MPLBACKEND": "Agg", "KMP_DUPLICATE_LIB_OK": "TRUE"}
    try:
        for rotulo, script in PASSOS:
            r = subprocess.run([sys.executable, str(ART / script)], cwd=ART, env=env,
                               capture_output=True, text=True)
            if r.returncode:
                print(f"{rotulo}: FALHOU\n{r.stderr[-1500:]}")
                return 1
            print(f"{rotulo}: executado")
        falhas = 0
        for f in GERADOS:
            igual = (ART / f).read_bytes() == antes[f]
            falhas += not igual
            print(f"  {f:22s} {'IDÊNTICO ao publicado' if igual else 'DIFERE do publicado'}")
    finally:
        # o repositório nunca fica com a versão regerada no lugar da publicada
        for f, b in antes.items():
            (ART / f).write_bytes(b)
    print(f"\n{len(GERADOS) - falhas} de {len(GERADOS)} arquivos gerados idênticos")
    return 1 if falhas else 0


if __name__ == "__main__":
    sys.exit(main())
