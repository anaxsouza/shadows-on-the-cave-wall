"""Reconstrói `dados_decoder/{genia,conll2003}_test.jsonl` a partir dos corpora públicos.

O segundo programa mede num contêiner sem rede, então lê o teste de um jsonl exportado
DOS MESMOS carregadores do primeiro (mesma ordem, mesmas entidades). O jsonl não é
distribuído, porque contém o texto dos corpora. Este script o refaz na máquina do
auditor, e confere cada arquivo contra o SHA-256 de `dados_decoder/MANIFESTO_test.json`.

Recuperado verbatim da célula que exportou os arquivos em 23/09/2026, que tinha ficado
fora do repositório.

O campo é `ner_char` e NÃO `ner`: os arquivos de treino do mesmo canal usam índice de
palavra em `ner`, e os carregadores dão posição de caractere. Nomes diferentes para
convenções diferentes, de propósito.
"""
import hashlib
import json
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
from src.core.loaders.biomedical.genia import GENIALoader  # noqa: E402
from src.core.loaders.conll.loader import CONLLLoader  # noqa: E402

D = RAIZ / "dados_decoder"
ESP = {"genia": 1854, "conll2003": 3453}


def main() -> int:
    man = json.loads((D / "MANIFESTO_test.json").read_text(encoding="utf-8"))["arquivos"]
    falhou = False
    for nome, L in (("genia", GENIALoader), ("conll2003", CONLLLoader)):
        ex = L().load_split("test")
        assert len(ex) == ESP[nome], f"{nome}: {len(ex)} sentenças, esperadas {ESP[nome]}"
        alvo = D / f"{nome}_test.jsonl"
        with alvo.open("w", encoding="utf-8") as fh:
            for e in ex:
                fh.write(json.dumps({
                    "tokenized_text": list(e.tokens),
                    "ner_char": [[int(s.start), int(s.end), str(s.label)] for s in e.entities],
                }, ensure_ascii=False) + "\n")
        h = hashlib.sha256(alvo.read_bytes()).hexdigest()
        ok = h == man[alvo.name]["sha256"]
        falhou |= not ok
        print(f"{alvo.name}: {len(ex)} sentenças | {'CONFERE' if ok else 'DIVERGE'} {h[:16]}")
    return 1 if falhou else 0


if __name__ == "__main__":
    sys.exit(main())
