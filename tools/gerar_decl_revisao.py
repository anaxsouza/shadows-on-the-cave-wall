"""Writes the declarations of the revision (decl-14 to decl-25), mechanically.

    python tools/gerar_decl_revisao.py

Origin: review of the CL manuscript by J. V. Miranda e Silva (30/09/2026); design approved by
the author on 01/10/2026 (SUGESTOES_C1_C4.md). Four tests:

    C4  decl-14-tipos-de-erro          error types (analysis of re-predicted outputs)
    C1  decl-15-janelas-de-controle    control windows of the same size in the same sentence
    C2  decl-16-previsao-bc5cdr        prediction of the sign on an unseen corpus (BC5CDR)
        decl-17..21                    the five extractors of the article on BC5CDR
    C3  decl-22-bert                   a token classifier (bert-base-cased + softmax)
        decl-23..25                    bert on GENIA, CoNLL-2003 and BC5CDR

Every executable configuration is a COPY of an already signed one with only the declaration id
and the model replaced; the script asserts that nothing else differs. No item is re-chosen.
"""
from __future__ import annotations

import difflib
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
from src.selective.preregistration import load_preregistration  # noqa: E402

CFG = RAIZ / "configs"
DOC = RAIZ / "docs" / "tese" / "declaracoes"
DATA = "2026-10-01"
PREAUT = ("Assinada sob pré-autorização expressa do autor, sem leitura do texto final "
          "(\"certo, aprovo tudo, dispare todos os testes e processos\", 01/10/2026). A "
          "contrapartida, registrada aqui e não omitida: nenhum parâmetro desta declaração foi "
          "escolhido nesta data; cada um é herdado de declaração assinada ou é a regra sem "
          "parâmetro livre aprovada pelo autor em SUGESTOES_C1_C4.md.")

# (new id, template id, template model, new model, corpus, note)
PONTOS = [
    ("decl-17-bc5cdr-gliner-base", "decl-04-escala", "urchade/gliner_large", "urchade/gliner_base",
     "BC5CDR", "GLiNER base sem ajuste (a configuração da decl-04 com o modelo da primeira escala)"),
    ("decl-18-bc5cdr-gliner-large", "decl-04-escala", "urchade/gliner_large", "urchade/gliner_large",
     "BC5CDR", "GLiNER large sem ajuste"),
    ("decl-19-bc5cdr-ajustado", "decl-05-ajustado-genia", "gliner_base-ft-genia", "gliner_base-ft-bc5cdr",
     "BC5CDR", "GLiNER base ajustado ao BC5CDR com a receita de tools/treinar_extrator.py"),
    ("decl-20-bc5cdr-decoder-05b", "decl-10-decoder-05b-genia", "qwen05b-ft-genia", "qwen05b-ft-bc5cdr",
     "BC5CDR", "Qwen2.5-0.5B ajustado ao BC5CDR com a receita de tools/treinar_decoder.py"),
    ("decl-21-bc5cdr-decoder-15b", "decl-12-decoder-15b-genia", "qwen15b-ft-genia", "qwen15b-ft-bc5cdr",
     "BC5CDR", "Qwen2.5-1.5B ajustado ao BC5CDR com a receita de tools/treinar_decoder.py"),
    ("decl-23-bert-genia", "decl-05-ajustado-genia", "gliner_base-ft-genia", "bert-base-cased-ft-genia",
     "GENIA", "bert-base-cased ajustado ao GENIA com a receita da decl-22"),
    ("decl-24-bert-conll", "decl-05-ajustado-genia", "gliner_base-ft-genia", "bert-base-cased-ft-conll2003",
     "CoNLL-2003", "bert-base-cased ajustado ao CoNLL-2003 com a receita da decl-22"),
    ("decl-25-bert-bc5cdr", "decl-05-ajustado-genia", "gliner_base-ft-genia", "bert-base-cased-ft-bc5cdr",
     "BC5CDR", "bert-base-cased ajustado ao BC5CDR com a receita da decl-22"),
]


def clonar(novo: str, molde: str, mod_velho: str, mod_novo: str) -> Path:
    src = (CFG / f"{molde}.yaml").read_text(encoding="utf-8")
    assert src.count(f"declaration_id: '{molde}'") == 1, molde
    assert src.count(f"model: '{mod_velho}'") == 1, (molde, mod_velho)
    out = src.replace(f"declaration_id: '{molde}'", f"declaration_id: '{novo}'")
    out = out.replace(f"model: '{mod_velho}'", f"model: '{mod_novo}'")
    cab = (f"# {novo}: copia de configs/{molde}.yaml; trocados SO o declaration_id e o model.\n"
           f"# Gerado por tools/gerar_decl_revisao.py em {DATA}. Nao editar.\n")
    out = cab + out
    dif = [l for l in difflib.unified_diff(src.splitlines(), out.splitlines(), lineterm="", n=0)
           if l[:1] in "+-" and not l.startswith(("+++", "---"))]
    mudou = {l[1:].strip() for l in dif}
    permitido = {f"declaration_id: '{molde}'", f"declaration_id: '{novo}'", f"model: '{mod_velho}'",
                 f"model: '{mod_novo}'", cab.splitlines()[0], cab.splitlines()[1]}
    assert mudou <= permitido, mudou - permitido
    p = CFG / f"{novo}.yaml"
    p.write_text(out, encoding="utf-8")
    return p


def doc_ponto(novo: str, molde: str, mod_novo: str, corpus: str, nota: str, guarda: str) -> None:
    pre = load_preregistration(CFG / f"{novo}.yaml")
    molde_pre = load_preregistration(CFG / f"{molde}.yaml")
    assert pre.analysis_hash == molde_pre.analysis_hash, "itens de análise divergiram do molde"
    txt = f"""# {novo}

**Estado:** ASSINADA em {DATA}, sob a declaração-guarda `{guarda}`. {PREAUT}

| | |
|---|---|
| `declaration_id` | `{novo}` |
| Hash da declaração | `{pre.declaration_hash}` |
| Hash de **medição** | `{pre.measurement_hash}` |
| Hash de análise | `{pre.analysis_hash}` (idêntico ao de `{molde}`) |
| Modelo | `{mod_novo}` |
| Corpus | {corpus}, partição de teste |
| Molde | `configs/{molde}.yaml`, copiado com só `declaration_id` e `model` trocados |

{nota}. Todos os itens de análise, as comparações (C1, C2, C3, T1–T4) e os critérios são os do
molde, palavra por palavra; o gerador `tools/gerar_decl_revisao.py` recusa a cópia se qualquer
outra linha diferir e confere que o hash de análise é o mesmo do molde. O que esta declaração
acrescenta ao molde está em `{guarda}.md`.
"""
    (DOC / f"{novo}.md").write_text(txt, encoding="utf-8")


def main() -> int:
    for novo, molde, mv, mn, corp, nota in PONTOS:
        clonar(novo, molde, mv, mn)
        guarda = "decl-22-bert" if "bert" in novo else "decl-16-previsao-bc5cdr"
        doc_ponto(novo, molde, mn, corp, nota, guarda)
        p = load_preregistration(CFG / f"{novo}.yaml")
        print(f"{novo:32s} decl={p.declaration_hash} med={p.measurement_hash}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
