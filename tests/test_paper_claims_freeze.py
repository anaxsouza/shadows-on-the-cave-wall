"""Afirmação publicada não muda de valor em silêncio.

A tese é redigida DEPOIS dos artigos e não é os artigos (decisão do autor,
02/09/2026: o programa exige ao menos um artigo publicado para a defesa). Isso
remove um risco e cria outro.

Remove: capítulo e manuscrito não compartilham texto, logo não há cópia a
divergir. O capítulo é reescrito com a voz e o comprimento da tese, que é o que
a objeção R8 da banca pede.

Cria: um artigo publicado congela números. Se o `sink_policy` do pré-registro
mudar depois de um artigo afirmar que a convenção vale 0,30 contra 0,75, a tese
passa a contradizer o que está publicado — e a contradição aparece na defesa,
não aqui. Este guarda compara o que cada artigo declara ter publicado com o que
o repositório usa hoje.

O mecanismo é um arquivo por artigo, `papers/<slug>/AFIRMACOES.yaml`, escrito no
momento da submissão. Enquanto nenhum artigo tiver esse arquivo, os testes
passam vazios de propósito: o guarda existe antes de haver o que guardar,
porque criá-lo depois da primeira submissão é criá-lo tarde.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

RAIZ = Path(__file__).parent.parent
PAPERS = RAIZ / "writing" / "papers"
CONFIG = RAIZ / "configs" / "config.yaml"


def declaracoes() -> list[tuple[str, dict]]:
    saida = []
    if not PAPERS.is_dir():
        return saida
    for p in sorted(PAPERS.iterdir()):
        if not p.is_dir() or p.name.startswith("_"):
            continue
        arq = p / "AFIRMACOES.yaml"
        if arq.is_file():
            saida.append((p.name, yaml.safe_load(arq.read_text(encoding="utf-8")) or {}))
    return saida


def test_manuscrito_declarado_tem_os_campos_exigidos():
    """Sem estado e sem data, o congelamento não significa nada."""
    for slug, d in declaracoes():
        for campo in ("estado", "data", "afirmacoes"):
            assert campo in d, f"{slug}/AFIRMACOES.yaml não tem '{campo}'"
        assert d["estado"] in ("rascunho", "submetido", "em_revisao", "aceito", "publicado"), (
            f"{slug}: estado {d['estado']!r} não é um dos previstos"
        )


@pytest.mark.parametrize("slug,decl", declaracoes(), ids=lambda x: x if isinstance(x, str) else "")
def test_item_de_preregistro_publicado_nao_mudou(slug, decl):
    """Se o artigo já saiu do rascunho, os valores que ele cita estão travados."""
    if decl.get("estado") == "rascunho":
        pytest.skip(f"{slug} ainda é rascunho; nada congelado")
    selective = yaml.safe_load(CONFIG.read_text(encoding="utf-8")).get("selective", {})
    divergentes = {
        chave: {"publicado": valor, "config": selective.get(chave)}
        for chave, valor in (decl.get("afirmacoes", {}).get("preregistro") or {}).items()
        if selective.get(chave) != valor
    }
    assert not divergentes, (
        f"{slug} está em estado '{decl['estado']}' e cita valores que o config já mudou: "
        f"{divergentes}. Ou o config volta, ou a tese vai contradizer o que está "
        f"publicado — e a contradição aparece na defesa."
    )
