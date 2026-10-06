"""Carregador do BC5CDR (decl-16): contagens, rótulos e posição de caractere. Pula sem o corpus."""
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parent.parent
pytestmark = pytest.mark.skipif(not (RAIZ / "dados_bc5cdr" / "test.json").exists(),
                                reason="dados_bc5cdr/ não está presente (o texto não é redistribuído)")


def test_contagens_rotulos_e_caracteres():
    from src.core.loaders.biomedical.bc5cdr import BC5CDRLoader
    ex = BC5CDRLoader().load_split("test")
    assert len(ex) == 5865          # a decl-16 diz 5.864: contagem de quebras de linha (ver o módulo)
    assert {e.label for x in ex for e in x.entities} == {"Chemical", "Disease"}
    assert sum(len(x.entities) for x in ex) == 9809
    for x in ex[:200]:
        assert x.text == " ".join(x.tokens)
        for e in x.entities:
            assert x.text[e.start:e.end] == e.text


def test_hash_divergente_falha_alto(tmp_path):
    from src.core.loaders.base.loader import DatasetLoadingError
    from src.core.loaders.biomedical.bc5cdr import BC5CDRLoader
    (tmp_path / "test.json").write_text('{"tags": [0], "tokens": ["x"]}\n')
    with pytest.raises(DatasetLoadingError):
        BC5CDRLoader(raiz=tmp_path).load_raw_dataset("test")
