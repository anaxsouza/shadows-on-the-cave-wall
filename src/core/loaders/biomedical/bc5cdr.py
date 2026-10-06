"""Carregador do BC5CDR (Li et al., 2016), versão em sentenças de `tner/bc5cdr`.

Corpus declarado em `decl-16-previsao-bc5cdr`: revisão `f68cdc7db924369241e7868656f583072acd4e90`,
separação oficial (treino 5.227, validação 5.329, teste 5.864 sentenças), rótulos `Chemical` e
`Disease`, anotação PLANA (BIO).

LEITURA LOCAL, E A RAZÃO. Os quatro arquivos baixados têm SHA-256 declarado na decl-16, e o texto do
corpus não é redistribuído (`dados_bc5cdr/` está no .gitignore). Por isso este carregador não vai ao
Hub: lê `dados_bc5cdr/{train,valid,test}.json` (um JSON por linha, `{tokens, tags}`) e CONFERE o
SHA-256 de cada arquivo contra o declarado antes de devolver qualquer sentença. Um arquivo diferente
do declarado falha alto, em vez de medir outro corpus com o nome deste.

A FORMA É A DOS OUTROS: `NERExample` com `text = ' '.join(tokens)` e entidades em posição de
CARACTERE nesse texto, extraídas por `extract_entities_from_bio` — a mesma função do CoNLL.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from datasets import Dataset

from ....shared.data.bio_extraction import extract_entities_from_bio
from ....shared.data.data_types import NERExample
from ....shared.data.nesting_utils import detect_nesting
from ....shared.model.config_manager import get_config_manager
from ..base.loader import BaseDatasetLoader, DatasetLoadingError

logger = logging.getLogger(__name__)

# SHA-256 dos arquivos baixados, copiados da decl-16 (assinada em 2026-10-01).
SHA256_DECLARADO = {
    "test.json": "20a04588b1a67c65203df698709a4384e15dac1597aa1cfb52365e7d0d6708b5",
    "train.json": "62fe247960ff1b4270cb71500fb99748aab371c2dce0ef5b048fc44cb0f82ec8",
    "valid.json": "d74d511b8708c645c51e98bfd3f191d54ec83a601b3c5dde84a25ad4363f7d8e",
    "label.json": "d1d6998c78bc510b526538212c3b8eee97f1dd45ded428fc1965139c368ed59f",
}
ARQUIVO_DA_PARTICAO = {"train": "train.json", "validation": "valid.json",
                       "val": "valid.json", "valid": "valid.json", "test": "test.json"}
# CONTAGEM REAL DOS ARQUIVOS DECLARADOS: 5.228 / 5.330 / 5.865 sentencas. A decl-16 diz 5.227 /
# 5.329 / 5.864, que sao as contagens de QUEBRAS DE LINHA (`wc -l`): o arquivo nao termina em
# newline, entao a ultima sentenca nao e contada. O SHA-256 declarado confere, logo o arquivo e o
# declarado; so o numero na prosa da declaracao esta uma unidade abaixo. A declaracao assinada nao
# e editada; a divergencia fica registrada aqui e no relatorio.
SENTENCAS_ESPERADAS = {"train": 5228, "validation": 5330, "test": 5865}


def _raiz_padrao() -> Path:
    env = os.environ.get("SENTINEL_BC5CDR")
    if env:
        return Path(env)
    cfg = get_config_manager().get_dataset_config("bc5cdr")
    local = Path(cfg.get("local_dir", "dados_bc5cdr"))
    if local.is_absolute():
        return local
    # relativo à raiz do repositório (src/core/loaders/biomedical/ -> 4 níveis acima)
    return Path(__file__).resolve().parents[4] / local


class BC5CDRLoader(BaseDatasetLoader):
    """BC5CDR plano, com `Chemical` e `Disease`."""

    ENTITY_TYPES = ["Chemical", "Disease"]

    def __init__(self, raiz: Optional[Path] = None, conferir_hash: bool = True):
        super().__init__()
        self._raiz = Path(raiz) if raiz else _raiz_padrao()
        self._conferir_hash = conferir_hash

    def get_dataset_name(self) -> str:
        return "bc5cdr"

    def get_huggingface_name(self) -> str:
        return get_config_manager().get_dataset_config("bc5cdr")["name"]

    def get_huggingface_config(self) -> Optional[str]:
        return None

    def get_huggingface_revision(self) -> Optional[str]:
        return get_config_manager().get_dataset_config("bc5cdr").get("revision")

    def get_entity_types(self) -> List[str]:
        return list(self.ENTITY_TYPES)

    # -- leitura local, com o hash declarado conferido --------------------------------
    def _ler(self, nome: str) -> bytes:
        caminho = self._raiz / nome
        if not caminho.is_file():
            raise DatasetLoadingError(
                f"{caminho} ausente. O BC5CDR não é redistribuído; baixe tner/bc5cdr na "
                f"revisão declarada na decl-16 e confira o SHA-256.")
        dados = caminho.read_bytes()
        if self._conferir_hash:
            h = hashlib.sha256(dados).hexdigest()
            if h != SHA256_DECLARADO[nome]:
                raise DatasetLoadingError(
                    f"{nome}: SHA-256 {h} != declarado {SHA256_DECLARADO[nome]} (decl-16). "
                    f"Medir assim seria medir outro corpus com o nome do declarado.")
        return dados

    def load_raw_dataset(self, split: str = "test") -> Dataset:
        if split not in ARQUIVO_DA_PARTICAO:
            raise ValueError(f"partição {split!r} desconhecida; use {sorted(ARQUIVO_DA_PARTICAO)}")
        chave = f"bc5cdr_{ARQUIVO_DA_PARTICAO[split]}"
        if chave in self._loaded_splits:
            return self._loaded_splits[chave]
        linhas = [json.loads(l) for l in self._ler(ARQUIVO_DA_PARTICAO[split]).decode("utf-8").splitlines() if l.strip()]
        ds = Dataset.from_dict({"tokens": [l["tokens"] for l in linhas],
                                "tags": [l["tags"] for l in linhas]})
        esperado = SENTENCAS_ESPERADAS["validation" if split in ("val", "valid") else split]
        if len(ds) != esperado:
            raise DatasetLoadingError(f"BC5CDR {split}: {len(ds)} sentencas, esperadas {esperado}")
        self._loaded_splits[chave] = ds
        return ds

    def extract_label_info(self, dataset: Dataset) -> None:
        mapa: Dict[str, int] = json.loads(self._ler("label.json").decode("utf-8"))
        self.label2id = dict(mapa)
        self.id2label = {i: l for l, i in mapa.items()}
        self.label_list = [self.id2label[i] for i in sorted(self.id2label)]

    def convert_to_examples(self, dataset: Dataset, max_examples: Optional[int] = None) -> List[NERExample]:
        self.extract_label_info(dataset)
        sub = dataset if not max_examples else dataset.select(range(min(max_examples, len(dataset))))
        exemplos: List[NERExample] = []
        for i, item in enumerate(sub):
            ex = self._converter(item, i)
            if ex is not None and self.validate_example(ex):
                exemplos.append(ex)
            else:
                logger.warning("BC5CDR: exemplo %d descartado", i)
        return exemplos

    def _converter(self, item: Dict[str, Any], i: int) -> Optional[NERExample]:
        tokens, tags = item["tokens"], item["tags"]
        if not tokens or len(tokens) != len(tags):
            return None
        rotulos = [self.id2label[int(t)] for t in tags]
        ents = detect_nesting(extract_entities_from_bio(tokens, rotulos))
        return NERExample(id=str(i), text=" ".join(tokens), tokens=tokens, labels=rotulos, entities=ents)
