"""
Architecture test: verify config.yaml model/dataset names match ARCHITECTURE.md.

Prevents model list drift and stale field artifacts (code_plan_role).
"""

import yaml
from pathlib import Path
import re

REPO_ROOT = Path(__file__).parent.parent
CONFIG_PATH = REPO_ROOT / "configs" / "config.yaml"
ARCH_PATH = REPO_ROOT / "docs" / "ARCHITECTURE.md"

# O conjunto é EXATO e não mínimo, e a diferença é o ponto: um subconjunto
# permitiria acrescentar modelo sem que ninguém notasse, e era assim que o
# roster de seis modelos convivia com uma tese que precisava de dois.
# DUAS escalas desde 15/09/2026. O conjunto é exato de propósito: modelo que
# aparece aqui sem estar no config, ou no config sem estar aqui, é deriva.
REQUIRED_MODELS = {"gliner-base", "gliner-large",
                   "gliner_base-ft-genia", "gliner_base-ft-conll2003"}

# Dois corpora, e a escolha é o desenho: CoNLL-2003 é plano, GENIA é aninhado, e
# o contraste entre os dois é metade da pergunta única. Os cinco domínios do
# CrossNER serviam à deriva de domínio, que é outra pergunta.
REQUIRED_DATASETS = {"conll2003", "genia"}

# Modelos e corpora que saíram. Nomeados para que a volta quebre a suíte.
MODELOS_APOSENTADOS = {
    "qwen2.5-1.5b-instruct", "qwen2.5-3b-instruct", "t5-base", "bart-base",
    # Aposentados em 02/09/2026 por não encontrarem entidade: são encoders crus,
    # e quem previa era um terceiro modelo. A atenção de um não explica a
    # confiança do outro.
    "bert-large", "deberta-v3-base",
}
CORPORA_APOSENTADOS = {
    "crossner_ai", "crossner_literature", "crossner_music",
    "crossner_politics", "crossner_science",
}


def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def test_config_models_sao_exatamente_os_declarados():
    config = load_config()
    config_models = set(config.get("models", {}).keys())
    assert config_models == REQUIRED_MODELS, (
        f"roster de modelos divergiu: faltam {REQUIRED_MODELS - config_models}, "
        f"sobram {config_models - REQUIRED_MODELS}"
    )


def test_config_datasets_sao_exatamente_os_declarados():
    config = load_config()
    config_datasets = set(config.get("datasets", {}).keys())
    assert config_datasets == REQUIRED_DATASETS, (
        f"corpora divergiram: faltam {REQUIRED_DATASETS - config_datasets}, "
        f"sobram {config_datasets - REQUIRED_DATASETS}"
    )


def test_todo_modelo_do_config_e_encoder_only():
    """O pipeline recusa outra arquitetura, então o config não pode oferecê-la.

    `NERPipeline._create_processor` levanta ValueError para qualquer coisa que
    não seja span-scoring. Um modelo decoder no config seria uma opção que
    falha só na hora de rodar — pior que não existir.
    """
    limites = load_config().get("model_limits", {})
    assert limites, "model_limits vazio"
    fora = {
        m: dados.get("architecture_type")
        for m, dados in limites.items()
        if dados.get("architecture_type") != "span-scoring"
    }
    assert not fora, f"modelos de arquitetura não suportada no config: {fora}"


def test_modelos_e_corpora_aposentados_nao_voltam():
    config = load_config()
    voltaram = (
        (set(config.get("models", {})) & MODELOS_APOSENTADOS)
        | (set(config.get("model_limits", {})) & MODELOS_APOSENTADOS)
        | (set(config.get("model_aliases", {}).values()) & MODELOS_APOSENTADOS)
        | (set(config.get("datasets", {})) & CORPORA_APOSENTADOS)
    )
    assert not voltaram, (
        f"{sorted(voltaram)} voltaram ao config. O contraste entre arquiteturas e a "
        f"deriva de domínio saíram da tese; ver docs/tese/reorg/corte.md."
    )


def test_todo_apelido_aponta_para_modelo_existente():
    config = load_config()
    modelos = set(config.get("models", {}))
    pendurados = {
        apelido: alvo
        for apelido, alvo in config.get("model_aliases", {}).items()
        if alvo not in modelos
    }
    assert not pendurados, f"apelidos apontando para modelo inexistente: {pendurados}"


def test_config_no_code_plan_artifacts():
    with open(CONFIG_PATH) as f:
        content = f.read()

    assert "code_plan_role" not in content, (
        "config.yaml still contains stale 'code_plan_role' fields"
    )
    assert "CODE PLAN" not in content, (
        "config.yaml still contains stale 'CODE PLAN' header"
    )


def test_architecture_md_has_model_table():
    with open(ARCH_PATH) as f:
        content = f.read()

    for model in REQUIRED_MODELS:
        assert model in content, (
            f"docs/ARCHITECTURE.md missing model '{model}'"
        )
