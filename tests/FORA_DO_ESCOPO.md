# Testes do SENTINEL que NÃO vieram para este repositório, e por quê

Este repositório contém só o código que produz os resultados do artigo (40 módulos).
Os testes abaixo protegem módulos do desenho anterior que não entram aqui. Rodá-los
exigiria trazer de volta esse código, que nenhum resultado usa.

| teste | o que protegia | por que não se aplica |
|---|---|---|
| test_import_integrity.py (inteiro) | o mapa de imports do AGENTS.md do SENTINEL | os módulos mapeados não estão aqui |
| test_no_dead_constants.py (inteiro) | src/shared/experiments/scientific_constants.py | o módulo não está aqui |
| test_no_orphan_config.py (inteiro) | toda seção do config.yaml tem consumidor | seções como `experiments` e `logging` só eram lidas pelo pipeline antigo. O config.yaml NÃO foi editado para passar no teste: ele é a declaração da decl-01, e editá-lo mudaria o hash |
| test_run_integrity.py: 2 funções | src/core/pipeline/ner_pipeline.py | o pipeline antigo não está aqui |
| test_doc_code_alignment.py: 1 função | src/shared/analysis/xai_utils.py | o analisador antigo não está aqui |
| test_analysis_metrics.py, test_reproducibility.py, validate_attributes.py, verify_figure_naming.py | módulos do desenho anterior | idem |
