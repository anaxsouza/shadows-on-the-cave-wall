# Declaração `decl-01-gliner-base`

**Hash do conteúdo:** `348e90cce8ca9874`
**Protocolo:** `docs/tese/PROTOCOLO.md`
**Estado:** ASSINADA em 2026-09-02. Os valores abaixo são compromisso: não mudam depois de
ver resultado. Uma medição sob outros valores exige uma declaração NOVA, com outro
`declaration_id` e outro hash — não a edição desta.
**Modelo:** `gliner-base` (`urchade/gliner_base`, 12 camadas, 12 cabeças)
**Corpora:** `conll2003` (plano) e `genia` (aninhado)

Cópia congelada, GERADA da seção `selective` de `configs/config.yaml` — nunca digitada,
porque cópia digitada divergiria da fonte sem que nada acusasse.

```yaml
declaration_id: decl-01-gliner-base
sink_policy: drop_from_denominator
layers:
- 0
- 1
- 2
- 3
- 4
- 5
- 6
- 7
- 8
- 9
- 10
- 11
layer_profile: true
heads: null
combination_rule: convex
calibration_fraction: 0.3
coverage_levels:
- 0.5
- 0.7
- 0.8
- 0.9
- 0.95
operating_point: derived_from_target_risk
added_value_ci_level: 0.95
n_bootstrap_resamples: 2000
target_risk_grid:
- 0.01
- 0.02
- 0.05
- 0.1
conformal_alpha: 0.05
loss_bound: 1.0
seed: 42
```

## Assinatura

Assinar é apagar a linha PROPOSTA acima e no `PREREGISTRO.md`, e commitar. A data desse
commit tem de ser anterior à data em que a medição rodar: é o commit que estabelece a
ordem, e nenhuma frase escrita depois a recupera.

| | |
|---|---|
| Assinada em | 2026-09-02 |
| Commit da assinatura | o commit que introduz esta alteração — `git log --diff-filter=M -1 -- docs/tese/declaracoes/decl-01-gliner-base.md` |
| Primeira medição sob ela | posterior a este commit, por construção |
