# Adenda 01 às declarações decl-10 a decl-13 — linha de base TREINADA

**Data:** 2026-09-23.
**Decidida por:** o autor, antes de qualquer execução da análise declarada.
**Estado da análise no momento desta adenda:** NÃO executada. Nenhum veredito,
nenhuma AURC e nenhuma carga de revisão de decl-10 a decl-13 foi calculada. O que
já foi visto, e é dito aqui para não ser omitido: AUC exploratória das sondas e de
`model_confidence` nos dois pontos de 0,5B (registrada em
`auc_05b_dois_corpora.csv`), na qual as sondas superam `model_confidence`. É essa
observação que motiva a adenda — e é por ela ter sido vista que a adenda precisa
estar datada e fixada antes do veredito.

## Por que existe

As sondas (`sonda_ocultos`, `sonda_atencao_cabecas`) são AJUSTADAS na partição de
calibração. `model_confidence` não é ajustada em nada. Uma vantagem da sonda sobre
`model_confidence` confunde duas coisas: informação no interior do modelo, e o
simples fato de ter sido treinada. Esta adenda acrescenta o comparador que separa
as duas: uma linha de base treinada da MESMA forma, na MESMA partição, mas que só
vê o que está FORA do modelo.

## O que se acrescenta — tudo fixado aqui, nada escolhido depois

**Escore novo:** `base_treinada`.

- **Entradas:** exatamente quatro colunas de `entities.csv`, nenhuma outra:
  `model_confidence`, `span_size`, `n_tokens`, `span_position`. As três últimas são
  as covariáveis declaradas em `src/selective/signals.py::COVARIAVEIS`
  (`n_tokens` é o nome, neste braço, de `sentence_length`).
- **Modelo:** regressão logística, entradas padronizadas, alvo = acerto
  (`1 - loss`), escore = probabilidade de acerto (alto = entregar, a mesma
  orientação de `model_confidence` e das sondas).
- **Partição:** a de `_separar_por_sentenca` do projeto, a mesma da análise e das
  sondas, com `calibration_fraction` e `seed` da declaração de cada ponto.
- **Duas versões, pareadas com as duas versões da sonda:**
  `base_treinada` com C = 1,0 fixo (par de `sonda_*`), e
  `base_treinada_ajustada` com C por validação cruzada na calibração, grade
  {0,001; 0,01; 0,1; 1; 10; 100}, a mesma das sondas (par de `sonda_*_ajustada`).

**Comparações novas, por ponto:** cada uma das quatro colunas de sonda contra a
`base_treinada` do MESMO regime de regularização, pelos DOIS critérios já
declarados — `paired_delta_aurc_ci_excludes_zero` e
`paired_review_load_ci_excludes_zero` — com as grades, a cobertura, o bootstrap e o
nível de confiança copiados literalmente da declaração do ponto. São 4 × 2 = 8
comparações por ponto, 32 no eixo.

## O que NÃO muda

- As 36 comparações declaradas de cada ponto rodam exatamente como assinadas.
- Os hashes de decl-10 a decl-13 não mudam: esta adenda é documento separado.
- As comparações novas são **exploratórias**, como as sondas: não alteram veredito
  confirmatório nenhum.
- A contagem por sorteio esperado, a 95%, entra ao lado de cada contagem, como na
  análise declarada: 0,025 × 8 = 0,20 por ponto e por direção.

## Leitura fixada antes do resultado

Se a sonda vence `base_treinada`: o ganho não é explicável por treino sobre o que
está fora do modelo. Se não vence: a vantagem sobre `model_confidence` pode ser
atribuída ao treino, e a leitura "o interior do modelo guarda informação" não se
sustenta por este teste.
