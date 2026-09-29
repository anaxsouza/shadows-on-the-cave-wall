"""Emite os YAML de decl-10 a decl-13 — o braço decoder — de UMA especificação.

POR QUE GERADO, E NÃO QUATRO ARQUIVOS ESCRITOS À MÃO

As quatro declarações diferem em exatamente três coisas: o modelo, o corpus e o
número de camadas. Todo o resto — regra de checkpoint, grades, critérios, o
conjunto de comparações — é idêntico por construção, porque um eixo de escala
cujos dois pontos foram declarados com critérios diferentes não é um eixo.

Escrever quatro arquivos criaria quatro fontes da mesma verdade, e elas
divergiriam na primeira correção. É o mesmo defeito que o vocabulário de rótulo
já custou a este projeto em 16/09/2026, e que `src/cli/commands/selective.py`
condena por escrito: duplicar cria duas fontes para uma entrada, "que é
exatamente o tipo de divergência que não aparece como erro".

A FONTE DA VERDADE passa a ser este arquivo, e os YAML são saída dele. O
`.md` continua sendo gerado do YAML por `gerar_docs_decl.py`, então a cadeia é
spec -> yaml -> md, com teste em cada elo.

ZERO PARÂMETRO LIVRE. Todo item de análise é copiado LITERALMENTE de
`decl-07-sinais-genia.yaml`, que é a declaração do braço encoder para extrator
ajustado — o arranjo mais próximo. Nada foi escolhido aqui, e o que MUDA em
relação a ela está enumerado em `O_QUE_MUDA` com a razão de cada mudança.
"""
from __future__ import annotations

from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent

# Os quatro pontos: (id, corpus, modelo, camadas). As camadas são item de
# MEDIÇÃO e diferem por arquitetura — 24 no 0,5B e 28 no 1,5B, lidos do
# config.json de cada um e não supostos. Declarar [0..23] num modelo de 28
# camadas mediria um subconjunto sem dizer qual.
PONTOS = [
    ("decl-10-decoder-05b-genia", "genia", "qwen05b-ft-genia", 24),
    ("decl-11-decoder-05b-conll", "conll2003", "qwen05b-ft-conll2003", 24),
    ("decl-12-decoder-15b-genia", "genia", "qwen15b-ft-genia", 28),
    ("decl-13-decoder-15b-conll", "conll2003", "qwen15b-ft-conll2003", 28),
]

# Os sinais CONFIRMATÓRIOS do braço decoder. A família de LOGITS não está aqui e
# a ausência é declarada, não esquecida: num extrator generativo o rótulo é texto
# que o modelo escreve, então não existe vetor fixo de escores de rótulo sobre o
# qual calcular margem, entropia ou máximo. A consequência colateral, que também
# se declara: a verificação de identidade `logit_max == model_confidence`, que o
# braço encoder usa como teste de encanamento, não tem análogo aqui.
SINAIS = ["span_mass", "row_entropy_causal", "row_max_causal",
          "hidden_norm", "hidden_dist_centroide", "hidden_delta_camadas"]

# As SONDAS, exploratórias. Separadas dos confirmatórios na própria estrutura, e
# não só por rótulo em comentário: elas entram com `kind` marcado e a apuração
# tem de contá-las à parte.
SONDAS = ["sonda_ocultos", "sonda_atencao_cabecas"]

# As DUAS linhas de base. `model_confidence` é a declarada desde a decl-02;
# `aggseq` entra por decisão do autor em 18/09/2026, e muda o que a conclusão
# pode dizer: de "os sinais internos não superam a confiança do modelo" para
# "não superam nem a confiança nem o melhor estimador publicado".
BASES = ["model_confidence", "aggseq"]

O_QUE_MUDA = """\
  # O QUE MUDA EM RELAÇÃO À decl-07, E A RAZÃO DE CADA MUDANÇA
  #
  # 1. `sink_policy: keep`. No encoder é `drop_from_denominator`. Aqui a máscara
  #    causal já restringe o denominador de cada linha às chaves permitidas, e o
  #    primeiro token é o sorvedouro de atenção por construção da arquitetura.
  #    Descontá-lo de novo mudaria a derivação do nulo EXATO, que é o que dá à
  #    família de atenção a imunidade a escala. O valor vem do que o medidor
  #    calcula (`expected_mass(k, T, sink_policy='keep')`), não de preferência.
  #
  # 2. `layers` cobre TODAS as camadas do modelo deste ponto — 24 no 0,5B, 28 no
  #    1,5B, lidas do config.json. No encoder eram 12.
  #
  # 3. A família de LOGITS sai, e a omissão é justificada: num extrator
  #    generativo não há vetor fixo de escores de rótulo. Perde-se com ela a
  #    verificação de identidade `logit_max == model_confidence`.
  #
  # 4. Entram `row_entropy_causal` e `row_max_causal` em lugar de `row_entropy` e
  #    `row_max`. NÃO é renomeação: sob máscara, o teto de cada linha depende da
  #    LARGURA dela, e a esperança do conjunto é a média de `log(i+1)` e de
  #    `1/(i+1)` sobre as linhas do trecho. Comparar contra `log|K|`, que é o teto
  #    bidirecional, seria comparar contra o nulo errado.
  #
  # 5. Entra a segunda linha de base `aggseq`, e com ela cada veredito ganha um
  #    par. A conta de multiplicidade dobra e isso é DECLARADO: com N comparações
  #    a 95%, esperam-se 0,025*N exclusões de zero por direção só por sorteio, e
  #    a apuração tem de reportar N ao lado de cada contagem.
  #
  # 6. Entram as SONDAS supervisionadas, em compartimento exploratório. Elas
  #    abandonam o nulo exato da família de atenção — pesos ajustados — e por
  #    isso ACOMPANHAM os escalares em vez de substituí-los.
"""


def comparacoes(corpus: str) -> list[str]:
    """As comparações, geradas mecanicamente. Sem seleção, logo sem favorito.

    A ausência de seleção é o que substitui aqui a correção de multiplicidade,
    pela mesma construção da decl-07: não há como escolher o sinal que deu certo
    depois de ver o resultado, porque TODOS entram, contra as DUAS bases, nos
    DOIS critérios.
    """
    L: list[str] = []

    L.append("""\
    - id: 'D1'
      kind: 'descriptive'
      question: >-
        A massa de atenção causal acompanha qual nulo? Relata R² contra
        `expected_causal`, que depende de tamanho, comprimento E POSIÇÃO, e
        contra `expected_bidir`, que depende só dos dois primeiros — na MESMA
        linha, para que "a posição acrescenta dimensão ao confundidor?" seja
        medida e não argumentada. É a pergunta própria deste braço.
""")
    L.append("""\
    - id: 'D2'
      kind: 'descriptive'
      question: >-
        Em cada meta de qualidade, qual a carga de revisão exigida por cada
        supervisor, e qual precisão, recall e F1 saem entre os entregues?
        Inclui a linha SEM supervisor como referência e declara "inalcançável"
        como resultado possível.
""")
    L.append("""\
    - id: 'C2'
      kind: 'verdict'
      question: 'A massa de atenção causal acrescenta sobre a geometria pura?'
      score: 'span_mass'
      against: 'geometric_fraction'
      criterion: 'paired_delta_aurc_ci_excludes_zero'
""")
    L.append("""\
    - id: 'C3'
      kind: 'verdict'
      question: 'O instrumento desconfundido acrescenta sobre a confiança do modelo?'
      score: 'model_confidence+enrichment'
      against: 'model_confidence'
      criterion: 'paired_delta_aurc_ci_excludes_zero'
""")

    n = 0
    for grupo, sinais, nota in (("confirmatório", SINAIS, ""),
                                ("EXPLORATÓRIO", SONDAS,
                                 " Compartimento exploratório: a sonda tem pesos "
                                 "ajustados na calibração e ABANDONA o nulo exato "
                                 "da família, logo não decide veredito sozinha.")):
        for s in sinais:
            for base in BASES:
                for kind, crit, unidade in (
                        ("verdict", "paired_delta_aurc_ci_excludes_zero", "AURC"),
                        ("task_verdict", "paired_review_load_ci_excludes_zero",
                         "carga de revisão")):
                    n += 1
                    pref = "S" if grupo == "confirmatório" else "E"
                    L.append(f"""\
    - id: '{pref}{n}'
      kind: '{kind}'
      question: '`{s}` acrescenta sobre `{base}` em {unidade}?{nota}'
      score: '{s}'
      against: '{base}'
      criterion: '{crit}'
""")
    return L


def yaml_de(did: str, corpus: str, modelo: str, camadas: int) -> str:
    nome_corpus = {"genia": "GENIA", "conll2003": "CoNLL-2003"}[corpus]
    comps = comparacoes(corpus)
    n_ver = sum(1 for c in comps if "criterion:" in c)
    cabeca = f"""\
# =============================================================================
# {did} — as famílias de sinal do DECODER contra a confiança do modelo
# =============================================================================
# GERADO POR `tools/gerar_decl_decoder.py`. Não editar à mão: as quatro
# declarações do braço decoder saem de UMA especificação, porque um eixo de
# escala cujos pontos foram declarados com critérios diferentes não é um eixo.
#
# Modelo: {modelo} · Corpus: {nome_corpus} · Camadas: {camadas}
#
# Regime de treino deste ponto: precisão mista bf16 com pesos-mestres em fp32 e
# momentos de Adam em 8 bits, em ml.g5.2xlarge. O regime é IGUAL nos dois pontos
# do eixo — inclusive o 0,5B foi REFEITO nele — porque dois ajustes sob regimes
# diferentes não são pontos comparáveis. A seleção de checkpoint e a medição
# rodam em fp32 sempre, para que a regra de seleção olhe o mesmo modelo que
# responde à hipótese.
#
# {n_ver} vereditos nesta declaração. Com N comparações a 95%, esperam-se
# 0,025*N exclusões de zero POR DIREÇÃO só por sorteio: {0.025 * n_ver:.2f} aqui.
# A apuração tem de reportar esse número ao lado de cada contagem, e contar por
# GRADE — as grades absoluta e do vão são réplicas sobre as MESMAS entidades, e
# empilhá-las infla o N. Foi o erro que a apuração de 16/09/2026 apanhou.

selective:
  declaration_id: '{did}'
  hash_version: 5

  # O modelo entra no hash de MEDIÇÃO: trocá-lo muda os números DENTRO da tabela.
  model: '{modelo}'

  # ---------------------------------------------------------------------------
  # MEDIÇÃO
  # ---------------------------------------------------------------------------
  sink_policy: 'keep'
  layers: [{", ".join(str(i) for i in range(camadas))}]
  heads: null

{O_QUE_MUDA}
  # ---------------------------------------------------------------------------
  # ANÁLISE — copiados LITERALMENTE de decl-07-sinais-genia.yaml
  # ---------------------------------------------------------------------------
  combination_rule: 'convex'
  calibration_fraction: 0.30
  coverage_levels: [0.5, 0.7, 0.8, 0.9, 0.95]
  operating_point: 'derived_from_target_risk'
  target_risk_grid: [0.01, 0.02, 0.05, 0.10]
  added_value_ci_level: 0.95
  n_bootstrap_resamples: 2000
  conformal_alpha: 0.05
  loss_bound: 1.0
  seed: 42
  comparison_scores: [span_size, geometric_fraction, sentence_length,
                      row_entropy_causal, row_max_causal,
                      hidden_norm, hidden_dist_centroide, hidden_delta_camadas,
                      sonda_ocultos, sonda_atencao_cabecas]
  stratify_by: [span_size, is_nested]
  span_size_bins: [[1, 1], [2, 2], [3, 4], [5, null]]
  task_gap_grid: [0.25, 0.50, 0.75, 0.90]
  task_quality_grid: [0.70, 0.80, 0.90, 0.95]
  task_metrics:
    - precision_delivered
    - recall_delivered
    - f1_delivered
    - review_load

  # EXPLORATÓRIO, como na decl-07: descreve sem decidir veredito.
  layer_profile: true

  # ---------------------------------------------------------------------------
  # AS COMPARAÇÕES — geradas mecanicamente, sem seleção
  # ---------------------------------------------------------------------------
  comparisons:
"""
    return cabeca + "\n".join(comps)


def main() -> None:
    for did, corpus, modelo, camadas in PONTOS:
        alvo = RAIZ / "configs" / f"{did}.yaml"
        alvo.write_text(yaml_de(did, corpus, modelo, camadas), encoding="utf-8")
        n = sum(1 for c in comparacoes(corpus) if "criterion:" in c)
        print(f"escrito {alvo.relative_to(RAIZ)}  | {modelo} | {camadas} camadas | "
              f"{n} vereditos | sorteio esperado {0.025 * n:.2f}/direcao")


if __name__ == "__main__":
    main()
