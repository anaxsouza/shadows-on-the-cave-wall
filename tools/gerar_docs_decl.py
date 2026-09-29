"""Gera os documentos congelados da decl-07 e decl-08 A PARTIR da fonte.

Gerado e não digitado: um teste compara documento com fonte e falha se
divergirem, então digitar à mão só adiaria a divergência.
"""
import sys
from pathlib import Path

import yaml

sys.path.insert(0, ".")
from src.selective.preregistration import load_preregistration  # noqa: E402
from src.selective.signals import por_nome  # noqa: E402

INFO = {
    "decl-07-sinais-genia": ("GENIA", "gliner_base-ft-genia",
                             "decl-05-ajustado-genia", "a068f4f0daf00399"),
    "decl-08-sinais-conll": ("CoNLL-2003", "gliner_base-ft-conll2003",
                             "decl-06-ajustado-conll", "4658eceda04c48df"),
    "decl-09-sinais-large": ("GENIA e CoNLL-2003 (uma declaração, dois corpora)",
                             "urchade/gliner_large",
                             "decl-04-escala", "5454128f1df4da6f"),

    # O BRAÇO DECODER, acrescentado em 18/09/2026. Os YAML destas quatro saem de
    # `tools/gerar_decl_decoder.py`, porque diferem apenas em modelo, corpus e
    # número de camadas — quatro arquivos à mão seriam quatro fontes da mesma
    # verdade. Os itens de análise são copiados literalmente de `decl-07`.
    "decl-10-decoder-05b-genia": ("GENIA", "qwen05b-ft-genia",
                                  "decl-07-sinais-genia", "7e83ec650511f2e5"),
    "decl-11-decoder-05b-conll": ("CoNLL-2003", "qwen05b-ft-conll2003",
                                  "decl-07-sinais-genia", "cd343817643081d8"),
    "decl-12-decoder-15b-genia": ("GENIA", "qwen15b-ft-genia",
                                  "decl-07-sinais-genia", "4d8a138acc663343"),
    "decl-13-decoder-15b-conll": ("CoNLL-2003", "qwen15b-ft-conll2003",
                                  "decl-07-sinais-genia", "e02dbb705bca90c1"),
}

# Qual BRAÇO, e isso muda duas seções do documento. Dicionário separado em vez de
# quinto elemento na tupla para não mexer na aritmética das entradas já geradas e
# assinadas — o hash delas tem de continuar reproduzindo.
DECODER = {"decl-10-decoder-05b-genia", "decl-11-decoder-05b-conll",
           "decl-12-decoder-15b-genia", "decl-13-decoder-15b-conll"}

for did, (nome, modelo, origem, medicao) in INFO.items():
    fonte = f"configs/{did}.yaml"
    p = load_preregistration(fonte)
    bruto = yaml.safe_load(Path(fonte).read_text(encoding="utf-8"))["selective"]
    L = [
        (f"# {did} — os sinais internos do DECODER contra a confiança do modelo"
         if did in DECODER else
         f"# {did} — as três famílias de sinal contra a confiança do modelo"),
        "",
        f"**GERADO DE `{fonte}`.** Não editar à mão: um teste compara este documento com a",
        "fonte executável e falha se divergirem.",
        "",
        # O ESTADO DE ASSINATURA é por declaração e não texto fixo. A primeira
        # versão desta função afirmava "ASSINADA em 2026-09-16" em TODAS, e com
        # isso as quatro do braço decoder, escritas em 18/09, sairiam declarando
        # uma assinatura que não existe. Uma declaração que se diz assinada sem
        # estar é falsidade de procedência — o oposto exato do que o instrumento
        # serve para garantir.
        *(["**Estado:** ASSINADA em 2026-09-22 pelo autor, com autorização explícita",
           "(\"assino as declarações, vamos avaliar os modelos decoder agora\").",
           "",
           "**O QUE FOI LIDO, dito com precisão porque a alternativa seria inflar a",
           "procedência:** o autor leu a explicação em linguagem simples do estudo — o que",
           "é atenção, por que o tamanho do trecho a contamina, e o que se mede — e não",
           "declarou ter lido o texto integral desta declaração. A contrapartida que torna",
           "isso defensável é a mesma das anteriores e é verificável: **zero parâmetro",
           "livre**. Todo item de análise é copiado literalmente de `decl-07-sinais-genia.yaml`,",
           "o conjunto de sinais vem do registro em `src/selective/signals.py` fixado em",
           "teste, e as 36 comparações são geradas mecanicamente por",
           "`tools/gerar_decl_decoder.py` sem seleção — não havia o que escolher aqui.",
           "",
           "Gerada em 2026-09-18, quatro dias antes da assinatura. A medição não rodou",
           "nesse intervalo. O treino dos pesos rodou, porque treinar não é medir.",
           "",
           "Como as anteriores, esta tem **zero parâmetro livre**: todo item de análise é",
           "copiado literalmente de `decl-07-sinais-genia.yaml` e as comparações são geradas",
           "mecanicamente de `tools/gerar_decl_decoder.py`, sem seleção — logo não há onde",
           "escolher o sinal favorito depois de ver o resultado."]
          if did in DECODER else
          ["**Estado:** ASSINADA em 2026-09-16, sob autorização explícita do autor concedida",
           "ANTES da leitura deste texto. O registro é honesto sobre isso: ele pré-autorizou, e a",
           "contrapartida é que esta declaração tem **zero parâmetro livre** — não havia o que",
           "escolher, e é isso que a torna defensável mesmo sem leitura prévia."]),
        "",
        f"**Hash da declaração:** `{p.declaration_hash}`",
        (f"**Hash de medição:** `{p.measurement_hash}` — PRÓPRIO: este braço exige medição "
         f"nova, e os itens de ANÁLISE vêm de `{origem}`"
         if did in DECODER else
         f"**Hash de medição:** `{p.measurement_hash}` — IDÊNTICO ao de `{origem}` (`{medicao}`)"),
        f"**Modelo:** `{modelo}` · **Corpus:** {nome}",
        "",
        "---",
        "",
        # ESTA SEÇÃO TAMBÉM É POR BRAÇO, e deixá-la incondicional produziu uma
        # CONTRADIÇÃO dentro do documento: a linha do hash acima já dizia
        # "PRÓPRIO" nas do decoder enquanto o título aqui afirmava "é o mesmo".
        # Foi apanhado em revisão, e é o mesmo defeito do texto de assinatura —
        # prosa escrita verdadeira para o caso de origem, mentindo no caso novo.
        # Duas ocorrências do MESMO erro no mesmo arquivo: todo trecho de prosa
        # deste gerador que afirma algo sobre a declaração é suspeito até ser
        # conferido contra o braço.
        *(["## Por que o hash de medição é PRÓPRIO, e por que isso importa",
           "",
           "Ao contrário das declarações do braço encoder, esta NÃO herda o hash de medição",
           "de nenhuma anterior, e não poderia: o modelo é outro, a arquitetura é outra, e a",
           "faixa de camadas cobre todas as deste modelo. Um hash herdado afirmaria que a",
           "tabela pode ser conferida contra colunas já medidas, e aqui não há colunas já",
           "medidas — este braço exige medição nova, e é isso que o hash próprio declara.",
           "",
           "O que É herdado, e apenas isso, são os itens de ANÁLISE: grades, critérios,",
           "fração de calibração, número de reamostragens, semente. Eles vêm literalmente da",
           "`decl-07`, e é essa herança que torna os dois braços comparáveis sem que a",
           "comparação dependa de uma escolha feita depois de ver resultado."]
          if did in DECODER else
          ["## Por que o hash de medição é o mesmo, e por que isso importa",
           "",
           "Os estados ocultos são medidos na **mesma faixa de camadas já declarada**. Não é",
           "conveniência: se a família de estados ocultos tivesse faixa própria, ela seria um",
           "parâmetro de medição novo, o hash mudaria, as tabelas já medidas deixariam de ser",
           "comparáveis, e escolher a faixa depois de ver o resultado voltaria a ser uma linha de",
           "código. Com o hash igual, a tabela é remedida sob a MESMA identidade e as colunas",
           "antigas têm de reproduzir exatamente — o que é, por si, uma verificação forte."]),
        "",
        "## Zero parâmetro livre — o que foi copiado e o que foi derivado",
        "",
        "| origem | o que veio de lá |",
        "|---|---|",
        (f"| `{origem}` | todo item de ANÁLISE, literalmente. Medição NÃO: é própria |"
         if did in DECODER else
         f"| `{origem}` | todo item de análise e de medição, literalmente |"),
        "| `src/selective/signals.py` | o conjunto de sinais e o ESTATUTO do nulo de cada um |",
        "| `CRITERIO_DO_TIPO` | o critério, fixo por tipo de veredito |",
        "| regra mecânica | um veredito por sinal, contra `model_confidence`, sem seleção |",
        "",
        "A ausência de seleção é o que substitui aqui a correção de multiplicidade: não há",
        "escolha de sinal favorito depois de ver o resultado porque **todos** entram.",
        "",
        "## Os sinais, com o estatuto do nulo de cada um",
        "",
        "| sinal | família | nulo | o que isso permite afirmar |",
        "|---|---|---|---|",
    ]
    for s_ in p.comparison_scores:
        try:
            spec = por_nome(s_)
        except Exception:
            L.append(f"| `{s_}` | geométrico | — | adversário já declarado nas anteriores |")
            continue
        forca = ("afirmação sobre o MODELO: a esperança sai da álgebra do orçamento fixo"
                 if spec.estatuto == "exato" else
                 "afirmação mais FRACA: forma suposta linear, coeficientes ajustados no mesmo dado")
        L.append(f"| `{s_}` | {spec.familia} | **{spec.estatuto}** | {forca} |")
    L += ["", "## A verificação de identidade embutida", ""]
    if did in DECODER:
        L += [
            "**Este braço NÃO TEM a verificação que o encoder tem, e a ausência é declarada.**",
            "No encoder, `logit_max` é por construção a mesma quantidade que",
            "`model_confidence`, e o veredito dele contra a confiança tem de dar delta",
            "exatamente zero — um teste de encanamento cujo valor esperado se conhece. Num",
            "extrator generativo o rótulo é TEXTO que o modelo escreve, então não existe vetor",
            "fixo de escores de rótulo, a família de logits não existe, e com ela se perde o",
            "teste. Dizer isso é melhor que omitir: este braço mede sem aquela rede.",
            "",
            "O que há em lugar dela, e é mais fraco por ser aditivo e não estrutural: a média",
            "das features de atenção POR CABEÇA tem de reproduzir o escalar agregado declarado.",
            "Medido em 18/09/2026 no 0,5B ajustado — correlação 1,0000 e erro absoluto médio",
            "0,00000 — e verificado também que os escalares saem bit-a-bit idênticos com e sem",
            "a extração de features, o que é a garantia de que a exploração não move o veredito.",
            "",
        ]
    else:
        L += [
            "`logit_max` é, por construção, o maior escore de rótulo do trecho — a **mesma**",
            "quantidade que `model_confidence`. O veredito `S9` dele contra a confiança tem de dar",
            "delta **exatamente zero**. Se não der, o encanamento está errado, e é melhor descobrir",
            "isso por um sinal cujo valor esperado se conhece do que por um cujo valor esperado se",
            "está medindo. A orientação dos dois é fixada por definição justamente para isso.",
            "",
        ]
    L += [
        "## A orientação do escore, e por que ela não é grau de liberdade",
        "",
        # O EXEMPLO da orientação também é por braço: citar a entropia dos
        # escores de rótulo numa declaração do decoder ilustraria a regra com um
        # sinal que ali não existe. Terceira ocorrência do mesmo erro neste
        # arquivo, e a razão de eu ter varrido a prosa inteira depois das duas
        # primeiras em vez de corrigir só o que a revisão apontou.
        "A curva entrega por escore decrescente. Para a confiança, alto significa entregar; para",
        ("a entropia das linhas causais, alto significa o contrário. Testar as duas orientações"
         if did in DECODER else
         "a entropia dos escores de rótulo, alto significa o contrário. Testar as duas orientações"),
        "e relatar a melhor dobraria as chances ao acaso.",
        "",
        "A orientação é decidida pelo **sinal da correlação entre o escore e o acerto na partição",
        "de CALIBRAÇÃO**, nunca na de avaliação — a mesma disciplina que o peso convexo já usa.",
        "É **invariante do protocolo** e não item por declaração: vale para todo sinal, em toda",
        "declaração, como a separação por sentença.",
        "",
        "## O que esta declaração NÃO pode afirmar",
        "",
        "Se um sinal de nulo **empírico** bater a confiança, a afirmação é mais fraca que a de um",
        "sinal de nulo **exato**, e a ressalva viaja com o número em toda tabela e todo texto: a",
        "forma da relação é suposta linear e os coeficientes são ajustados no mesmo dado em que o",
        "resíduo é avaliado. Um nulo exato não tem nenhuma das duas fraquezas.",
        "",
        "## A fonte, congelada",
        "",
        "```yaml",
        yaml.safe_dump({"selective": bruto}, sort_keys=False, allow_unicode=True).rstrip(),
        "```",
    ]
    Path(f"docs/tese/declaracoes/{did}.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"{did}.md: {len(L)} linhas | decl {p.declaration_hash} | medicao {p.measurement_hash}")
