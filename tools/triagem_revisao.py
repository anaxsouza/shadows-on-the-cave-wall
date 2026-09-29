"""A REGRA DE TRIAGEM da revisão sistemática, como ARQUIVO e não estado de kernel.

POR QUE ESTE ARQUIVO EXISTE, e é uma correção de defeito meu

A triagem de 17/09/2026 rodou como expressões regulares num kernel, e nada disso
foi salvo como código. Quando o kernel reiniciou, a regra SE PERDEU — e uma
revisão sistemática cujo instrumento de triagem não existe mais não é
reproduzível por terceiros, que é a única propriedade que a distingue de uma
leitura seletiva. O protocolo foi salvo; o instrumento que o aplica, não.

RECONSTRUÇÃO, E A DIVERGÊNCIA DECLARADA

Este arquivo é uma RECONSTRUÇÃO da regra emendada, e ela reproduz 97,95% das 977
decisões guardadas em `triagem_completa.csv` — 20 divergências. NÃO é a regra
original, e dizer isso importa: qualquer registro triado por este arquivo foi
triado por um instrumento ligeiramente diferente do que triou os 977. A
alternativa — apresentá-la como idêntica — afirmaria uma reprodutibilidade que
não se verificou.

A partir daqui a regra é esta, e é conferível: quem quiser repetir a triagem roda
`valida_contra(triagem_completa.csv)` e vê o mesmo 0,9795.
"""
from __future__ import annotations

import re

SINAL = re.compile(r"attention|hidden state|hidden representation|\bprobe|logit|softmax|"
    r"token[- ]level probab|perplexit|entropy|margin|confidence|uncertaint|calibrat|"
    r"verbali[sz]|self[- ]consistency|semantic entropy|likelihood|p\(true\)|"
    r"self[- ]evaluat|self[- ]assess|self[- ]knowledge", re.I)
USO = re.compile(r"abstain|abstention|selective predict|selective classif|reject option|"
    r"defer|routing|route to|filter|flag|hallucination detect|error detect|"
    r"risk[- ]coverage|human review|human-in-the-loop|escalat|misclassification detect|"
    r"failure predict|mispredict|correctness predict|predict(ing)? correctness|"
    r"confidence estimation|confidence score|quality estimation|"
    r"know what .{0,20}(don't|do not) know|when to trust|trustworth", re.I)
NER = re.compile(r"named entity|\bNER\b|sequence label|entity extraction|entity recognition", re.I)
DEC = re.compile(r"decoder[- ]only|autoregressive|generative|GPT|LLaMA|Qwen|Mistral|"
    r"instruction[- ]tun|seq2seq|encoder[- ]decoder|T5|BART|large language model|\bLLM", re.I)
SURVEY = re.compile(r"\bsurvey\b|\breview\b|position paper|\boverview\b|systematic review|\btutorial\b", re.I)
VISAO = re.compile(r"\bimage\b|\bvisual\b|\bspeech\b|\baudio\b|\bvideo\b|captioning|\bASR\b", re.I)

# O componente de USO é exigido em SEPARADO do de SINAL, e a razão é a emenda 2 do
# protocolo: sem isso, uma frase que só diz "confidence" satisfazia o critério (i)
# sozinha, e a tela incluía trabalho sem componente de decisão nenhum.
CODIGOS = {"E1": "sem confiança/abstenção/detecção de erro E sem NER por decoder",
           "E2": "modalidade exclusivamente visão, áudio ou fala",
           "E3": "survey, artigo de posição ou overview"}


def tria(titulo: str, resumo: str) -> tuple[str, str, str]:
    """(decisão, código de exclusão, critério de inclusão).

    A tela é um FILTRO DE RECALL, não a decisão final: a inclusão se decide na
    extração, onde modelo e escala precisam ser determináveis. O `DEC` casa
    `large language model` de forma larga de propósito — prefere-se um falso
    positivo que a extração descarta a um falso negativo que ninguém revê.
    """
    b = f"{titulo} {resumo}"
    if SURVEY.search(titulo):
        return "EXCLUIR", "E3", ""
    if VISAO.search(titulo) and not SINAL.search(b):
        return "EXCLUIR", "E2", ""
    i = bool(SINAL.search(b) and USO.search(b))
    ii = bool(NER.search(b) and DEC.search(b))
    if i and ii:
        return "INCLUIR", "", "i+ii"
    if i:
        return "INCLUIR", "", "i"
    if ii:
        return "INCLUIR", "", "ii"
    return "EXCLUIR", "E1", ""


def valida_contra(caminho_csv: str) -> float:
    """Fração das decisões guardadas que esta regra reproduz. Esperado: 0,9795."""
    import csv
    n = ok = 0
    with open(caminho_csv, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            n += 1
            ok += tria(r["titulo"], r.get("resumo", ""))[0] == r["decisao"]
    return ok / n if n else float("nan")
