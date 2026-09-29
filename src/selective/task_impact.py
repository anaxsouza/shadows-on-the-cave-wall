"""O impacto na TAREFA: carga de revisão a qualidade fixa.

POR QUE ESTE MÓDULO EXISTE

A decl-02 decidiu em ΔAURC, que é área sob uma curva. Ninguém em NER decide nada
com uma área. Aqui a mesma medição sai em duas unidades que se usam: a qualidade
do que o sistema entrega, e quanto trabalho humano isso custa.

A RESTRIÇÃO QUE ESCOLHEU A UNIDADE

O supervisor só REMOVE predições; nunca cria. As entidades anotadas que o
extrator não apontou são invisíveis para ele, então o recall tem teto fixo — o
recall do extrator — e abster-se só o baixa. Por isso a unidade declarada é
carga de revisão A QUALIDADE FIXA, e não ganho de F1: é a pergunta que o arranjo
pode responder.

A CONVENÇÃO, E DE ONDE ELA VEM

Declara-se a qualidade alvo e MAXIMIZA-SE a cobertura sujeita a ela. É a mesma
convenção de Geifman & El-Yaniv (2017) que a declaração já usa para o risco —
precisão alvo `q` é risco alvo `1-q` entre os entregues —, e não o inverso.
Declarar cobertura e ler a qualidade obtida seria inconsistente com a garantia.

O QUE ELE NÃO DECIDE

Nada. As metas vêm de `prereg.task_quality_grid`, as métricas de
`prereg.task_metrics`, o nível e as reamostragens da declaração. A partição, a
ordenação e a semente são as da decl-02, intocadas.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .preregistration import Preregistration
from .quality_grid import absolute_targets, base_precision, gap_closed


class TaskImpactError(RuntimeError):
    """A tabela ou a declaração não permitem calcular o impacto na tarefa."""


# Rótulo único para "nenhuma cobertura atinge esta meta". É valor declarado e não
# ausência: a decl-03 nomeia "inalcançável" como resultado possível, e a
# alternativa (célula vazia) seria lida como falha de execução.
INALCANCAVEL = "inalcançável"


def _fronteiras(ordenado: np.ndarray) -> np.ndarray:
    """Tamanhos de prefixo que NÃO partem um grupo de empate.

    Empate importa: dois escores iguais não têm ordem entre si, e cortar no meio
    de um grupo empatado reportaria uma cobertura que o supervisor não consegue
    realizar — ele não sabe qual dos empatados entregar primeiro.
    """
    muda = np.flatnonzero(np.diff(ordenado) != 0) + 1
    return np.append(muda, ordenado.size)


@dataclass(frozen=True)
class OperatingPoint:
    """O ponto de operação que atinge uma meta de qualidade, se existir."""

    target: float
    reachable: bool
    review_load: float = float("nan")
    coverage: float = float("nan")
    precision: float = float("nan")
    recall: float = float("nan")
    f1: float = float("nan")
    n_delivered: int = 0


def operating_point(
    losses: np.ndarray,
    score: np.ndarray,
    target: float,
    n_gold: int,
) -> OperatingPoint:
    """Maior cobertura cuja precisão entre os entregues atinge `target`.

    `score` alto significa ENTREGAR — a convenção do resto do módulo. A busca é
    pelo MAIOR prefixo que atinge a meta, e não pelo primeiro: precisão não é
    monótona na cobertura, então parar no primeiro que atinge entregaria menos
    do que o supervisor consegue e inflaria a carga de revisão dele.
    """
    perdas = np.asarray(losses, dtype=float)
    s = np.asarray(score, dtype=float)
    if perdas.shape != s.shape or perdas.ndim != 1:
        raise TaskImpactError(f"perda e escore incompatíveis: {perdas.shape} e {s.shape}")
    if perdas.size == 0:
        raise TaskImpactError("amostra vazia")
    if n_gold <= 0:
        raise TaskImpactError(
            f"n_gold = {n_gold}: sem o total de entidades anotadas não há denominador para o "
            f"recall, e inventar um denominador seria inventar o recall")

    n = perdas.size
    ordem = np.argsort(-s, kind="stable")
    acertos = np.cumsum(1.0 - perdas[ordem])
    m = _fronteiras(s[ordem])
    precisao = acertos[m - 1] / m

    ok = np.flatnonzero(precisao >= target)
    if ok.size == 0:
        return OperatingPoint(target=float(target), reachable=False)

    i = int(ok[-1])                      # o MAIOR prefixe que atinge a meta
    entregues = int(m[i])
    corretos = float(acertos[entregues - 1])
    prec = corretos / entregues
    rec = corretos / float(n_gold)
    f1 = 0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec)
    return OperatingPoint(
        target=float(target), reachable=True,
        review_load=1.0 - entregues / n, coverage=entregues / n,
        precision=prec, recall=rec, f1=f1, n_delivered=entregues,
    )


@dataclass(frozen=True)
class ReviewLoadDelta:
    """Diferença de carga de revisão entre dois supervisores, com intervalo."""

    target: float
    score_label: str
    against_label: str
    review_load_score: float
    review_load_against: float
    delta: float
    ci_low: float
    ci_high: float
    level: float
    n_resamples: int
    n_entities: int
    n_groups: int
    n_degenerate: int
    contains_zero: bool
    verdict: str

    def render(self) -> str:
        if not np.isfinite(self.delta):
            return (f"  meta {self.target:.0%}: {self.verdict}")
        return (f"  meta {self.target:.0%}: revisão {self.review_load_score:.1%} "
                f"({self.score_label}) contra {self.review_load_against:.1%} "
                f"({self.against_label}) | Δ {self.delta:+.4f} "
                f"IC [{self.ci_low:+.4f}, {self.ci_high:+.4f}] -> {self.verdict}")


def paired_review_load_delta(
    losses: np.ndarray,
    score: np.ndarray,
    against: np.ndarray,
    groups: np.ndarray,
    target: float,
    n_gold: int,
    *,
    level: float,
    n_resamples: int,
    seed: int,
    labels: tuple[str, str] = ("escore", "base"),
) -> ReviewLoadDelta:
    """Δ carga de revisão com intervalo por reamostragem de CONGLOMERADO.

    `groups` são as sentenças, e reamostrar sentenças e não entidades não é
    detalhe: entidades da mesma sentença compartilham a matriz de atenção, então
    o número de unidades independentes é o de sentenças. Tratá-las como
    independentes estreitaria o intervalo — e intervalo estreito demais é o que
    faz uma diferença "excluir o zero" quando não devia.

    Sinal de `delta`: NEGATIVO significa que `score` exige MENOS revisão, ou
    seja, é melhor. É a direção em que a tese se refuta.

    Reamostragem em que alguma das duas ordenações não atinge a meta é
    DEGENERADA: não há carga de revisão a comparar. Ela é contada e excluída, e a
    contagem viaja no resultado — se for grande, o intervalo descreve um
    subconjunto das reamostragens e isso tem de estar visível.
    """
    perdas = np.asarray(losses, dtype=float)
    a = np.asarray(score, dtype=float)
    b = np.asarray(against, dtype=float)
    g = np.asarray(groups)
    if not (perdas.shape == a.shape == b.shape == g.shape):
        raise TaskImpactError("perda, escores e grupos têm de ter o mesmo tamanho")

    p_a = operating_point(perdas, a, target, n_gold)
    p_b = operating_point(perdas, b, target, n_gold)
    grupos = np.unique(g)
    if not (p_a.reachable and p_b.reachable):
        quais = " e ".join(
            n for n, p in ((labels[0], p_a), (labels[1], p_b)) if not p.reachable)
        return ReviewLoadDelta(
            target=float(target), score_label=labels[0], against_label=labels[1],
            review_load_score=p_a.review_load, review_load_against=p_b.review_load,
            delta=float("nan"), ci_low=float("nan"), ci_high=float("nan"),
            level=float(level), n_resamples=0, n_entities=int(perdas.size),
            n_groups=int(grupos.size), n_degenerate=0, contains_zero=False,
            verdict=f"{INALCANCAVEL} para {quais}",
        )

    # Índices por sentença, uma vez: no laço só se concatena.
    por_grupo = {gr: np.flatnonzero(g == gr) for gr in grupos}
    chaves = list(por_grupo)
    rng = np.random.default_rng(seed)
    deltas, degeneradas = [], 0
    for _ in range(int(n_resamples)):
        sorteados = rng.integers(0, len(chaves), size=len(chaves))
        idx = np.concatenate([por_grupo[chaves[j]] for j in sorteados])
        # n_gold é do corpus e não da reamostragem: ele entra no recall, que não
        # é o alvo da busca. A carga de revisão não depende dele.
        ra = operating_point(perdas[idx], a[idx], target, n_gold)
        rb = operating_point(perdas[idx], b[idx], target, n_gold)
        if not (ra.reachable and rb.reachable):
            degeneradas += 1
            continue
        deltas.append(ra.review_load - rb.review_load)

    if not deltas:
        return ReviewLoadDelta(
            target=float(target), score_label=labels[0], against_label=labels[1],
            review_load_score=p_a.review_load, review_load_against=p_b.review_load,
            delta=p_a.review_load - p_b.review_load, ci_low=float("nan"),
            ci_high=float("nan"), level=float(level), n_resamples=0,
            n_entities=int(perdas.size), n_groups=int(grupos.size),
            n_degenerate=degeneradas, contains_zero=False,
            verdict="INCONCLUSIVO: toda reamostragem foi degenerada",
        )

    d = np.asarray(deltas)
    alfa = (1.0 - float(level)) / 2.0
    lo, hi = np.quantile(d, [alfa, 1.0 - alfa])
    delta = p_a.review_load - p_b.review_load
    contem_zero = bool(lo <= 0.0 <= hi)
    if contem_zero:
        veredito = "NÃO DISTINGUE (IC contém zero)"
    elif hi < 0:
        veredito = f"{labels[0]} EXIGE MENOS revisão"
    else:
        veredito = f"{labels[0]} EXIGE MAIS revisão"
    return ReviewLoadDelta(
        target=float(target), score_label=labels[0], against_label=labels[1],
        review_load_score=p_a.review_load, review_load_against=p_b.review_load,
        delta=float(delta), ci_low=float(lo), ci_high=float(hi), level=float(level),
        n_resamples=int(d.size), n_entities=int(perdas.size), n_groups=int(grupos.size),
        n_degenerate=int(degeneradas), contains_zero=contem_zero, verdict=veredito,
    )


@dataclass
class TaskImpactReport:
    """As linhas do relato, mais as ressalvas que viajam com elas."""

    corpus: str
    declaration_id: str
    declaration_hash: str
    n_gold: int
    linhas: list[dict] = field(default_factory=list)
    notas: list[str] = field(default_factory=list)

    def render(self) -> str:
        out = [f"TAREFA — {self.corpus} sob {self.declaration_id} ({self.declaration_hash})",
               f"  entidades anotadas (denominador do recall): {self.n_gold}"]
        for n in self.notas:
            out.append(f"  * {n}")
        return "\n".join(out)


def run_task_impact(
    table_dir: str | Path,
    prereg: Preregistration,
    corpus: str,
    n_gold: int,
) -> TaskImpactReport:
    """Executa T1 e os vereditos de tarefa declarados, e nada além.

    Reusa o carregamento, os escores derivados, a separação por sentença e a
    combinação convexa de `comparisons`: são os mesmos objetos que a decl-02
    usou, e reimplementá-los aqui abriria caminho para divergirem.
    """
    from .comparisons import (
        _carregar_tabela,
        _combinar,
        _escores_geometricos,
        _separar_por_sentenca,
        sentence_lengths,
    )

    if not prereg.task_quality_grid:
        raise TaskImpactError(
            f"{prereg.declaration_id} não declara task_quality_grid: sem a grade no hash a meta "
            f"poderia ser escolhida depois de ver a carga de revisão")

    d = Path(table_dir)
    t = _carregar_tabela(d, sentence_lengths(d / "attention"))
    _escores_geometricos(t, prereg)
    cal, aval = _separar_por_sentenca(t, prereg)

    rel = TaskImpactReport(corpus=corpus, declaration_id=prereg.declaration_id,
                           declaration_hash=prereg.declaration_hash, n_gold=int(n_gold))
    rel.notas.append(
        f"partição de avaliação: {aval['loss'].size} entidades em "
        f"{np.unique(aval['sentence_id']).size} sentenças; calibração com "
        f"{np.unique(cal['sentence_id']).size} sentenças, separada por SENTENÇA")
    rel.notas.append(
        "o recall usa o total de entidades anotadas do SPLIT INTEIRO como denominador, "
        "incluindo as das sentenças de calibração; é o recall do sistema, não da partição")

    # --- a grade do VÃO, derivada da CALIBRAÇÃO -------------------------------
    #
    # A meta absoluta sai de `p_base + fracao * (1 - p_base)`, e `p_base` vem da
    # partição de CALIBRAÇÃO. Derivá-la da partição de avaliação seria fixar o
    # alvo olhando o dado que produz o veredito. A fração é declarada e está no
    # hash; a meta absoluta é consequência da medição, não escolha.
    metas_vao = ()
    if prereg.task_gap_grid:
        metas_vao = absolute_targets(prereg.task_gap_grid, cal["loss"])
        p_cal = metas_vao[0].base_precision
        rel.notas.append(
            "grade do VÃO em vigor: precisão de base da CALIBRAÇÃO = "
            f"{p_cal:.4f}; metas derivadas " +
            ", ".join(f"{t_.gap_fraction:.0%} -> {t_.absolute:.4f}" for t_ in metas_vao) +
            ". A fração está no hash; a meta absoluta é derivada, e vem da calibração "
            "porque derivá-la da avaliação fixaria o alvo olhando a resposta")

    supervisores = tuple(prereg.comparison_scores) + ("span_mass", "model_confidence")
    escores = {nome: _combinar(nome, cal, aval, prereg, rel.notas) for nome in supervisores}

    # --- T1: a tabela descritiva, com a linha SEM supervisor ------------------
    n = aval["loss"].size
    corretos_total = float((1.0 - aval["loss"]).sum())
    prec_total = corretos_total / n
    rec_total = corretos_total / float(n_gold)
    rel.linhas.append(dict(
        corpus=corpus, comparacao="T1", tipo="descriptive", supervisor="(sem supervisor)",
        grade="—", meta="—", meta_absoluta="", vao_fechado=0.0,
        alcancavel="sim", review_load=0.0, coverage=1.0,
        precision=round(prec_total, 4), recall=round(rec_total, 4),
        f1=round(2 * prec_total * rec_total / (prec_total + rec_total), 4), n_delivered=n))
    for nome in supervisores:
        grades = ([("absoluta", f"{q:.2f}", q, "") for q in prereg.task_quality_grid] +
                  [("vao", f"{t_.gap_fraction:.2f}", t_.absolute, f"{t_.absolute:.4f}")
                   for t_ in metas_vao])
        for grade, etiqueta, q, absoluta in grades:
            p = operating_point(aval["loss"], escores[nome], q, n_gold)
            rel.linhas.append(dict(
                corpus=corpus, comparacao="T1", tipo="descriptive", supervisor=nome,
                grade=grade, meta=etiqueta, meta_absoluta=absoluta,
                vao_fechado=(round(gap_closed(p.precision, prec_total), 4)
                             if p.reachable else ""),
                alcancavel="sim" if p.reachable else "não",
                review_load=round(p.review_load, 4) if p.reachable else INALCANCAVEL,
                coverage=round(p.coverage, 4) if p.reachable else "",
                precision=round(p.precision, 4) if p.reachable else "",
                recall=round(p.recall, 4) if p.reachable else "",
                f1=round(p.f1, 4) if p.reachable else "",
                n_delivered=p.n_delivered))

    # --- os vereditos declarados ---------------------------------------------
    for c in prereg.comparisons:
        if c.kind != "task_verdict":
            continue
        a = _combinar(c.score, cal, aval, prereg, rel.notas)
        b = _combinar(c.against, cal, aval, prereg, rel.notas)
        grades = ([("absoluta", f"{q:.2f}", q, "") for q in prereg.task_quality_grid] +
                  [("vao", f"{t_.gap_fraction:.2f}", t_.absolute, f"{t_.absolute:.4f}")
                   for t_ in metas_vao])
        for grade, etiqueta, q, absoluta in grades:
            r = paired_review_load_delta(
                aval["loss"], a, b, aval["sentence_id"], q, n_gold,
                level=prereg.added_value_ci_level,
                n_resamples=prereg.n_bootstrap_resamples,
                seed=prereg.seed, labels=(c.score, c.against))
            rel.linhas.append(dict(
                corpus=corpus, comparacao=c.id, tipo="task_verdict", supervisor=c.score,
                contra=c.against, grade=grade, meta=etiqueta, meta_absoluta=absoluta,
                alcancavel="não" if INALCANCAVEL in r.verdict else "sim",
                review_load=round(r.review_load_score, 4) if np.isfinite(r.review_load_score) else INALCANCAVEL,
                review_load_contra=round(r.review_load_against, 4) if np.isfinite(r.review_load_against) else INALCANCAVEL,
                delta=round(r.delta, 4) if np.isfinite(r.delta) else "",
                ci_low=round(r.ci_low, 4) if np.isfinite(r.ci_low) else "",
                ci_high=round(r.ci_high, 4) if np.isfinite(r.ci_high) else "",
                n_entidades=r.n_entities, n_sentencas=r.n_groups,
                n_reamostragens=r.n_resamples, n_degeneradas=r.n_degenerate,
                veredito=r.verdict))
            if r.n_degenerate and r.n_resamples:
                rel.notas.append(
                    f"{c.id} grade {grade} meta {etiqueta}: {r.n_degenerate} de "
                    f"{r.n_degenerate + r.n_resamples} reamostragens degeneradas (a meta não "
                    f"foi atingida nelas); o intervalo descreve as {r.n_resamples} restantes")
    return rel


__all__ = [
    "INALCANCAVEL",
    "OperatingPoint",
    "ReviewLoadDelta",
    "TaskImpactError",
    "TaskImpactReport",
    "operating_point",
    "paired_review_load_delta",
    "run_task_impact",
]
