"""As comparações DECLARADAS, executadas a partir da declaração.

POR QUE ESTE MÓDULO NÃO DECIDE NADA

Ele não sabe quais comparações fazer: lê isso de `prereg.comparisons`. Não sabe
onde as faixas de tamanho cortam: lê de `prereg.span_size_bins`. Não sabe quantas
reamostragens nem qual nível: lê da declaração. Um valor padrão aqui seria uma
escolha que alguém pode ter feito depois de ver a curva, e é exatamente o que a
declaração existe para impedir.

O QUE ELE CALCULA

- comparações `descriptive`: retrato, sem veredito. R² da massa contra a fração
  geométrica e a mediana da razão observado/esperado.
- comparações `verdict`: diferença de AURC pareada com intervalo por
  reamostragem de SENTENÇAS, e o veredito pelo critério declarado.
- o relato estratificado pelas faixas declaradas e por aninhamento.
- a correlação de cada escore com o comprimento da sentença, que é o controle que
  Saghir (2026) recomenda e que torna o confundidor visível em vez de suposto.

QUAL PARTIÇÃO CADA COISA USA, E POR QUÊ

`C3` combina dois sinais com um peso, e esse peso tem de ser ajustado fora da
partição onde o ganho é medido — senão o ganho medido inclui o próprio ajuste.
Então os VEREDITOS saem da partição de avaliação (os 70% restantes), com o peso
ajustado nos 30% de calibração.

A comparação descritiva usa a partição INTEIRA: ela não ajusta nada e não produz
veredito, então dividir só perderia precisão da descrição. Cada linha do relatório
declara qual partição a produziu.
"""
from __future__ import annotations

import csv
import json
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .geometry import enrichment as _enrichment
from .geometry import expected_mass, geometric_residual
from .preregistration import Preregistration
from .risk_coverage import DeltaAURC, aurc, delta_aurc_paired_bootstrap, risk_coverage_curve

#: Rótulos que o relatório usa para blocos de RELATO, não para comparações
#: declaradas. Existe como constante para que acrescentar um bloco seja um ato
#: explícito: o teste que verifica "executou exatamente o declarado" subtrai
#: estes, então uma comparação INVENTADA continua quebrando a suíte.
ROTULOS_DE_RELATO = ("controle", "escores")

__all__ = [
    "ROTULOS_DE_RELATO",
    "ComparisonsError",
    "ComparisonsReport",
    "sentence_lengths",
    "run_declared_comparisons",
]

#: Colunas exigidas na tabela. `token_indices` entrou em 03/09/2026 e é o que
#: permite recompor a geometria: sem ela, k não é recuperável da tabela.
COLUNAS_EXIGIDAS = (
    "sentence_id", "loss", "model_confidence", "span_mass", "is_nested", "token_indices",
)


class ComparisonsError(RuntimeError):
    """Entrada que não permite executar o que a declaração manda."""


def sentence_lengths(attention_dir: Path) -> dict[str, int]:
    """T por sentença, do arquivo ao lado da tabela ou, na falta dele, das fatias.

    T é UM INTEIRO por sentença, e exigir as matrizes de atenção para obtê-lo
    acoplava a análise a 1,5 GB de tensor por corpus. Isso deixou de ser só
    desperdício quando a medição passou a rodar em outra máquina: trazer de volta
    a tabela por entidade custa megabytes, trazer os tensores custa gigabytes, e
    a análise ficava impossível sem eles. Agora a medição grava
    `sentence_lengths.csv` ao lado de `entities.csv`.

    A leitura das fatias continua como caminho alternativo, e não por gentileza:
    as medições feitas antes desta mudança não têm o arquivo, e recalculá-las
    para obter um número que já está no disco seria perder a partição
    confirmatória por questão de formato.
    """
    from numpy.lib import format as npy_format

    fora: dict[str, int] = {}
    lado = Path(attention_dir).parent / "sentence_lengths.csv"
    if lado.is_file():
        with lado.open(encoding="utf-8-sig", newline="") as fh:
            for linha in csv.DictReader(fh):
                fora[linha["sentence_id"]] = int(linha["n_tokens"])
        if fora:
            return fora

    fatias = sorted(Path(attention_dir).glob("shard_*.npz"))
    if not fatias:
        raise ComparisonsError(
            f"nem {lado} nem fatias de atenção em {attention_dir}: sem uma das duas "
            f"não há comprimento de sentença, e `geometric_fraction` e `sentence_length` "
            f"dependem dele"
        )
    for fatia in fatias:
        with zipfile.ZipFile(fatia) as z:
            for nome in z.namelist():
                if not nome.endswith(".npy"):
                    continue
                with z.open(nome) as fh:
                    maior, menor = npy_format.read_magic(fh)
                    # O nome público muda com a versão do formato .npy; o privado
                    # não existe em toda versão do numpy. Despachar pela versão
                    # lida é o que não quebra quando o numpy sobe.
                    leitor = getattr(npy_format, f"read_array_header_{maior}_{menor}")
                    forma, _, _ = leitor(fh)
                fora[nome[: -len(".npy")]] = int(forma[-1])
    return fora


def _carregar_tabela(table_dir: Path, lengths: dict[str, int]) -> dict[str, np.ndarray]:
    caminho = Path(table_dir) / "entities.csv"
    if not caminho.is_file():
        raise ComparisonsError(f"tabela ausente: {caminho}")
    with caminho.open(encoding="utf-8", newline="") as fh:
        linhas = list(csv.DictReader(fh))
    if not linhas:
        raise ComparisonsError(f"{caminho} está vazia")
    faltando = [c for c in COLUNAS_EXIGIDAS if c not in linhas[0]]
    if faltando:
        raise ComparisonsError(
            f"{caminho} não tem as colunas {faltando}. `token_indices` entrou no contrato em "
            f"03/09/2026 e sem ela a geometria não é recuperável — remeça o split."
        )
    sem_T = {l["sentence_id"] for l in linhas if l["sentence_id"] not in lengths}
    if sem_T:
        raise ComparisonsError(
            f"{len(sem_T)} sentenças da tabela não têm matriz de atenção correspondente "
            f"(ex.: {sorted(sem_T)[:3]}). Tabela e fatias vêm de execuções diferentes."
        )
    k = np.array([len(l["token_indices"].split("|")) for l in linhas], dtype=float)
    T = np.array([lengths[l["sentence_id"]] for l in linhas], dtype=float)
    fora = {
        "sentence_id": np.array([l["sentence_id"] for l in linhas]),
        "loss": np.array([float(l["loss"]) for l in linhas]),
        "model_confidence": np.array([float(l["model_confidence"]) for l in linhas]),
        "span_mass": np.array([float(l["span_mass"]) for l in linhas]),
        "is_nested": np.array([float(l["is_nested"]) for l in linhas]),
        "span_size": k,
        "sentence_length": T,
    }

    # OS SINAIS DAS TRÊS FAMÍLIAS, quando a tabela os traz.
    #
    # Opcionais de propósito: as tabelas medidas antes de 16/09/2026 não têm
    # estas colunas, e exigir todas tornaria as medições anteriores ilegíveis por
    # este módulo. Quem pede um sinal ausente recebe erro nomeado em `_combinar`,
    # o que é melhor que um erro de carregamento que não diz qual sinal falta.
    #
    # `span_position` entra aqui e não é derivado: é o primeiro token do trecho,
    # e sem a coluna não há como recuperá-lo de `k` e `T`. Ele é covariável da
    # desconfundição empírica e, no caso causal, dimensão própria do confundidor.
    OPCIONAIS = (
        "span_position",
        "row_entropy", "row_max", "emitted_mass",
        "hidden_norm", "hidden_dist_centroide", "hidden_delta_camadas",
        "logit_margin", "logit_entropy", "logit_max",
        # O BRAÇO DECODER, acrescentado em 23/09/2026 pela adenda 02. Tabelas do
        # encoder não trazem nenhuma destas colunas, então o caminho delas não
        # muda. `expected_causal` é o nulo exato sob máscara causal, e a presença
        # dele é o que troca o nulo em `_escores_geometricos`.
        "expected_causal",
        "row_entropy_causal", "row_max_causal",
        "aggseq",
        "sonda_ocultos", "sonda_atencao_cabecas",
        "sonda_ocultos_ajustada", "sonda_atencao_cabecas_ajustada",
        "base_treinada", "base_treinada_ajustada",
    )
    for nome in OPCIONAIS:
        if nome in linhas[0]:
            fora[nome] = np.array([float(l[nome]) for l in linhas])
    return fora


def _escores_geometricos(t: dict[str, np.ndarray], prereg: Preregistration) -> None:
    """Acrescenta os escores derivados, todos com a política de sumidouro declarada."""
    k, T = t["span_size"], t["sentence_length"]
    if "expected_causal" in t:
        # O NULO CAUSAL, pela adenda 02 (decisão do autor, 23/09/2026). Sob
        # máscara causal a parte esperada do trecho depende também da POSIÇÃO,
        # e o nulo bidirecional `expected_mass(k, T)` é o nulo errado. Mesmas
        # três quantidades do encoder, com o nulo trocado e nada mais.
        esp = t["expected_causal"]
        if esp.size < 3 or np.allclose(esp, esp[0]):
            raise ComparisonsError("nulo causal constante ou amostra < 3: nada a remover")
        t["geometric_fraction"] = esp
        t["enrichment"] = t["span_mass"] / esp
        a_, b_ = np.polyfit(esp, t["span_mass"], 1)
        t["geometric_residual"] = t["span_mass"] - (a_ * esp + b_)
        return
    t["geometric_fraction"] = expected_mass(k, T, sink_policy=prereg.sink_policy)
    t["enrichment"] = _enrichment(t["span_mass"], k, T, sink_policy=prereg.sink_policy)
    t["geometric_residual"] = geometric_residual(
        t["span_mass"], k, T, sink_policy=prereg.sink_policy
    )


def _separar_por_sentenca(
    t: dict[str, np.ndarray], prereg: Preregistration
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Calibração e avaliação separadas por SENTENÇA, não por entidade.

    Entidades da mesma sentença compartilham a matriz de atenção; separá-las
    colocaria contexto dos dois lados. Vazamento que não aparece como erro,
    aparece como resultado bom.
    """
    sentencas = np.unique(t["sentence_id"])
    if sentencas.size < 2:
        raise ComparisonsError(f"{sentencas.size} sentença(s): não há como separar")
    rng = np.random.default_rng(prereg.seed)
    ordem = rng.permutation(sentencas)
    n_cal = min(max(1, int(round(prereg.calibration_fraction * sentencas.size))),
                sentencas.size - 1)
    cal = set(ordem[:n_cal].tolist())
    m = np.array([s in cal for s in t["sentence_id"]])
    return ({k: v[m] for k, v in t.items()}, {k: v[~m] for k, v in t.items()})


# Sinais cuja ORIENTAÇÃO é fixada por teoria, não por dado. Para os outros, a
# orientação sai da correlação com acerto na CALIBRAÇÃO (ver `_orientar`).
#
# `model_confidence` e `logit_max` são a mesma quantidade e são confiança por
# definição: alto significa entregar. Fixá-los aqui, em vez de deixá-los na regra
# geral, tem uma função — é o que faz a verificação de identidade de `logit_max`
# valer: se a orientação dele viesse do dado, ele poderia sair invertido em
# relação à confiança e o delta deixaria de ser exatamente zero por uma razão
# que não é defeito de encanamento.
ORIENTACAO_FIXA = {"model_confidence": +1.0, "logit_max": +1.0}


def _orientar(
    nome: str,
    cal: dict[str, np.ndarray],
    aval: dict[str, np.ndarray],
    notas: list[str],
) -> np.ndarray:
    """O escore na orientação em que ALTO significa entregar.

    POR QUE ISTO EXISTE, e por que não é grau de liberdade.

    A curva risco-cobertura entrega por escore decrescente. Para a confiança,
    alto significa entregar. Para a entropia dos escores de rótulo, alto
    significa o contrário — o modelo está indeciso, e essa entidade deveria ir
    para revisão. Um sinal na orientação errada sai pior que o acaso por razão
    trivial, e relatar isso como achado seria erro grosseiro.

    Testar as DUAS orientações e reportar a melhor dobraria as chances ao acaso.
    A saída é a mesma disciplina que o peso convexo já usa neste projeto: a
    orientação é decidida pelo sinal da correlação entre o escore e o acerto na
    partição de CALIBRAÇÃO, e nunca na de avaliação. Nada é escolhido depois de
    ver o veredito, e nada é escolhido por mim.

    INVARIANTE DO PROTOCOLO, não item por declaração: vale para todo sinal, em
    toda declaração, como a separação por sentença.
    """
    x_cal, x_av = cal[nome], aval[nome]
    if nome in ORIENTACAO_FIXA:
        return ORIENTACAO_FIXA[nome] * x_av
    acerto = 1.0 - cal["loss"]
    if np.std(x_cal) == 0 or np.std(acerto) == 0:
        notas.append(
            f"{nome}: escore ou acerto constante na calibração, orientação indeterminada; "
            f"mantida como medida (+1). Um escore constante não ordena nada, e o veredito "
            f"sai como não distingue por essa razão e não por empate de desempenho")
        return x_av
    r = float(np.corrcoef(x_cal, acerto)[0, 1])
    sinal = 1.0 if r >= 0 else -1.0
    notas.append(
        f"{nome}: orientação {'+' if sinal > 0 else '-'}1 (correlação com acerto na "
        f"CALIBRAÇÃO r={r:+.4f}). Decidida antes de olhar a avaliação, por invariante do "
        f"protocolo, e não por escolha")
    return sinal * x_av


def _combinar(
    nome: str,
    cal: dict[str, np.ndarray],
    aval: dict[str, np.ndarray],
    prereg: Preregistration,
    notas: list[str],
) -> np.ndarray:
    """Escore combinado, com o peso AJUSTADO NA CALIBRAÇÃO.

    Aceita `a+b`: dois sinais padronizados pela média e desvio da CALIBRAÇÃO e
    combinados convexamente por um peso varrido lá. Usar média e desvio da
    avaliação vazaria a partição de teste para dentro do escore.
    """
    partes = [p.strip() for p in nome.split("+")]
    for p_ in partes:
        if p_ not in aval:
            raise ComparisonsError(
                f"sinal {p_!r} não está na tabela. Colunas disponíveis: "
                f"{sorted(k for k, v in aval.items() if v.dtype.kind == 'f')}. "
                f"Tabelas medidas antes de 16/09/2026 não têm as colunas das três "
                f"famílias de sinal — remeça o split sob a declaração que as pede."
            )
    # A ORIENTAÇÃO É REGRA DA VERSÃO 5 DO PROTOCOLO, e só vale para declarações
    # assinadas sob ela. Ela entrou em 15/09/2026 com decl-07/08; as declarações
    # anteriores (hash_version 2 a 4) foram assinadas e analisadas SEM ela, com o
    # escore na orientação em que foi medido. Aplicá-la a elas mudava vereditos
    # publicados — C2 no CoNLL e C3 no GENIA — e foi medido em 24/09/2026: com a
    # regra condicionada à versão, as dez tabelas confirmatórias do braço encoder
    # reproduzem com diferença 0,0; sem a condição, quatro não reproduzem.
    # Uma regra de protocolo nova não pode reescrever o passado.
    if prereg.hash_version < 5:
        orientar = lambda p_, c_, a_, n_: a_[p_]  # noqa: E731
    else:
        orientar = _orientar
    if len(partes) == 1:
        return orientar(partes[0], cal, aval, notas)
    if len(partes) != 2:
        raise ComparisonsError(f"combinação de {len(partes)} sinais não implementada: {nome!r}")
    if prereg.combination_rule != "convex":
        raise ComparisonsError(
            f"regra {prereg.combination_rule!r} não implementada neste módulo; "
            f"a declaração vigente diz 'convex'"
        )
    # Cada parte é ORIENTADA antes de padronizar: combinar um sinal na
    # orientação certa com outro na inversa produz peso que corrige a orientação
    # em vez de pesar informação.
    X_cal = np.column_stack([orientar(p, cal, cal, notas) for p in partes])
    X_av = np.column_stack([orientar(p, cal, aval, notas) for p in partes])
    mu, sd = X_cal.mean(axis=0), X_cal.std(axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    Z_cal, Z_av = (X_cal - mu) / sd, (X_av - mu) / sd

    melhor_w, melhor = 0.0, np.inf
    for w in np.linspace(0.0, 1.0, 101):
        a = risk_coverage_curve(cal["loss"], w * Z_cal[:, 0] + (1 - w) * Z_cal[:, 1]).aurc
        if a < melhor:
            melhor, melhor_w = float(a), float(w)
    notas.append(
        f"{nome}: peso {melhor_w:.2f} em {partes[0]} e {1 - melhor_w:.2f} em {partes[1]}, "
        f"ajustado na calibração (AURC {melhor:.4f}). Peso em 0,00 ou 1,00 é a combinação "
        f"COLAPSANDO para um sinal isolado — o que distingue 'descartou o outro' de 'empatou'."
    )
    return melhor_w * Z_av[:, 0] + (1 - melhor_w) * Z_av[:, 1]


def _faixas(k: np.ndarray, bins: Sequence[tuple[int, int | None]]) -> list[tuple[str, np.ndarray]]:
    fora = []
    for lo, hi in bins:
        m = (k >= lo) if hi is None else ((k >= lo) & (k <= hi))
        rot = f"k>={lo}" if hi is None else (f"k={lo}" if hi == lo else f"k={lo}-{hi}")
        fora.append((rot, m))
    return fora


@dataclass
class ComparisonsReport:
    declaration_id: str
    declaration_hash: str
    corpus: str
    linhas: list[dict[str, Any]] = field(default_factory=list)
    notas: list[str] = field(default_factory=list)

    def render(self) -> str:
        out = [f"Declaração {self.declaration_id} · hash {self.declaration_hash} · {self.corpus}"]
        for l in self.linhas:
            if l["tipo"] == "descriptive":
                out.append(f"  {l['comparacao']} [{l['estrato']}] {l['metrica']} = {l['valor']}")
            else:
                out.append(
                    f"  {l['comparacao']} [{l['estrato']}] ΔAURC {l['delta']:+.4f} "
                    f"IC [{l['ci_low']:+.4f}, {l['ci_high']:+.4f}] -> {l['veredito']}"
                )
        if self.notas:
            out.append("  Ressalvas:")
            out += [f"    - {n}" for n in self.notas]
        return "\n".join(out)


def run_declared_comparisons(
    table_dir: str | Path, prereg: Preregistration, corpus: str
) -> ComparisonsReport:
    """Executa exatamente as comparações que a declaração manda, e nada além."""
    table_dir = Path(table_dir)
    lengths = sentence_lengths(table_dir / "attention")
    t = _carregar_tabela(table_dir, lengths)
    _escores_geometricos(t, prereg)

    rel = ComparisonsReport(prereg.declaration_id, prereg.declaration_hash, corpus)
    cal, aval = _separar_por_sentenca(t, prereg)
    rel.notas.append(
        f"partição por sentença: {np.unique(cal['sentence_id']).size} sentenças de calibração "
        f"({cal['loss'].size} entidades) e {np.unique(aval['sentence_id']).size} de avaliação "
        f"({aval['loss'].size} entidades)."
    )

    for comp in prereg.comparisons:
        # ROTEAMENTO POR TIPO, e ele não é organização de código.
        #
        # Uma comparação `task_verdict` declara o critério de CARGA DE REVISÃO, e
        # quem a julga é `task_impact.run_task_impact`. Este runner julga por
        # ΔAURC. Antes desta guarda, tudo que não fosse `descriptive` caía no
        # ramo de AURC — inclusive `task_verdict` —, e o relatório imprimia
        # "T2 ΔAURC +0.1048 -> PIOR que a base" para uma comparação cujo critério
        # declarado é outro. Reportar quantidade NÃO declarada com etiqueta de
        # veredito é o defeito mais grave que este arquivo poderia ter.
        #
        # Só apareceu com a `decl-04`, a primeira declaração a trazer os dois
        # tipos no mesmo arquivo: a `decl-02` tinha apenas AURC e a `decl-03`
        # apenas tarefa.
        if comp.kind == "task_verdict":
            rel.notas.append(
                f"{comp.id} não é julgada aqui: o critério declarado dela é "
                f"{comp.criterion!r}, de carga de revisão, e quem julga é "
                f"task_impact.run_task_impact. Omitida deste relatório de propósito."
            )
            continue

        if comp.kind == "descriptive":
            # Partição inteira: não ajusta nada e não produz veredito.
            for rot, m in [("todos", np.ones(t["loss"].size, dtype=bool))] + _faixas(
                t["span_size"], prereg.span_size_bins
            ):
                if m.sum() < 3:
                    continue
                massa, esp = t["span_mass"][m], t["geometric_fraction"][m]
                r = float(np.corrcoef(massa, esp)[0, 1])
                for metrica, valor in (
                    ("R2_massa_vs_fracao_geometrica", round(r * r, 4)),
                    ("mediana_observado_sobre_esperado", round(float(np.median(massa / esp)), 4)),
                    ("n_entidades", int(m.sum())),
                ):
                    rel.linhas.append(dict(
                        comparacao=comp.id, tipo=comp.kind, particao="teste inteiro",
                        estrato=rot, metrica=metrica, valor=valor))
            continue

        escore_a = _combinar(comp.against, cal, aval, prereg, rel.notas)
        escore_b = _combinar(comp.score, cal, aval, prereg, rel.notas)
        estratos = [("todos", np.ones(aval["loss"].size, dtype=bool))]
        estratos += _faixas(aval["span_size"], prereg.span_size_bins)
        estratos += [("aninhado", aval["is_nested"] > 0), ("plano", aval["is_nested"] == 0)]

        for rot, m in estratos:
            if m.sum() < 10 or np.unique(aval["loss"][m]).size < 2:
                rel.linhas.append(dict(
                    comparacao=comp.id, tipo=comp.kind, particao="avaliação", estrato=rot,
                    n_entidades=int(m.sum()), veredito="NÃO AVALIADO (estrato pequeno "
                    "ou de perda constante)"))
                continue
            try:
                d = delta_aurc_paired_bootstrap(
                    aval["loss"][m], escore_a[m], escore_b[m],
                    groups=aval["sentence_id"][m],
                    n_resamples=prereg.n_bootstrap_resamples,
                    level=prereg.added_value_ci_level,
                    seed=prereg.seed,
                    labels=(comp.against, comp.score),
                )
            except RuntimeError as e:
                rel.linhas.append(dict(
                    comparacao=comp.id, tipo=comp.kind, particao="avaliação", estrato=rot,
                    n_entidades=int(m.sum()), veredito=f"NÃO AVALIADO ({e})"))
                continue
            rel.linhas.append(dict(
                comparacao=comp.id, tipo=comp.kind, particao="avaliação", estrato=rot,
                escore=comp.score, contra=comp.against,
                aurc_contra=round(d.aurc_a, 4), aurc_escore=round(d.aurc_b, 4),
                delta=round(d.delta, 4), ci_low=round(d.ci_low, 4), ci_high=round(d.ci_high, 4),
                n_entidades=d.n_entities, n_sentencas=d.n_groups,
                n_degeneradas=d.n_degenerate_resamples,
                veredito=("NÃO ACRESCENTA (IC contém zero)" if d.contains_zero
                          else ("ACRESCENTA" if d.delta < 0 else "PIOR que a base")),
            ))

    # A AURC de CADA escore declarado, na partição de avaliação. Declarar um
    # escore de comparação e não reportar o desempenho dele seria declarar sem
    # usar — e é o que dá conteúdo à frase "a massa tem de vencer estes".
    base = float(aval["loss"].mean())
    rel.linhas.append(dict(
        comparacao="escores", tipo="descriptive", particao="avaliação", estrato="todos",
        metrica="aurc_acaso", valor=round(base, 4)))
    for nome in prereg.comparison_scores + ("span_mass", "model_confidence"):
        a = aurc(risk_coverage_curve(aval["loss"], aval[nome]))
        rel.linhas.append(dict(
            comparacao="escores", tipo="descriptive", particao="avaliação", estrato="todos",
            metrica=f"aurc_{nome}", valor=round(float(a), 4),
            veredito="melhor que o acaso" if a < base else "NÃO melhor que o acaso"))

    # Controle de Saghir (2026): a correlação escore-comprimento, relatada e não suposta.
    for nome in prereg.comparison_scores + ("span_mass", "model_confidence"):
        rel.linhas.append(dict(
            comparacao="controle", tipo="descriptive", particao="teste inteiro", estrato="todos",
            metrica=f"corr_{nome}_vs_comprimento",
            valor=round(float(np.corrcoef(t[nome], t["sentence_length"])[0, 1]), 4)))
    return rel
