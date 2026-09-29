"""Os dois testes da pergunta única, sobre uma tabela de entidades medidas.

O QUE ESTE MÓDULO CONSOME, E POR QUE ELE NÃO CARREGA MODELO

A separação é deliberada: a passagem de inferência produz uma TABELA por
entidade, e os dois testes são funções dessa tabela. Assim o teste estatístico é
reexecutável em segundos sobre resultado já medido, sem GPU, sem rede e sem
depender de a inferência ser determinística. Mudar o alpha do pré-registro e
reexecutar não é uma nova rodada de modelo, é uma releitura.

A tabela esperada é um CSV com uma linha por entidade PREDITA:

    sentence_id          agrupador da partição — ver a sutileza abaixo
    loss                 perda por entidade, em [0, loss_bound]. Com perda 0/1,
                         1 = a predição está errada (fronteira ou tipo)
    model_confidence   a confiança que o próprio modelo já fornece, agregada
                         por entidade pela regra do item 1 do pré-registro
    span_mass            a massa de atenção sobre o span, de attention_mass.py
    is_nested            0 ou 1, a estratificação do contraste plano/aninhado

A SUTILEZA QUE MAIS IMPORTA AQUI: A PARTIÇÃO É POR SENTENÇA

Calibração e avaliação têm de ser separadas por SENTENÇA, não por entidade. Duas
entidades da mesma sentença compartilham a mesma matriz de atenção e o mesmo
contexto: separá-las entre calibração e teste deixa passar informação de uma
partição para a outra, e o limiar sai otimista. Isso é o tipo de vazamento que
não aparece como erro — aparece como resultado bom. Por isso o agrupamento é por
`sentence_id` e a fração declarada no pré-registro é fração de SENTENÇAS.

O QUE CADA TESTE DECIDE

`floor`: a massa de atenção ordena erro melhor que a abstenção aleatória? A
comparação é contra um escore constante, cuja AURC é exatamente a taxa de erro
base, com intervalo por reamostragem pareada. Falhar aqui encerra a tese, e é
por isso que é o piso.

`added-value`: o sinal combinado (confiança do modelo mais massa de atenção,
combinados pela regra do item 3, ajustada na calibração) ordena melhor que a
confiança do modelo sozinha? E o ganho é maior nas entidades aninhadas? A
segunda parte é o que a pergunta afirma de específico, e é ela que distingue
esta tese de aplicar predição seletiva a NER em geral.
"""

from __future__ import annotations

import json

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .conformal import CRCThreshold, crc_threshold
from .preregistration import Preregistration, load_preregistration
from .risk_coverage import (
    DeltaAURC,
    delta_aurc_paired_bootstrap,
    random_abstention_risk,
    risk_coverage_curve,
)

__all__ = ["SelectiveResult", "SelectiveRunner", "EntityTableError"]

# Colunas EXIGIDAS pelos testes, que é subconjunto das EMITIDAS por
# measurement.py: `token_indices` é emitida para a análise de perímetro (recompor
# a massa sob outra leitura sem repetir inferência) e nenhum teste a consome.
# Exigir aqui o conjunto emitido acoplaria o veredito a uma coluna que ele ignora.
COLUNAS = ("sentence_id", "loss", "model_confidence", "span_mass", "is_nested")


class EntityTableError(RuntimeError):
    """A tabela por entidade não existe, ou não tem o que os testes exigem."""


@dataclass(frozen=True)
class SelectiveResult:
    test: str
    model_key: str
    dataset_name: str
    split: str
    n_entities: int
    n_sentences: int
    n_calibration_entities: int
    n_test_entities: int
    base_risk: float
    prereg: Preregistration
    delta: DeltaAURC
    threshold: CRCThreshold
    # Cobertura OBTIDA ao risco alvo declarado, e não risco a uma cobertura
    # fixa. A direção vem de Geifman & El-Yaniv (2017): declara-se r* e
    # maximiza-se a cobertura sujeita a ele.
    operating_point_coverage: float
    # Tabela de relato: cobertura obtida em cada risco alvo da grade. É a forma
    # da Tabela 1 de Geifman & El-Yaniv (2017). Zero numa linha não é erro: é
    # "nem abstendo-se ao máximo este sinal alcança esse risco".
    coverage_by_risk: tuple[tuple[float, float], ...]
    nested_delta: DeltaAURC | None = None
    flat_delta: DeltaAURC | None = None
    notes: tuple[str, ...] = field(default=())

    def render(self) -> str:
        linhas = [
            f"TESTE: {self.test}",
            f"modelo {self.model_key} | corpus {self.dataset_name} | partição {self.split}",
            f"{self.n_entities} entidades em {self.n_sentences} sentenças "
            f"({self.n_calibration_entities} na calibração, {self.n_test_entities} na avaliação)",
            f"taxa de erro base (cobertura total): {self.base_risk:.4f}",
            "",
            self.delta.render(),
            "",
            f"GARANTIA em r* = {self.prereg.target_risk:.3f}: cobertura obtida "
            f"{self.operating_point_coverage:.1%}",
            "cobertura obtida por risco alvo (relato):",
            *[
                f"    r* = {r:.2f}  ->  "
                + (f"{c:.1%}" if c > 0 else "inalcançável em qualquer cobertura")
                for r, c in self.coverage_by_risk
            ],
            self.threshold.render(),
        ]
        if self.nested_delta is not None and self.flat_delta is not None:
            linhas += [
                "",
                "Contraste aninhado x plano — a parte específica da pergunta:",
                f"  aninhadas (n = {self.nested_delta.n_entities}): "
                f"ΔAURC {self.nested_delta.delta:+.4f} "
                f"[{self.nested_delta.ci_low:+.4f}, {self.nested_delta.ci_high:+.4f}]",
                f"  planas    (n = {self.flat_delta.n_entities}): "
                f"ΔAURC {self.flat_delta.delta:+.4f} "
                f"[{self.flat_delta.ci_low:+.4f}, {self.flat_delta.ci_high:+.4f}]",
                "  Os dois intervalos são separados por estrato e NÃO constituem teste da "
                "diferença entre estratos; compará-los por sobreposição de intervalo é o "
                "erro que a leitura desta tabela convida a cometer.",
            ]
        if self.notes:
            linhas += ["", "Ressalvas desta execução:"] + [f"  - {n}" for n in self.notes]
        linhas += ["", self.prereg.render()]
        return "\n".join(linhas)

    def to_dict(self) -> dict[str, Any]:
        def d(x: DeltaAURC | None) -> dict[str, Any] | None:
            return None if x is None else {
                "aurc_a": x.aurc_a, "aurc_b": x.aurc_b, "delta": x.delta,
                "ci_low": x.ci_low, "ci_high": x.ci_high, "level": x.level,
                "n_resamples": x.n_resamples, "n_entities": x.n_entities,
                "contains_zero": x.contains_zero,
                "n_degenerate_resamples": x.n_degenerate_resamples,
                "labels": list(x.labels),
            }

        return {
            "test": self.test,
            "model_key": self.model_key,
            "dataset_name": self.dataset_name,
            "split": self.split,
            "n_entities": self.n_entities,
            "n_sentences": self.n_sentences,
            "n_calibration_entities": self.n_calibration_entities,
            "n_test_entities": self.n_test_entities,
            "base_risk": self.base_risk,
            "delta": d(self.delta),
            "nested_delta": d(self.nested_delta),
            "flat_delta": d(self.flat_delta),
            "operating_point_coverage": self.operating_point_coverage,
            "coverage_by_risk": [[r, c] for r, c in self.coverage_by_risk],
            "threshold": {
                "value": self.threshold.threshold,
                "coverage": self.threshold.coverage,
                "empirical_risk": self.threshold.empirical_risk,
                "alpha": self.threshold.alpha,
                "bound": self.threshold.bound,
                "feasible": self.threshold.feasible,
                "monotone": self.threshold.monotone,
                "max_monotonicity_violation": self.threshold.max_monotonicity_violation,
            },
            "preregistration": {
                "source": self.prereg.source,
                "sink_policy": self.prereg.sink_policy,
                "layers": list(self.prereg.layers),
                "heads": None if self.prereg.heads is None else list(self.prereg.heads),
                "combination_rule": self.prereg.combination_rule,
                "calibration_fraction": self.prereg.calibration_fraction,
                "coverage_levels": list(self.prereg.coverage_levels),
                "target_risk": self.prereg.target_risk,
                "declaration_id": self.prereg.declaration_id,
                "declaration_hash": self.prereg.declaration_hash,
                "measurement_hash": self.prereg.measurement_hash,
                "added_value_ci_level": self.prereg.added_value_ci_level,
                "n_bootstrap_resamples": self.prereg.n_bootstrap_resamples,
                "conformal_alpha": self.prereg.conformal_alpha,
                "loss_bound": self.prereg.loss_bound,
                "seed": self.prereg.seed,
            },
            "notes": list(self.notes),
        }


class SelectiveRunner:
    """Roda um dos dois testes sobre a tabela de entidades já medida."""

    def __init__(
        self,
        model_key: str,
        dataset_name: str,
        config_path: str = "configs/config.yaml",
        split: str = "test",
        max_samples: int = -1,
        seed: int | None = None,
        output_dir: str | None = None,
    ) -> None:
        self.model_key = model_key
        self.dataset_name = dataset_name
        self.config_path = config_path
        self.split = split
        self.max_samples = max_samples
        self.output_dir = Path(output_dir) if output_dir else Path("results")
        self.prereg = load_preregistration(config_path)
        self.seed = self.prereg.seed if seed is None else int(seed)
        # Ressalvas levantadas ANTES de a lista de notas da execução existir —
        # a conferência de procedência acontece ao carregar a tabela, e o que
        # ela apurar tem de chegar ao relatório em vez de se perder.
        self.pending_notes: list[str] = []
        # Ressalvas levantadas dentro do cálculo por estrato, que não tem a
        # lista de notas em mão.
        self._notas_estrato: list[str] = []

    @property
    def table_path(self) -> Path:
        return self.output_dir / self.model_key / self.dataset_name / self.split / "entities.csv"

    def plan(self, test: str) -> list[str]:
        return [
            f"TESTE {test} — modelo {self.model_key}, corpus {self.dataset_name}, "
            f"partição {self.split}",
            f"tabela de entidades: {self.table_path}"
            f"{'' if self.table_path.is_file() else '  (AUSENTE — rode a passagem de inferência)'}",
            f"partição de calibração: {self.prereg.calibration_fraction:.0%} das SENTENÇAS "
            f"(por sentença, não por entidade, para não vazar contexto de atenção)",
            f"semente: {self.seed}",
            "",
            self.prereg.render(),
        ]

    def run(self, test: str) -> SelectiveResult:
        if test not in ("floor", "added-value"):
            raise ValueError(f"teste desconhecido: {test!r}")
        tabela = self._load_table()
        cal, aval = self._split_by_sentence(tabela)
        notas: list[str] = list(self.pending_notes)
        self._notas_estrato = []
        if self.max_samples > 0:
            notas.append(
                f"execução limitada a max_samples = {self.max_samples}; "
                f"os intervalos refletem essa amostra e não o corpus inteiro"
            )

        perdas_aval = aval["loss"]
        base = random_abstention_risk(perdas_aval)

        if test == "floor":
            escore_a = np.full_like(perdas_aval, 0.5)  # AURC = taxa de erro base
            escore_b = aval["span_mass"]
            rotulos = ("abstenção aleatória", "massa de atenção")
            escore_cal = cal["span_mass"]
            escore_aval_ponto = escore_b
            aninhado = plano = None
        else:
            combinado_cal, combinado_aval = self._combine(cal, aval, notas)
            escore_a = aval["model_confidence"]
            escore_b = combinado_aval
            rotulos = ("confiança do modelo", "combinado")
            escore_cal = combinado_cal
            escore_aval_ponto = combinado_aval
            aninhado, plano = self._stratified_deltas(aval, escore_a, escore_b, rotulos)

        delta = delta_aurc_paired_bootstrap(
            perdas_aval,
            escore_a,
            escore_b,
            # Por SENTENÇA: entidades da mesma sentença compartilham a matriz de
            # atenção. Ver o docstring da função — sortear entidades produziria
            # intervalo estreito demais.
            groups=aval["sentence_id"],
            n_resamples=self.prereg.n_bootstrap_resamples,
            level=self.prereg.added_value_ci_level,
            seed=self.seed,
            labels=rotulos,
        )
        if delta.n_degenerate_resamples:
            notas.append(
                f"{delta.n_degenerate_resamples} reamostragens degeneradas descartadas "
                f"(todas as entidades sorteadas com a mesma perda)"
            )

        limiar = crc_threshold(
            cal["loss"], escore_cal, alpha=self.prereg.conformal_alpha,
            loss_bound=self.prereg.loss_bound,
        )
        if not limiar.monotone:
            notas.append(
                "risco empírico não monótono no limiar durante a calibração: o limiar "
                "devolvido não carrega a garantia formal do teorema"
            )
        curva = risk_coverage_curve(perdas_aval, escore_aval_ponto)

        return SelectiveResult(
            test=test,
            model_key=self.model_key,
            dataset_name=self.dataset_name,
            split=self.split,
            n_entities=int(tabela["loss"].size),
            n_sentences=int(np.unique(tabela["sentence_id"]).size),
            n_calibration_entities=int(cal["loss"].size),
            n_test_entities=int(aval["loss"].size),
            base_risk=base,
            prereg=self.prereg,
            delta=delta,
            threshold=limiar,
            # Cobertura OBTIDA ao risco alvo, e não risco a uma cobertura fixa:
            # Geifman & El-Yaniv (2017) fixam r* e maximizam a cobertura sujeita a
            # ele, e o controle conformal declara alpha com o mesmo papel.
            operating_point_coverage=curva.coverage_at_risk(self.prereg.target_risk),
            coverage_by_risk=tuple(
                (r, curva.coverage_at_risk(r)) for r in self.prereg.target_risk_grid
            ),
            nested_delta=aninhado,
            flat_delta=plano,
            notes=tuple(notas) + tuple(self._notas_estrato),
        )

    # ------------------------------------------------------------------ dados ---

    def _conferir_procedencia(self, tabela: Path) -> None:
        """A tabela tem de ter sido medida sob uma declaração ADMISSÍVEL.

        Admissível é a própria, ou aquela de quem esta declaração HERDA a medição
        — e a herança é declarada em `inherits_measurement_from`, não inferida.
        Sem essa conferência, trocar um item e rodar sobre a tabela antiga daria
        resultado sem erro nenhum, com metade dos parâmetros de cada declaração.

        A comparação é pelo hash de MEDIÇÃO quando o arquivo o traz, e pelo hash
        da declaração de origem quando não (medições anteriores a 03/09/2026 só
        gravavam este). Ausência de qualquer um dos dois vira ressalva no
        relatório, não recusa: o resultado sai, com a procedência declarada
        incompleta.
        """
        proc = tabela.parent / "MEDIDA.json"
        if not proc.is_file():
            self.pending_notes.append(
                f"procedência NÃO verificada: {proc.name} não existe ao lado da tabela."
            )
            return
        try:
            dados = json.loads(proc.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            self.pending_notes.append(f"procedência NÃO verificada: {proc.name} ilegível ({e}).")
            return

        # A chave é `preregistro`, que é como measurement.py a escreve. Ela já
        # esteve como "preregistration" aqui e o guarda ficou INERTE: caía na
        # ressalva de procedência incompleta em vez de conferir coisa alguma.
        bloco = dados.get("preregistro") or dados.get("preregistration") or dados
        # QUAL MODELO produziu a tabela. Conferência separada do hash porque as
        # versões 1 a 3 do conjunto de itens não cobrem o modelo: para elas o
        # bloco `modelo` é o único registro, e exigir casamento seria exigir de
        # uma declaração algo que ela não declara.
        modelo_gravado = (dados.get("modelo") or {}).get("id")
        if self.prereg.model is not None:
            if modelo_gravado is None:
                raise EntityTableError(
                    f"{self.prereg.declaration_id} declara o modelo "
                    f"{self.prereg.model!r}, e {proc.name} não grava qual modelo produziu a "
                    f"tabela. Tabela anterior ao registro de modelo não serve a uma declaração "
                    f"que fixa modelo: é justamente a confusão entre escalas que a versão 4 "
                    f"do hash existe para impedir."
                )
            if modelo_gravado != self.prereg.model:
                raise EntityTableError(
                    f"tabela produzida por {modelo_gravado!r}, mas está sendo testada sob "
                    f"{self.prereg.declaration_id}, que declara {self.prereg.model!r}. "
                    f"Modelos diferentes produzem números diferentes DENTRO da tabela."
                )
        elif modelo_gravado is not None:
            self.pending_notes.append(
                f"a declaração {self.prereg.declaration_id} é da versão "
                f"{self.prereg.hash_version} e NÃO declara modelo; a tabela foi produzida por "
                f"{modelo_gravado!r}, registrado mas não conferido pelo hash."
            )

        med = bloco.get("measurement_hash")
        decl = bloco.get("declaration_hash")
        ident = bloco.get("declaration_id")

        if med is None and decl is None:
            self.pending_notes.append(
                f"procedência NÃO verificada: {proc.name} não grava measurement_hash "
                f"nem declaration_hash."
            )
            return

        herda = self.prereg.inherits_measurement_from
        if med is not None:
            if med != self.prereg.measurement_hash:
                raise EntityTableError(
                    f"tabela medida com parâmetros de medição {med}, mas está sendo testada "
                    f"sob {self.prereg.measurement_hash} ({self.prereg.declaration_id}). "
                    f"Sumidouro, camadas ou cabeças diferem: a tabela não serve."
                )
            if decl is not None and decl != self.prereg.declaration_hash:
                origem = f"{ident or '?'} ({decl})"
                if herda and decl == herda[1]:
                    self.pending_notes.append(
                        f"medição herdada de {origem}, conforme declarado em "
                        f"inherits_measurement_from — parâmetros de medição idênticos."
                    )
                else:
                    raise EntityTableError(
                        f"tabela medida sob {origem} e testada sob "
                        f"{self.prereg.declaration_id} ({self.prereg.declaration_hash}), "
                        f"e esta declaração não herda daquela. Declare "
                        f"inherits_measurement_from ou remeça."
                    )
            return

        # Só o hash da declaração: aceita se for o próprio ou o declarado como origem.
        if decl == self.prereg.declaration_hash:
            return
        if herda and decl == herda[1]:
            self.pending_notes.append(
                f"medição herdada de {herda[0]} ({decl}), conforme declarado. A tabela não "
                f"grava measurement_hash (medida antes de 03/09/2026), então a igualdade dos "
                f"parâmetros de medição vem da DECLARAÇÃO e não do arquivo."
            )
            return
        raise EntityTableError(
            f"tabela medida sob {ident or '?'} ({decl}), sendo rodada sob "
            f"{self.prereg.declaration_id} ({self.prereg.declaration_hash}). "
            f"Trocar de declaração exige remedir, ou declarar inherits_measurement_from."
        )

    def _load_table(self) -> dict[str, np.ndarray]:
        import csv

        caminho = self.table_path
        if not caminho.is_file():
            raise EntityTableError(
                f"tabela de entidades ausente: {caminho}\n"
                f"Os dois testes são funções de uma tabela por entidade — este módulo não "
                f"carrega modelo por desenho (ver o docstring). Produza a tabela com a "
                f"passagem de inferência, com as colunas: {', '.join(COLUNAS)}."
            )
        self._conferir_procedencia(caminho)

        with caminho.open(encoding="utf-8", newline="") as fh:
            linhas = list(csv.DictReader(fh))
        if not linhas:
            raise EntityTableError(f"{caminho} está vazia")
        faltando = [c for c in COLUNAS if c not in linhas[0]]
        if faltando:
            raise EntityTableError(
                f"{caminho} não tem as colunas {faltando}; presentes: {list(linhas[0])}"
            )
        if self.max_samples > 0:
            sentencas = []
            vistas: set[str] = set()
            for l in linhas:
                if l["sentence_id"] not in vistas:
                    vistas.add(l["sentence_id"])
                    sentencas.append(l["sentence_id"])
                if len(vistas) >= self.max_samples:
                    break
            linhas = [l for l in linhas if l["sentence_id"] in vistas]
        return {
            "sentence_id": np.array([l["sentence_id"] for l in linhas]),
            "loss": np.array([float(l["loss"]) for l in linhas]),
            "model_confidence": np.array([float(l["model_confidence"]) for l in linhas]),
            "span_mass": np.array([float(l["span_mass"]) for l in linhas]),
            "is_nested": np.array([float(l["is_nested"]) for l in linhas]),
        }

    def _split_by_sentence(
        self, t: dict[str, np.ndarray]
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        sentencas = np.unique(t["sentence_id"])
        if sentencas.size < 2:
            raise EntityTableError(
                f"{sentencas.size} sentença(s): não há como separar calibração de avaliação"
            )
        rng = np.random.default_rng(self.seed)
        ordem = rng.permutation(sentencas)
        n_cal = max(1, int(round(self.prereg.calibration_fraction * sentencas.size)))
        n_cal = min(n_cal, sentencas.size - 1)
        cal_ids = set(ordem[:n_cal].tolist())
        mascara = np.array([s in cal_ids for s in t["sentence_id"]])
        if mascara.all() or (~mascara).all():
            raise EntityTableError("a separação por sentença deixou uma das partições vazia")
        return ({k: v[mascara] for k, v in t.items()}, {k: v[~mascara] for k, v in t.items()})

    # ------------------------------------------------------------ combinação ---

    def _combine(
        self,
        cal: dict[str, np.ndarray],
        aval: dict[str, np.ndarray],
        notas: list[str],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Combina confiança e massa pela regra do item 3, AJUSTADA na calibração.

        Ajustar na calibração é o que torna a combinação honesta: os coeficientes
        não podem ser escolhidos olhando a partição de avaliação, senão o ganho
        medido inclui o próprio ajuste.
        """
        regra = self.prereg.combination_rule
        X_cal = np.column_stack([cal["model_confidence"], cal["span_mass"]])
        X_aval = np.column_stack([aval["model_confidence"], aval["span_mass"]])

        if regra == "product":
            return X_cal.prod(axis=1), X_aval.prod(axis=1)

        if regra == "rank_average":
            def rank(x: np.ndarray) -> np.ndarray:
                ordem = np.argsort(np.argsort(x))
                return ordem / max(1, x.size - 1)

            return (
                (rank(X_cal[:, 0]) + rank(X_cal[:, 1])) / 2,
                (rank(X_aval[:, 0]) + rank(X_aval[:, 1])) / 2,
            )

        if regra == "convex":
            # Combinação CONVEXA com UM peso, ajustado na calibração:
            #     s(w) = w * confiança + (1 - w) * massa
            # É o precedente citável (Zheng et al. 2026), e a razão de ser um
            # peso só não é economia — é legibilidade do resultado. Com w em
            # [0, 1], "a combinação acrescenta algo?" se lê direto: w perto de 0
            # ou de 1 é a combinação COLAPSANDO para um sinal isolado. Foi o que
            # aconteceu com eles (o segundo sinal deu AUROC 0,49-0,54, mal acima
            # do acaso), e é por isso que o peso ajustado é reportado.
            #
            # Os dois sinais são padronizados na calibração antes de combinar,
            # porque suas escalas não são comparáveis: a confiança do GLiNER
            # satura perto de 1 e a massa vive perto de 0,1. Sem padronizar, o
            # peso mediria diferença de escala e não de informação. Média e
            # desvio vêm da CALIBRAÇÃO e são aplicados à avaliação — usar os da
            # avaliação vazaria a partição de teste para dentro do escore.
            mu, sd = X_cal.mean(axis=0), X_cal.std(axis=0)
            sd = np.where(sd > 0, sd, 1.0)
            Z_cal, Z_aval = (X_cal - mu) / sd, (X_aval - mu) / sd

            grade = np.linspace(0.0, 1.0, 101)
            perdas_cal = cal["loss"].astype(float)
            melhor_w, melhor_aurc = 0.0, np.inf
            for w in grade:
                s = w * Z_cal[:, 0] + (1.0 - w) * Z_cal[:, 1]
                a = risk_coverage_curve(perdas_cal, s).aurc
                if a < melhor_aurc:
                    melhor_aurc, melhor_w = float(a), float(w)
            notas.append(
                f"combinação convexa: peso {melhor_w:.2f} na confiança do modelo e "
                f"{1 - melhor_w:.2f} na massa de atenção, ajustado na calibração "
                f"(AURC {melhor_aurc:.4f}). Peso em 0,00 ou 1,00 é a combinação "
                f"colapsando para um sinal isolado."
            )
            return (
                melhor_w * Z_cal[:, 0] + (1.0 - melhor_w) * Z_cal[:, 1],
                melhor_w * Z_aval[:, 0] + (1.0 - melhor_w) * Z_aval[:, 1],
            )

        raise ValueError(f"regra de combinação desconhecida: {regra!r}")

    def _stratified_deltas(
        self,
        aval: dict[str, np.ndarray],
        escore_a: np.ndarray,
        escore_b: np.ndarray,
        rotulos: tuple[str, str],
    ) -> tuple[DeltaAURC | None, DeltaAURC | None]:
        saida: list[DeltaAURC | None] = []
        for alvo in (1.0, 0.0):
            m = aval["is_nested"] == alvo
            if m.sum() < 10:
                saida.append(None)
                continue
            try:
                saida.append(
                    delta_aurc_paired_bootstrap(
                        aval["loss"][m], escore_a[m], escore_b[m],
                        groups=aval["sentence_id"][m],
                        n_resamples=self.prereg.n_bootstrap_resamples,
                        level=self.prereg.added_value_ci_level,
                        seed=self.seed,
                        labels=rotulos,
                    )
                )
            except RuntimeError as e:
                # Estrato pequeno ou de perda constante: TODAS as reamostragens saem
                # degeneradas e não há intervalo a formar. Isso é um estrato NÃO
                # AVALIADO, não um resultado nulo, e a distinção é a coisa toda: um
                # corpus plano (CoNLL-2003 tem zero entidades aninhadas na anotação)
                # produz um punhado de predições no estrato aninhado, e reportar
                # "sem diferença" ali seria afirmar ausência a partir de amostra
                # que não permite afirmar nada.
                #
                # Antes desta correção a exceção subia e derrubava o teste inteiro,
                # levando junto o estrato plano, que tinha n = 5.677 e era
                # perfeitamente avaliável.
                self._notas_estrato.append(
                    f"estrato {'aninhado' if alvo else 'plano'} NÃO AVALIADO "
                    f"(n = {int(m.sum())}): {e}"
                )
                saida.append(None)
        return saida[0], saida[1]
