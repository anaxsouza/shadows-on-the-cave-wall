"""A passagem de medição: produz a tabela por entidade e guarda a atenção.

É o único item no caminho crítico dos dois artigos (`docs/tese/PLANO_PAPERS.md`).
Nada em `src/selective/` roda antes dela existir, porque os dois testes da
pergunta única são funções da tabela que ela emite.

O QUE ELA EMITE

1. `results/<modelo>/<corpus>/<split>/entities.csv` — uma linha por entidade
   PREDITA, com as cinco colunas que `runner.py` consome. Predita e não anotada,
   e a distinção é o desenho: predição seletiva decide o que ENTREGAR, então a
   unidade é o que o modelo produziu, e `loss` é o acerto dessa produção contra
   a anotação. Uma entidade de ouro que o modelo não encontrou não é candidata a
   abstenção — é um falso negativo, e ele entra na medida de desempenho, não na
   curva risco-cobertura.

2. `results/<modelo>/<corpus>/<split>/attention/shard_NNN.npz` — o tensor de
   atenção completo, todas as camadas e todas as cabeças, `float16`, em fatias
   de `shard_size` sentenças. Decisão do autor em 02/09/2026, com o custo
   conhecido: ~7,3 GB por 10 mil sentenças a 32 tokens, para 24 camadas e 16
   cabeças.

A FRICÇÃO QUE O TENSOR COMPLETO REMOVE, E COMO ELA VOLTA

Guardar todas as camadas permite reescolher a faixa depois de ver o resultado, e
a faixa é o item 2 do pré-registro. Ter o dado não viola nada; o que ele remove
é o atrito que protege a escolha declarada. Duas coisas devolvem esse atrito:

- a faixa usada no cálculo vem do pré-registro carregado, nunca de argumento
  desta função — quem quiser outra faixa tem de editar `configs/config.yaml`, o
  que é mudança versionada e visível;
- o `entities.csv` grava a faixa que produziu cada `span_mass` em
  `MEDIDA.json`, ao lado, de modo que uma tabela medida com uma faixa e citada
  como outra não passa em silêncio.

O QUE ELA NÃO FAZ

Não carrega dado de rede por conta própria e não decide orçamento de disco: o
pré-cálculo de tamanho roda ANTES da primeira escrita e recusa a execução se o
projetado exceder `max_disk_gb`. Encher o disco de quem roda não é falha
aceitável de uma passagem que leva horas.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from .attention_mass import span_attention_mass
from .geometry import n_allowed_keys
from .preregistration import Preregistration

# Colunas do contrato consumido por runner.py::_load_table. A ordem é a do
# arquivo e está fixada em teste: mudança silenciosa aqui quebra o runner longe
# daqui.
# `token_indices` foi acrescentada em 02/09/2026 e a razão é um defeito que só
# apareceu ao tentar usar o que estava guardado: o tensor de atenção completo é
# guardado para NÃO ser preciso repetir a inferência, mas sem os índices de token
# da entidade não há como recompor a massa sob outra leitura — o propósito de
# guardar tudo ficava derrotado por uma coluna ausente. Formato "3|4|5", inteiros
# separados por barra, que sobrevive a CSV sem aspas nem escape.
# As seis colunas do contrato original, mais as dos SINAIS das três famílias.
#
# A ordem agrupa por família e não por tipo, e o motivo é de leitura: quem abrir
# o CSV tem de ver de qual família cada coluna é sem consultar outro arquivo. O
# estatuto do nulo de cada uma está em `signals.py` e é ele que manda.
COLUNAS = (
    "sentence_id", "loss", "model_confidence", "span_mass", "is_nested", "token_indices",
    # covariáveis geométricas: entram como colunas porque são o que a
    # desconfundição empírica regride, e porque `span_position` não é derivável
    # das outras duas.
    "span_size", "sentence_length", "span_position",
    # família de ATENÇÃO, nulo EXATO. Nada aqui é ajustado ao dado.
    "row_entropy", "row_max", "emitted_mass",
    # família de ESTADOS OCULTOS, nulo EMPÍRICO. Medidos na MESMA faixa de
    # camadas já declarada, então nenhum parâmetro de medição novo entra e o
    # `measurement_hash` não muda.
    "hidden_norm", "hidden_dist_centroide", "hidden_delta_camadas",
    # família de LOGITS, nulo EMPÍRICO. Os escores são sigmoide por par
    # (trecho, rótulo) e não somam 1 — a entropia é normalizada por log(n).
    "logit_margin", "logit_entropy", "logit_max",
)

# O perfil por camada sai em arquivo SEPARADO, e a separação é a proteção: a
# tabela que decide o veredito tem uma coluna de massa, medida na faixa
# declarada. Se as 12 massas por camada estivessem em entities.csv, escolher a
# melhor depois de ver o resultado seria uma linha de código.
COLUNAS_PERFIL = ("sentence_id", "entity_idx", "layer", "span_mass")

BYTES_FLOAT16 = 2


class MeasurementError(RuntimeError):
    """Falha da passagem de medição, com a causa dita."""


@dataclass
class DiskEstimate:
    """Projeção de disco, calculada antes de escrever o primeiro byte."""

    n_sentences: int
    n_layers: int
    n_heads: int
    mean_tokens: float
    max_tokens: int
    projected_bytes: int
    budget_bytes: int

    @property
    def projected_gb(self) -> float:
        return self.projected_bytes / 1024**3

    @property
    def fits(self) -> bool:
        return self.projected_bytes <= self.budget_bytes

    def render(self) -> str:
        return (
            f"Atenção a persistir: {self.n_layers} camadas x {self.n_heads} cabeças, "
            f"float16, {self.n_sentences} sentenças\n"
            f"  comprimento médio {self.mean_tokens:.1f} tokens, máximo {self.max_tokens}\n"
            f"  projetado {self.projected_gb:.2f} GB  |  orçamento "
            f"{self.budget_bytes / 1024**3:.2f} GB  |  "
            f"{'cabe' if self.fits else 'NÃO CABE'}"
        )


def estimate_disk(
    token_counts: Sequence[int],
    n_layers: int,
    n_heads: int,
    max_disk_gb: float,
) -> DiskEstimate:
    """Projeta o disco a partir dos comprimentos REAIS, não de uma média suposta.

    O custo é quadrático no comprimento, então a média dos T² é o que importa e
    não o quadrado da média — usar a segunda subestima, e subestimar disco é o
    erro que interrompe uma execução de horas no meio.
    """
    if not len(token_counts):
        raise MeasurementError("nenhuma sentença: nada a estimar")
    t = np.asarray(token_counts, dtype=np.int64)
    if (t <= 0).any():
        raise MeasurementError("comprimento de sentença não positivo na estimativa")
    total = int(n_layers) * int(n_heads) * int((t * t).sum()) * BYTES_FLOAT16
    return DiskEstimate(
        n_sentences=int(t.size),
        n_layers=int(n_layers),
        n_heads=int(n_heads),
        mean_tokens=float(t.mean()),
        max_tokens=int(t.max()),
        projected_bytes=total,
        budget_bytes=int(max_disk_gb * 1024**3),
    )


@dataclass
class EntityRow:
    """Uma entidade predita, medida."""

    sentence_id: str
    loss: int
    model_confidence: float
    span_mass: float
    is_nested: int
    token_indices: str
    span_size: int = 0
    sentence_length: int = 0
    span_position: int = 0
    row_entropy: float = float("nan")
    row_max: float = float("nan")
    emitted_mass: float = float("nan")
    hidden_norm: float = float("nan")
    hidden_dist_centroide: float = float("nan")
    hidden_delta_camadas: float = float("nan")
    logit_margin: float = float("nan")
    logit_entropy: float = float("nan")
    logit_max: float = float("nan")

    def as_dict(self) -> dict[str, Any]:
        """Uma linha do CSV, na ordem de `COLUNAS`.

        Gerada por iteração sobre `COLUNAS` e não escrita à mão: com dezoito
        colunas, uma lista digitada divergiria da tupla no primeiro sinal novo, e
        a divergência apareceria como coluna trocada em vez de erro.
        """
        fora: dict[str, Any] = {}
        for nome in COLUNAS:
            v = getattr(self, nome)
            fora[nome] = round(v, 6) if isinstance(v, float) else v
        return fora


def _sinais_do_span(
    *,
    attn: "np.ndarray",
    ocultos: Any,
    idx: Sequence[int],
    label_scores: Sequence[float],
    prereg: Preregistration,
    sid: str,
    notes: list[str],
) -> dict[str, Any]:
    """Os sinais das três famílias, para uma entidade predita.

    TRÊS DECISÕES DE MÉTODO FICAM AQUI, e nenhuma delas é escolha livre:

    1. Os estados ocultos usam a MESMA faixa de camadas já declarada em
       `prereg.layers`. Isso não é conveniência — é o que impede a família de
       estados ocultos de trazer um parâmetro de medição novo. Se ela tivesse
       faixa própria, o `measurement_hash` mudaria, as tabelas já medidas
       deixariam de ser comparáveis, e escolher a faixa depois de ver o
       resultado voltaria a ser uma linha de código.
    2. A entropia da linha e o máximo da linha são médias sobre as CONSULTAS do
       span, com o denominador da política de sumidouro declarada. São os dois
       sinais cujo nulo é exato (`log|K|` e `1/|K|`).
    3. A entropia dos escores de rótulo é normalizada por `log(n_rótulos)`. Os
       escores são sigmoide por par (trecho, rótulo) e não somam 1, então a
       entropia bruta não é comparável entre corpora com números diferentes de
       rótulo — o GENIA tem 5 e o CoNLL 4.
    """
    import numpy as _np

    K = n_allowed_keys(attn.shape[-1], sink_policy=prereg.sink_policy)
    T = int(attn.shape[-1])
    fora: dict[str, Any] = {
        "span_size": len(idx),
        "sentence_length": T,
        "span_position": int(min(idx)),
    }

    # --- família de ATENÇÃO, nulo exato --------------------------------------
    camadas = list(prereg.layers)
    cabecas = list(prereg.heads) if prereg.heads else list(range(attn.shape[1]))
    bloco = attn[_np.ix_(camadas, cabecas)].astype(float)      # [L, H, T, T]
    media = bloco.mean(axis=(0, 1))                            # [T, T]
    permitidas = _np.arange(1, T) if prereg.sink_policy == "drop_from_denominator" else _np.arange(T)
    linhas_span = media[_np.asarray(idx)][:, permitidas]
    soma = linhas_span.sum(axis=1, keepdims=True)
    renorm = linhas_span / _np.where(soma > 0, soma, 1.0)
    with _np.errstate(divide="ignore", invalid="ignore"):
        termo = _np.where(renorm > 0, renorm * _np.log(renorm), 0.0)
    fora["row_entropy"] = float((-termo.sum(axis=1)).mean())
    fora["row_max"] = float(renorm.max(axis=1).mean())
    # A massa que o span EMITE para fora dele: complemento da recebida dentro do
    # próprio bloco de consultas do span.
    dentro = [j for j in idx if j in set(permitidas.tolist())]
    fora["emitted_mass"] = float(1.0 - renorm[:, [list(permitidas).index(j) for j in dentro]].sum(axis=1).mean()) if dentro else 1.0

    # --- família de ESTADOS OCULTOS, nulo empírico ---------------------------
    if ocultos is None:
        fora["hidden_norm"] = float("nan")
        fora["hidden_dist_centroide"] = float("nan")
        fora["hidden_delta_camadas"] = float("nan")
        # A frase "sem estados ocultos" é a CHAVE de deduplicação e aparece na
        # própria mensagem. Guarda e mensagem divergentes fizeram a ressalva ser
        # repetida uma vez por sentença — defeito apanhado pelos testes, e do
        # tipo que enche o relatório e esconde as outras ressalvas.
        if not any("SEM ESTADOS OCULTOS" in n.upper() for n in notes):
            notes.append(
                "medição SEM ESTADOS OCULTOS: o adaptador não os devolveu, então a família "
                "correspondente sai como NaN e as comparações que dependem dela ficam sem "
                "dado. Nenhum valor foi inventado.")
    else:
        # `hidden_states` traz L+1 tensores (embeddings + uma por camada), então
        # a camada `c` de `prereg.layers` é o índice `c + 1`.
        H = _np.stack([_np.asarray(ocultos[c + 1][0], dtype=float) for c in camadas])  # [L, T, d]
        span = H[:, _np.asarray(idx), :]                       # [L, k, d]
        fora["hidden_norm"] = float(_np.linalg.norm(span, axis=-1).mean())
        centroide = H.mean(axis=1, keepdims=True)              # [L, 1, d]
        fora["hidden_dist_centroide"] = float(
            _np.linalg.norm(span.mean(axis=1, keepdims=True) - centroide, axis=-1).mean())
        fora["hidden_delta_camadas"] = float(
            _np.linalg.norm(span[-1].mean(axis=0) - span[0].mean(axis=0)))

    # --- família de LOGITS, nulo empírico ------------------------------------
    s = _np.asarray(sorted(label_scores, reverse=True), dtype=float)
    if s.size == 0:
        fora["logit_margin"] = float("nan")
        fora["logit_entropy"] = float("nan")
        fora["logit_max"] = float("nan")
        notes.append(f"{sid}: trecho sem distribuição de escores de rótulo (casamento por "
                     f"posição falhou); família de logits sai como NaN nesta entidade")
    else:
        fora["logit_max"] = float(s[0])
        fora["logit_margin"] = float(s[0] - s[1]) if s.size > 1 else float(s[0])
        total = s.sum()
        if total > 0 and s.size > 1:
            p = s / total
            with _np.errstate(divide="ignore", invalid="ignore"):
                h = -_np.where(p > 0, p * _np.log(p), 0.0).sum()
            fora["logit_entropy"] = float(h / _np.log(s.size))
        else:
            fora["logit_entropy"] = 0.0
    return fora


@dataclass
class MeasurementReport:
    output_dir: Path
    n_sentences: int
    n_predicted: int
    n_correct: int
    n_gold: int
    n_shards: int
    bytes_written: int
    n_profile_rows: int
    estimate: DiskEstimate
    prereg_layers: tuple[int, ...]
    sink_policy: str
    notes: list[str] = field(default_factory=list)

    @property
    def error_rate(self) -> float:
        """Taxa de erro entre as PREDITAS — é a horizontal do acaso da curva."""
        if not self.n_predicted:
            return float("nan")
        return 1.0 - self.n_correct / self.n_predicted

    def render(self) -> str:
        recall = f"{self.n_correct / self.n_gold:.3f}" if self.n_gold else "n/d"
        linhas = [
            f"Medição escrita em {self.output_dir}",
            f"  sentenças            {self.n_sentences}",
            f"  entidades preditas   {self.n_predicted}  (ouro: {self.n_gold})",
            f"  taxa de erro base    {self.error_rate:.4f}   <- a horizontal do acaso",
            f"  recuperação          {recall}   (não entra na curva; é desempenho)",
            f"  atenção              {self.n_shards} fatias, "
            f"{self.bytes_written / 1024**3:.2f} GB",
            f"  faixa de camadas     {list(self.prereg_layers)}  (do pré-registro)",
            f"  sumidouro            {self.sink_policy}  (do pré-registro)",
        ]
        if self.n_profile_rows:
            linhas.append(
                f"  perfil por camada    {self.n_profile_rows} linhas  <- EXPLORATÓRIO: "
                f"descreve onde o sinal mora, não decide veredito"
            )
        if self.notes:
            linhas += ["", "Ressalvas desta execução:"] + [f"  - {n}" for n in self.notes]
        return "\n".join(linhas)


def _stack_attention(attentions: Iterable[Any]) -> np.ndarray:
    """Empilha a saída do modelo em [camadas, cabeças, T, T], float16.

    A saída de um transformer é uma tupla de tensores [lote, cabeças, T, T], uma
    por camada. Aqui o lote é sempre 1: sentenças têm comprimentos diferentes e
    lote com preenchimento colocaria massa de atenção sobre token de
    preenchimento, o que contaminaria o denominador da proporção. Perder o lote
    custa tempo; contaminar o denominador custa o resultado.
    """
    camadas = []
    for a in attentions:
        arr = a.detach().cpu().numpy() if hasattr(a, "detach") else np.asarray(a)
        if arr.ndim == 4:
            if arr.shape[0] != 1:
                raise MeasurementError(
                    f"lote {arr.shape[0]} != 1: a passagem é por sentença, ver docstring"
                )
            arr = arr[0]
        if arr.ndim != 3:
            raise MeasurementError(f"atenção com forma inesperada {arr.shape}")
        camadas.append(arr.astype(np.float16))
    if not camadas:
        raise MeasurementError(
            "o modelo não devolveu atenção; carregue com output_attentions=True"
        )
    return np.stack(camadas, axis=0)


def measure(
    *,
    sentences: Sequence[Any],
    predict: Any,
    prereg: Preregistration,
    output_dir: Path | str,
    model_id: str,
    max_disk_gb: float = 20.0,
    shard_size: int = 500,
    max_samples: int = -1,
) -> MeasurementReport:
    """Roda a passagem e escreve a tabela e as fatias de atenção.

    `predict(sentence)` é o adaptador para o modelo e devolve, por sentença:
      tokens            list[str]
      attentions        tupla por camada de tensores [1, cabeças, T, T]
      predicted         list[dict] com `token_indices`, `confidence`, `correct`
                        e `is_nested`
      n_gold            int, quantidade de entidades anotadas na sentença

    `model_id` é OBRIGATÓRIO e sem valor padrão. Até 15/09/2026 o MEDIDA.json não
    registrava qual modelo produziu a tabela: o único traço era o nome do
    diretório, e nome de diretório não é procedência. Com um modelo só isso não
    machucava; no momento em que existe um segundo, duas tabelas de escalas
    diferentes carregariam o mesmo measurement_hash. Um valor padrão aqui
    deixaria o registro silenciosamente vazio, que é exatamente o defeito.

    Manter o modelo atrás de `predict` é o que torna esta função testável sem
    rede e sem GPU: os testes passam um adaptador determinístico. Não é
    abstração por gosto — é a diferença entre um teste de 20 ms e um que baixa
    1,3 GB.
    """
    saida = Path(output_dir)
    attn_dir = saida / "attention"
    if max_samples is not None and max_samples > 0:
        sentences = list(sentences)[:max_samples]
    if not len(sentences):
        raise MeasurementError("nenhuma sentença a medir")

    # --- pré-voo: mede antes de escrever ------------------------------------
    # Uma passada de tokenização é barata perto de horas de inferência, e é o
    # que permite recusar por orçamento ANTES de gastar as horas.
    prevoo = [predict(s, tokens_only=True) for s in sentences]
    comprimentos = [len(p["tokens"]) for p in prevoo]
    n_layers = int(prevoo[0]["n_layers"])
    n_heads = int(prevoo[0]["n_heads"])
    est = estimate_disk(comprimentos, n_layers, n_heads, max_disk_gb)
    if not est.fits:
        raise MeasurementError(
            f"{est.render()}\n\nA execução foi recusada ANTES de escrever qualquer byte. "
            f"Aumente max_disk_gb se o disco comporta, ou reduza com max_samples."
        )

    attn_dir.mkdir(parents=True, exist_ok=True)
    linhas: list[EntityRow] = []
    fatia: dict[str, np.ndarray] = {}
    n_shards = bytes_written = n_pred = n_ok = n_gold = 0
    notes: list[str] = []

    def descarrega() -> None:
        nonlocal fatia, n_shards, bytes_written
        if not fatia:
            return
        caminho = attn_dir / f"shard_{n_shards:03d}.npz"
        np.savez_compressed(caminho, **fatia)
        bytes_written += caminho.stat().st_size
        n_shards += 1
        fatia = {}

    perfil: list[dict[str, Any]] = []
    for i, sent in enumerate(sentences):
        sid = str(getattr(sent, "sentence_id", None) or f"s{i}")
        r = predict(sent)
        attn = _stack_attention(r["attentions"])
        fatia[sid] = attn
        # Ausente quando o adaptador é antigo ou sintético: a família de estados
        # ocultos sai como NaN e a ressalva é registrada, em vez de o valor ser
        # inventado ou a execução morrer.
        ocultos = r.get("hidden_states")
        n_gold += int(r.get("n_gold", 0))

        for k, ent in enumerate(r["predicted"]):
            idx = list(ent["token_indices"])
            if not idx:
                notes.append(f"{sid}: entidade predita sem token mapeado, descartada")
                continue
            massa = span_attention_mass(
                attn,
                idx,
                sink_policy=prereg.sink_policy,
                layers=prereg.layers,
                heads=prereg.heads,
            )
            if prereg.layer_profile:
                # Uma linha por camada: descreve onde o sinal mora, sem tocar no
                # número que decide o veredito.
                for c in range(attn.shape[0]):
                    m_c = span_attention_mass(
                        attn, idx, sink_policy=prereg.sink_policy,
                        layers=[c], heads=prereg.heads,
                    )
                    perfil.append(
                        {"sentence_id": sid, "entity_idx": k, "layer": c,
                         "span_mass": round(float(m_c.mass), 6)}
                    )
            n_pred += 1
            acerto = bool(ent["correct"])
            n_ok += int(acerto)
            sinais = _sinais_do_span(
                attn=attn,
                ocultos=ocultos,
                idx=idx,
                label_scores=ent.get("label_scores") or [],
                prereg=prereg,
                sid=sid,
                notes=notes,
            )
            linhas.append(
                EntityRow(
                    sentence_id=sid,
                    loss=0 if acerto else 1,
                    model_confidence=float(ent["confidence"]),
                    span_mass=float(massa.mass),
                    is_nested=int(bool(ent["is_nested"])),
                    token_indices="|".join(str(int(v)) for v in idx),
                    **sinais,
                )
            )
        if len(fatia) >= shard_size:
            descarrega()
    descarrega()

    if not linhas:
        raise MeasurementError(
            "nenhuma entidade predita em nenhuma sentença: a tabela sairia vazia e "
            "os dois testes não teriam insumo. Verifique o modelo e o mapeamento."
        )

    tabela = saida / "entities.csv"
    with tabela.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(COLUNAS))
        w.writeheader()
        w.writerows(l.as_dict() for l in linhas)

    if perfil:
        with (saida / "layer_profile.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(COLUNAS_PERFIL))
            w.writeheader()
            w.writerows(perfil)

    rel = MeasurementReport(
        output_dir=saida,
        n_sentences=len(sentences),
        n_predicted=n_pred,
        n_correct=n_ok,
        n_gold=n_gold,
        n_shards=n_shards,
        bytes_written=bytes_written,
        n_profile_rows=len(perfil),
        estimate=est,
        prereg_layers=tuple(prereg.layers),
        sink_policy=prereg.sink_policy,
        notes=notes,
    )

    # Procedência ao lado da tabela: uma medida citada com faixa diferente da que
    # a produziu é erro que não se detecta olhando o CSV.
    (saida / "MEDIDA.json").write_text(
        json.dumps(
            {
                # QUEM produziu a tabela. Primeira coisa do registro porque é a
                # primeira pergunta de quem compara duas medições.
                "modelo": {
                    "id": model_id,
                    "camadas": n_layers,
                    "cabecas": n_heads,
                },
                "colunas": list(COLUNAS),
                "n_sentencas": rel.n_sentences,
                "n_entidades_preditas": rel.n_predicted,
                "taxa_de_erro_base": rel.error_rate,
                "preregistro": {
                    "layers": list(prereg.layers),
                    "heads": list(prereg.heads) if prereg.heads else None,
                    "sink_policy": prereg.sink_policy,
                    "declaration_id": prereg.declaration_id,
                    "declaration_hash": prereg.declaration_hash,
                    # Hash só dos parâmetros que PRODUZEM a tabela. É por ele que
                    # uma declaração de análise nova reusa esta medição.
                    "measurement_hash": prereg.measurement_hash,
                    "source": str(prereg.source),
                    # O modelo declarado, quando a versão do conjunto de itens o
                    # cobre. Em versão anterior fica None, e a ausência é
                    # informação: aquela declaração não fixa modelo.
                    "model": prereg.model,
                },
                "atencao": {
                    "camadas": n_layers,
                    "cabecas": n_heads,
                    "dtype": "float16",
                    "fatias": n_shards,
                    "bytes": bytes_written,
                },
                "perfil_por_camada": {
                    "linhas": len(perfil),
                    "natureza": "exploratorio",
                    "nao_decide_veredito": True,
                },
                "ressalvas": notes,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return rel
