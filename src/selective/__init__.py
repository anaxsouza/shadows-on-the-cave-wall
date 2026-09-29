"""Predição seletiva de NER: o que a tese contribui.

Este pacote é a contribuição, não infraestrutura. Ele responde uma pergunta:

    A massa de atenção sobre o span adiciona poder de decisão sobre a confiança
    que o próprio modelo já fornece, permitindo abstenção por entidade — e o
    ganho é maior em entidades aninhadas que em planas?

Quatro módulos, e a divisão é por responsabilidade e não por conveniência:

- `attention_mass`: o instrumento. Proporção da massa de atenção sobre os tokens
  do span, razão adimensional. Só depende de numpy, para ser testável sem modelo.
- `risk_coverage`: a medida. Curva risco-cobertura, AURC e a diferença de AURC
  com intervalo por reamostragem pareada.
- `conformal`: a garantia. Controle conformal de risco, que escolhe o ponto
  operacional único com risco esperado controlado em alpha.
- `preregistration`: o que não pode ser escolhido depois de ver o resultado.

Antes da reorganização, nenhuma linha disto existia no repositório: a busca por
`risk_coverage`, `aurc`, `abstention`, `selective` e `conformal` em `src/`
retornava zero arquivos (docs/tese/reorg/corte.md).
"""

from .attention_mass import SpanAttentionMass, span_attention_mass
from .measurement import (
    COLUNAS,
    DiskEstimate,
    EntityRow,
    MeasurementError,
    MeasurementReport,
    estimate_disk,
    measure,
)
from .decoder_adapter import (
    DecoderAdapter,
    DecoderAdapterError,
    ancorar,
    parse_saida,
    prompt_de,
)
from .geometry import (
    GeometryError,
    causal_mass,
    enrichment,
    expected_mass,
    expected_mass_causal,
    geometric_residual,
    n_allowed_keys,
)
from .signals import (
    ATENCAO,
    ESTADOS_OCULTOS,
    FAMILIAS,
    LOGITS,
    SignalError,
    SignalSpec,
    TODOS as SINAIS,
    expected_row_entropy,
    expected_row_max,
    por_nome,
    residuo_empirico,
    row_entropy,
)
from .quality_grid import (
    QualityGridError,
    RelativeTarget,
    absolute_targets,
    base_precision,
    gap_closed,
)
from .conformal import CRCThreshold, crc_threshold
from .preregistration import Preregistration, PreregistrationError, load_preregistration
from .risk_coverage import (
    DeltaAURC,
    RiskCoverageCurve,
    aurc,
    delta_aurc_paired_bootstrap,
    random_abstention_risk,
    risk_coverage_curve,
)

__all__ = [
    "ATENCAO",
    "ESTADOS_OCULTOS",
    "FAMILIAS",
    "LOGITS",
    "QualityGridError",
    "RelativeTarget",
    "SINAIS",
    "SignalError",
    "SignalSpec",
    "absolute_targets",
    "base_precision",
    "expected_row_entropy",
    "expected_row_max",
    "gap_closed",
    "por_nome",
    "residuo_empirico",
    "row_entropy",
    "DecoderAdapter",
    "DecoderAdapterError",
    "GeometryError",
    "ancorar",
    "causal_mass",
    "parse_saida",
    "prompt_de",
    "enrichment",
    "expected_mass",
    "expected_mass_causal",
    "geometric_residual",
    "n_allowed_keys",
    "COLUNAS",
    "DiskEstimate",
    "EntityRow",
    "MeasurementError",
    "MeasurementReport",
    "estimate_disk",
    "measure",
    "SpanAttentionMass",
    "span_attention_mass",
    "CRCThreshold",
    "crc_threshold",
    "Preregistration",
    "PreregistrationError",
    "load_preregistration",
    "DeltaAURC",
    "RiskCoverageCurve",
    "aurc",
    "delta_aurc_paired_bootstrap",
    "random_abstention_risk",
    "risk_coverage_curve",
]
