"""A grade de qualidade RELATIVA à precisão de base do extrator.

POR QUE ESTE MÓDULO EXISTE

A grade de metas absolutas — 70%, 80%, 90%, 95% de precisão entre os entregues —
foi declarada quando o extrator errava perto de metade. Com o extrator ajustado
ao corpus ela ficou parcialmente vazia: o `gliner_base-ft-conll2003` já entrega
0,8997 de precisão SEM abster-se de nada, então as metas de 70% e 80% são
atingidas com carga de revisão zero e a comparação entre supervisores não
distingue nada. Não porque os supervisores sejam equivalentes — porque não há o
que supervisionar.

O defeito não é o valor das metas. É que uma meta ABSOLUTA mede a qualidade do
ARRANJO e não a pergunta: a mesma meta é trivial num extrator bom e inalcançável
num ruim, então nenhuma comparação entre modelos significa coisa alguma. Trocar
70/80/90/95 por outros quatro números repetiria o defeito no próximo extrator.

O CONSERTO

Declarar a fração do VÃO que a meta fecha, e não a precisão em si. O vão é a
distância entre a precisão que o extrator entrega sem abster-se e a perfeição:

    meta_absoluta = p_base + fracao * (1 - p_base)

Fechar 50% do vão exige abstenção em QUALQUER extrator que não seja perfeito, e
exige um esforço comparável em extratores de qualidade diferente. É isso que
torna a comparação entre modelos interpretável.

O PONTO DELICADO, E COMO ELE FICA FECHADO

A meta passa a depender de uma quantidade MEDIDA (`p_base`), o que parece abrir
grau de liberdade. Não abre, por duas razões, e as duas são verificáveis:

1. O que se declara é a FRAÇÃO, fixada antes de qualquer medição e dentro do
   hash. A meta absoluta que sai dela é determinada pelo dado, não escolhida —
   o mesmo estatuto do ponto operacional, que já é derivado do risco alvo neste
   projeto desde a decl-02.
2. `p_base` vem da partição de CALIBRAÇÃO, nunca da de avaliação. Derivar a meta
   da mesma partição que produz o veredito seria fixar o alvo olhando a resposta.
   A separação por sentença que já existe para a calibração serve aqui sem
   mudança.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class QualityGridError(ValueError):
    """Grade ou precisão de base que não permite derivar as metas."""


@dataclass(frozen=True)
class RelativeTarget:
    """Uma meta da grade relativa, com a conta que a produziu à vista."""

    gap_fraction: float
    base_precision: float
    absolute: float

    def render(self) -> str:
        return (f"fecha {self.gap_fraction:.0%} do vão: precisão de base "
                f"{self.base_precision:.4f} -> meta {self.absolute:.4f}")


def base_precision(loss: np.ndarray) -> float:
    """A precisão sem abstenção nenhuma: a fração de predições corretas.

    Recebe o vetor de perda 0/1 da partição de CALIBRAÇÃO. É o `p_base` da
    fórmula, e o ponto de encontro de todas as curvas risco-cobertura em
    cobertura total.
    """
    l = np.asarray(loss, dtype=float)
    if l.size == 0:
        raise QualityGridError("partição vazia: sem predições não há precisão de base")
    if not np.all((l == 0) | (l == 1)):
        raise QualityGridError("perda tem de ser 0/1 para virar precisão por contagem")
    return float((1.0 - l).mean())


def absolute_targets(gap_fractions, loss_calibration: np.ndarray) -> tuple[RelativeTarget, ...]:
    """As metas absolutas que a grade relativa determina nesta medição.

    A fração é declarada; a meta absoluta é DERIVADA. Fração fora de (0, 1) é
    recusada: fechar 0% do vão é não exigir nada, e fechar 100% é exigir precisão
    perfeita, que nenhum ponto de operação com cobertura positiva atinge num
    extrator que erra.
    """
    p = base_precision(loss_calibration)
    if not 0.0 < p < 1.0:
        raise QualityGridError(
            f"precisão de base {p:.4f} fora de (0, 1): com extrator perfeito ou "
            f"totalmente errado o vão não define meta")
    fora = []
    for f in gap_fractions:
        f = float(f)
        if not 0.0 < f < 1.0:
            raise QualityGridError(
                f"fração do vão tem de estar em (0, 1): recebida {f}. Fechar 0% não "
                f"exige nada e fechar 100% exige precisão perfeita")
        fora.append(RelativeTarget(gap_fraction=f, base_precision=p,
                                   absolute=p + f * (1.0 - p)))
    return tuple(fora)


def gap_closed(precision_achieved: float, base: float) -> float:
    """Quanto do vão um ponto de operação de fato fechou.

    É a leitura inversa, e serve ao relato: permite dizer "este supervisor fechou
    38% do vão" em vez de "atingiu precisão 0,85", que só é interpretável quem
    souber a precisão de base de cor.
    """
    if not 0.0 < base < 1.0:
        raise QualityGridError(f"precisão de base {base:.4f} fora de (0, 1)")
    return float((precision_achieved - base) / (1.0 - base))
