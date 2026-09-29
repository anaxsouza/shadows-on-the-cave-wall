"""O impacto na tarefa, sobre dado construído à mão.

Construído e não medido: aqui se verifica que a conta é a conta — maior cobertura
sujeita à meta, empate não partido, reamostragem por sentença, e "inalcançável"
como resultado e não como erro. Se a atenção funciona é outra pergunta, e ela se
responde no teste.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.selective.task_impact import (
    INALCANCAVEL,
    TaskImpactError,
    operating_point,
    paired_review_load_delta,
)


class TestPontoDeOperacao:
    def test_escore_perfeito_entrega_tudo_o_que_esta_correto(self):
        """Sem erro entre os entregues, a meta é atingida na cobertura máxima."""
        perdas = np.array([0.0, 0.0, 0.0, 1.0, 1.0])
        escore = np.array([0.9, 0.8, 0.7, 0.2, 0.1])   # alto = entregar
        p = operating_point(perdas, escore, 0.99, n_gold=10)
        assert p.reachable
        assert p.n_delivered == 3
        assert p.precision == pytest.approx(1.0)
        assert p.review_load == pytest.approx(2 / 5)
        assert p.recall == pytest.approx(3 / 10), "o denominador é o total ANOTADO"
        assert p.coverage == pytest.approx(3 / 5)

    def test_meta_inalcancavel_e_resultado_declarado_e_nao_erro(self):
        """Nenhuma cobertura atinge a meta: a linha diz isso, e não estoura."""
        perdas = np.array([1.0, 1.0, 1.0, 1.0])
        p = operating_point(perdas, np.array([0.9, 0.8, 0.7, 0.6]), 0.50, n_gold=8)
        assert p.reachable is False
        assert not np.isfinite(p.review_load)

    def test_pega_o_maior_prefixo_e_nao_o_primeiro_que_atinge(self):
        """Precisão não é monótona: parar no primeiro inflaria a carga de revisão.

        Aqui o prefixo de 1 tem precisão 1,0 e o de 5 também atinge 0,60. Entregar
        só 1 seria atribuir ao supervisor uma carga de revisão de 80% quando ele
        consegue 0%.
        """
        perdas = np.array([0.0, 1.0, 0.0, 0.0, 0.0])
        escore = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
        p = operating_point(perdas, escore, 0.60, n_gold=5)
        assert p.n_delivered == 5
        assert p.review_load == pytest.approx(0.0)

    def test_empate_nao_e_partido(self):
        """Cortar no meio de um grupo empatado reportaria cobertura irrealizável.

        Os três do meio têm o mesmo escore: o supervisor não sabe qual deles
        entregar primeiro, então 1, 2 ou 3 deles não é um ponto de operação.
        """
        perdas = np.array([0.0, 0.0, 1.0, 1.0, 1.0])
        escore = np.array([0.9, 0.5, 0.5, 0.5, 0.1])
        p = operating_point(perdas, escore, 0.90, n_gold=5)
        assert p.n_delivered == 1, "só o prefixo de 1 não parte o empate"

    def test_sem_total_anotado_recusa_em_vez_de_inventar_denominador(self):
        with pytest.raises(TaskImpactError, match="n_gold"):
            operating_point(np.array([0.0, 1.0]), np.array([1.0, 0.0]), 0.5, n_gold=0)


class TestDeltaDeCargaDeRevisao:
    @staticmethod
    def _dados(n_sent=40, por_sent=3, seed=1):
        """Um escore forte e um FRACO MAS INFORMATIVO, com estrutura de sentença.

        Fraco e não ruído puro, e a distinção foi apanhada pelos próprios testes:
        um escore de puro ruído ordena na taxa de erro base (~0,5) em qualquer
        cobertura, então NÃO atinge meta de 90% nenhuma e a comparação devolve
        "inalcançável" — correto, mas não exercita o intervalo. O caso real é
        este: a confiança do modelo informa, só informa menos.

        O dado tem EFEITO DE SENTENÇA, e isso também foi apanhado pelos testes:
        sem correlação dentro do conglomerado, reamostrar sentenças não alarga o
        intervalo — e está certo que não alargue. O conglomerado importa porque
        entidades da mesma sentença compartilham a matriz de atenção, logo acerto
        e escore são correlacionados dentro dela. Dado sem essa estrutura não
        exercita o que se quer verificar.
        """
        rng = np.random.default_rng(seed)
        n = n_sent * por_sent
        sent = np.repeat([f"s{i}" for i in range(n_sent)], por_sent)
        u = np.repeat(rng.normal(0, 1.2, size=n_sent), por_sent)   # efeito de sentença
        p_erro = 1.0 / (1.0 + np.exp(-u))
        perdas = (rng.random(n) < p_erro).astype(float)
        comum = 0.35 * u                                            # componente compartilhada
        bom = 1.0 - perdas + comum + rng.normal(0, 0.05, size=n)    # alto = entregar
        fraco = 1.0 - perdas + comum + rng.normal(0, 0.45, size=n)
        return perdas, bom, fraco, sent

    def test_o_escore_mais_informativo_exige_menos_revisao(self):
        perdas, bom, fraco, sent = self._dados()
        r = paired_review_load_delta(
            perdas, bom, fraco, sent, target=0.90, n_gold=int((1 - perdas).sum()),
            level=0.95, n_resamples=300, seed=7, labels=("forte", "fraco"))
        assert r.delta < 0, "carga menor para o escore informativo"
        assert r.ci_high < 0 and not r.contains_zero
        assert "MENOS revisão" in r.verdict

    def test_escore_contra_si_mesmo_nao_distingue(self):
        perdas, bom, _, sent = self._dados()
        r = paired_review_load_delta(
            perdas, bom, bom, sent, target=0.90, n_gold=int((1 - perdas).sum()),
            level=0.95, n_resamples=200, seed=3, labels=("a", "a"))
        assert r.delta == pytest.approx(0.0)
        assert r.contains_zero and "NÃO DISTINGUE" in r.verdict

    def test_uma_sentenca_so_da_intervalo_de_largura_ZERO(self):
        """O mecanismo do conglomerado, verificado de forma determinística.

        Com UMA sentença só, toda reamostragem de conglomerado sorteia a mesma
        sentença e devolve exatamente a mesma amostra — logo todo delta é
        idêntico e o intervalo tem largura zero. Se a reamostragem fosse por
        entidade, a largura seria positiva. É assinatura exata e não tendência.

        NOTA DE MÉTODO, registrada porque eu errei aqui primeiro: a versão
        anterior deste teste exigia que reamostrar sentenças ALARGASSE o
        intervalo em relação a reamostrar entidades. Isso é teorema para
        estatística suave sob correlação positiva dentro do conglomerado, e
        carga de revisão NÃO é suave — é o maior prefixo que atinge a meta, uma
        função degrau. No dado com efeito de sentença o intervalo por entidade
        saiu mais LARGO (0,567 contra 0,492) e o veredito mudou. A justificativa
        para reamostrar sentenças é o desenho amostral — as unidades
        independentes são sentenças —, não a largura que sai.
        """
        perdas = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
        escore = np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1])
        base = np.array([0.5, 0.1, 0.9, 0.8, 0.2, 0.7])
        uma = np.array(["s0"] * 6)
        r = paired_review_load_delta(
            perdas, escore, base, uma, target=0.90, n_gold=6, level=0.95,
            n_resamples=200, seed=9, labels=("escore", "base"))
        assert r.n_groups == 1
        assert r.ci_low == pytest.approx(r.ci_high), (
            f"IC [{r.ci_low}, {r.ci_high}] tem largura positiva com uma sentença só: "
            f"a reamostragem não está respeitando o conglomerado"
        )
        assert r.ci_low == pytest.approx(r.delta)

    def test_a_unidade_de_reamostragem_nao_muda_a_estimativa_pontual(self):
        perdas, bom, fraco, sent = self._dados(n_sent=20, por_sent=6, seed=5)
        comum = dict(target=0.90, n_gold=int((1 - perdas).sum()), level=0.95,
                     n_resamples=200, seed=11)
        por_sentenca = paired_review_load_delta(perdas, bom, fraco, sent, **comum)
        por_entidade = paired_review_load_delta(
            perdas, bom, fraco, np.arange(perdas.size).astype(str), **comum)
        assert por_sentenca.n_groups == 20 and por_entidade.n_groups == perdas.size
        assert por_sentenca.delta == pytest.approx(por_entidade.delta), (
            "a unidade de reamostragem é do intervalo, não da estimativa"
        )

    def test_meta_inalcancavel_viaja_como_veredito_e_nao_como_excecao(self):
        perdas = np.ones(30)
        sent = np.repeat([f"s{i}" for i in range(10)], 3)
        r = paired_review_load_delta(
            perdas, np.arange(30.0), np.arange(30.0)[::-1], sent, target=0.9,
            n_gold=30, level=0.95, n_resamples=100, seed=1, labels=("a", "b"))
        assert INALCANCAVEL in r.verdict
        assert not np.isfinite(r.delta)

    def test_reamostragens_degeneradas_sao_contadas_e_nao_silenciadas(self):
        """Meta no limite: parte das reamostragens não a atinge.

        A contagem tem de viajar no resultado — se for grande, o intervalo
        descreve um subconjunto das reamostragens, e isso não pode ficar
        invisível.
        """
        rng = np.random.default_rng(2)
        sent = np.repeat([f"s{i}" for i in range(12)], 2)
        perdas = rng.integers(0, 2, size=24).astype(float)
        a = 1.0 - perdas + rng.normal(0, 0.6, size=24)
        r = paired_review_load_delta(
            perdas, a, 1.0 - perdas + rng.normal(0, 0.8, size=24), sent, target=0.999,
            n_gold=int((1 - perdas).sum()), level=0.95, n_resamples=200, seed=4,
            labels=("a", "b"))
        assert r.n_degenerate + r.n_resamples == 200 or INALCANCAVEL in r.verdict
