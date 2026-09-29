# Recuperado da linhagem do artefato que produziu a peça publicada; só os CAMINHOS
# foram trocados para o repositório. Rode por reproduzir/derivadas.py, que o executa
# em saida/derivadas/ e compara o que ele escreve com o publicado.
from pathlib import Path as _P
RAIZ = _P(__file__).resolve().parents[2]
import csv as _csv

linhas = [
    dict(corpus="GENIA", teste="piso", n_avaliacao=4038, comparacao="massa vs abstenção aleatória",
         aurc_a=0.4809, aurc_b=0.4390, delta=-0.0419, ic_baixo=-0.0575, ic_alto=-0.0267,
         contem_zero="não", veredito="massa ordena melhor que o acaso — CRITÉRIO ATENDIDO"),
    dict(corpus="CoNLL-2003", teste="piso", n_avaliacao=5701, comparacao="massa vs abstenção aleatória",
         aurc_a=0.5310, aurc_b=0.5629, delta=+0.0319, ic_baixo=+0.0186, ic_alto=+0.0442,
         contem_zero="não", veredito="ACASO ordena melhor que a massa — REFUTADO com sinal invertido"),
    dict(corpus="GENIA", teste="valor adicionado", n_avaliacao=4038, comparacao="combinado vs confiança do modelo",
         aurc_a=0.3162, aurc_b=0.3150, delta=-0.0012, ic_baixo=-0.0037, ic_alto=+0.0016,
         contem_zero="sim", veredito="REFUTADO — peso convexo 0,97/0,03 (colapso)"),
    dict(corpus="CoNLL-2003", teste="valor adicionado", n_avaliacao=5701, comparacao="combinado vs confiança do modelo",
         aurc_a=0.2593, aurc_b=0.2603, delta=+0.0010, ic_baixo=-0.0003, ic_alto=+0.0026,
         contem_zero="sim", veredito="REFUTADO — peso convexo 0,99/0,01 (colapso)"),
    dict(corpus="GENIA", teste="valor adicionado / estrato aninhado", n_avaliacao=510,
         comparacao="combinado vs confiança", aurc_a="", aurc_b="", delta=+0.0018,
         ic_baixo=-0.0023, ic_alto=+0.0061, contem_zero="sim", veredito="sem efeito detectável"),
    dict(corpus="GENIA", teste="valor adicionado / estrato plano", n_avaliacao=3528,
         comparacao="combinado vs confiança", aurc_a="", aurc_b="", delta=-0.0013,
         ic_baixo=-0.0040, ic_alto=+0.0020, contem_zero="sim", veredito="sem efeito detectável"),
    dict(corpus="CoNLL-2003", teste="valor adicionado / estrato aninhado", n_avaliacao=24,
         comparacao="combinado vs confiança", aurc_a="", aurc_b="", delta="", ic_baixo="", ic_alto="",
         contem_zero="", veredito="NÃO AVALIADO — reamostragens degeneradas; ausência de dado, não de efeito"),
]
with open("resultado_piloto.csv", "w", newline="", encoding="utf-8") as fh:
    w = _csv.DictWriter(fh, fieldnames=list(linhas[0]))
    w.writeheader(); w.writerows(linhas)