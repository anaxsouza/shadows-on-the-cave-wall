"""As features que a SONDA SUPERVISIONADA lê, e por que elas não são os escalares.

POR QUE ESTE ARQUIVO EXISTE

O braço encoder mediu estados ocultos por três escalares não supervisionados —
norma, distância ao centroide, variação entre camadas — e a atenção por médias
sobre camadas e cabeças. Os dois instrumentos falharam. A revisão sistemática de
17/09/2026 mostrou que isso deixa aberta a objeção mais forte que existe contra
um nulo de estados ocultos: **a literatura que ENCONTRA sinal ali quase nunca usa
resumo não supervisionado, usa sonda treinada** — regressão logística sobre o
vetor de ativação, ajustada numa partição de calibração. E o estudo de sete
arquiteturas de 0,5B a 9B mede que a feature mais forte dele era atenção **por
cabeça**, não agregada.

A distinção é epistemicamente grande, e é a razão de a exploração valer o custo:

- uma NORMA falhar diz que aquele resumo escalar não decide;
- uma SONDA falhar diz que a informação não está linearmente recuperável do
  vetor, o que é uma afirmação sobre a representação e não sobre o resumo.

O nulo do segundo tipo é muito mais difícil de atacar. E se a sonda ENCONTRAR
sinal, a tese melhora em vez de cair: passa a distinguir resumo de representação,
que é uma afirmação mais fina do que "sinais internos não acrescentam".

O QUE ESTE ARQUIVO NÃO FAZ, E É DELIBERADO

Não toca em nenhum escalar declarado. `_sinais_de_atencao` e `_sinais_ocultos`
continuam calculando exatamente o que as declarações dizem, com os mesmos
valores — a persistência aqui é PURAMENTE ADITIVA, um arquivo a mais ao lado do
`entities.csv`. A razão: um veredito pré-registrado que mudasse de valor porque
eu acrescentei exploração deixaria de ser pré-registrado.

E a exploração fica em compartimento próprio: a sonda entra como UMA comparação
por família — uma agregação supervisionada — e não como 336 comparações por
cabeça. Sem isso, cada cabeça seria uma chance a mais de exclusão de zero por
sorteio, que é o erro de contagem que a apuração de 16/09/2026 apanhou.
"""
from __future__ import annotations

import numpy as np

# float16 e não float32: são features de entrada de uma regressão logística, não
# quantidade de veredito, e a resolução relativa de 1e-3 do float16 está três
# ordens de grandeza abaixo do ruído amostral de 5.307 trechos. Corta o arquivo
# pela metade — 228 MB contra 456 MB por corpus no 0,5B.
DTYPE = np.float16

# Três features por (camada, cabeça), e a razão de serem estas: são as MESMAS
# quantidades dos escalares declarados, antes da média que apaga a identidade de
# cabeça. Assim a sonda e o escalar respondem sobre a mesma grandeza, e a
# diferença entre eles é só o instrumento — que é a comparação que interessa.
FEATURES_CABECA = ("massa", "entropia_linha", "maximo_linha")


def features_por_cabeca(attn, idx) -> np.ndarray:
    """[camadas, cabecas, 3] — massa, entropia e máximo de linha POR CABEÇA.

    `attn` é [camadas, cabeças, T, T] com máscara causal já aplicada. `idx` são
    as posições de token do trecho.

    A entropia e o máximo são calculados sobre as chaves PERMITIDAS de cada
    linha (`media[q, :q+1]`), pela mesma razão que o caminho declarado: contar os
    zeros da máscara como massa distribuída baixaria a entropia por aritmética.
    """
    A = np.asarray(attn, dtype=np.float32)
    n_cam, n_cab = A.shape[0], A.shape[1]
    out = np.zeros((n_cam, n_cab, len(FEATURES_CABECA)), dtype=np.float32)
    ii = list(idx)
    for c in range(n_cam):
        for h in range(n_cab):
            M = A[c, h]
            recebida = M[:, ii].sum()
            total = M.sum()
            out[c, h, 0] = recebida / total if total > 0 else 0.0
            ents, maxs = [], []
            for q in ii:
                perm = M[q, : q + 1]
                s = perm.sum()
                if s <= 0:
                    continue
                p = perm / s
                nz = p[p > 0]
                ents.append(float(-(nz * np.log(nz)).sum()))
                maxs.append(float(p.max()))
            out[c, h, 1] = float(np.mean(ents)) if ents else 0.0
            out[c, h, 2] = float(np.mean(maxs)) if maxs else 0.0
    return out.astype(DTYPE)


def features_ocultas(ocultos, idx) -> np.ndarray | None:
    """[camadas+1, d] — o vetor médio do trecho em CADA camada.

    Média sobre as posições do trecho, e não concatenação: o número de tokens
    varia por trecho e a sonda precisa de dimensão fixa. A média é a agregação
    que o caminho declarado já usa em `_sinais_ocultos`, então a sonda lê o mesmo
    objeto que a norma leu — de novo, a diferença é o instrumento.

    TODAS as camadas, e não só a última: a literatura reporta que a camada de
    melhor sondagem é intermediária, e escolher a camada depois de ver o
    resultado seria grau de liberdade. Guardar todas e declarar a regra de
    seleção antes é o que evita isso.
    """
    if ocultos is None:
        return None
    h = np.asarray(ocultos, dtype=np.float32)            # [camadas+1, T, d]
    ii = list(idx)
    return h[:, ii, :].mean(axis=1).astype(DTYPE)        # [camadas+1, d]


class Acumulador:
    """Junta as features de todos os trechos e escreve UM npz.

    A chave de junção é `linha_id`, o índice da linha no `entities.csv`, e não o
    `sentence_id`: um trecho é a unidade da análise e há vários por sentença.
    Sem essa chave o npz não seria alinhável com os rótulos de acerto, e uma
    sonda ajustada contra rótulo desalinhado produziria número plausível e
    errado — a falha silenciosa que este projeto já pagou uma vez.
    """

    def __init__(self):
        self.cabeca: list[np.ndarray] = []
        self.oculta: list[np.ndarray] = []
        self.linha_id: list[int] = []

    def junta(self, linha_id: int, attn, ocultos, idx) -> None:
        self.linha_id.append(int(linha_id))
        self.cabeca.append(features_por_cabeca(attn, idx))
        o = features_ocultas(ocultos, idx)
        self.oculta.append(o if o is not None else np.zeros((1, 1), dtype=DTYPE))

    def salvar(self, caminho) -> dict:
        if not self.linha_id:
            return {"features": "nenhuma"}
        C = np.stack(self.cabeca)
        tem_oculta = all(o.ndim == 2 and o.shape[1] > 1 for o in self.oculta)
        O = np.stack(self.oculta) if tem_oculta else np.zeros((len(self.linha_id), 0, 0), DTYPE)
        np.savez_compressed(caminho, linha_id=np.asarray(self.linha_id, dtype=np.int32),
                            cabeca=C, oculta=O,
                            nomes_features_cabeca=np.asarray(FEATURES_CABECA))
        return {"features_arquivo": str(caminho),
                "features_cabeca_forma": list(C.shape),
                "features_oculta_forma": list(O.shape),
                "features_nomes_cabeca": list(FEATURES_CABECA),
                "features_dtype": str(np.dtype(DTYPE))}
