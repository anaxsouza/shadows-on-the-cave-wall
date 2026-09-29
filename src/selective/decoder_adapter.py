"""O adaptador do decoder: extração generativa e atenção causal.

POR QUE ESTE MÓDULO EXISTE, separado do adaptador do encoder

O extrator do encoder pontua trechos: ele recebe o texto e devolve, para cada
trecho candidato, um escore por rótulo. Um decoder autorregressivo não faz isso
— ele ESCREVE as entidades. As duas diferenças que decorrem disso não são de
implementação:

1. A CONFIANÇA é outra quantidade. No encoder é o escore sigmoide do trecho; num
   decoder é a verossimilhança da sequência que ele gerou. Ambas são "o número
   que o próprio modelo reporta sobre aquela saída", e é contra esse número que
   todo sinal interno tem de se provar — mas elas não são a mesma função.

2. A ATENÇÃO é mascarada. A consulta na posição `i` só vê `i+1` chaves, e as
   demais entradas da linha são zero porque a máscara as zerou, não porque o
   modelo não tenha olhado. Normalizar por um conjunto fixo de chaves — o que o
   adaptador do encoder faz corretamente para o caso dele — deprimiria a massa
   das consultas iniciais por razão puramente aritmética. Daí `causal_mass`.

AS DUAS LEITURAS, e por que medir as duas em vez de escolher

Há dois conjuntos de consultas defensáveis num decoder generativo:

- **prompt**: as posições dos tokens da menção DENTRO do texto de entrada. É o
  análogo direto do caso bidirecional, e é o único em que a derivação
  `expected_mass_causal(a, k, T)` se aplica exatamente — porque consultas e
  chaves vivem na mesma sequência. Esta é a leitura CONFIRMATÓRIA.
- **geração**: as posições em que o modelo ESCREVE a menção. É onde ele decide,
  e portanto a leitura que um praticante consideraria relevante. Mas aqui as
  consultas estão na saída e o trecho está na entrada, e o nulo derivado não
  cobre essa configuração. Esta leitura é EXPLORATÓRIA, e o rótulo viaja com
  ela.

Medir as duas custa uma passagem para frente a mais e transforma uma escolha de
desenho numa comparação medida. Escolher uma e justificar em prosa seria mais
barato e menos defensável.

ATENÇÃO EAGER É OBRIGATÓRIA. Com a implementação padrão (`sdpa`), pedir
`output_attentions=True` devolve tupla VAZIA — zero camadas, sem erro nenhum.
Uma medição feita sem `attn_implementation="eager"` produziria nada em silêncio,
e o construtor recusa em vez de deixar isso passar.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

SEP = " ## "


class DecoderAdapterError(RuntimeError):
    """O adaptador não pôde produzir uma medição válida."""


@dataclass(frozen=True)
class MencaoGerada:
    """Uma menção que o modelo escreveu, já ancorada no texto de entrada."""

    texto: str
    rotulo: str
    char_ini: int
    char_fim: int
    confianca: float
    pos_geracao: tuple[int, ...]

    def __post_init__(self) -> None:
        if not 0.0 <= self.confianca <= 1.0:
            raise DecoderAdapterError(
                f"confiança {self.confianca} fora de [0, 1]: é exp da média dos log-probs")


def prompt_de(texto: str, rotulos: Sequence[str]) -> str:
    """O MESMO prompt do treino.

    Treinar com um formato e medir com outro mediria a capacidade de generalizar
    formato de instrução, não a de extrair entidades.
    """
    return f"Text: {texto}\nEntity types: {', '.join(sorted(rotulos))}\nEntities:\n"


def parse_saida(gerado: str) -> list[tuple[str, str]]:
    """As menções geradas, na ORDEM em que foram escritas.

    A ordem importa: ela é o que liga cada menção às posições de geração dela, e
    um conjunto perderia essa ligação.
    """
    fora, _ = parse_saida_contando(gerado)
    return fora


def parse_saida_contando(gerado: str) -> tuple[list[tuple[str, str]], int]:
    """Como `parse_saida`, mais a CONTAGEM de linhas descartadas por formato.

    POR QUE A CONTAGEM EXISTE

    Descarte silencioso não é ausência. O teste de fumaça com o modelo base
    mostrou o caso: ele gerou `IL-2`, `T cell`, `Human blood` — menções
    plausíveis, todas SEM o separador `##` — e o parser descartou as três sem
    deixar rastro. Num modelo ajustado isso deve ser raro, mas "deve ser raro" é
    exatamente o tipo de suposição que precisa de número: se o extrator perder o
    formato em parte das sentenças, a taxa de erro medida sairia otimista,
    porque as menções mal formatadas não entram no denominador.

    A contagem sobe no `MEDIDA.json` como ressalva da execução.
    """
    fora: list[tuple[str, str]] = []
    descartadas = 0
    for linha in gerado.strip().splitlines():
        cru = linha.strip()
        if not cru or cru.lower() == "none":
            continue
        if SEP not in cru:
            descartadas += 1
            continue
        m, _, r = cru.partition(SEP)
        m, r = m.strip(), r.strip()
        if m and r and m.lower() != "none":
            fora.append((m, r))
        else:
            descartadas += 1
    return fora, descartadas


def ancorar(texto: str, mencao: str, usados: list[tuple[int, int]]) -> tuple[int, int] | None:
    """Onde, no texto de entrada, está a menção que o modelo escreveu.

    Um decoder pode gerar uma string que não aparece no texto — é alucinação, e
    isso é resultado e não erro: a menção fica sem âncora e é contada como tal,
    em vez de ser silenciosamente descartada ou casada por aproximação.

    Ocorrências já consumidas são puladas, para que duas menções idênticas na
    mesma sentença ancorem em posições diferentes em vez de na mesma.
    """
    inicio = 0
    while True:
        i = texto.find(mencao, inicio)
        if i < 0:
            return None
        par = (i, i + len(mencao))
        if par not in usados:
            return par
        inicio = i + 1


def indices_de_token(offsets: Sequence[tuple[int, int]], char_ini: int,
                     char_fim: int, desloc: int) -> list[int]:
    """Índices de token que cobrem o intervalo de caracteres, no espaço do prompt.

    `desloc` é onde o texto da sentença começa DENTRO do prompt: os offsets são
    do prompt inteiro, e o intervalo vem do texto. Somar o deslocamento aqui em
    vez de no chamador evita o erro de indexação mais provável deste módulo.
    """
    a, b = char_ini + desloc, char_fim + desloc
    return [i for i, (ini, fim) in enumerate(offsets)
            if ini is not None and fim > ini and ini < b and fim > a]


class DecoderAdapter:
    """Extração generativa mais atenção causal, de uma passagem por sentença.

    O modelo fica atrás desta classe e os testes a substituem por uma função
    determinística — é a mesma separação que mantém a análise do encoder
    testável em milissegundos, sem baixar pesos.
    """

    def __init__(self, modelo: Any, tokenizador: Any, rotulos: Sequence[str],
                 *, max_new_tokens: int = 96) -> None:
        impl = getattr(getattr(modelo, "config", None), "_attn_implementation", None)
        if impl is not None and impl != "eager":
            raise DecoderAdapterError(
                f"modelo carregado com attn_implementation={impl!r}. Só 'eager' devolve a "
                f"matriz de atenção: com 'sdpa' o pedido volta VAZIO, sem erro, e a medição "
                f"produziria nada em silêncio.")
        self.m, self.tok = modelo, tokenizador
        self.rotulos = sorted(rotulos)
        self.max_new_tokens = int(max_new_tokens)

    def __call__(self, sentenca: str, *, tokens_only: bool = False) -> dict[str, Any]:
        import torch

        p = prompt_de(sentenca, self.rotulos)
        desloc = p.index(sentenca)
        cod = self.tok(p, return_tensors="pt", return_offsets_mapping=True,
                       add_special_tokens=False)
        offsets = [tuple(x) for x in cod.pop("offset_mapping")[0].tolist()]
        cod = {k: v.to(self.m.device) for k, v in cod.items()}
        n_prompt = int(cod["input_ids"].shape[1])
        if tokens_only:
            return {"tokens": self.tok.convert_ids_to_tokens(cod["input_ids"][0])}

        with torch.no_grad():
            ger = self.m.generate(**cod, max_new_tokens=self.max_new_tokens,
                                  do_sample=False, output_scores=True,
                                  return_dict_in_generate=True,
                                  pad_token_id=self.tok.pad_token_id)
        ids_saida = ger.sequences[0][n_prompt:]
        texto_gerado = self.tok.decode(ids_saida, skip_special_tokens=True)

        # Log-prob por token gerado, para a confiança de cada menção.
        logps = []
        for passo, pontos in enumerate(ger.scores):
            lp = torch.log_softmax(pontos[0].float(), dim=-1)
            logps.append(float(lp[int(ids_saida[passo])]))

        # A passagem que devolve a atenção. Duas: o prompt sozinho (leitura
        # confirmatória, em que a derivação vale) e prompt+saída (exploratória).
        with torch.no_grad():
            # Atenção E estados ocultos da MESMA passagem. Rodar duas vezes
            # custaria o dobro e abriria a porta para as duas medidas vierem de
            # estados diferentes do modelo.
            so_prompt = self.m(**cod, output_attentions=True, output_hidden_states=True)
            completo = self.m(input_ids=ger.sequences, output_attentions=True)
        if not so_prompt.attentions:
            raise DecoderAdapterError(
                "o modelo não devolveu atenção mesmo com output_attentions=True — "
                "quase certamente attn_implementation != 'eager'")
        attn_prompt = np.stack([a[0].float().cpu().numpy() for a in so_prompt.attentions])
        attn_total = np.stack([a[0].float().cpu().numpy() for a in completo.attentions])

        # Casar cada menção gerada com as posições de saída que a escreveram.
        pecas = self.tok.convert_ids_to_tokens(ids_saida)
        mencoes, usados, cursor = [], [], 0
        pares, n_descartadas = parse_saida_contando(texto_gerado)
        for texto_m, rot in pares:
            par = ancorar(sentenca, texto_m, usados)
            pos = []
            alvo = texto_m
            while cursor < len(pecas) and alvo:
                peca = self.tok.convert_tokens_to_string([pecas[cursor]]).strip()
                if peca and peca in alvo:
                    pos.append(cursor)
                    alvo = alvo[alvo.index(peca) + len(peca):].strip()
                cursor += 1
            conf = float(np.exp(np.mean([logps[i] for i in pos]))) if pos else 0.0
            if par is None:
                mencoes.append({"texto": texto_m, "rotulo": rot, "ancorada": False,
                                "confidence": conf, "pos_geracao": tuple(pos)})
                continue
            idx = indices_de_token(offsets, par[0], par[1], desloc)
            mencoes.append({"texto": texto_m, "rotulo": rot, "ancorada": True,
                            "char_ini": par[0], "char_fim": par[1],
                            "token_indices": idx, "confidence": min(1.0, max(0.0, conf)),
                            "pos_geracao": tuple(pos)})
            usados.append(par)

        ocultos = getattr(so_prompt, "hidden_states", None)
        return {
            "tokens": self.tok.convert_ids_to_tokens(cod["input_ids"][0]),
            "n_prompt": n_prompt,
            "attentions_prompt": attn_prompt,
            "attentions_total": attn_total,
            # [camadas+1, T, d] — inclui o embedding de entrada, como no encoder.
            # `None` quando o modelo não os devolve: a família correspondente sai
            # vazia e a ressalva é registrada, em vez de inventar valor.
            "hidden_states": (np.stack([h[0].float().cpu().numpy() for h in ocultos])
                              if ocultos else None),
            "mencoes": mencoes,
            # Linhas que o modelo escreveu e o parser descartou por formato.
            # Descarte silencioso não é ausência: sem este número, um extrator
            # que perdesse o formato produziria taxa de erro otimista.
            "n_descartadas": n_descartadas,
            "texto_gerado": texto_gerado,
        }


__all__ = [
    "SEP",
    "DecoderAdapter",
    "DecoderAdapterError",
    "MencaoGerada",
    "ancorar",
    "indices_de_token",
    "parse_saida",
    "parse_saida_contando",
    "prompt_de",
]
