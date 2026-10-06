"""O adaptador do GLiNER: de onde saem a previsão e a atenção, do MESMO modelo.

Esta é a peça que liga o modelo à passagem de medição (`measurement.py`). Ela
existe separada por uma razão de teste: `measure()` recebe uma função `predict` e
não um modelo, então a passagem inteira é testável sem rede e sem pesos. Aqui é
o único lugar que toca o modelo de verdade.

POR QUE O GLINER, E NÃO UM ETIQUETADOR

A pergunta compara a atenção do modelo com a confiança DO MESMO modelo. Até
02/09/2026 o repositório listava `bert-large` e `deberta-v3-base` como os
modelos, mas nenhum dos dois encontra entidade — são encoders crus — e quem
previa era um terceiro modelo. Atenção de um explicando confiança de outro é
medida sem sentido.

E o modelo tem de pontuar TRECHOS, não etiquetar tokens: etiquetagem BIO não
prevê entidade aninhada, porque exigiria dois rótulos no mesmo token, e metade
da pergunta é sobre o aninhado.

OS CAMINHOS INTERNOS, VERIFICADOS RODANDO EM 02/09/2026 (gliner 0.2.28)

    m = GLiNER.from_pretrained("urchade/gliner_base")   -> UniEncoderSpanGLiNER
    m.predict_entities(texto, rotulos)
        -> [{"start": 0, "end": 12, "text": "Barack Obama",
             "label": "person", "score": 0.9957}, ...]
    m.data_processor.transformer_tokenizer               -> tokenizador, com
        return_offsets_mapping funcionando ('▁Barack' -> [0, 6])
    m.model.token_rep_layer.bert_layer.model            -> o transformer
        aceita output_attentions=True e devolve 12 camadas de [1, 12, T, T],
        linhas somando 1

São caminhos internos de um pacote de terceiros e podem mudar de versão: cada um
é verificado na construção, com mensagem que diz qual quebrou.

DOIS FATORES QUE MUDAM A MEDIDA E ESTÃO DECLARADOS FORA DAQUI

1. A faixa de camadas vem do pré-registro. Este modelo tem 12 camadas, e uma
   faixa fora do intervalo é recusada em vez de truncada em silêncio.
2. As descrições de rótulo que se dão ao GLiNER mudam o que ele prevê — são
   entrada do modelo, não detalhe de implementação. Ficam em
   `configs/config.yaml`, seção `datasets.<corpus>.gliner_labels`, para serem
   versionadas e citáveis.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .preregistration import Preregistration


class AdapterError(RuntimeError):
    """Falha do adaptador, com o caminho interno que quebrou."""


@dataclass
class GoldEntity:
    """Entidade anotada, em offsets de caractere sobre o texto reconstruído."""

    start: int
    end: int
    label: str
    is_nested: bool = False


def _resolver(obj: Any, caminho: str) -> Any:
    """Segue um caminho de atributos e diz exatamente qual elo quebrou."""
    atual = obj
    for parte in caminho.split("."):
        if not hasattr(atual, parte):
            raise AdapterError(
                f"o caminho interno do GLiNER mudou: '{caminho}' quebrou em '{parte}' "
                f"({type(atual).__name__} não tem esse atributo). Verificado em "
                f"gliner 0.2.28; veja o docstring de gliner_adapter.py."
            )
        atual = getattr(atual, parte)
    return atual


class GLiNERAdapter:
    """Fornece a `measure()` a função `predict` que ela espera.

    O contrato de saída é o de `measurement.measure`: `tokens`, `attentions`,
    `predicted` (com `token_indices`, `confidence`, `correct`, `is_nested`) e
    `n_gold`.
    """

    def __init__(
        self,
        model: Any,
        *,
        prereg: Preregistration,
        labels: Sequence[str],
        label_to_corpus: Mapping[str, str],
        threshold: float = 0.5,
    ) -> None:
        self.model = model
        self.prereg = prereg
        self.labels = list(labels)
        self.label_to_corpus = dict(label_to_corpus)
        self.threshold = float(threshold)

        self.tokenizer = _resolver(model, "data_processor.transformer_tokenizer")
        self.transformer = _resolver(model, "model.token_rep_layer.bert_layer.model")

        cfg = getattr(self.transformer, "config", None)
        self.n_layers = int(getattr(cfg, "num_hidden_layers", 0) or 0)
        self.n_heads = int(getattr(cfg, "num_attention_heads", 0) or 0)
        if not self.n_layers or not self.n_heads:
            raise AdapterError(
                "não foi possível ler num_hidden_layers/num_attention_heads da config "
                "do transformer do GLiNER"
            )

        # A recusa que impede a medida de sair errada em silêncio: uma faixa que
        # não existe neste modelo não pode ser truncada por conveniência, porque
        # o número medido passaria a ser de outras camadas que as declaradas.
        fora = [c for c in prereg.layers if not 0 <= c < self.n_layers]
        if fora:
            raise AdapterError(
                f"a faixa de camadas pré-registrada {list(prereg.layers)} não existe neste "
                f"modelo: ele tem {self.n_layers} camadas (índices 0 a {self.n_layers - 1}), "
                f"e {fora} está fora. Truncar mudaria em silêncio quais camadas produziram a "
                f"medida. Corrija 'selective.layers' em configs/config.yaml e registre em "
                f"docs/tese/PREREGISTRO.md."
            )
        if prereg.heads:
            fora_h = [h for h in prereg.heads if not 0 <= h < self.n_heads]
            if fora_h:
                raise AdapterError(
                    f"cabeças pré-registradas {list(prereg.heads)} fora do intervalo: o "
                    f"modelo tem {self.n_heads} (0 a {self.n_heads - 1}); {fora_h} não existe"
                )

        rotulos_desconhecidos = set(self.label_to_corpus) - set(self.labels)
        if rotulos_desconhecidos:
            raise AdapterError(
                f"o mapa de rótulos cita descrições que não são dadas ao modelo: "
                f"{sorted(rotulos_desconhecidos)}"
            )

    # -- o contrato de measure() --------------------------------------------

    def __call__(self, example: Any, tokens_only: bool = False) -> dict[str, Any]:
        texto, ouro = _texto_e_ouro(example)
        lote = self.tokenizer(
            texto, return_tensors="pt", return_offsets_mapping=True, truncation=True
        )
        offsets = lote["offset_mapping"][0].tolist()
        tokens = self.tokenizer.convert_ids_to_tokens(lote["input_ids"][0])

        if tokens_only:
            return {"tokens": tokens, "n_layers": self.n_layers, "n_heads": self.n_heads}

        import torch

        with torch.no_grad():
            saida = self.transformer(
                input_ids=lote["input_ids"],
                attention_mask=lote["attention_mask"],
                output_attentions=True,
                # Os estados ocultos saem da MESMA passagem para frente que a
                # atenção. Rodar duas vezes custaria o dobro e abriria a porta
                # para as duas medidas virem de estados diferentes do modelo.
                output_hidden_states=True,
            )
        atencoes = getattr(saida, "attentions", None)
        if not atencoes:
            raise AdapterError(
                "o transformer do GLiNER não devolveu atenção mesmo com "
                "output_attentions=True"
            )
        ocultos = getattr(saida, "hidden_states", None)
        if not ocultos:
            raise AdapterError(
                "o transformer do GLiNER não devolveu estados ocultos mesmo com "
                "output_hidden_states=True; sem eles a família de estados ocultos "
                "não pode ser medida"
            )

        preditas = self.model.predict_entities(texto, self.labels, threshold=self.threshold)

        # A DISTRIBUIÇÃO DE ESCORES SOBRE OS RÓTULOS, por trecho.
        #
        # `predict_entities` devolve só o melhor rótulo de cada trecho, e a
        # família de logits precisa dos outros para ter margem e entropia. Com
        # `multi_label=True` e limiar zero, cada trecho volta uma vez por rótulo.
        #
        # Duas ressalvas que viajam com qualquer número dessa família: os escores
        # são SIGMOIDE POR PAR (trecho, rótulo) e não um softmax sobre classes,
        # então não somam 1 e a entropia tem de ser normalizada; e o limiar zero
        # devolve todo trecho candidato, então o casamento com as predições é por
        # posição exata de caractere.
        por_span: dict[tuple[int, int], list[float]] = {}
        try:
            todos = self.model.predict_entities(
                texto, self.labels, threshold=0.0, multi_label=True)
        except TypeError as e:  # pragma: no cover - depende da versão do gliner
            raise AdapterError(
                f"esta versão do gliner não aceita multi_label/threshold=0.0 em "
                f"predict_entities, e sem isso a família de logits não é medível: {e}"
            ) from e
        for p in todos:
            por_span.setdefault((int(p["start"]), int(p["end"])), []).append(float(p["score"]))

        saida_preditas = []
        for p in preditas:
            idx = _tokens_do_span(offsets, int(p["start"]), int(p["end"]))
            rotulo_corpus = self.label_to_corpus.get(str(p["label"]), str(p["label"]))
            escores = sorted(por_span.get((int(p["start"]), int(p["end"])), []), reverse=True)
            saida_preditas.append(
                {
                    "token_indices": idx,
                    "confidence": float(p["score"]),
                    "correct": _casa_ouro(int(p["start"]), int(p["end"]), rotulo_corpus, ouro),
                    "is_nested": _e_aninhada(int(p["start"]), int(p["end"]), ouro),
                    # Vazio quando o casamento por posição falha; a medição
                    # registra isso como ressalva em vez de inventar um valor.
                    "label_scores": escores,
                    "char_span": (int(p["start"]), int(p["end"])),
                }
            )
        return {
            "tokens": tokens,
            "attentions": atencoes,
            "hidden_states": ocultos,
            "predicted": saida_preditas,
            "n_gold": len(ouro),
        }

    def para_remedicao(self, example: Any) -> dict[str, Any]:
        """A passagem da REMEDIÇÃO (decl-14 e decl-15): a mesma previsão e a mesma atenção.

        Mesma tokenização, mesmo transformer, mesmo `predict_entities` com o
        mesmo limiar que `__call__`. Não repete a passada de `multi_label` com
        limiar zero (só alimenta a família de logits, que a remedição não
        usa) nem pede estados ocultos (não mudam a atenção). Acrescenta o que
        a medição original descartava: o intervalo de caracteres e o rótulo de
        cada previsão, as fronteiras das anotadas e os offsets dos tokens.
        """
        texto, ouro = _texto_e_ouro(example)
        lote = self.tokenizer(
            texto, return_tensors="pt", return_offsets_mapping=True, truncation=True
        )
        offsets = lote["offset_mapping"][0].tolist()
        import torch

        with torch.no_grad():
            saida = self.transformer(
                input_ids=lote["input_ids"],
                attention_mask=lote["attention_mask"],
                output_attentions=True,
            )
        atencoes = getattr(saida, "attentions", None)
        if not atencoes:
            raise AdapterError("o transformer do GLiNER não devolveu atenção")
        preditas = self.model.predict_entities(texto, self.labels, threshold=self.threshold)
        out = []
        for p in preditas:
            ini, fim = int(p["start"]), int(p["end"])
            rot = self.label_to_corpus.get(str(p["label"]), str(p["label"]))
            out.append({
                "token_indices": _tokens_do_span(offsets, ini, fim),
                "confidence": float(p["score"]),
                "correct": _casa_ouro(ini, fim, rot, ouro),
                "is_nested": _e_aninhada(ini, fim, ouro),
                "char_span": (ini, fim),
                "label": rot,
            })
        return {
            "attentions": atencoes,
            "offsets": offsets,
            "predicted": out,
            "gold": [(g.start, g.end, g.label) for g in ouro],
            "n_tokens": len(offsets),
        }


# -- funções puras, testáveis sem modelo ------------------------------------


def _tokens_do_span(offsets: Sequence[Sequence[int]], inicio: int, fim: int) -> list[int]:
    """Índices dos tokens que sobrepõem o span de caracteres [inicio, fim).

    Tokens especiais têm offset (0, 0) e são descartados: eles não pertencem ao
    span, e incluir o [CLS] no numerador da massa inflaria a medida com
    exatamente o token que a convenção de sumidouro existe para tratar.
    """
    saida = []
    for i, (a, b) in enumerate(offsets):
        if a == b:
            continue
        if a < fim and b > inicio:
            saida.append(i)
    return saida


def _casa_ouro(inicio: int, fim: int, rotulo: str, ouro: Sequence[GoldEntity]) -> bool:
    """Casamento ESTRITO: mesma fronteira e mesmo rótulo.

    Estrito e não parcial porque a unidade da curva é o que se entrega ao
    cliente: um trecho com fronteira errada é entrega errada, e chamá-lo de
    acerto parcial infla o desempenho na direção do resultado desejado.
    """
    return any(g.start == inicio and g.end == fim and g.label == rotulo for g in ouro)


def _e_aninhada(inicio: int, fim: int, ouro: Sequence[GoldEntity]) -> bool:
    """Aninhada quando o span está estritamente dentro de outra entidade de ouro.

    Definido pela ANOTAÇÃO e não pela predição: se dependesse do que o modelo
    previu, o estrato mudaria com a qualidade do modelo e os dois estratos
    deixariam de ser comparáveis entre execuções.
    """
    for g in ouro:
        if g.start <= inicio and fim <= g.end and (g.end - g.start) > (fim - inicio):
            return True
    return False


def _texto_e_ouro(example: Any) -> tuple[str, list[GoldEntity]]:
    """Aceita o NERExample do repositório ou um dicionário equivalente.

    O texto é reconstruído por junção com espaço e os offsets do ouro são
    calculados sobre ESSA reconstrução — a mesma string que vai ao modelo. Usar
    uma reconstrução para o modelo e outra para o ouro desalinharia os offsets, e
    o desalinhamento apareceria como erro de predição e não como defeito de
    código.
    """
    if isinstance(example, Mapping):
        texto = example.get("text")
        tokens = example.get("tokens")
        entidades = example.get("entities") or []
    else:
        texto = getattr(example, "text", None)
        tokens = getattr(example, "tokens", None)
        entidades = getattr(example, "entities", None) or []

    if texto is None:
        if not tokens:
            raise AdapterError("exemplo sem 'text' e sem 'tokens': nada a medir")
        texto = " ".join(tokens)

    ouro: list[GoldEntity] = []
    for e in entidades:
        if isinstance(e, Mapping):
            ini, f = e.get("start"), e.get("end")
            rot = e.get("label") or e.get("type")
        else:
            ini, f = getattr(e, "start", None), getattr(e, "end", None)
            rot = getattr(e, "label", None) or getattr(e, "type", None)
        if ini is None or f is None or rot is None:
            raise AdapterError(f"entidade de ouro sem start/end/label: {e!r}")
        ouro.append(GoldEntity(start=int(ini), end=int(f), label=str(rot)))

    for g in ouro:
        g.is_nested = _e_aninhada(g.start, g.end, ouro)
    return str(texto), ouro
