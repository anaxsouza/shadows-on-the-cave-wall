"""A passagem de medição do braço decoder.

O QUE ELA PRODUZ, e a diferença em relação à do encoder

A tabela por entidade, com as colunas do contrato mais as da geometria CAUSAL.
Três decisões ficam aqui, e nenhuma é livre:

1. NÃO persiste os tensores de atenção, e a omissão é declarada. No braço
   encoder eles foram guardados porque a varredura de invariância recalcula
   leituras a partir deles. Aqui seriam ~3 GB por corpus e o limite de
   transferência é 256 MB, então ficariam PRESOS na máquina remota — e o acesso
   a ela depende de um token de SSO que expira. Guardar o que não se pode
   buscar é pior que não guardar: cria a ilusão de que o dado está disponível.
   Consequência honesta: a varredura de leituras não fica disponível neste braço
   sem remedir.

2. O texto medido é o MESMO que o treino viu: as palavras unidas por espaço. O
   carregador dá offsets de caractere do texto ORIGINAL, e usá-los aqui casaria
   contra um texto diferente do que o modelo leu. Os trechos de ouro são
   derivados dos índices de PALAVRA, exatamente como no export de treino.

3. O estrato aninhado vem da ANOTAÇÃO, não da predição — invariante do
   protocolo. Se dependesse do previsto, o estrato mudaria com a qualidade do
   modelo e as execuções deixariam de ser comparáveis.

AS COLUNAS QUE EXISTEM SÓ AQUI

`expected_causal` é o nulo com máscara, que depende de três números: tamanho do
trecho, comprimento da sentença e POSIÇÃO. `expected_bidir` é o nulo sem
máscara, que depende de dois. Guardar as duas na mesma linha é o que permite
medir, e não argumentar, se a posição acrescenta dimensão ao confundidor.

A FAMÍLIA DE ATENÇÃO INTEIRA, com nulos CAUSAIS

`row_entropy_causal` e `row_max_causal` vêm com os nulos exatos do caso
mascarado — `log(T!)/T` e `H_T/T` —, e não com os bidirecionais. A diferença foi
medida antes de se decidir isto: o teto do máximo bidirecional é 3,4x a 5,3x
menor que o causal, e a razão cresce com o comprimento. Usar o bidirecional
faria todo máximo observado parecer alto por razão aritmética.

A FAMÍLIA DE LOGITS NÃO ENTRA, e a omissão é declarada

Num extrator de trechos existe uma distribuição sobre rótulos por trecho, e
dela saem margem, entropia e máximo. Num decoder generativo não existe: o
modelo ESCREVE o rótulo, e construir uma distribuição sobre rótulos exigiria
escolher quais posições do vocabulário contam. Essa escolha é definicional e
precisaria da própria declaração — e o braço encoder já testou a família, sem
encontrar nada que sobrevivesse. Declarar aqui um análogo escolhido por mim
acrescentaria um grau de liberdade para responder uma pergunta já respondida.
"""
from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, ".")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from rotulos import descricao, descricoes  # noqa: E402
from features_sonda import Acumulador  # noqa: E402

from src.selective.decoder_adapter import DecoderAdapter  # noqa: E402
from src.selective.geometry import (  # noqa: E402
    causal_mass,
    expected_mass,
    expected_mass_causal,
)
from src.selective.signals import (  # noqa: E402
    row_ceiling_entropy,
    row_ceiling_max,
    row_entropy,
)

COLUNAS = (
    "sentence_id", "loss", "model_confidence", "is_nested", "token_indices",
    # geometria, as três covariáveis do caso causal
    "span_size", "span_position", "n_tokens",
    # os dois nulos, lado a lado
    "expected_causal", "expected_bidir",
    # as duas leituras da massa
    "causal_mass_prompt", "causal_mass_geracao", "causal_mass_autofoco",
    # o resto da FAMÍLIA DE ATENÇÃO, com os nulos causais exatos
    "row_entropy_causal", "row_max_causal",
    "expected_row_entropy_causal", "expected_row_max_causal",
    # a família de ESTADOS OCULTOS, de nulo empírico
    "hidden_norm", "hidden_dist_centroide", "hidden_delta_camadas",
    # o que o modelo escreveu e não estava no texto
    "ancorada",
    # A CHAVE DE JUNÇÃO COM O AggSeq, acrescentada em 23/09/2026 e NO FIM, para
    # nenhuma coluna existente mudar de posição. `medir_aggseq.py` declara a
    # junção por (sentence_id, mencao, rotulo), e esta tabela nunca gravou as
    # duas últimas — as 16 comparações contra `aggseq` de cada declaração não
    # tinham como ser executadas. Defeito meu de desenho, apanhado ao montar a
    # análise, antes de qualquer veredito. O texto e o rótulo são exatamente os
    # que o parser devolveu, os mesmos que o medidor de feixe usa.
    "mencao", "rotulo",
)

CARREGADOR = {
    "genia": ("src.core.loaders.biomedical.genia", "GENIALoader"),
    "conll2003": ("src.core.loaders.conll.loader", "CONLLLoader"),
}


def trechos_de_ouro(exemplo, corpus: str) -> tuple[str, set[tuple[int, int, str]], set[tuple[int, int]]]:
    """Texto unido por espaço e os trechos de ouro NESSE espaço de caracteres.

    Derivados dos índices de palavra, como no export de treino. Usar os offsets
    do texto original casaria contra um texto que o modelo não leu.

    O RÓTULO DE OURO passa por `rotulos.descricao`, e isto é obrigatório e não
    cosmético: o carregador entrega `CELL_TYPE` e o modelo foi ajustado para
    escrever `cell type`. Comparar a saída gerada contra a sigla do carregador
    contava todo acerto daquela classe como erro — nas 4 classes do CoNLL e em
    3 das 5 do GENIA. Ver `tools/rotulos.py` para o mapa e a medição do defeito.
    """
    toks = exemplo.tokens
    texto = " ".join(toks)
    inicios, cursor = [], 0
    for t in toks:
        inicios.append(cursor)
        cursor += len(t) + 1
    ouro, intervalos = set(), set()
    for e in exemplo.entities:
        i0 = next((i for i, (a, b) in enumerate(zip(inicios, toks))
                   if a <= e.start < a + len(b)), None)
        i1 = next((i for i, (a, b) in enumerate(zip(inicios, toks))
                   if a < e.end <= a + len(b)), None)
        if i0 is None or i1 is None or i1 < i0:
            continue
        ci, cf = inicios[i0], inicios[i1] + len(toks[i1])
        ouro.add((ci, cf, descricao(corpus, e.label)))
        intervalos.add((ci, cf))
    return texto, ouro, intervalos


def aninhada(intervalo: tuple[int, int], todos: set[tuple[int, int]]) -> bool:
    """Estrito: contido em OUTRO intervalo anotado, e não igual a ele."""
    a, b = intervalo
    return any(c <= a and b <= d and (c, d) != (a, b) for c, d in todos)



def _sinais_de_atencao(attn, idx, T: int) -> dict[str, float]:
    """Entropia e máximo das linhas DO TRECHO, com os nulos causais ao lado.

    As linhas do trecho são as consultas que ele emite, e cada uma tem largura
    própria: a linha `i` vê `i+1` chaves. Medir a entropia sobre a largura total
    contaria os zeros da máscara como massa nula distribuída, o que baixaria a
    entropia por razão aritmética — o mesmo erro que o denominador fixo causa na
    massa.

    Os nulos vêm com a linha porque o estatuto tem de viajar com o número: aqui
    eles são EXATOS, derivados da contagem de chaves permitidas.
    """
    media = np.asarray(attn, dtype=float).mean(axis=(0, 1))       # [T, T]
    ents, maxs = [], []
    for q in idx:
        permitidas = media[q, : q + 1]
        s = permitidas.sum()
        if s <= 0:
            continue
        linha = (permitidas / s).reshape(1, -1)
        ents.append(float(row_entropy(linha)[0]))
        maxs.append(float(linha.max()))
    if not ents:
        return {"row_entropy_causal": "", "row_max_causal": "",
                "expected_row_entropy_causal": "", "expected_row_max_causal": ""}
    # O teto de cada linha depende da LARGURA DELA — `log(w)` e `1/w` —, e a
    # esperança do conjunto é a média desses tetos. NÃO é `log(T!)/T`, que é a
    # média sobre TODAS as T linhas da matriz: essa é outra quantidade, e
    # confundir as duas subestima o teto (medido: 1,6638 contra os 2,4826
    # corretos, para as linhas 10 a 12). Foi o defeito que o teste com o nulo
    # literal apanhou.
    e_ent = float(np.mean([row_ceiling_entropy(int(q) + 1) for q in idx]))
    e_max = float(np.mean([row_ceiling_max(int(q) + 1) for q in idx]))
    return {
        "row_entropy_causal": round(float(np.mean(ents)), 8),
        "row_max_causal": round(float(np.mean(maxs)), 8),
        "expected_row_entropy_causal": round(e_ent, 8),
        "expected_row_max_causal": round(e_max, 8),
    }


def _sinais_ocultos(ocultos, idx) -> dict[str, float]:
    """A família de estados ocultos, de nulo EMPÍRICO.

    Ausente quando o adaptador não os devolve: sai vazio e a ressalva fica no
    MEDIDA.json, em vez de o valor ser inventado ou a execução morrer.
    """
    if ocultos is None:
        return {"hidden_norm": "", "hidden_dist_centroide": "", "hidden_delta_camadas": ""}
    h = np.asarray(ocultos, dtype=float)                 # [camadas+1, T, d]
    ultima, penultima = h[-1], h[-2]
    v = ultima[list(idx)].mean(axis=0)
    centro = ultima.mean(axis=0)
    return {
        "hidden_norm": round(float(np.linalg.norm(v)), 8),
        "hidden_dist_centroide": round(float(np.linalg.norm(v - centro)), 8),
        "hidden_delta_camadas": round(
            float(np.linalg.norm(v - penultima[list(idx)].mean(axis=0))), 8),
    }


def main(corpus: str, split: str, caminho_modelo: str, saida: str, max_samples: int = 0):
    # DOIS CAMINHOS PARA O MESMO CORPUS, e o do jsonl existe porque os
    # carregadores vivem em `src/core/loaders/`, uma árvore que puxa `datasets`
    # e rede — inviável dentro de um contêiner faturado. O jsonl foi exportado
    # DOS MESMOS carregadores, então a ORDEM é a mesma e as duas leituras têm de
    # produzir a mesma tabela. Isso não é suposto: foi verificado linha a linha
    # em 23/09/2026 antes de qualquer medição em GPU.
    #
    # O campo é `ner_char` e não `ner`: os arquivos de TREINO do mesmo canal
    # usam índice de palavra em `ner`, e os carregadores dão posição de
    # caractere. Nomes diferentes para convenções diferentes, de propósito.
    dados = Path(os.environ.get("SENTINEL_DADOS", ""))
    arq = dados / f"{corpus}_{split}.jsonl" if str(dados) else None
    if arq is not None and arq.exists():
        from dataclasses import dataclass

        @dataclass
        class _Ent:
            start: int
            end: int
            label: str

        @dataclass
        class _Ex:
            tokens: list
            entities: list

        exemplos = []
        with arq.open(encoding="utf-8") as fh:
            for linha in fh:
                d = json.loads(linha)
                if "ner_char" not in d:
                    raise SystemExit(
                        f"{arq} sem `ner_char`. Recusando cair para `ner`, que "
                        f"neste canal está em índice de palavra — lê-lo como "
                        f"caractere daria entidades erradas SEM erro nenhum.")
                exemplos.append(_Ex([str(t) for t in d["tokenized_text"]],
                                    [_Ent(int(a), int(b), str(r))
                                     for a, b, r in d["ner_char"]]))
        print(f"corpus lido de {arq} ({len(exemplos)} sentenças)", flush=True)
    else:
        mod, classe = CARREGADOR[corpus]
        import importlib
        loader = getattr(importlib.import_module(mod), classe)()
        exemplos = loader.load_split(split)
    if max_samples:
        exemplos = exemplos[:max_samples]
    # O prompt recebe as descrições DECLARADAS do corpus, não as observadas na
    # amostra. Duas razões: são as que o treino usou, e derivar da amostra faria
    # o prompt mudar com `--max-samples` — uma corrida de 6 sentenças daria ao
    # modelo um vocabulário menor que a de 1.854, e as duas não seriam
    # comparáveis. `descricao` levanta se o carregador trouxer rótulo não
    # declarado, para que corpus novo falhe alto em vez de medir errado.
    vistos = sorted({str(e.label) for ex in exemplos for e in ex.entities})
    for r in vistos:
        descricao(corpus, r)
    rotulos = descricoes(corpus)
    print(f"{corpus}/{split}: {len(exemplos)} sentenças | rótulos {rotulos} "
          f"(do carregador: {vistos})", flush=True)

    tok = AutoTokenizer.from_pretrained(caminho_modelo)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # EAGER obrigatório: com sdpa, output_attentions devolve tupla VAZIA sem erro.
    # DISPOSITIVO vem do ambiente, com `cuda` como padrão para não mudar o
    # comportamento de quem já roda isto. A razão de existir: foi MEDIDO que a
    # medição não precisa de GPU: 4,43 s por sentença em CPU de 1 thread, e o
    # número é o COEFICIENTE entre duas corridas (n=20 em 104,0 s, n=60 em
    # 281,1 s, medidas em 18/09/2026), não uma média — a média de uma corrida só
    # embute os 15,5 s de carga do modelo e superestima. Os dois valores que este
    # comentário já trouxe estavam errados por razões diferentes: 2,53 vinha de
    # outro regime de geração, e 3,49 era 69,8/20 feito de cabeça, com a carga
    # dentro e sem célula por trás. E o
    # processo decidido é GPU só para treino. Com `.to("cuda")` fixo, o passo
    # de medição só rodava em máquina faturada — que é o desperdício que o
    # processo existe para eliminar.
    dispositivo = os.environ.get("SENTINEL_DISPOSITIVO", "cuda")
    m = AutoModelForCausalLM.from_pretrained(
        caminho_modelo, dtype=torch.float32,
        attn_implementation="eager").to(dispositivo).eval()
    # OPT-IN. Sem SENTINEL_FEATURES=1 nada muda: nenhum arquivo a mais, nenhum
    # escalar diferente. A exploracao nao pode alterar o caminho confirmatorio.
    acum = Acumulador() if os.environ.get("SENTINEL_FEATURES") == "1" else None
    ad = DecoderAdapter(m, tok, rotulos)

    linhas, n_alucinadas, n_ouro, n_descartadas = [], 0, 0, 0
    for i, ex in enumerate(exemplos):
        texto, ouro, intervalos = trechos_de_ouro(ex, corpus)
        n_ouro += len(ouro)
        try:
            r = ad(texto)
        except Exception as e:                     # noqa: BLE001
            print(f"  s{i}: FALHOU {type(e).__name__}: {str(e)[:110]}", flush=True)
            continue
        T = int(r["n_prompt"])
        A_p, A_t = r["attentions_prompt"], r["attentions_total"]
        H = r.get("hidden_states")          # [camadas+1, T, d], ou None
        n_descartadas += int(r.get("n_descartadas", 0))
        for men in r["mencoes"]:
            if not men.get("ancorada"):
                n_alucinadas += 1
                linhas.append({c: "" for c in COLUNAS} | {
                    "sentence_id": f"s{i}", "loss": 1,
                    "model_confidence": round(float(men["confidence"]), 6),
                    "is_nested": 0, "token_indices": "", "ancorada": 0,
                    "mencao": men["texto"], "rotulo": men["rotulo"]})
                continue
            idx = men["token_indices"]
            if not idx:
                continue
            ci, cf = men["char_ini"], men["char_fim"]
            k, a0 = len(idx), int(min(idx))
            acerto = (ci, cf, men["rotulo"]) in ouro
            linhas.append({
                "sentence_id": f"s{i}",
                "loss": 0 if acerto else 1,
                "model_confidence": round(float(men["confidence"]), 6),
                "is_nested": int(aninhada((ci, cf), intervalos)),
                "token_indices": "|".join(str(t) for t in idx),
                "span_size": k,
                "span_position": a0,
                "n_tokens": T,
                "expected_causal": round(float(expected_mass_causal(a0, k, T)), 8),
                "expected_bidir": round(float(expected_mass(k, T, sink_policy="keep")), 8),
                "causal_mass_prompt": round(causal_mass(A_p, idx), 8),
                "causal_mass_geracao": (
                    round(causal_mass(A_t, idx, query_positions=list(men["pos_geracao"])), 8)
                    if men["pos_geracao"] else ""),
                # AUTOFOCO: as linhas DO TRECHO olhando para as chaves do trecho.
                # É a direção "emitida" do caso bidirecional, e sob máscara ela
                # não é redundante com a recebida: a linha do primeiro token do
                # trecho não vê os tokens seguintes dele.
                "causal_mass_autofoco": round(causal_mass(A_p, idx, query_positions=idx), 8),
                **_sinais_de_atencao(A_p, idx, T),
                **_sinais_ocultos(H, idx),
                "ancorada": 1,
                "mencao": men["texto"], "rotulo": men["rotulo"],
            })
            if acum is not None:
                # `len(linhas) - 1`: a linha acabou de ser acrescentada, e este e
                # o indice dela no entities.csv. E a chave de juncao com o rotulo
                # de acerto; sem ela a sonda se ajustaria contra rotulo
                # desalinhado e daria numero plausivel e errado.
                acum.junta(len(linhas) - 1, A_p, H, idx)
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(exemplos)} sentenças, {len(linhas)} linhas", flush=True)

    d = Path(saida)
    d.mkdir(parents=True, exist_ok=True)
    with (d / "entities.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(COLUNAS), extrasaction="ignore")
        w.writeheader()
        w.writerows(linhas)
    ancoradas = [l for l in linhas if l.get("ancorada") == 1]
    erro = (sum(int(l["loss"]) for l in ancoradas) / len(ancoradas)) if ancoradas else float("nan")
    meta = {
        "corpus": corpus, "split": split, "modelo": caminho_modelo,
        "n_sentencas": len(exemplos), "n_entidades_ouro": n_ouro,
        "n_entidades_preditas": len(ancoradas),
        "n_alucinadas": n_alucinadas,
        "n_linhas_descartadas_por_formato": n_descartadas,
        "taxa_de_erro_base": round(erro, 6),
        "camadas": int(m.config.num_hidden_layers),
        "cabecas": int(m.config.num_attention_heads),
        "attn_implementation": "eager",
        "tensores_persistidos": False,
        "por_que_nao_persiste": (
            "~3 GB por corpus contra limite de transferencia de 256 MB: ficariam presos na "
            "maquina remota, cujo acesso depende de token de SSO que expira. A varredura de "
            "leituras NAO fica disponivel neste braco sem remedir."),
        "texto_medido": "palavras unidas por espaco, o MESMO que o treino viu",
        "estrato_aninhado": "da ANOTACAO, invariante do protocolo",
    }
    # FEATURES DA SONDA, quando pedidas. O registro diz tambem que os escalares
    # NAO mudaram: eles saem do mesmo caminho de codigo com ou sem esta flag, e
    # quem ler o MEDIDA.json precisa poder saber disso sem ler o fonte.
    if acum is not None:
        meta.update(acum.salvar(d / "features_sonda.npz"))
        meta["features_alteram_escalares"] = False
        meta["features_compartimento"] = (
            "EXPLORATORIO: a sonda entra como UMA comparacao por familia, "
            "ajustada so na particao de calibracao. Nao e veredito pre-registrado.")
    (d / "MEDIDA.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1),
                                   encoding="utf-8")
    print(f"\n{len(linhas)} linhas ({len(ancoradas)} ancoradas, {n_alucinadas} alucinadas) "
          f"| erro base {erro:.4f} | {d}", flush=True)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    # OS ARGUMENTOS CAEM PARA O AMBIENTE quando ausentes: a SageMaker chama
    # `python medir_decoder.py` SEM argumento nenhum, e com `required=True` o
    # job morreria na partida. Localmente nada muda — argumento explícito ganha.
    ap.add_argument("--corpus", default=os.environ.get("SENTINEL_CORPORA"),
                    choices=sorted(CARREGADOR))
    ap.add_argument("--split", default=os.environ.get("SENTINEL_SPLIT", "test"))
    ap.add_argument("--modelo", default=os.environ.get("SENTINEL_AJUSTADO"))
    ap.add_argument("--saida", default=os.environ.get("SENTINEL_SAIDA"))
    ap.add_argument("--max-samples", type=int,
                    default=int(os.environ.get("SENTINEL_LIMITE", "0")))
    a = ap.parse_args()
    faltando = [n for n in ("corpus", "modelo", "saida") if not getattr(a, n)]
    if faltando:
        raise SystemExit(f"faltam {faltando}: passe por argumento ou por "
                         f"SENTINEL_CORPORA / SENTINEL_AJUSTADO / SENTINEL_SAIDA")
    # O canal monta o TARBALL, não o conteúdo extraído — a SageMaker só extrai
    # o `model.tar.gz` de saída, nunca os de entrada.
    mp = Path(a.modelo)
    if not (mp / "config.json").exists():
        tgz = sorted(mp.glob("*.tar.gz"))
        if tgz:
            import tarfile as _tf
            alvo = Path("/tmp/ajustado_extraido"); alvo.mkdir(parents=True, exist_ok=True)
            with _tf.open(tgz[0]) as t:
                t.extractall(alvo, filter="data")
            mp = alvo
        achados = [p for p in mp.rglob("config.json")]
        if len(achados) != 1:
            raise SystemExit(f"{len(achados)} config.json sob {mp}: recusando escolher")
        a.modelo = str(achados[0].parent)
        print(f"peso encontrado em {a.modelo}", flush=True)
    main(a.corpus, a.split, a.modelo, a.saida, a.max_samples)
