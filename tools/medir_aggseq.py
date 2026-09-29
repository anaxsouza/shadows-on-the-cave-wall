"""AggSeq — a confiança por AGREGAÇÃO SOBRE O FEIXE, medida à parte.

POR QUE ESTE ARQUIVO EXISTE, E POR QUE SEPARADO

A linha de base do estudo é `model_confidence`: a média geométrica das
probabilidades dos tokens que o modelo escreveu para aquele trecho. A revisão
sistemática de 17/09/2026 encontrou um estimador melhor NA TAREFA EXATA —
rotulação sequencial generativa — que agrega a probabilidade de SEQUÊNCIA sobre
o feixe de busca em vez de ler um caminho só, e reduz o erro de calibração
esperado no NER de 0,075 para 0,030 (ACL UncertaiNLP 2024, mT5-base ~580M).

A consequência para o relato, e é ela que justifica o custo: com esta linha de
base no lugar, a afirmação deixa de ser "os sinais internos não superam a
confiança do modelo" e passa a ser "não superam nem a confiança nem o melhor
estimador publicado". A segunda é muito mais difícil de atacar.

SEPARADO do `medir_decoder.py` por três razões, e nenhuma é organização:

1. CUSTO. A passagem do adaptador é GULOSA. Não há feixe para ler — eu afirmei
   que havia e estava errado. Medido no 0,5B ajustado, CPU de 1 thread: guloso
   2,31 s/sentença, feixe 5 com 5 retornos 7,25 s (3,13x). Sobre as 5.307
   sentenças e os dois pontos do eixo, são +21,4 h. Por isso esta passagem roda
   na GPU, num job que acaba — decisão do autor em 18/09/2026, e exceção
   DECLARADA à regra "GPU só para treino", cujo propósito (não pagar máquina
   ociosa) um job em lote serve.

2. O CAMINHO CONFIRMATÓRIO NÃO PODE MUDAR. Os sinais internos continuam medidos
   pelo `medir_decoder.py` em CPU, exatamente como declarado. Se o feixe entrasse
   naquele arquivo, o número de um veredito pré-registrado passaria a depender de
   uma decisão tomada depois da declaração.

3. ALINHAMENTO. O que se pontua são os trechos que a passagem GULOSA prediz —
   são as predições do sistema. Para cada um, o AggSeq é a massa de probabilidade
   do feixe que contém aquele par (menção, rótulo). Isso é bem definido mesmo
   quando o primeiro feixe difere do guloso, e é por isso que a medição pode
   rodar em outro lugar: a chave de junção é (sentence_id, mencao, rotulo), que
   não depende de onde a conta foi feita.

k = 5, E A ESCOLHA FOI MEDIDA. Feixe 3 custa 6,54 s e feixe 5 custa 7,25 s: o
custo é dominado pela passagem em lote, não por k. Havendo feixe, k=5 sai quase
de graça sobre k=3, então o maior poder é escolhido antes de ver resultado.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# DOIS LAYOUTS, e o caminho de import serve os dois. No repositório este
# arquivo está em `tools/` e os módulos em `../src/`; no contêiner tudo é PLANO
# em `/opt/ml/code`, com `src/selective/` ao lado. A versão anterior inseria só
# `RAIZ` e `RAIZ/tools`, que no contêiner apontam para `/opt/ml` e um diretório
# inexistente — ela funcionava lá por ACIDENTE, porque a SageMaker já põe
# `/opt/ml/code` no PYTHONPATH. Inserir o diretório do próprio arquivo tira o
# acidente do caminho crítico.
AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parent
for p in (RAIZ, RAIZ / "tools", AQUI):
    if p.is_dir():
        sys.path.insert(0, str(p))
from rotulos import descricoes                                   # noqa: E402
from src.selective.decoder_adapter import parse_saida_contando, prompt_de  # noqa: E402

FEIXES = 5
MAX_NEW = 96
COLUNAS = ("sentence_id", "mencao", "rotulo", "aggseq", "n_feixes_com",
           "aggseq_top1", "massa_total_feixe")


def aggseq_da_sentenca(m, tok, texto: str, rotulos, dispositivo: str) -> dict:
    """Massa de probabilidade do feixe, por par (menção, rótulo).

    `sequences_scores` do transformers é o log-prob da sequência JÁ normalizado
    pelo comprimento (length_penalty=1.0 por padrão). Uso-o como está e declaro:
    desnormalizar favoreceria sequências curtas, e escolher a normalização depois
    de ver o resultado seria grau de liberdade sobre a linha de base.

    O denominador é a soma sobre os feixes RETORNADOS, não sobre todo o espaço —
    é uma probabilidade condicionada ao feixe, que é o que a referência agrega.
    Chamar isso de probabilidade absoluta seria afirmar mais do que se mede.
    """
    cod = tok(prompt_de(texto, rotulos), return_tensors="pt")
    cod = {k: v.to(m.device) for k, v in cod.items()}
    n_prompt = int(cod["input_ids"].shape[1])
    with torch.no_grad():
        ger = m.generate(**cod, max_new_tokens=MAX_NEW, num_beams=FEIXES,
                         num_return_sequences=FEIXES, do_sample=False,
                         output_scores=True, return_dict_in_generate=True,
                         pad_token_id=tok.pad_token_id)
    escores = ger.sequences_scores.float().cpu().numpy()
    # Softmax sobre os log-probs dos feixes: estável e equivalente a normalizar
    # exp(score) pela soma. Sem isso, scores muito negativos subfluem para zero e
    # o denominador viraria 0 — falha que produziria aggseq = nan em silêncio.
    pesos = np.exp(escores - escores.max())
    pesos = pesos / pesos.sum()
    descartadas = 0
    por_par: dict[tuple[str, str], float] = {}
    contagem: dict[tuple[str, str], int] = {}
    topo: set[tuple[str, str]] = set()
    for b in range(len(escores)):
        saida = tok.decode(ger.sequences[b][n_prompt:], skip_special_tokens=True)
        # `parse_saida_contando` e nao `parse_saida`: a primeira devolve a LISTA
        # e a segunda a tupla com a contagem de descartes. Eu desempacotei a
        # lista por engano e funcionou por acidente quando havia exatamente duas
        # mencoes. E a contagem importa aqui pela razao do docstring dela:
        # descarte silencioso nao e ausencia — se um feixe perde o formato, a
        # massa dele nao e atribuida a mencao nenhuma e o aggseq sai otimista.
        pares, desc = parse_saida_contando(saida)
        descartadas += desc
        vistos = set()
        for men, rot in pares:
            ch = (men.strip(), rot.strip())
            if ch in vistos:      # repetição DENTRO de um feixe não soma duas vezes:
                continue          # a massa é da sequência, não da menção.
            vistos.add(ch)
            por_par[ch] = por_par.get(ch, 0.0) + float(pesos[b])
            contagem[ch] = contagem.get(ch, 0) + 1
        if b == 0:
            topo = vistos
    return {"pares": por_par, "contagem": contagem, "topo": topo,
            "massa_total": float(pesos.sum()), "descartadas": descartadas}


def main() -> None:
    ap = argparse.ArgumentParser()
    # OS ARGUMENTOS CAEM PARA O AMBIENTE quando ausentes, porque a SageMaker
    # chama `python medir_aggseq.py` SEM argumento nenhum — com `required=True`
    # o job morreria na partida. O caminho do peso ajustado vem do canal
    # `ajustado`, montado pela submissão.
    ap.add_argument("--corpus", default=os.environ.get("SENTINEL_CORPORA"))
    ap.add_argument("--split", default=os.environ.get("SENTINEL_SPLIT", "test"))
    ap.add_argument("--modelo", default=os.environ.get("SENTINEL_AJUSTADO"))
    ap.add_argument("--saida", default=os.environ.get("SENTINEL_SAIDA"))
    ap.add_argument("--max-samples", type=int,
                    default=int(os.environ.get("SENTINEL_LIMITE", "0")))
    a = ap.parse_args()
    faltando = [n for n in ("corpus", "modelo", "saida") if not getattr(a, n)]
    if faltando:
        raise SystemExit(
            f"faltam {faltando}: passe por argumento ou por SENTINEL_CORPORA / "
            f"SENTINEL_AJUSTADO / SENTINEL_SAIDA. Falhar aqui e nao adivinhar, "
            f"porque um caminho errado mediria outro peso em silencio.")
    # O canal monta o tarball JA extraido ou o proprio .tar.gz, conforme o job.
    # Descobrir qual e olhando, em vez de supor.
    mp = Path(a.modelo)
    # O CANAL MONTA O TARBALL, não o conteúdo — a SageMaker extrai só o
    # `model.tar.gz` de SAÍDA, nunca os de entrada. Medido em 22/09/2026: o
    # canal `ajustado` continha model.tar.gz e nenhum config.json, e o guard
    # abaixo recusou medir. Extrair aqui é o que faltava.
    if not (mp / "config.json").exists():
        tgz = sorted(mp.glob("*.tar.gz"))
        if tgz:
            import tarfile as _tf
            alvo = Path("/tmp/ajustado_extraido")
            alvo.mkdir(parents=True, exist_ok=True)
            with _tf.open(tgz[0]) as t:
                t.extractall(alvo, filter="data")
            print(f"tarball {tgz[0].name} extraido em {alvo}", flush=True)
            mp = alvo
            a.modelo = str(alvo)
    if not (mp / "config.json").exists():
        sub = [p for p in mp.rglob("config.json")]
        if len(sub) != 1:
            raise SystemExit(f"{len(sub)} config.json sob {mp}: nao da para escolher sozinho")
        a.modelo = str(sub[0].parent)
        print(f"peso encontrado em {a.modelo}", flush=True)

    # LÊ O MESMO JSONL QUE O TREINO LEU, e não os carregadores do projeto.
    # Três razões, e a terceira é a que decide:
    #  1. os carregadores vivem em `src/core/loaders/`, uma árvore grande que
    #     puxa `datasets` e rede — embarcá-la num contêiner faturado é frágil;
    #  2. o contêiner já monta este canal, porque o treino depende dele;
    #  3. PROCEDÊNCIA: assim é VERIFICÁVEL que o feixe mediu o mesmo corpus que
    #     o treino viu, em vez de um corpus re-derivado no dia da medição.
    # O `test` foi exportado DOS MESMOS carregadores em 22/09/2026, o que
    # preserva a ORDEM — é ela que faz `aggseq.csv` casar com `entities.csv`
    # por `sentence_id`. Os hashes estão em `dados_decoder/MANIFESTO_test.json`.
    dados = Path(os.environ.get("SENTINEL_DADOS", "dados_decoder"))
    arq = dados / f"{a.corpus}_{a.split}.jsonl"
    if not arq.exists():
        raise SystemExit(f"{arq} ausente. O canal `dados` precisa conter o split "
                         f"{a.split}; falhar aqui e nao cair para outro corpus.")
    exemplos = []
    with arq.open(encoding="utf-8") as fh:
        for linha in fh:
            exemplos.append({"texto": " ".join(json.loads(linha)["tokenized_text"])})
    if a.max_samples:
        exemplos = exemplos[: a.max_samples]
    rotulos = descricoes(a.corpus)

    dispositivo = os.environ.get("SENTINEL_DISPOSITIVO", "cuda")
    tok = AutoTokenizer.from_pretrained(a.modelo)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # `eager` NÃO é necessário aqui: este arquivo não lê atenção nenhuma, só
    # probabilidade de sequência. Deixar o padrão é mais rápido, e a diferença
    # não toca em nada declarado — a atenção continua vindo do outro caminho.
    m = AutoModelForCausalLM.from_pretrained(
        a.modelo, dtype=torch.float32).to(dispositivo).eval()

    linhas, n_desc = [], 0
    for i, ex in enumerate(exemplos):
        texto = ex["texto"]
        try:
            r = aggseq_da_sentenca(m, tok, texto, rotulos, dispositivo)
        except Exception as e:                            # noqa: BLE001
            print(f"  s{i}: FALHOU {type(e).__name__}: {str(e)[:110]}", flush=True)
            continue
        n_desc += r["descartadas"]
        for (men, rot), massa in r["pares"].items():
            linhas.append({
                "sentence_id": f"s{i}", "mencao": men, "rotulo": rot,
                "aggseq": round(massa, 8),
                "n_feixes_com": r["contagem"][(men, rot)],
                "aggseq_top1": int((men, rot) in r["topo"]),
                "massa_total_feixe": round(r["massa_total"], 8),
            })
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(exemplos)} sentenças, {len(linhas)} pares", flush=True)

    d = Path(a.saida)
    d.mkdir(parents=True, exist_ok=True)
    with (d / "aggseq.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(COLUNAS), extrasaction="ignore")
        w.writeheader()
        w.writerows(linhas)
    (d / "AGGSEQ.json").write_text(json.dumps({
        "corpus": a.corpus, "split": a.split, "modelo": a.modelo,
        "n_sentencas": len(exemplos), "n_pares": len(linhas),
        "n_linhas_descartadas_por_formato": n_desc,
        "feixes": FEIXES, "max_new_tokens": MAX_NEW,
        "dispositivo": dispositivo,
        "escore": "sequences_scores do transformers, log-prob JA normalizado por comprimento",
        "denominador": "soma sobre os feixes RETORNADOS — probabilidade condicionada ao feixe",
        "chave_de_juncao": "(sentence_id, mencao, rotulo)",
        "papel": "LINHA DE BASE, nao sinal interno: e um estimador sobre a SAIDA",
        "excecao_declarada": (
            "roda na GPU, fora da regra 'GPU so para treino'. O proposito da regra "
            "(nao pagar maquina ociosa) fica servido por um job em lote que acaba. "
            "Os sinais internos continuam medidos em CPU pelo medir_decoder.py."),
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n{len(linhas)} pares de {len(exemplos)} sentenças | {d}", flush=True)


if __name__ == "__main__":
    main()
