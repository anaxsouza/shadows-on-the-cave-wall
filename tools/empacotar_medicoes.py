"""Monta `dados_medidos/`: as tabelas de medição que a análise lê, sem texto de corpus.

O QUE SAI E O QUE NÃO SAI. Sai tudo o que a análise declarada lê: acerto, confiança,
sinais, índices de token, comprimento das frases. Não sai nenhum texto: a coluna
`mencao` do braço decoder, que carrega o trecho escrito pelo modelo, é trocada pelo
seu SHA-256 truncado em 16 hexadecimais. O CoNLL-2003 é texto da Reuters e não pode
ser redistribuído. A troca preserva a junção com o feixe (a chave é a mesma função
aplicada aos dois lados), e a análise não usa o texto para mais nada.

Uso: python tools/empacotar_medicoes.py <SENTINEL>/results <medicoes_decoder> dados_medidos [<medicao_decl07_08>]

As medições de decl-07/08 ficam em pasta PRÓPRIA (`gliner_base-ft-*__sinais`): elas
têm as colunas das três famílias, e as de decl-05/06 não. Os pesos e as predições são
os mesmos, e isso é conferido em reproduzir/remedir.py.
"""
import hashlib, json, shutil, sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.selective.comparisons import sentence_lengths

ENCODER = [("gliner-base", "genia"), ("gliner-base", "conll2003"), ("gliner-large", "genia"),
           ("gliner-large", "conll2003"), ("gliner_base-ft-genia", "genia"),
           ("gliner_base-ft-conll2003", "conll2003")]
DECODER = [("qwen05b-ft-genia", "genia"), ("qwen05b-ft-conll2003", "conll2003"),
           ("qwen15b-ft-genia", "genia"), ("qwen15b-ft-conll2003", "conll2003")]


def h(txt: str) -> str:
    return hashlib.sha256(txt.encode("utf-8")).hexdigest()[:16]


def main(res: Path, dec: Path, out: Path, sinais: Path | None = None) -> None:
    man = {}
    for mod, corpus in ENCODER:
        o = out / mod / corpus / "test"; o.mkdir(parents=True, exist_ok=True)
        d = res / mod / corpus / "test"
        shutil.copy2(d / "entities.csv", o / "entities.csv")
        L = sentence_lengths(d / "attention")          # dos tensores, se o csv não existir
        pd.DataFrame(sorted(L.items()), columns=["sentence_id", "n_tokens"]) \
          .to_csv(o / "sentence_lengths.csv", index=False)
        shutil.copy2(d / "MEDIDA.json", o / "MEDIDA.json")
        man[f"{mod}/{corpus}"] = len(L)
    # A VALIDAÇÃO do gliner-base e os perfis por camada: lidos pelo piloto (decl-01) e
    # pela monotonicidade, que são análises da partição de validação ou exploratórias.
    for corpus in ("genia", "conll2003"):
        for split in ("validation", "test"):
            d = res / "gliner-base" / corpus / split
            o = out / "gliner-base" / corpus / split; o.mkdir(parents=True, exist_ok=True)
            for arq in ("entities.csv", "layer_profile.csv", "MEDIDA.json"):
                shutil.copy2(d / arq, o / arq)
            if split == "validation":
                L = sentence_lengths(d / "attention")
                pd.DataFrame(sorted(L.items()), columns=["sentence_id", "n_tokens"]) \
                  .to_csv(o / "sentence_lengths.csv", index=False)
    for mod, corpus in (ENCODER[4:] if sinais else []):
        o = out / f"{mod}__sinais" / corpus / "test"; o.mkdir(parents=True, exist_ok=True)
        d = sinais / mod / corpus / "test"
        shutil.copy2(d / "entities.csv", o / "entities.csv")
        L = sentence_lengths(d / "attention")
        pd.DataFrame(sorted(L.items()), columns=["sentence_id", "n_tokens"]) \
          .to_csv(o / "sentence_lengths.csv", index=False)
        shutil.copy2(d / "MEDIDA.json", o / "MEDIDA.json")
        man[f"{mod}__sinais/{corpus}"] = len(L)
    for mod, corpus in DECODER:
        o = out / mod / corpus / "test"; o.mkdir(parents=True, exist_ok=True)
        e = pd.read_csv(dec / "chave" / mod / corpus / "test" / "entities.csv",
                        keep_default_na=False, dtype=str)
        e["mencao"] = e["mencao"].map(h)
        e.to_csv(o / "entities.csv", index=False)
        shutil.copy2(dec / "gpu_feat" / mod / corpus / "test" / "sondas.csv", o / "sondas.csv")
        shutil.copy2(dec / "chave" / mod / corpus / "test" / "MEDIDA.json", o / "MEDIDA.json")
        a = pd.read_csv(dec / "aggseq" / f"{mod}.csv", keep_default_na=False, dtype=str)
        a["mencao"] = a["mencao"].map(h)
        a.to_csv(o / "aggseq.csv", index=False)
        man[f"{mod}/{corpus}"] = int(e["sentence_id"].nunique())
    arqs = {str(p.relative_to(out)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(out.rglob("*")) if p.is_file() and p.name != "MANIFESTO.json"}
    (out / "MANIFESTO.json").write_text(json.dumps(
        {"sentencas_por_ponto": man, "sha256": arqs}, indent=1, ensure_ascii=False))
    print(f"{len(arqs)} arquivos em {out}")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]),
         Path(sys.argv[4]) if len(sys.argv) > 4 else None)
