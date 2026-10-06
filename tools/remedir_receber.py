"""Recebe a saída de um job de remedição Qwen: baixa, extrai, tira o texto do corpus e roda a guarda.

Uso: python tools/remedir_receber.py 05b-genia
Do model.tar.gz do job guarda só `remedicao.csv` (sem a coluna `mencao`, que é
TEXTO DO CORPUS) e `MEDIDA_remedicao.json`; o `entities.csv` do job (que tem a
menção) não é guardado. A guarda compara com `saida/decoder/<ponto>/<corpus>/test/entities.csv`.
"""
import csv, io, json, os, sys, tarfile
from pathlib import Path
import boto3
sys.path.insert(0, str(Path(__file__).resolve().parent))
from remedir_guarda import guarda

RAIZ = Path(__file__).resolve().parents[1]
BUCKET = os.environ["SENTINEL_BUCKET"]  # private bucket, set by whoever runs the jobs
PONTOS = {"05b-genia": ("qwen05b-ft-genia", "genia"), "05b-conll": ("qwen05b-ft-conll2003", "conll2003"),
          "15b-genia": ("qwen15b-ft-genia", "genia"), "15b-conll": ("qwen15b-ft-conll2003", "conll2003")}

def main(p):
    ponto, corpus = PONTOS[p]
    s3 = boto3.client("s3", region_name="sa-east-1")
    corpo = s3.get_object(Bucket=BUCKET, Key=f"sentinel/saida/remedir-{p}/output/model.tar.gz")["Body"].read()
    dest = RAIZ / "saida/remedicao" / ponto / corpus / "test"
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(corpo)) as tf:
        nomes = tf.getnames()
        rem = tf.extractfile("remedicao.csv").read().decode("utf-8")
        meta = json.loads(tf.extractfile("MEDIDA_remedicao.json").read())
    leitor = csv.DictReader(io.StringIO(rem))
    campos = [c for c in leitor.fieldnames if c != "mencao"]
    with (dest / "remedicao.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=campos, extrasaction="ignore")
        w.writeheader(); w.writerows(leitor)
    (dest / "MEDIDA_remedicao.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False), encoding="utf-8")
    g = guarda(dest / "remedicao.csv", RAIZ / "saida/decoder" / ponto / corpus / "test/entities.csv", tipo="decoder")
    (dest / "guarda.json").write_text(json.dumps(g, indent=1, ensure_ascii=False), encoding="utf-8")
    print(nomes); print(json.dumps(g, indent=1))

if __name__ == "__main__":
    main(sys.argv[1])
