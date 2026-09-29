"""A impressão digital dos pesos de um checkpoint.

POR QUE ESTE ARQUIVO EXISTE

O `measurement_hash` do pré-registro identifica o modelo pelo NOME. Um retreino
produz o mesmo nome com pesos diferentes, e o guarda aceitaria — então a
identidade do modelo tem de incluir o conteúdo. Em 16/09/2026 isso foi feito à
mão na instância, com uma receita que NÃO ficou registrada: os dois valores
gravados (`743a9a791ba81ae7cf9f7b0184ba3dbe` e `fa7e25b022cd5f3f5da80e5e0611bf9a`,
em `docs/tese/resultados/procedencia/`) não reproduzem por md5 nem por sha256,
por arquivo nem por nenhuma das 24 combinações testadas. Um número que não se
sabe recalcular não é procedência — é anotação.

A RECEITA, declarada aqui e não escolhida depois

sha256 sobre os bytes dos ARQUIVOS DE PESO apenas (`*.bin`, `*.safetensors`,
`*.pt`, `*.pth`), em ordem alfabética de nome-base, com o nome-base e o tamanho
entrando no fluxo antes de cada conteúdo. Hex completo, 64 caracteres.

Três escolhas, e a razão de cada uma:

1. SÓ OS PESOS. Tokenizador e config ficam fora. Eles não são o que o treino
   produz, e incluí-los faria um reempacotamento de tokenizador parecer um
   modelo diferente — falso positivo no guarda, que é o defeito que desativa
   guardas.

2. O NOME-BASE E O TAMANHO ENTRAM NO FLUXO. Sem isso, um checkpoint dividido em
   dois arquivos (`model-00001-of-00002`) e a concatenação dos mesmos bytes num
   arquivo só dariam o mesmo valor, e são checkpoints diferentes de carregar.

3. OS BYTES, E NÃO OS TENSORES. Um hash sobre o `state_dict` seria mais limpo
   conceitualmente — detectaria mudança de peso e não de empacotamento — mas
   exigiria o carregador certo para cada tipo de modelo (GLiNER e HF causal são
   diferentes), e um erro de carregador produziria valor plausível e errado. O
   custo aceito, declarado: reempacotar o MESMO modelo de `.bin` para
   `.safetensors` muda a impressão. É falso positivo, não falso negativo, e é o
   lado seguro.

O comprimento distingue as receitas sem ambiguidade: a antiga tem 32
caracteres, esta tem 64. Nenhuma remedição é necessária — o que o valor prova é
identidade do peso a partir de agora, não retroativamente.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

PADROES = ("*.bin", "*.safetensors", "*.pt", "*.pth")

# O ESTADO DE RETOMADA NÃO É O MODELO, e a exclusão é load-bearing, não
# cosmética. Um checkpoint do `Trainer` traz `optimizer.pt` (momentos do Adam,
# o DOBRO do tamanho dos pesos), `scheduler.pt` e `rng_state.pth`. Sem esta
# lista, `*.pt` os capturaria e a impressão passaria a depender de estado que o
# modelo carregado nem lê — e no braço encoder ela cobriu só `pytorch_model.bin`,
# então o mesmo campo significaria coisas diferentes nos dois braços. Isso é
# pior que não ter o campo: um valor que se lê como comparável e não é.
EXCLUIDOS = ("optimizer.pt", "scheduler.pt", "rng_state.pth", "training_args.bin")
RECEITA = "sha256-bytes-dos-pesos-v1"


def arquivos_de_peso(diretorio: Path) -> list[Path]:
    """Os arquivos de peso, sem repetição e em ordem determinística."""
    achados: set[Path] = set()
    for padrao in PADROES:
        achados.update(diretorio.glob(padrao))
    return sorted((p for p in achados if p.name not in EXCLUIDOS),
                  key=lambda p: p.name)


def impressao_digital(diretorio: str | Path, bloco: int = 8 << 20) -> dict:
    """sha256 dos bytes dos pesos. Ver a receita no topo do arquivo.

    Devolve também a lista de arquivos e tamanhos, porque um valor sozinho não
    diz sobre o que foi calculado — e um hash cujo escopo não se sabe tem o
    mesmo problema que o `measurement_hash` por nome.
    """
    d = Path(diretorio)
    pesos = arquivos_de_peso(d)
    if not pesos:
        raise FileNotFoundError(
            f"nenhum arquivo de peso {PADROES} em {d} — um hash de diretório "
            f"vazio seria constante, e constante não identifica nada")
    h = hashlib.sha256()
    detalhe = []
    for p in pesos:
        tamanho = p.stat().st_size
        h.update(p.name.encode("utf-8"))
        h.update(str(tamanho).encode("utf-8"))
        with p.open("rb") as fh:
            while pedaco := fh.read(bloco):
                h.update(pedaco)
        detalhe.append({"arquivo": p.name, "bytes": tamanho})
    return {"checkpoint_sha256_v1": h.hexdigest(), "receita": RECEITA,
            "arquivos": detalhe}


if __name__ == "__main__":
    # Conferência do auditor: python tools/impressao_digital.py <dir> [<dir> ...]
    # Compara cada diretório com PESOS.json pelo nome da pasta.
    import json
    import sys

    manifesto = {p["nome"]: p["impressao"] for p in
                 json.load(open(Path(__file__).resolve().parents[1] / "PESOS.json"))["pesos"]}
    falhou = False
    for arg in sys.argv[1:]:
        d = Path(arg)
        got = impressao_digital(d)["checkpoint_sha256_v1"]
        esp = manifesto.get(d.name.rstrip("/"))
        estado = "sem entrada em PESOS.json" if esp is None else ("CONFERE" if got == esp else "DIVERGE")
        falhou |= estado != "CONFERE"
        print(f"{d.name:28s} {estado:10s} {got}")
    sys.exit(1 if falhou else 0)
