"""O comando `selective`: os dois testes da pergunta única.

Este arquivo é só a superfície de linha de comando. A decisão científica mora em
`src/selective/` (curva risco-cobertura, AURC, valor adicionado, controle
conformal de risco) e os parâmetros que não podem ser escolhidos depois de ver o
resultado moram em `docs/tese/PREREGISTRO.md`.

Uma escolha de desenho que vale explicar, porque é o que responde à objeção R4
(cortes numéricos sem sistemática): não há opção de limiar aqui. A curva
risco-cobertura É a varredura de todos os limiares, e o ponto operacional único
é calibrado em partição separada, com o nível declarado antes de rodar. Um
`--threshold` nesta CLI seria exatamente o grau de liberdade que a banca
criticou.
"""

from __future__ import annotations

import argparse
import sys

TESTES = ("floor", "added-value")


def build_selective_parser(p: argparse.ArgumentParser) -> None:
    # A medição é uma ETAPA do comando único, não um comando novo.
    #
    # Ela é uma fase diferente dos testes: produz dado (roda o modelo sobre o
    # corpus e escreve a tabela por entidade) em vez de responder à pergunta.
    # Mas um segundo comando quebraria o "um comando só" decidido em 02/09/2026,
    # e a fase não precisa de comando próprio para ficar explícita — precisa de
    # ser explícita, e é isso que a flag faz. Sem ela nada roda o modelo: os
    # testes recusam se a tabela não existir, em vez de medir por conta própria.
    # Medir por conta própria transformaria um teste estatístico de segundos numa
    # execução de horas sem que ninguém tivesse pedido.
    grupo = p.add_mutually_exclusive_group(required=True)
    grupo.add_argument(
        "--test",
        choices=TESTES,
        dest="test",
        help=(
            "floor: a massa de atenção calibrada bate a abstenção aleatória? "
            "added-value: o ganho é incremental sobre o softmax e maior no aninhado?"
        ),
    )
    grupo.add_argument(
        "--measure",
        action="store_true",
        help=(
            "Etapa de MEDIÇÃO: roda o modelo sobre o corpus e escreve a tabela por "
            "entidade mais as fatias de atenção. É o que os testes consomem, e é a "
            "única etapa que carrega o modelo."
        ),
    )
    p.add_argument("--model", required=True, help="Chave do modelo (ver configs/config.yaml)")
    p.add_argument(
        "--dataset",
        required=True,
        choices=("conll2003", "genia"),
        help="conll2003 (plano) ou genia (aninhado) — o contraste é metade da pergunta",
    )
    p.add_argument("--config", default="configs/config.yaml",
                   help="A DECLARAÇÃO em vigor: só itens pré-registrados, e é o que o hash cobre")
    p.add_argument("--registry", default=None,
                   help="Registro OPERACIONAL (modelos e descrições de rótulo); "
                        "padrão configs/config.yaml. Separado da declaração para que uma "
                        "declaração nova possa medir sem duplicar entrada do modelo")
    p.add_argument("--split", default="test", help="Partição de avaliação")
    p.add_argument(
        "--max-samples",
        type=int,
        default=-1,
        dest="max_samples",
        help="Limite de sentenças (-1 = todas). Declare no relatório se usar.",
    )
    p.add_argument("--seed", type=int, default=42, help="Semente")
    p.add_argument("--output-dir", dest="output_dir", help="Diretório de saída dos resultados")
    p.add_argument("--dry-run", action="store_true", help="Mostra o plano sem executar")


def run_selective(args: argparse.Namespace) -> int:
    if getattr(args, "measure", False):
        return _run_measure(args)

    from src.selective.runner import SelectiveRunner

    runner = SelectiveRunner(
        model_key=args.model,
        dataset_name=args.dataset,
        config_path=args.config,
        split=args.split,
        max_samples=args.max_samples,
        seed=args.seed,
        output_dir=args.output_dir,
    )

    if args.dry_run:
        for linha in runner.plan(args.test):
            print(linha)
        return 0

    try:
        resultado = runner.run(args.test)
    except NotImplementedError as e:
        print(f"Ainda não implementado: {e}", file=sys.stderr)
        return 2

    print(resultado.render())
    return 0


def _run_measure(args: argparse.Namespace) -> int:
    """A etapa de medição: a única que carrega o modelo.

    Ela existe separada dos testes por uma razão que não é de organização: os
    testes são estatística sobre uma tabela e rodam em segundos; medir é rodar o
    modelo sobre milhares de sentenças. Juntar as duas faria um teste que se
    esperava rápido virar uma execução de horas.
    """
    import yaml

    from src.selective.gliner_adapter import GLiNERAdapter
    from src.selective.measurement import measure
    from src.selective.preregistration import load_preregistration

    # DOIS arquivos, e a separação não é conveniência.
    #
    # `--config` é a DECLARAÇÃO: só o que foi pré-registrado, e é o que o hash
    # cobre. `--registry` é o registro OPERACIONAL: quais modelos existem e quais
    # descrições de rótulo se dão a eles. Antes de 15/09/2026 os dois vinham do
    # mesmo arquivo, e a consequência era concreta: uma declaração nova
    # (decl-02, decl-03, decl-04) não podia ser usada para medir, porque o
    # arquivo dela só tem a seção `selective`. A mensagem de erro chegou a
    # nomear `configs/config.yaml` enquanto lia outro arquivo.
    #
    # Os rótulos ficam no registro e NÃO na declaração de propósito: duplicá-los
    # por declaração criaria duas fontes para uma entrada do modelo, que é
    # exatamente o tipo de divergência que não aparece como erro.
    prereg = load_preregistration(args.config)
    registro = args.registry or "configs/config.yaml"
    cfg = yaml.safe_load(open(registro, encoding="utf-8"))

    mapa = (cfg.get("gliner_labels") or {}).get(args.dataset)
    if not mapa:
        print(
            f"{registro} não declara gliner_labels para '{args.dataset}'. As descrições de "
            f"rótulo são ENTRADA do modelo — o GLiNER pontua trechos contra elas — então não "
            f"há padrão em código para elas.",
            file=sys.stderr,
        )
        return 2

    modelo_id = ((cfg.get("models") or {}).get(args.model) or {}).get("model_name")
    if not modelo_id:
        print(f"modelo '{args.model}' não está em {registro}", file=sys.stderr)
        return 2

    # O modelo pedido tem de ser o DECLARADO, quando a declaração fixa modelo.
    # Sem esta conferência, medir com --model gliner-base sob a decl-04 escreveria
    # uma tabela do base carregando o measurement_hash do large, e o guarda de
    # procedência do runner a aceitaria: o erro entraria pela porta da medição,
    # que é onde ele é mais difícil de ver depois.
    if prereg.model is not None and prereg.model != modelo_id:
        print(
            f"{prereg.declaration_id} declara o modelo {prereg.model!r}, e --model "
            f"{args.model!r} resolve para {modelo_id!r}. Medir assim produziria uma tabela de "
            f"um modelo carregando a declaração de outro.",
            file=sys.stderr,
        )
        return 2

    saida = args.output_dir or f"results/{args.model}/{args.dataset}/{args.split}"

    if args.dry_run:
        print(prereg.render())
        print()
        print(f"MEDIÇÃO (plano, nada foi executado)")
        print(f"  modelo            {args.model}  ->  {modelo_id}")
        print(f"  corpus            {args.dataset}, partição {args.split}")
        print(f"  descrições dadas  {', '.join(mapa)}")
        print(f"  rótulos do corpus {', '.join(sorted(set(mapa.values())))}")
        print(f"  limite            {'todas' if args.max_samples < 0 else args.max_samples} sentenças")
        print(f"  saída             {saida}/entities.csv")
        return 0

    # O MODELO CARREGA ANTES DO CORPUS, e a ordem não é arbitrária.
    #
    # Onde numpy/scipy trazem um runtime OpenMP e o wheel do torch traz outro, o
    # processo ABORTA (SIGSEGV) ao inicializar o modelo depois da pilha do
    # `datasets` — observado nesta máquina em 02/09/2026, com KMP_DUPLICATE_LIB_OK
    # já ligado. Carregar o modelo primeiro não tem custo e não acontece.
    #
    # É um defeito de ambiente e não do repositório, mas a ordem é a diferença
    # entre a medição rodar e morrer no meio, então fica fixada aqui com a razão.
    from gliner import GLiNER

    modelo = GLiNER.from_pretrained(modelo_id)
    modelo.eval()

    from src.core.loaders.biomedical.genia import GENIALoader
    from src.core.loaders.conll.loader import CONLLLoader

    Loader = {"conll2003": CONLLLoader, "genia": GENIALoader}[args.dataset]
    exemplos = Loader().load_split(args.split)
    if args.max_samples and args.max_samples > 0:
        exemplos = exemplos[: args.max_samples]
    adaptador = GLiNERAdapter(
        modelo, prereg=prereg, labels=list(mapa), label_to_corpus=mapa
    )
    relatorio = measure(
        sentences=exemplos, predict=adaptador, prereg=prereg, output_dir=saida,
        model_id=modelo_id,
    )
    print(relatorio.render())
    return 0
