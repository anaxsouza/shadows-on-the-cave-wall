# Nota sobre `monotonicidade_por_tamanho.csv` (24/09/2026)

A tabela gerada em 18/09/2026 não pode ser refeita por ninguém. O script sorteava cada
reamostragem com a semente `SEED_BASE + hash((corpus, escore))`, e o `hash` de texto do
Python muda a cada processo (`PYTHONHASHSEED`). A versão guardada em
`monotonicidade_por_tamanho_18-09-NAO-REPRODUTIVEL.csv` é a daquele dia, intacta.

A tabela `monotonicidade_por_tamanho.csv` que está aqui vem do mesmo código, com a
semente trocada por `zlib.crc32`, que é a mesma em toda máquina. Duas execuções dão o
mesmo SHA-256.

**O que muda:** as violações médias nas frações intermediárias, que são o ruído da
reamostragem, na segunda casa decimal. **O que não muda:** os valores na fração 1,00, que
não dependem de sorteio, e o padrão por fração. A leitura em `monotonicidade_leitura.md`
cita números da versão de 18/09. Os expoentes que ela relata ainda precisam ser
reconferidos contra esta versão.

É uma análise exploratória da partição de validação, e não decide nenhum veredito.
