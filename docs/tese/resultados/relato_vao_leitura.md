# A grade do vão — relato exploratório dos modelos já medidos

**Estatuto: EXPLORATÓRIO, e a razão não é cerimônia.** Os vereditos destes arranjos já são
confirmatórios sob as declarações assinadas deles (`decl-04`, `decl-05`, `decl-06`). O que este
documento faz é reapresentar as MESMAS tabelas por entidade sob um instrumento de relato melhor.
Mudança de relato não é veredito novo. Se a grade do vão mudar algum veredito em relação à grade
absoluta, isso não pode ser afirmado como confirmatório — viraria candidato a uma declaração
irmã v5, assinada, herdando a medição.

**Dado:** `relato_vao_exploratorio.csv` (324 linhas) e `relato_vao_notas.txt`
**Configuração:** `configs/relato-vao-*.yaml`, que não se chamam `decl-` de propósito

---

## O defeito que a grade conserta, medido

A grade absoluta declarava metas de precisão fixas: 70%, 80%, 90%, 95%. O problema é que uma
meta absoluta mede a qualidade do **arranjo** e não a pergunta. Contando quantas metas exigem
carga de revisão **zero** — isto é, são atingidas sem abster-se de nada:

| grade | metas triviais |
|---|---|
| absoluta | **3 de 16** |
| do vão | **0 de 16** |

Nas três triviais não havia o que supervisionar, e a comparação entre supervisores não
distinguia nada — o que é fácil de confundir com "os supervisores são equivalentes".

## As metas que a grade do vão deriva

A fração é declarada; a meta absoluta é derivada de `p_base + fração × (1 − p_base)`, com
`p_base` medido na partição de **calibração**:

| arranjo | `p_base` (calibração) | 25% | 50% | 75% | 90% |
|---|---:|---:|---:|---:|---:|
| `gliner-large` · GENIA | 0,5741 | 0,6806 | 0,7870 | 0,8935 | 0,9574 |
| `gliner-large` · CoNLL | 0,5356 | 0,6517 | 0,7678 | 0,8839 | 0,9536 |
| ajustado · GENIA | 0,7448 | 0,8086 | 0,8724 | 0,9362 | 0,9745 |
| ajustado · CoNLL | 0,8915 | 0,9186 | 0,9457 | 0,9729 | 0,9891 |

Note o que isso faz: a mesma fração declarada produz metas muito diferentes em arranjos de
qualidade diferente, e é exatamente por isso que ela é comparável. "Fechar metade do vão" é o
mesmo **esforço** em qualquer extrator; "atingir 80% de precisão" não é.

## O veredito, agora legível em todos os arranjos

`T2` — a massa de atenção contra a confiança do modelo, carga de revisão exigida:

| arranjo | vão | massa | confiança | veredito |
|---|---:|---:|---:|---|
| `gliner-large` · GENIA | todas | inalcançável | 0,37 a 1,00 | a massa não atinge meta nenhuma |
| `gliner-large` · CoNLL | 25% | 0,9988 | 0,2267 | massa exige mais (Δ +0,7721) |
| | 90% | 0,9994 | 0,9910 | massa exige mais (Δ +0,0084) |
| ajustado · GENIA | 25% | 0,9493 | **0,1170** | massa exige mais (Δ +0,8323) |
| | 90% | 0,9997 | 0,9753 | massa exige mais (Δ +0,0244) |
| ajustado · CoNLL | 25% | 0,9998 | **0,0294** | massa exige mais (Δ +0,9703) |
| | 50% | 0,9998 | 0,1189 | massa exige mais (Δ +0,8809) |

A leitura mais clara está no arranjo melhor: para fechar um quarto do vão no CoNLL ajustado, a
confiança do modelo exige revisar **2,9%** das predições e a massa de atenção exige revisar
**99,98%**. Mesma exigência de qualidade, mesma tabela, ordenações diferentes.

## Conteúdo NOVO que a grade absoluta escondia

No CoNLL ajustado, fechar **75% e 90% do vão é inalcançável até para a confiança do modelo** —
não existe cobertura em que ordenar pela confiança atinja 0,9729 de precisão entre os entregues.
A grade absoluta não mostrava isso porque a meta de 0,95 era atingível com 14,3% de revisão, e
0,95 fica **abaixo** de 75% do vão daquele arranjo.

Isso é um limite do supervisor, não da atenção, e é informativo: mesmo o melhor sinal disponível
tem um teto de qualidade entregável, e ele está entre metade e três quartos do vão.

## O que ficou de fora, e por quê

O `gliner-base` **não** entra. A tabela dele foi medida quando a identidade de medição ainda não
incluía o modelo (`measurement_hash` `8a798025fb755a32`), e uma configuração `hash_version` 5
inclui — então o guarda de procedência recusa, corretamente. Não é defeito: é a correção da
versão 4 funcionando. Incluí-lo exigiria remedir o `gliner-base` sob identidade que registra o
modelo, o que é medição nova para melhorar relato, e não fiz.

Os quatro arranjos que entram cobrem precisão de base de **0,5356 a 0,8915**, que é faixa
suficiente para a demonstração de invariância.
