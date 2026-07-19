# Capítulo — Reformulação para previsão de ocorrência de fogo: resultados e discussão

> **Versão:** 01/06/2026. Documento de síntese consolidando o pipeline novo
> (`scripts/previsao_ocorrencia/`) e seus experimentos. Pronto para servir de base
> ao capítulo de Resultados e Discussão do TCC. Numerários reais (relatórios em
> `modelos/relatorios/`). Referências: chaves de `REFERENCIAS_TCC.md` e do
> `ESTADO_DA_ARTE_E_PROPOSTAS_MELHORIA.md` (§7).

---

## 1. Motivação: do índice ao fenômeno

O pipeline original do trabalho classificava o **índice Risco de Fogo do INPE**
(`RiscoFogo`) em três faixas. A inspeção dos dados revelou duas fragilidades de
fundo: (i) o alvo era a **saída de outro modelo** (o índice do INPE, ele próprio
derivado de dias-sem-chuva, precipitação e umidade — variáveis que também eram
features), gerando circularidade; e (ii) a base era *presence-only* — todas as
~934 mil linhas eram focos reais, sem classe negativa. Empiricamente, ~10% dos
focos ocorriam onde o índice marcava risco mínimo (0,0–0,1), mostrando que o
rótulo herdava os pontos cegos do índice (caso Pium-TO generalizado).

Este capítulo reformula o problema para **previsão de ocorrência de fogo
observado**, com classe negativa real, horizonte temporal explícito e validação
sem vazamento — alinhado ao estado da arte (previsão de fogo observado, não de
índice) [huot2022nextday; karasante2025firecastnet; deandrade2024lstmgru].

---

## 2. Materiais e métodos do pipeline reformulado

**Unidade e alvo.** Grade de células de 0,25° (~27 km, granularidade MERRA-2/SeasFire)
× mês. Universo: 4.168 células *fire-prone* (com ≥1 foco em 2014–2023). Para cada
(célula, mês *t*), o alvo é binário: **houve foco no mês *t+1*?** Presença = focos
observados (INPE); **ausência** = cell-months sem foco — a classe negativa antes
inexistente. Período: 2015–2023 (após *burn-in* de 12 meses). Total: **450.144**
cell-months, **prevalência 24,7%**.

**Features causais (≤ t).** Para corrigir o vazamento temporal do pipeline
original (a validação rolling-origin documentava queda de 82,9%→70,1% de acurácia),
todas as features usam apenas dados até *t*: histórico de fogo defasado
(`focos_lag1-3`, `fogo_lag1-3`, somas móveis 3/6/12 m com `shift`),
`meses_desde_fogo`, sazonalidade (`mes_sin/cos`, `estacao_seca`), localização
(`LatBin/LonBin`), `FRP_lag1` e clima do mês corrente.

**Modelo e validação.** `HistGradientBoostingClassifier` (scikit-learn),
`class_weight='balanced'`. Validação **espaço-temporal em bloco** [roberts2017cv]:
*rolling-origin* por ano (treina < Y, testa = Y, para Y ∈ {2021, 2022, 2023}) e
*leave-region-out* (5 blocos de longitude). Métricas: **PR-AUC** (Average
Precision, robusta a desbalanceamento), ROC-AUC, Brier.

---

## 3. Resultados

### 3.1 Desempenho preditivo e baselines

*(rolling-origin 2021–23; `modelos/relatorios/previsao_ocorrencia_validacao.json`)*

| Validação | Modelo PR-AUC | Modelo ROC-AUC | Persistência | Climatologia |
|---|---|---|---|---|
| Temporal | **0,777** | 0,904 | 0,449 | 0,715 |
| Espacial (leave-region-out) | **0,759** | 0,895 | — | 0,167–0,335 |

O modelo supera os dois baselines. Na validação **espacial**, a climatologia
colapsa (PR-AUC 0,17–0,34) ao prever regiões nunca vistas, enquanto o modelo
mantém 0,759 — indício de **generalização** (dinâmica transferível), não
memorização de taxa-base local.

### 3.2 Benchmark contra índices físicos (FWI e Risco de Fogo INPE)

*(`fwi.py` implementa o FWI canadense [vanwagner1985fwi], validado contra os
valores de referência publicados; `modelos/relatorios/benchmark_p3_fwi.json`)*

| Preditor de fogo observado (t+1) | PR-AUC | ROC-AUC |
|---|---|---|
| **Modelo ML** (dataset completo) | **0,777** | 0,904 |
| Índice INPE RiscoFogo (como forecast) | 0,294 | 0,553 |
| *Amostra 100 células (2023):* ML | **0,858** | — |
| → FWI canadense real (clima diário NASA POWER) | 0,608 | 0,781 |
| → INPE RiscoFogo | 0,398 | — |

Na amostra (mesmas linhas), **ML 0,858 > FWI 0,608 > RiscoFogo 0,398**. O FWI real
separa fisicamente bem (média 6,0 em meses com fogo vs 1,5 sem), validando a
implementação. **O modelo de ML supera o índice físico operacional na previsão de
fogo observado** — o resultado que sustenta a tese: o ML agrega valor sobre o
índice que o pipeline original apenas reproduzia [digiuseppe2024geff].

### 3.3 Incerteza calibrada (conformal prediction)

*(`conformal.py`; LAC classe-condicional [sadinle2019lac; mortier2024conformal];
`modelos/relatorios/previsao_ocorrencia_conformal.json`)*

Cobertura empírica **88,8%** (alvo 90%; 88,6% não-fogo / 89,6% fogo); 82,9% das
predições são *singletons* confiantes e 17,1% são marcadas como **ambíguas** com
base estatística — substituto principiado para a heurística *gap top-2*.
ECE 0,099, Brier 0,123.

### 3.4 Experimentos controlados de covariáveis

Para identificar **o que** melhora a previsão, cada família de covariável foi
adicionada às features-base e medida por PR-AUC na validação em bloco (Δ = ganho
sobre o histórico de fogo).

| Driver | Tipo | Fonte | Δ PR-AUC geral |
|---|---|---|---|
| Clima real + FWI (P5) | dinâmico, meteorológico | NASA POWER diário [594 células] | **+0,001** |
| Distância a cidade | estático, antrópico | Natural Earth | **+0,001** |
| Desmatamento PRODES | dinâmico, antrópico | TerraBrasilis [prodes-terrabrasilis] | **+0,003** |

No agregado, **nenhuma covariável agrega de forma expressiva** sobre o histórico de
fogo. Para o clima isso foi medido diretamente: ML-base (só histórico) PR-AUC 0,617
vs ML-gridded (+ clima real + FWI) 0,617 (`previsao_ocorrencia_gridded.json`).
A interpretação é que o histórico de fogo já é um *proxy a jusante* da propensão
climática e da pressão humana (clima/uso do solo → fogo → histórico).

### 3.5 Análise estratificada: recorrência vs nova ignição

O agregado, porém, esconde a estrutura do problema. Estratificando o teste de 2023
por presença de fogo recente na célula:

| Estrato | n | prevalência | base PR-AUC | ROC-AUC |
|---|---|---|---|---|
| **Recorrente** (foco nos últimos 12 m) | 39.549 | 0,316 | **0,805** | 0,891 |
| **Nova ignição** (sem foco recente) | 10.467 | 0,054 | **0,230** | 0,857 |

O modelo é **excelente onde o fogo recorre** (persistência) e **fraco onde o fogo
é novo** (PR-AUC 0,23) — e a nova ignição é justamente o caso crítico para alerta
precoce. É nesse regime, onde a persistência é inútil por construção, que uma
covariável pode provar seu valor.

O teste decisivo usou o **desmatamento PRODES** (features causais: desmatamento da
célula nos anos anteriores; `modelos/relatorios/driver_desmatamento.json`):

| Estrato (média 2021–23) | base PR-AUC | + desmatamento | Δ |
|---|---|---|---|
| Geral | 0,777 | 0,781 | +0,003 |
| Recorrente | 0,791 | 0,794 | +0,003 |
| **Nova ignição** | **0,206** | **0,231** | **+0,025** |

O desmatamento é o **único** dos três drivers testados que agrega valor não-trivial
— e o faz **exatamente no estrato de nova ignição** (até +0,040 em 2022). Confirma
o **mecanismo causal**: o desmatamento **precede a primeira queimada** de uma célula
(queima de área recém-desmatada), informação que o histórico de fogo, por definição,
não contém — coerente com o fogo amazônico ser majoritariamente antrópico
[aragao2018].

**Robustez do achado (DETER mensal).** O resultado foi replicado com uma fonte
independente e de maior resolução temporal: o **DETER** (alertas de desmatamento com
data, INPE [prodes-terrabrasilis]), agregado por célula/**mês**, com features causais
mensais (área desmatada nos meses anteriores; `modelos/relatorios/driver_deter.json`):

| Estrato (média 2021–23) | base PR-AUC | + DETER mensal | Δ | (PRODES anual) |
|---|---|---|---|---|
| Geral | 0,777 | 0,779 | +0,001 | +0,003 |
| Recorrente | 0,791 | 0,792 | +0,001 | +0,003 |
| **Nova ignição** | **0,206** | **0,233** | **+0,026** | (+0,025) |

Duas conclusões: (i) o sinal **desmatamento → nova ignição é robusto** — duas fontes
independentes (PRODES anual e DETER mensal) convergem (+0,025 e +0,026 PR-AUC); (ii) a
**resolução mensal não amplificou** o ganho sobre a anual, indicando que, para a
previsão de fogo mensal, o que importa é *ter sido desmatado recentemente* (escala
anual/sazonal), não o mês exato do desmatamento. A cicatriz de fogo (`CICATRIZ_DE_QUEIMADA`)
foi **excluída** do DETER para evitar vazamento do alvo.

### 3.6 Robustez a famílias de modelo (com intervalos de confiança)

Para verificar que os achados não dependem do algoritmo, comparam-se quatro famílias
no mesmo conjunto de features (base + DETER), validação rolling-origin (predições
out-of-time 2021–23), com **IC 95% por cluster bootstrap por célula** (respeita a
autocorrelação espacial; `modelos/relatorios/comparacao_familias.json`):

| Família | PR-AUC geral [IC 95%] | ROC-AUC | Brier | PR-AUC nova ignição [IC 95%] |
|---|---|---|---|---|
| **HistGradientBoosting** | **0,776 [0,769–0,783]** | 0,904 | 0,123 | **0,232 [0,209–0,258]** |
| RandomForest | 0,769 [0,762–0,776] | 0,901 | **0,113** | 0,204 [0,185–0,232] |
| ExtraTrees | 0,761 [0,754–0,768] | 0,897 | 0,129 | 0,191 [0,174–0,217] |
| LogisticRegression | 0,699 [0,690–0,707] | 0,873 | 0,143 | 0,117 [0,108–0,131] |

Leitura: os três ensembles de árvore são **estatisticamente equivalentes** (ICs
sobrepostos — p.ex. HGB 0,776 vs RF 0,769 têm ICs que se cruzam), enquanto a
**regressão logística é significativamente inferior** (IC não sobreposto). Ou seja,
os achados são **robustos à escolha de família** — o desempenho ~0,77 (geral) e ~0,23
(nova ignição) não é artefato de um algoritmo. Nota: o RandomForest tem o **melhor
Brier** (0,113), sendo mais bem calibrado "de fábrica" que o HGB — alternativa válida
caso a calibração probabilística seja prioritária.

### 3.7 Validação do rótulo contra área queimada independente (MapBiomas Fogo)

O rótulo (`fogo=1` se ≥1 foco INPE na célula-mês) vem de **detecção de fogo ativo**.
Para verificá-lo contra um produto **independente e de modalidade diferente**, usou-se
o **MapBiomas Fogo** (cicatriz de área queimada, Landsat 30 m [mapbiomasfogo]). O raster
exige Earth Engine/rasterio (indisponível no ambiente); usou-se a estatística oficial de
**área queimada por estado/ano** (Coleção 3), correlacionada com a contagem de focos INPE
por (estado da Amazônia Legal, ano), 2014–2023 (`modelos/relatorios/validacao_rotulo_mapbiomas.json`):

| Medida | Valor |
|---|---|
| Spearman **interanual médio dentro de cada estado** | **0,814** (9/9 estados positivos: 0,69–0,94) |
| Spearman global (entre estados) | 0,275 |
| Pearson (log) global | 0,173 (n.s.) |

Leitura: a **dinâmica temporal** do rótulo é fortemente corroborada pelo MapBiomas — quando
a área queimada de um estado sobe num ano, os focos INPE sobem junto (ρ≈0,81). A correlação
**entre estados** é fraca porque a contagem de focos não é proxy linear da *magnitude* em
hectares (escala focos↔área varia por tipo de fogo/região) — caveat conhecido dos focos.
Como o rótulo aqui é **ocorrência binária** (e não magnitude), a fraqueza de escala não o
afeta; o que importa — *houve fogo?* — é validado pela forte concordância interanual.

---

## 4. Discussão e síntese

Os experimentos, sob validação honesta, sustentam uma tese coesa:

1. **A ocorrência mensal de fogo (0,25°) é dominada por persistência espaço-temporal
   + sazonalidade.** O histórico de fogo defasado é o preditor dominante, tornando
   **redundantes** tanto a reanálise meteorológica (incluindo o FWI operacional)
   quanto a acessibilidade humana estática.

2. **O modelo de ML supera os índices físicos** (FWI, Risco de Fogo do INPE) na
   previsão de fogo observado — diferença essencial frente ao pipeline original,
   que apenas reproduzia o índice.

3. **O problema difícil e operacionalmente relevante é a nova ignição** (PR-AUC ~0,2),
   e ali o único sinal útil entre os testados é o **desmatamento recente** — não a
   meteorologia. O achado é **robusto a duas fontes independentes** (PRODES anual e
   DETER mensal, ambas ~+0,025 PR-AUC na nova ignição). Isso reposiciona o alerta
   precoce de fogo na Amazônia como, em primeira ordem, um problema de **detecção de
   fronteira de desmatamento**, mais do que de fire weather.

4. **A incerteza é quantificável com garantia** (conformal), distinguindo predições
   confiantes (83%) de ambíguas (17%) — relevante para uso operacional.

O valor metodológico está em **como** se chegou a isso: experimentos controlados que
mostram não só *que* o modelo funciona, mas *o que* contribui, *onde* e *por quê* —
incluindo resultados negativos informativos (clima e acessibilidade redundantes).

---

## 5. Limitações

- **Granularidade mensal/0,25°.** Picos diários de tempo seco e a dinâmica
  sub-mensal são diluídos; a hipótese de que o clima passe a importar em escala
  sub-mensal não foi testada (trabalho em curso).
- **Domínio restrito a células *fire-prone*.** O modelo responde "dada uma célula
  propensa, queima no próximo mês?", não cobre o interior de floresta densa.
- **Clima gridado parcial.** O NASA POWER aplicou *rate-limit*; o P5 foi medido em
  594/4.168 células (subconjunto enviesado para alta atividade). Dado o ganho nulo,
  completar a grade não foi priorizado.
- **Teto da nova ignição (~0,23 PR-AUC).** Esgotaram-se as dimensões do desmatamento:
  PRODES (anual) e DETER (mensal) dão ganho equivalente (~+0,026), a resolução mensal
  não amplificou, e *idade* do desmatamento, *estoque acumulado* e *fronteira*
  (desmatamento na vizinhança) **não agregam nada além do volume de desmatamento
  recente local** (`modelos/relatorios/driver_frontier.json`). O ganho satura em
  ~+0,026 e o teto em ~0,23 PR-AUC parece um **limite com estes dados** — parte da
  ignição é provavelmente irredutível (decisão humana estocástica) ou exige dados de
  outra natureza (transição de uso do solo classe-a-classe, variáveis socioeconômicas).
- **Rótulo via focos do INPE.** Validado contra a área queimada do MapBiomas (§3.7):
  forte concordância interanual intra-estado (ρ≈0,81). Resta a validação **cel-a-cel**
  com o raster do MapBiomas (exige Earth Engine/rasterio) para checar omissão/comissão
  no nível da célula — não feita por restrição de ambiente.
- **Tuning de hiperparâmetros.** As famílias foram comparadas com IC 95% (§3.6) e os
  ensembles de árvore são estatisticamente equivalentes, mas não houve busca de
  hiperparâmetros por família — um *tuning* dedicado (p.ex. Optuna) poderia render
  ganhos marginais. As métricas, porém, já vêm com intervalos de confiança.

---

## 6. Conclusão do capítulo

A reformulação elevou o trabalho de "reprodução de um índice (84% acurácia, mas com
vazamento e sem classe negativa)" para "**previsão de fogo observado, validada sem
vazamento, que supera o índice físico operacional, com incerteza calibrada**"
(PR-AUC 0,78 temporal; 0,76 espacial). Mais importante, identificou — por
experimentação controlada — que o gargalo é a **nova ignição** e que o sinal que a
endereça é o **desmatamento recente**, não a meteorologia. Esse é um achado de
mecanismo, citável e operacionalmente acionável.

**Produto operacional.** O pipeline foi integrado em um produto: o modelo final
(base + desmatamento DETER) foi serializado (`modelos/modelo_ocorrencia.pkl`) com
limiares conformais (cobertura calibrada 0,900) e gera uma **previsão por célula** de
P(fogo no próximo mês) com rótulo conformal e drivers (`dataset_forecast_celulas.csv`),
servida por um app Flask (`app_forecast.py`) com mapa interativo
(`mapa_forecast_ocorrencia.html`) — independente do app original. Verificações de
sanidade conferem com a fenologia do fogo (ex.: Roraima alto em dezembro; sul do Pará
baixo em dezembro, pois sua estação de fogo é jul–out).

**Artefatos:** pipeline em `scripts/previsao_ocorrencia/` (ver `README.md`);
relatórios em `modelos/relatorios/previsao_ocorrencia_*.json`,
`benchmark_p3_fwi.json`, `driver_*.json`; datasets `dataset_ocorrencia_mensal.csv`,
`dataset_forecast_celulas.csv`; modelo `modelos/modelo_ocorrencia.pkl`.
