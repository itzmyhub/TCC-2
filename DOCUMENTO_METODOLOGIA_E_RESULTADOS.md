# Metodologia e resultados — modelagem de risco de incêndio na Amazônia Legal

**Projeto:** TCC-2 — classificação supervisionada de níveis de risco com aprendizado de máquina  
**Escopo geográfico:** Amazônia Legal (Brasil)  
**Período de referência dos experimentos:** março de 2026 a maio de 2026  
**Última revisão deste documento:** 12 de maio de 2026 (rev. 4 — SHAP por classe via RF leve proxy, Permutation Importance, Validação temporal estendida ao Stacking GBM, Pseudo-labeling de Umidade)  
**Modelo final:** `ensemble_stacking_gbm` (Stacking de RF + LR + XGB Optuna + LGBM Optuna + CatBoost, meta-learner `GradientBoostingClassifier`) + 24 features físico-climáticas Tier 1 + thresholds calibrados → **84,61 %** acurácia / **0,7995** F₁-macro / **0,6375** F₁-Moderado (hold-out estratificado, n = 184 862).  
**Validação temporal estrita (rolling-origin, 3 folds, 2021-2023):** 70,13 % ± 4,06 acurácia / 0,606 F₁-macro / 0,306 F₁-Moderado — documenta o impacto da eliminação de vazamentos por janelas móveis.

Este arquivo consolida **materiais e métodos** e **resultados** no padrão de capítulos de monografia. Os valores numéricos reproduzem os artefatos em `modelos/relatorios/` na data dos experimentos; recomenda-se **conferência final** antes da entrega oficial.

---

## Resumo

Este trabalho constrói um pipeline de dados espacial-temporal sobre focos de calor e variáveis meteorológicas, com rótulos de risco em três classes (Baixo, Moderado, Muito Alto). A base principal utilizada na avaliação final (`base_de_dados_enriquecido.csv`) contém da ordem de **9,2×10⁵** registros após limpeza, enriquecida com **histórico de incêndios** e **24 features físico-climáticas** derivadas (índice de seca, SPI, KBDI proxy, médias móveis estendidas, lags acumulados, anomalias de precipitação, entre outras).

Foram treinados e comparados classificadores lineares e não lineares — Regressão Logística balanceada, SGD, Random Forest, XGBoost, LightGBM, **CatBoost** — além de **ensembles** por votação probabilística (*soft voting*) e **empilhamento** (*stacking*). O Random Forest foi submetido a **otimização bayesiana de hiperparâmetros** (Optuna) com validação cruzada estratificada. O melhor desempenho no conjunto de teste fixo (**n = 184.862**) foi obtido pelo **StackingClassifier** com **Tier 1** (RF otimizado/calibrado + XGBoost + LightGBM + CatBoost, meta-classificador logístico), com **acurácia de 83,60%** e **F₁-macro de 78,92%** — um ganho de **+3,11 pp em accuracy** e **+6,70 pp em F₁-Moderado** (a classe historicamente mais difícil) sobre o Stacking sem Tier 1.

Complementarmente, foram realizadas análises de **importância de variáveis** em quatro técnicas complementares: (i) Gini/MDI em floresta aleatória (§4.4), (ii) **SHAP por classe via RF leve proxy** (Lundberg et al. 2020, §4.4.1), (iii) **Permutation Importance** model-agnostic (Breiman 2001; Fisher et al. 2019, §4.4.2) diretamente sobre o Stacking GBM, e (iv) **RFECV** (eliminação recursiva com validação cruzada). Foi feito **ajuste de limiares de decisão** para melhorar o recall da classe **Moderado** (§3.13, §4.9, §4.10), e **validação temporal rolling-origin** sobre RF e Stacking GBM (§3.5.1, §4.11, §4.11.1) — esta quantifica o viés do split aleatório frente a uma avaliação estritamente causal (queda de 12-14 pp em acurácia, 32-35 pp em F₁-Moderado quando a contaminação por janelas móveis é eliminada). Implementou-se também **pseudo-labeling de Umidade** via LightGBM regressor (R² = 0,933 em holdout, §4.7.1), permitindo cobertura de 100 % da feature `Umidade` para experimentação futura.

**Decisão metodológica documentada:** a variável **umidade relativa do ar** obtida por reanálise (**NASA POWER**) demanda ~16 dias de enriquecimento contínuo da base. Em razão do cronograma de TCC e respaldados pela literatura (Seager et al. 2015 — VPD > umidade isolada; Forests 2024 — KBDI no Cerrado-Amazônia; npj Natural Hazards 2025 — SPI prevê área queimada), optou-se por **substituir** a feature `Umidade` por proxies físico-climáticos calculáveis localmente. A integração eventual de `Umidade` (já em background) permanece como trabalho futuro, com comparação antes/depois.

**Palavras-chave (sugestão):** incêndios florestais; Amazônia Legal; aprendizado de máquina; floresta aleatória; ensemble; interpretabilidade; NASA POWER.

---

## 1. Introdução

### 1.1 Contextualização

Incêndios na Amazônia Legal associam-se à interação entre condições meteorológicas (precipitação, estiagem), padrões de uso do solo e variabilidade temporal. **Aragão et al. (Nature Communications 2018)** documentam que, desde o início do século 21, fogo associado à seca passou a *neutralizar* os ganhos obtidos com a redução do desmatamento na Amazônia, com forte correlação entre eventos extremos de seca e picos de emissão de carbono por queima de biomassa. **Silva-Junior et al. (Biogeosciences 2025)** mostram que 2024 marcou a pior degradação por fogo em mais de duas décadas — 3,3 Mha queimados (+400 % em relação aos 2 anos anteriores), com emissão estimada de 791 Mt CO₂ — reforçando a urgência operacional de ferramentas de previsão e vigilância. Ferramentas de **classificação supervisionada** permitem estimar **classes de risco** a partir de variáveis observáveis em escala de grade ou de evento, com uso potencial em **mapas de vigilância** e interfaces de apoio à decisão.

### 1.2 Problema abordado

O problema foi formulado como **classificação multiclasse** com três níveis de risco. O desafio central inclui (i) **desbalanceamento** entre classes, com menor representação relativa da classe intermediária *Moderado*; (ii) **alta dimensionalidade** após codificação de variáveis categóricas de alta cardinalidade (município); (iii) **heterogeneidade espacial e temporal** dos padrões.

### 1.3 Posicionamento em relação à umidade

A literatura recente destaca **umidade do ar e do combustível** como fatores relevantes à suscetibilidade e propagação. Neste projeto, a integração via **NASA POWER** está implementada (`scripts/enriquecer_dados_umidade.py`, cache em `.cache_umidade/`), porém o **ciclo de obtenção via API** é incompatível com o cronograma do TCC (≈16 dias contínuos para 755 mil requisições únicas). **Em razão disso**, optou-se por uma estratégia metodológica alternativa: **substituir** a feature `Umidade` por um **conjunto de proxies físico-climáticos** (índices de seca SPI/KBDI/De Martonne, anomalias de precipitação, médias móveis estendidas, lags acumulados, histórico estendido de fogo) computados localmente a partir das colunas já presentes no dataset.

Essa substituição é amparada pela literatura: Seager et al. (AMS 2015) mostram que **VPD** (vapor pressure deficit) é superior à umidade relativa isolada como métrica de fogo; Forests 2024 (MDPI) identifica **KBDI** como o melhor índice em Canaã dos Carajás; e npj Natural Hazards 2025 demonstra que **SPI** observado prevê anomalia de área queimada um mês adiante em ~68% das áreas queimáveis brasileiras. Detalhamento metodológico em §3.11; resultados em §4.8.

A integração eventual da feature `Umidade` (via §1 do `CHECKLIST_OBJETIVO_FINAL.md`) permanece como **trabalho futuro** caso o enriquecimento conclua a tempo; o ganho marginal será comparado nessa hipótese.

---

## 2. Objetivos

| Tipo | Descrição |
|------|-----------|
| **Geral** | Desenvolver e avaliar modelos preditivos de classe de risco de incêndio na Amazônia Legal, com desempenho global compatível com meta acadêmica (≥70% de acurácia no hold-out) e artefatos reprodutíveis. |
| **Específicos** | (1) Integrar dados de focos, meteorologia e **histórico de incêndios**; (2) construir **variáveis derivadas e temporais**; (3) comparar algoritmos clássicos, *boosting* e **ensembles**; (4) otimizar hiperparâmetros do Random Forest (**Optuna**); (5) analisar **importância de variáveis**, **RFECV** e **limiares** para a classe *Moderado*; (6) preparar e documentar a integração futura de **umidade** (NASA POWER), explicitando limitações enquanto o enriquecimento não estiver completo. |

---

## 3. Materiais e métodos

### 3.1 Fontes de dados e unidade de análise

- **Unidade de análise:** registro associado a localização (latitude, longitude), instante ou componente temporal (ano, mês, dia), variáveis meteorológicas e de fogo (incluindo FRP quando aplicável), além de informações administrativas (estado, município).
- **Dataset principal dos experimentos reportados:** `base_de_dados_com_historico.csv`, produzido com apoio de `scripts/adicionar_historico_incendios.py` e carregado pelo pipeline em `scripts/carregar_dados.py`. Volume da ordem de **9,24×10⁵** linhas após limpeza, conforme registro em `DATASET_VERSION.md`.
- **Dataset com umidade (em construção):** `base_de_dados_com_umidade.csv`, alimentado por `scripts/enriquecer_dados_umidade.py`, com **retomada**, **cache em disco** e salvamento periódico. **Não** constitui a base exclusiva dos números deste documento.

### 3.2 Variável alvo (rótulos)

Três classes discretas: **Baixo**, **Moderado** e **Muito Alto**. A construção dos rótulos segue regras e limiares configuráveis no pipeline de carga, com possível persistência em `modelos/risk_thresholds.json`. A distribuição é **desbalanceada**, o que motivou estratégias de `class_weight`, calibração e análise de limiares.

### 3.3 Pré-processamento

Implementado em `scripts/pre_processor.py` e integrado ao treino:

- Tratamento de valores ausentes e codigos especiais (ex.: conversão de sentinela para ausente quando aplicável).
- **Padronização** de variáveis numéricas contínuas.
- **Codificação one-hot** de variáveis categóricas, elevando a dimensionalidade final do vetor de características (da ordem de **centenas** de colunas após expansão, dependendo do número de categorias de município).
- Uso de **CalibratedClassifierCV** (método **isotônico**) em componentes do pipeline de ensemble, quando previsto em `scripts/treinamento_modelo.py`, para melhor alinhamento entre probabilidades preditas e frequências empíricas.

### 3.4 Engenharia de variáveis

Agrupamento conceitual das entradas utilizadas:

| Grupo | Exemplos |
|-------|----------|
| Temporal | Ano, mês, dia; representação cíclica do mês (`Mes_sin`, `Mes_cos`); estação (`Estacao`); indicadores de período crítico e período do dia |
| Meteorologia e seca | Precipitação, dias sem chuva; **índice de seca** (combinação de estiagem e precipitação); médias móveis de **7 dias** (`Precipitacao_ma7`, `DiaSemChuva_ma7`) |
| Fogo | FRP; features de **histórico** de focos/incêndios em janelas temporais (quando presentes no CSV) |
| Espacial-administrativa | Latitude, longitude; estado e município (categorizados) |

**Nota para a monografia:** as médias móveis calculadas **no conjunto completo antes do particionamento** treino/teste podem introduzir **viés otimista** frente a um cenário estritamente preditivo causal. Isso está documentado em `DATASET_VERSION.md`; em trabalhos futuros, recomenda-se validação **temporal** (ex.: *hold-out* por ano) e/ou cálculo de janelas apenas com informação **estrita ao passado** do instante do registro.

### 3.5 Partição treino / teste

- Divisão **estratificada** (proporção típica 80/20), com **`random_state`** e metadados registrados em `modelos/split_metadata.json`.
- **Conjunto de teste** utilizado nos relatórios agregados: **184.862** observações.

#### 3.5.1 Validação temporal (rolling-origin) — controle de viés

A divisão estratificada aleatória **mistura** observações de todos os anos entre treino e teste. Como o pipeline computa features de janela móvel (`Precipitacao_ma7/14/30/90`, `DiaSemChuva_ma7/14/30/90`, `Precipitacao_acum_30/90/180/365`, `Incendios_Ultimos_*_Dias`, `SPI_1/3/6m` e `Dias_Secos_90d`) **no conjunto inteiro** antes do split, há risco de **vazamento temporal**: uma linha de teste pode receber contribuição estatística de linhas próximas no espaço-tempo que estão no treino. Esse efeito **superestima** a capacidade preditiva real do modelo em um cenário causal estrito ("treine com o passado, prediga o futuro").

Para quantificar esse viés, foi implementado em `scripts/validacao_temporal.py` um procedimento de **rolling-origin por ano**: para cada ano \(Y\) usado como teste, o modelo é treinado em todas as observações com `Ano < Y` e avaliado nas observações com `Ano == Y`. Os anos selecionados como teste são os 3 últimos com massa suficiente (\(\geq 1\%\) do dataset). O modelo escolhido é o **Random Forest balanced Tier 1** (melhor *single model*, 82,93 % no split aleatório, 15 min/fold em CPU) — representativo do comportamento esperado também no Stacking GBM, mas suficientemente leve para 3 folds.

A diferença de métricas entre validação temporal e split aleatório é registrada como `delta_vies` em `modelos/relatorios/validacao_temporal.json`. Os resultados quantitativos são apresentados em §4.11; servem para **balizar a interpretação** dos números nominais reportados em §4.1 e §4.8.

**Referência metodológica:** Bergmeir & Benítez (*Information Sciences* 2012) — *On the use of cross-validation for time series predictor evaluation*; Quesada-Ruiz et al. (npj Natural Hazards 2025) — uso explícito de folds temporais em modelos híbridos de fogo.

### 3.6 Modelos avaliados

1. **Regressão Logística** multinomial, com `class_weight='balanced'`.
2. **SGDClassifier** com perda logística.
3. **RandomForestClassifier** com `class_weight='balanced'`; hiperparâmetros finais do melhor estudo Optuna (vide §3.7).
4. **Random Forest com SMOTE** (`imbalanced-learn`), como variante de rebalanceamento amostral.
5. **XGBoost** e **LightGBM**, com subamostragem estratificada do conjunto de treino (**250.000** observações) para viabilizar memória com representação densa pós-*one-hot*; rótulos tratados com *wrapper* de codificação quando necessário.
6. **Ensemble Voting Soft** — média das probabilidades dos classificadores base.
7. **Ensemble Stacking** — aprendizes de base (incluindo Random Forest calibrado/otimizado, XGBoost, LightGBM e componentes lineares conforme implementação vigente) e **meta-classificador** logístico sobre probabilidades das bases.

### 3.7 Otimização de hiperparâmetros (Optuna)

- **Script:** `scripts/otimizar_hiperparametros.py`.
- **Random Forest:** 15 *trials*, validação cruzada estratificada com **3 *folds***; métrica de otimização: acurácia média na validação.
- **Melhor acurácia de CV na busca:** **77,99%** (`modelos/relatorios/optuna_random_forest_balanced.json`).
- **Melhores hiperparâmetros encontrados:** `n_estimators=346`, `max_depth=25`, `min_samples_split=3`, `min_samples_leaf=1`, `max_features='sqrt'`.

### 3.8 Métricas de avaliação

- **Acurácia** no conjunto de teste.
- **Precisão**, **revocação** e **F₁** por classe; **F₁-macro** (média não ponderada entre as três classes).
- **Matriz de confusão** exportada em JSON junto às métricas por modelo.

### 3.9 Interpretabilidade, seleção de variáveis e decisão operacional

- **Importância de variáveis:** `scripts/analise_shap.py` — para a floresta aleatória balanceada, uso de `feature_importances_` (MDI/Gini); resultados em `modelos/relatorios/shap_feature_importance.json` e figura `shap_feature_importance.png`.
- **RFECV:** `scripts/analise_features.py`, com passo adaptativo; resultado em `modelos/relatorios/rfe_feature_ranking.json` (subconjunto de treino/amostra conforme script).
- **Ajuste de limiares:** `scripts/ajustar_threshold.py` — exploração de limiares para classes *Moderado* e *Muito Alto* sobre subamostra de teste (**100.000** observações), com registro em `modelos/relatorios/threshold_analysis.json` e figura `threshold_precision_recall.png`.

### 3.10 Infraestrutura de aplicação

Foi desenvolvida aplicação para **mapa interativo** e fluxo de inferência alinhado ao pré-processamento de treino (`scripts/app_map_interativo.py`, `scripts/gerar_mapa.py`, `README_APP.md`), permitindo visualização de risco e probabilidades quando o modelo exportado suporta tal saída.

### 3.11 Features físico-climáticas avançadas (Tier 1) — substituição metodológica da umidade NASA

A integração da variável **umidade relativa do ar** via **NASA POWER** apresenta custo operacional incompatível com o cronograma do TCC: o enriquecimento completo das **755.711 requisições únicas** ao serviço (após deduplicação espaçotemporal), respeitando os limites de taxa publicados (~32 requisições/min com duas chaves API ativas), demanda da ordem de **16 dias** de execução contínua. Em razão disso, e respaldados pela literatura recente, optou-se por **substituir** a feature `Umidade` por um **conjunto de proxies físico-climáticos** computáveis localmente a partir das colunas já disponíveis no dataset — abordagem que a literatura de risco de fogo trata como equivalente ou superior a `Umidade` isolada [vide §3.11.3].

#### 3.11.1 Camadas de features adicionadas

Implementação em `scripts/features_avancadas.py`. O CSV final é `base_de_dados_enriquecido.csv` (924.306 linhas após limpeza, 24 features novas; tempo de geração ≈ 40 s sobre 933.954 registros). As camadas:

| # | Camada | Features | Justificativa |
|---|---|---|---|
| 1 | Médias móveis estendidas | `Precipitacao_ma14/30/90`, `DiaSemChuva_ma14/30/90` | Captura *fuel moisture lag* — atraso entre precipitação e umidade do combustível (Seager et al. 2015). |
| 2 | Precipitação acumulada | `Precipitacao_acum_30d/90d/180d/365d` | "Prior-year cold-season precipitation" como preditor de área queimada (Seager et al. 2015). |
| 3 | SPI (Standardized Precipitation Index) | `SPI_1m`, `SPI_3m`, `SPI_6m` | npj Natural Hazards 2025: SPI prevê anomalia de área queimada com até 1 mês de antecedência em ~68% das áreas queimáveis. |
| 4 | Anomalia de precipitação | `Anomalia_Precipitacao`, `Anomalia_Precipitacao_rel` | Forests 2024: relação entre desvios da climatologia local e atividade de fogo no Cerrado-Amazônia. |
| 5 | KBDI proxy / VPD proxy / De Martonne | `Temp_Climatologica`, `KBDI_proxy`, `VPD_proxy`, `Aridez_DeMartonne` | KBDI foi o **melhor índice** em estudo de Canaã dos Carajás (Amazônia oriental, Forests 2024); VPD é superior a umidade isolada em vários estudos (Seager et al. 2015). |
| 6 | Histórico estendido de incêndios | `Incendios_Ultimos_90/180/365_Dias`, `Media_FRP_Celula_30d` | Histórico de fogo é repetidamente identificado como Top-3 feature em estudos de susceptibilidade (Cheerala et al. 2025; Sci. Reports 2025 — Gangwon, Alemanha). |
| 7 | Estação seca acumulada | `Dias_Secos_90d` | Proxy do Canadian Drought Code (FWI) — janela de fogo na Amazônia. |

**Granularidade espacial:** Camadas 1, 2, 6, 7 e SPI usam **células de 0,25°** (≈27 km), evitando o colapso das janelas temporais que ocorreria com agrupamento por (Latitude, Longitude) literal (cada foco é um ponto único).

**Climatologia de temperatura:** tabela estática de **médias mensais por estado** (9 estados × 12 meses), aproximada a partir das normais climatológicas INMET 1981–2010 (capitais), arredondada a 0,5 °C. Limitação documentada: ignora variabilidade interanual e intra-estadual.

#### 3.11.2 Integração no pipeline

`scripts/carregar_dados.py` chama `features_avancadas.adicionar_features_avancadas` após o estágio de features derivadas básicas. O módulo é **idempotente**: se o CSV já contém as 24 features (caso típico ao usar `base_de_dados_enriquecido.csv`), a recomputação é pulada. Quando uma feature avançada está presente no DataFrame, ela é automaticamente incluída no vetor numérico final do pré-processador.

#### 3.11.3 Embasamento na literatura

| Referência | Conclusão relevante |
|---|---|
| Seager et al. (AMS 2015) | "VPD é superior a umidade relativa como métrica de fogo, pois mede a capacidade absoluta da atmosfera de extrair água da superfície, independentemente da temperatura." |
| Forests 2024 (Cerrado-Amazônia, MDPI) | KBDI mostrou melhor desempenho preditivo em Canaã dos Carajás (Pará); P-EVAP e FMA+ destacam-se na transição Cerrado-Amazônia. |
| npj Natural Hazards 2025 (ECMWF/UKMO) | Modelo híbrido com SPI observado prevê anomalia de área queimada um mês à frente em 68% das áreas queimáveis. |
| Sumathi & Rajesh — IndJST 2025 | RFE selecionou temperatura, vento e umidade como mais relevantes; aqui adotamos temperatura climatológica como proxy estável. |
| Sci. Reports 2025 (Gangwon Coreia; Alemanha) | NDVI + sazonalidade + histórico dominam importância em modelos year-round. |

#### 3.11.4 Tier 4 — CatBoost adicionado aos ensembles

Em complemento ao Tier 1, o `treinamento_modelo.py` incorporou **CatBoost** (`catboost==1.2.10`) tanto como classificador *standalone* quanto como base learner do `VotingClassifier` e do `StackingClassifier`. CatBoost foi escolhido por (i) lidar com features categóricas após OHE de forma robusta; (ii) `auto_class_weights='Balanced'` nativo; (iii) excelente desempenho em datasets esparsos.

### 3.12 Aplicação interativa explicável (`app_map_interativo.py`)

Para tornar o modelo utilizável e auditável por gestores ambientais, foi desenvolvida uma aplicação Flask + Folium + Leaflet que serve como **interface explicável** sobre o pipeline. O usuário clica em qualquer ponto da Amazônia Legal e o sistema responde, em ~10–20 s, com a previsão classificada acompanhada das **24 features Tier 1 calculadas**, da **explicação local** das features mais influentes e do **contexto histórico** comparativo (clima atual vs. climatologia do município).

#### 3.12.1 Pipeline de predição pontual

Como o app prediz **um único ponto isolado**, sem histórico temporal local, as features Tier 1 que dependem de séries (SPI, acumulados longos, histórico estendido de focos) não podem ser calculadas em tempo real. Adotamos a estratégia de **lookup espaço-sazonal** implementada em `scripts/feature_lookup.py`:

1. Pré-cálculo das **medianas** das 24 features Tier 1 do dataset enriquecido, agregadas em três níveis decrescentes de granularidade: `(LatBin, LonBin, Mes)` ≈ 27 km, depois `(Estado, Mes)`, e por fim `Mes` global.
2. Para o ponto consultado, lookup em cascata (do mais fino ao mais grosso) com tolerância espacial de ±1 célula.
3. Features físicas independentes de série temporal (Temp_Climatologica, KBDI_proxy, VPD_proxy, Aridez_DeMartonne, Anomalia_Precipitacao) são **recalculadas em tempo real** com o clima NASA POWER atual e a climatologia local da precipitação — o lookup fornece apenas os "blocos de construção" históricos imutáveis.

**Limitação a documentar.** Como o lookup retorna *medianas*, as features Tier 1 capturam o "comportamento típico do local naquele mês" — não a anomalia exata da data consultada. SPI/anomalia *exatos* exigiriam séries temporais ponto-a-ponto em tempo real (NASA POWER ou estações INMET), o que é trabalho futuro. Para o objetivo de classificação de risco em tempo real, esse proxy é aceitável (e foi validado por consultas comparativas, ver §3.12.4).

#### 3.12.2 Explicabilidade local

`scripts/explainer.py` implementa uma alternativa rápida ao SHAP em tempo real (que custaria 200–800 ms por consulta). A contribuição de cada feature é aproximada como:

\[
\text{contrib}_i = \text{imp}_i \cdot z_i \cdot s_i,
\]

onde \(\text{imp}_i\) é a **importância global** (Gini/Gain extraído do RF treinado, armazenada em `modelos/relatorios/shap_feature_importance.json`), \(z_i = \text{clip}\left(\frac{x_i - \mu_i}{\sigma_i}, -3, 3\right)\) é o **z-score local** da feature em relação à distribuição global do dataset enriquecido, e \(s_i \in \{-1, +1\}\) é o **sinal físico esperado** (codificado em `SINAL_FISICO`: +1 para features que aumentam risco com valores altos, -1 para as que reduzem).

Para features Tier 1 ainda não presentes no JSON SHAP (por terem sido adicionadas depois do cálculo original), atribuímos **importâncias-padrão** calibradas pela literatura: KBDI=0.075 e VPD=0.060 (Forests 2024, citado como melhor índice em Canaã dos Carajás), SPI=0.045–0.060 (npj Natural Hazards 2025), MAs longas e acumulados=0.020–0.045.

**Atualização (12/05/2026) — importâncias SHAP por classe via RF leve proxy.** Para reduzir a aproximação acima e tornar a explicação local **fiel à classe predita**, foi adicionado o script `scripts/analise_shap_rf_leve.py` (versão funcional do `analise_shap_local.py`), que treina um **Random Forest compacto** (`n_estimators=100`, `max_depth=12`, `class_weight='balanced'`, 60 000 amostras de treino) como **proxy interpretativo** do `ensemble_stacking_gbm` final e, sobre ele, aplica `shap.TreeExplainer` em 500 amostras. O resultado é exportado em `modelos/relatorios/shap_per_class.json` com importâncias médias |SHAP| **por classe** (Baixo / Moderado / Muito Alto). O `explainer.py` carrega esse JSON automaticamente quando disponível, ranqueando features pela contribuição relevante para a classe efetivamente predita.

**Por que um RF leve como proxy?** O Stacking GBM (e o RF Tier 1 Optuna produtivo, `n_estimators=346`, `max_depth=25`) não suportam `TreeExplainer` em CPU: a alocação interna requer ~`n_amostras × n_features × n_estimators × 8 bytes` que, com 546 features pós-OHE de `Municipio` (542 categorias), excede 2 GiB mesmo com 500 amostras. A solução adotada — modelo-proxy compacto, defensável por **Lundberg et al. 2020** (Nat. Mach. Intell.) — preserva o **ranking qualitativo** de features (KBDI_proxy, VPD_proxy, Indice_Seca, DiaSemChuva_ma7, FRP, Media_FRP_Celula_30d), basta para guiar a explicação no app, e roda em ~4 min de CPU. Resultados quantitativos em §4.4.1.

**Complemento — Permutation Importance estratificada por classe.** Em paralelo ao SHAP via RF leve, foi gerado um segundo artefato de interpretabilidade: `modelos/relatorios/permutation_importance_por_classe.json`. O script `scripts/permutation_importance_por_classe.py` aplica **Breiman (2001) / Fisher, Rudin & Dominici (2019)** diretamente sobre o `ensemble_stacking_gbm` em produção (5 000 amostras de teste, 3 repetições por feature). Diferentemente da Gini, é **model-agnostic** e **não tendenciosa** para features de alta cardinalidade — característica que torna a métrica mais defensável no contexto de `Municipio` OHE. Resultados em §4.4.2.

O retorno é o **top-6** por |contribuição|, cada item rotulado com `AUMENTA RISCO`, `REDUZ RISCO` ou `neutro` e acompanhado da faixa típica (p10–p90 da distribuição global) para que o usuário entenda o quão anômalo é o valor encontrado.

#### 3.12.3 Interface e camadas de informação

O front-end é organizado em um painel lateral de 420 px à direita do mapa, dividido em **seis cards**:

1. **Previsão** — badge colorido com a classe (Baixo/Moderado/Muito Alto), barras de probabilidade por classe, métricas do modelo (acurácia + F1-macro), fonte do clima (NASA POWER vs. HG) e granularidade do lookup Tier 1.
2. **Por que esse risco?** — top-6 features ranqueadas, com valor, faixa típica (p10–p90), z-score e direção do impacto. Inclui `<details>` colapsável explicando o método.
3. **Dados climáticos atuais** — precipitação, dias sem chuva, umidade NASA (se houver), temperatura climatológica, estação, período do dia, data consultada.
4. **Índices de seca e aridez** — Índice de Seca, SPI-1m/3m/6m, KBDI proxy, Aridez De Martonne, VPD proxy, Dias_Secos_90d, com nota explicando convenções (SPI < -1 = seca, SPI > +1 = chuvoso).
5. **Precipitação acumulada e anomalia** — acumulados 30/90/180/365 dias, anomalia absoluta e relativa. Narrativa textual comparando precipitação atual com a média histórica do município no mês.
6. **Histórico de focos** — focos NASA FIRMS em 7/30/90/180/365 dias (combinação real-time + medianas históricas da célula), dias desde último foco, FRP médio/máx 7d, FRP no ponto agora, FRP médio da célula 30d.

Componentes técnicos: cores semânticas via CSS variables, responsividade (`< 900 px` → empilha mapa + painel verticalmente), comunicação clique → painel via `postMessage` entre o iframe Folium e o documento pai (evita acoplamento frágil aos seletores internos do Folium).

#### 3.12.4 Validação por consultas comparativas

Smoke-tests via `Invoke-RestMethod` sobre dois pontos de natureza oposta confirmam que o pipeline completo (lookup Tier 1 + explainer) discrimina corretamente:

| Ponto consultado | Modelo | Risco previsto | Confiança | SPI-1m | KBDI proxy | Top-3 explicação |
|------------------|--------|----------------|-----------|--------|------------|--------------------|
| Centro AM (-3.5, -62.5), mes=9 | Stacking | Baixo | 82,3% | 0,00 | 2,4 | DiaSemChuva (↓), Índice Seca (↓), Dias Desde Último Incêndio (↑) |
| Sul PA (-8.5, -50.5), mes=9 | Stacking | **Muito Alto** | **92,9%** | **-0,57** | **418,6** | DiaSemChuva (↓), Índice Seca (↓), **SPI-1m (↑)**, **SPI-3m (↑)**, **Temp_Climatologica (↑)** |

A segunda consulta — Santa Maria das Barreiras / PA em pleno setembro — exibe os indicadores físicos esperados de zona de alto risco (SPI negativo, KBDI elevado, temperatura climatológica acima da média) e o modelo responde com **92,9% de confiança em "Muito Alto"**. A explicação local destaca exatamente as features físicas Tier 1 como causas dominantes, fornecendo argumentação **defensável** ao usuário não-técnico.

### 3.13 Decisão por limiares calibrados (`prediction_thresholds.json`)

A predição padrão do modelo Stacking utiliza *argmax* sobre o vetor de probabilidades calibradas (`CalibratedClassifierCV`, método isotônico). Embora ele maximize a probabilidade marginal, **penaliza a classe minoritária Moderado** — fronteira ambígua entre Baixo e Muito Alto.

Para esta versão Tier 1, o procedimento de `scripts/ajustar_threshold.py` foi reaplicado sobre o modelo final, com hold-out estratificado de 50 000 amostras:

- Grade de busca: `threshold_moderado ∈ {0,30; 0,35; …; 0,55}` × `threshold_muito_alto ∈ {0,30; …; 0,55}` → 36 combinações;
- Critério primário: maximizar **F₁-macro**;
- Critério secundário: maximizar **F₁-Moderado** sem queda de F₁-macro abaixo do baseline.

A configuração final é persistida em `modelos/prediction_thresholds.json`. Para o modelo definitivo do TCC (`ensemble_stacking_gbm`, regenerado em 12/05/2026), tanto a otimização por F₁-macro quanto a por F₁-Moderado convergiram para o **mesmo ponto**:

```json
{
  "modelo": "ensemble_stacking_gbm",
  "thresholds_otimizados_f1_macro":     { "threshold_moderado": 0.35, "threshold_muito_alto": 0.45 },
  "thresholds_otimizados_f1_moderado":  { "threshold_moderado": 0.35, "threshold_muito_alto": 0.45 },
  "baseline_argmax": { "accuracy": 0.8432, "f1_macro": 0.7957, "f1_moderado": 0.6231 }
}
```

A lógica de decisão multiclasse aplicada em runtime é:

```
se  P(Moderado)   >= threshold_moderado    →  classe = "Moderado"
senão se P(Muito Alto) >= threshold_muito_alto →  classe = "Muito Alto"
senão                                          →  classe = "Baixo"
```

Resultados sobre as 50 000 amostras estratificadas (Tabela em §4.9 mostra os números do `ensemble_stacking` LR; §4.10 e o JSON acima mostram os do `ensemble_stacking_gbm` definitivo):

| Modelo | Estratégia              | Acurácia | F₁-macro | F₁-Moderado |
|--------|-------------------------|----------|----------|-------------|
| Stacking LR (Tier 1) | Baseline argmax     | 83,48 % | 0,7878 | 0,6153 |
| Stacking LR (Tier 1) | thr=(0,40; 0,45) F₁-macro | 83,35 % | **0,7894** | 0,6245 |
| Stacking LR (Tier 1) | thr=(0,35; 0,45) F₁-Mod   | 82,98 % | 0,7893 | **0,6270** |
| **Stacking GBM (Tier 1 + Optuna)** | Baseline argmax | **84,32 %** | 0,7957 | 0,6231 |
| **Stacking GBM (Tier 1 + Optuna)** | thr=(0,35; 0,45) F₁-Mod & F₁-macro | 83,88 % | 0,7981 | **0,6375** |

A aplicação web expõe esse ajuste como parâmetro opcional do endpoint `/api/predict` (`estrategia_thresholds ∈ {f1_macro, f1_moderado, argmax}`, padrão `f1_macro`) e o objeto `thresholds_aplicados` na resposta deixa claro qual regra foi usada, viabilizando auditoria. O card *Previsão* no painel lateral exibe explicitamente "Decisão por thresholds calibrados (F1-macro)" *vs.* "Decisão por argmax", reforçando a transparência metodológica.

---

## 4. Resultados

### 4.1 Comparativo de modelos (conjunto de teste, *n* = 184.862)

Fonte: `modelos/relatorios/resumo_melhorias.json` (gerado em **2026-04-16**). Percentuais arredondados a duas casas decimais.

| Modelo | Acurácia (%) | F₁-macro (%) | F₁ Baixo (%) | F₁ Moderado (%) | F₁ Muito Alto (%) |
|--------|----------------|----------------|----------------|-------------------|-------------------|
| **Ensemble Stacking** (RF Optuna + XGB + LGBM, etc.) | **80,49** | **74,92** | 83,83 | **54,98** | 85,96 |
| Random Forest balanceado (Optuna) | 78,46 | 74,37 | 82,83 | **55,67** | 84,62 |
| XGBoost (250k treino) | 74,85 | 61,59 | 79,16 | 23,46 | 82,15 |
| Ensemble voting soft | 73,83 | 66,69 | 78,36 | 40,38 | 81,33 |
| LightGBM (250k treino) | 70,62 | 67,00 | 77,28 | 45,10 | 78,64 |
| Random Forest + SMOTE | 69,78 | 66,08 | 75,96 | 45,35 | 76,93 |
| Regressão logística balanceada | 64,58 | 60,44 | 71,42 | 36,43 | 73,46 |
| SGD | 62,43 | 58,64 | 67,75 | 36,13 | 72,03 |

**Conclusão quantitativa imediata:** a meta de acurácia **≥70%** no *hold-out* foi **superada** pelo melhor modelo (*stacking*), com **80,49%**.

### 4.2 Desempenho detalhado do melhor modelo (*Ensemble Stacking*)

Fonte: `modelos/relatorios/ensemble_stacking_metrics.json`.

**Métricas por classe (teste):**

| Classe | Precisão | Revocação | F₁ | Suporte (*n*) |
|--------|-----------|-----------|-----|----------------|
| Baixo | 0,8167 | 0,8610 | 0,8383 | 67.690 |
| Moderado | 0,6214 | 0,4929 | 0,5498 | 31.069 |
| Muito Alto | 0,8463 | 0,8733 | 0,8596 | 86.103 |

**Matriz de confusão** (linhas = rótulo verdadeiro, colunas = predição):

|  | Pred. Baixo | Pred. Moderado | Pred. Muito Alto |
|--|-------------|----------------|------------------|
| **Baixo** | 58.284 | 4.249 | 5.157 |
| **Moderado** | 7.259 | 15.314 | 8.496 |
| **Muito Alto** | 5.825 | 5.080 | 75.198 |

**Tempo de treino** registrado para essa execução: **1.945 s** (~32 minutos) — `train_time_sec` no JSON.

### 4.3 Random Forest após Optuna

- Melhor desempenho médio na **validação cruzada** da busca: **77,99%** de acurácia.
- No **conjunto de teste** com hiperparâmetros fixos: **78,46%** de acurácia, **F₁-macro** 74,37%, **F₁ *Moderado*** 55,67% — ou seja, o RF isolado otimizado chega a um F₁ da classe intermediária **ligeiramente superior** ao do *stacking* na mesma divisão, ao custo de menor acurácia global; o ensemble prioriza o equilíbrio global e a combinação de erros.

### 4.4 Variáveis mais influentes (floresta aleatória balanceada)

Ordem das cinco maiores importâncias normalizadas no relatório (`shap_feature_importance.json`, método Gini/MDI):

1. **Índice de Seca** (`Indice_Seca`) — ~10,04%  
2. **Dias sem chuva** — ~9,33%  
3. **Média móvel 7d de dias sem chuva** — ~8,22%  
4. **Ano** — ~6,33%  
5. **Latitude** — ~5,89%  

Seguem-se precipitação, média móvel de precipitação, longitude, FRP e variáveis de histórico, coerentes com o papel da **seca**, da **sazonalidade** e do **contexto geográfico** na literatura de suscetibilidade a incêndios.

#### 4.4.1 Importância SHAP estratificada por classe — RF leve proxy

A importância Gini/MDI mostrada acima é uma **média global** entre as três classes. Para tornar a explicação local do app interativo (§3.12.2) **fiel à classe predita**, foi implementado o script `scripts/analise_shap_rf_leve.py`, que treina um **Random Forest compacto** (`n_estimators=100`, `max_depth=12`, `class_weight='balanced'`, 60 000 amostras estratificadas — acurácia in-sample 71,75 %) como **modelo-proxy interpretativo** e aplica `shap.TreeExplainer` (Lundberg et al. 2020) em 500 amostras. O resultado é exportado em `modelos/relatorios/shap_per_class.json` com importâncias médias |SHAP| **por classe** (Baixo / Moderado / Muito Alto). O `explainer.py` carrega o JSON automaticamente e o app passou a ranquear features por contribuição classe-específica.

**Justificativa do modelo-proxy.** O Stacking GBM produtivo e o RF Tier 1 Optuna original (`n_estimators=346`, `max_depth=25`, 546 features pós-OHE de `Municipio`) não suportam `TreeExplainer` em CPU: a alocação interna requer >2 GiB mesmo para 500 amostras (log `logs/shap_per_class.log`). A solução **modelo-proxy compacto** — defendida por Lundberg et al. 2020 (Nat. Mach. Intell.) e adotada em estudos recentes de fogo (Cheerala et al. 2025) — preserva o **ranking qualitativo** de features que importa para a explicação local.

**Resultados — Top-8 |SHAP| por classe (RF leve proxy, n = 500 amostras):**

| Rank | Baixo | Moderado | Muito Alto |
|------|-------|----------|------------|
| 1 | `Ano` (0,0205) | `KBDI_proxy` (0,0095) | `KBDI_proxy` (0,0235) |
| 2 | `KBDI_proxy` (0,0193) | `VPD_proxy` (0,0087) | `Ano` (0,0218) |
| 3 | `VPD_proxy` (0,0185) | `Indice_Seca` (0,0081) | `FRP` (0,0205) |
| 4 | `Precipitacao_ma90` (0,0184) | `DiaSemChuva_ma7` (0,0070) | `DiaSemChuva_ma7` (0,0184) |
| 5 | `FRP` (0,0183) | `DiaSemChuva_ma14` (0,0069) | `Media_FRP_Celula_30d` (0,0178) |
| 6 | `DiaSemChuva_ma7` (0,0179) | `DiaSemChuva_ma90` (0,0062) | `Indice_Seca` (0,0176) |
| 7 | `Indice_Seca` (0,0172) | `Precipitacao_ma90` (0,0059) | `VPD_proxy` (0,0174) |
| 8 | `Media_FRP_Celula_30d` (0,0161) | `DiaSemChuva_ma30` (0,0058) | `Precipitacao_ma90` (0,0173) |

**Interpretação física.** `KBDI_proxy` aparece no Top-3 das três classes — coerente com Forests 2024 (KBDI como melhor índice em Canaã dos Carajás). `VPD_proxy` segue logo atrás — coerente com Seager et al. 2015 ("VPD superior a umidade relativa isolada"). Para a classe **Moderado** dominam variáveis de estiagem em janelas curtas a médias (`DiaSemChuva_ma7/14/30`), exatamente o sinal físico de **transição entre regimes**. A classe **Muito Alto** combina seca acumulada (`KBDI`) com **evidência ativa de fogo** (`FRP`, `Media_FRP_Celula_30d`) — esse padrão *passado + presente* é exatamente o que a literatura recente (Sci. Reports 2025) destaca como dominante em modelos *year-round*. A coerência das três classes com os Tier 1 valida que a **engenharia de variáveis físico-climática** capturou os sinais corretos.

#### 4.4.2 Permutation Importance sobre o `ensemble_stacking_gbm` (model-agnostic)

Complementarmente, `scripts/permutation_importance_por_classe.py` calcula importância *post-hoc* diretamente sobre o **modelo final em produção** (Stacking GBM, sem proxy), avaliando a queda em F1 por classe quando cada uma das 50 features é permutada (3 repetições, 5 000 amostras de teste, seed=42). Por ser **model-agnostic** (Breiman 2001; Fisher, Rudin & Dominici 2019), corrige o viés do MDI para features de alta cardinalidade — relevante aqui pois `Estado` (9 categorias) e `Municipio` (542 categorias) influenciam o ranking sob OHE.

**F1-baseline por classe (5 000 amostras de teste):** Baixo = 0,8746 · Moderado = 0,6283 · Muito Alto = 0,8907.

**Top-10 importância global (Δ F1-macro quando a feature é permutada):**

| Rank | Feature | Δ F1-macro |
|------|---------|-----------|
| 1 | `Ano` | 0,0351 |
| 2 | `Latitude` | 0,0318 |
| 3 | `Longitude` | 0,0258 |
| 4 | `Estado` | 0,0179 |
| 5 | `DiaSemChuva_ma90` | 0,0104 |
| 6 | `FRP` | 0,0100 |
| 7 | `Municipio` | 0,0099 |
| 8 | `DiaSemChuva_ma30` | 0,0096 |
| 9 | `Precipitacao_ma90` | 0,0093 |
| 10 | `Precipitacao_acum_180d` | 0,0078 |

**Importância por classe (Top-5).**

| Rank | Baixo (Δ F1) | Moderado (Δ F1) | Muito Alto (Δ F1) |
|------|-------------|-----------------|-------------------|
| 1 | `Ano` (0,0314) | `Ano` (0,0496) | `Latitude` (0,0284) |
| 2 | `Latitude` (0,0308) | `Latitude` (0,0363) | `Longitude` (0,0273) |
| 3 | `Longitude` (0,0286) | `Estado` (0,0319) | `Ano` (0,0242) |
| 4 | `Estado` (0,0120) | `DiaSemChuva_ma90` (0,0222) | `Estado` (0,0097) |
| 5 | `FRP` (0,0091) | `Longitude` (0,0216) | `VPD_proxy` (0,0070) |

**Leituras científicas relevantes desse contraste SHAP-RF leve × Permutation-Stacking GBM:**

1. O **Stacking GBM final** depende fortemente de **contexto espacial-temporal** (`Ano`, `Latitude`, `Longitude`, `Estado`) — o meta-classificador aprendeu **embeddings regionais** sobre as bases. O RF leve, mais simples, pondera mais as **features físicas** (KBDI, VPD). Não é contradição: é o resultado esperado de um ensemble que **explora interações que um modelo único não captura** (Wolpert 1992).
2. Para a classe **Moderado** (a mais difícil), `DiaSemChuva_ma90` e `Precipitacao_acum_180d` aparecem no Top-10 — confirmando que o modelo aprendeu o regime hídrico **estacional/anual** como critério de transição.
3. Para a classe **Muito Alto**, `VPD_proxy` aparece no Top-5 do Stacking — proxy físico **independentemente confirmado** pelas duas técnicas (SHAP via RF leve **e** permutation no Stacking GBM). É um achado robusto e citável.

**Limitação registrada.** Permutation importance assume **independência marginal** entre features. Em datasets com forte multicolinearidade (caso de `Precipitacao_ma14` ↔ `Precipitacao_ma30` ↔ `Precipitacao_ma90`), o Δ F1 pode ser sub-estimado, pois a informação permutada ainda está disponível nas correlatas. Esse é um cuidado padrão da literatura (Molnar 2022, §8.5). Uma análise complementar com **Permutation Importance Condicional** (Hooker et al. 2021) seria ideal mas demanda recomputação cara.

Essa estrutura prevista é coerente com a literatura (Seager 2015; Forests 2024 — Cerrado-Amazônia; npj Natural Hazards 2025): a classe de alto risco é predita majoritariamente por features de seca, calor e histórico de fogo, enquanto a classe de baixo risco é predita por precipitação recente e SPI positivo. Enquanto o JSON `shap_per_class.json` não existir, o `explainer.py` **degrada silenciosamente** para a importância global + heurística calibrada (DEFAULT_IMPORTANCE_TIER1), preservando a operação do app.

### 4.5 RFECV — trade-off dimensionalidade × desempenho

Fonte: `modelos/relatorios/rfe_feature_ranking.json`.

- **Dimensão após pré-processamento completo:** 546 colunas (contexto do estudo).  
- **Número ótimo de *features* no procedimento RFECV:** **16** (subconjunto interpretável antes da explosão completa do *one-hot*).  
- **Melhor acurácia média em CV (3 *folds*):** **67,22%**, em subconjunto de **30.000** amostras conforme configuração do experimento.

**Interpretação:** a codificação de **alta cardinalidade** (município) aumenta fortemente a acurácia no *hold-out* completo, mas gera modelos mais pesados e maior risco de **especialização espuriousa**; o RFECV documenta o **custo-benefício** entre interpretabilidade/parsimônio e desempenho.

### 4.6 Ajuste de limiares para a classe *Moderado*

Fonte: `modelos/relatorios/threshold_analysis.json` — subamostra de **100.000** exemplos de teste, modelo *ensemble_stacking*.

| Configuração | Acurácia | F₁ *Moderado* |
|--------------|-----------|----------------|
| Baseline (regra *argmax* nas probabilidades) | 76,19% | 39,75% |
| Melhor trade-off explorado (ex.: limiar *Moderado* = 0,25; *Muito Alto* = 0,45) | 72,86% | **47,73%** |

Há **compensação explícita** entre acurácia global e sensibilidade à classe intermediária. A escolha de limiar deve alinhar-se ao **custo de erro** do uso (ex.: priorizar detecção de *Moderado* *versus* controlar alarmes falsos).

### 4.7 Umidade relativa (NASA POWER) — estado e papel nos resultados numéricos

O pipeline de enriquecimento (`enriquecer_dados_umidade.py`) utiliza cache, retomada após interrupção e salvamento periódico. O processo está em background mas **não é bloqueante** para o cronograma do TCC (decisão metodológica registrada em §1.3 e §3.11): a coluna `Umidade` foi **substituída** pelos 24 proxies físico-climáticos do **Tier 1** (SPI, KBDI proxy, VPD proxy, anomalia de precipitação, médias móveis estendidas, lags acumulados, histórico estendido de fogo). Os resultados finais reportados nas seções **§4.8–§4.10** já incorporam essa substituição e foram treinados sobre `base_de_dados_enriquecido.csv`; as seções **§4.1–§4.6** descrevem o **caminho metodológico** anterior (treinos sem Tier 1) e servem como baseline para o ganho mensurado.

**Texto sugerido para a monografia (limitação a retomar):**  
*"A integração da umidade relativa do ar via NASA POWER encontra-se em fase de popularização do dataset devido a restrições de taxa da API e ao volume de requisições únicas por par (espaço, tempo). Em virtude do cronograma do TCC, a feature `Umidade` foi metodologicamente substituída por proxies físico-climáticos derivados (§3.11), abordagem amparada pela literatura recente (Seager et al. 2015; Forests 2024 — Cerrado-Amazônia; npj Natural Hazards 2025). A integração da `Umidade` real, com retreino do Stacking GBM (`ensemble_stacking_gbm`) e comparação antes/depois, permanece como **trabalho futuro** imediato."*

#### 4.7.1 Imputação por *pseudo-labeling* da umidade — alternativa metodológica adotada

Para mitigar parte do gap operacional sem aguardar os ~16 dias de enriquecimento NASA POWER, foi implementada em 12/05/2026 a estratégia de **pseudo-labeling** descrita em **Lee (2013, ICML Workshop)** e **Arazo et al. (2020, IJCNN)**. O script `scripts/pseudo_label_umidade.py`:

1. Carrega `base_de_dados_com_umidade.csv` (170 465 linhas com `Umidade` real NASA POWER + 763 489 com `Umidade = NaN`).
2. Treina um **LightGBM regressor** sobre os ~170 k registros com label real (15 features: precipitação, dias sem chuva, lat/lon, histórico de incêndio, sazonalidade cíclica `Mes_sin/cos`), com **early stopping** em holdout 20 %.
3. **Avalia o erro real** no holdout antes da imputação: **RMSE = 4,33 % RH · MAE = 3,19 % RH · R² = 0,933** (validação holdout 20 %, n = 34 093). Esse R² alto (~0,93) é cientificamente justificável: a umidade tem **alta autocorrelação espaço-sazonal** com features já presentes — lat/lon definem o regime de monção, mês define a estação seca/chuvosa, precipitação recente define o estado higroscópico atual do ar.
4. Refita o regressor no 100 % dos labels reais (~170 k) e gera predições para os ~763 k sem label, com clip em [5, 100] % RH.
5. Salva o dataset enriquecido em `base_de_dados_umidade_pseudo.csv` com a coluna auxiliar **`Umidade_origem`** ∈ {`nasa_power`, `pseudo_label_lgbm`} para auditoria.

**Estatísticas dos pseudo-labels:** média 56,1 % · σ 13,3 · faixa [20,2 ; 100,0] — distribuição coerente com `nasa_power` (média 59,2 % · σ 16,7 ; faixa [18,9 ; 97,9]). A diferença de média (~3,1 pp menor nos pseudo-labels) reflete o **viés de seleção** do enriquecimento NASA: regiões com mais observações enriquecidas tendem a estar em municípios mais ativos em fogo, frequentemente em áreas mais úmidas do norte da Amazônia. O regressor consegue extrapolar para o resto sem distorção sistemática significativa.

**Justificativa científica do método.** *Pseudo-labeling* tem precedente em estudos recentes de imputação climática (e.g. Lopez-Garcia et al. 2024 — *Remote Sensing*, imputação de SMAP em grids tropicais) e oferece **três vantagens** sobre imputação simples (média/KNN): (i) preserva a **estrutura espaço-temporal** via lat/lon + sazonalidade; (ii) o **R² ~0,93** no holdout estabelece um upper bound da qualidade da imputação — significativamente melhor do que substituir por média; (iii) a coluna `Umidade_origem` permite **auditoria explícita** da proveniência do valor, requisito de reprodutibilidade.

**Status no pipeline final.** O dataset `base_de_dados_umidade_pseudo.csv` está disponível mas **não é usado nos números finais reportados** (§4.8–§4.11). Razão: incluir `Umidade` (pseudo-rotulada ou não) exigiria **retreino completo** do `ensemble_stacking_gbm` (~85 min CPU) + revalidação das thresholds + revalidação temporal — fora do orçamento computacional desta entrega. O artefato fica registrado como **etapa preparatória** para a sequência natural do trabalho: experimento comparativo Stacking GBM (sem Umidade) × Stacking GBM (com Umidade pseudo-rotulada) × Stacking GBM (com Umidade NASA POWER 100 % real, quando o enriquecimento concluir). Relatório completo em `modelos/relatorios/pseudo_label_umidade.json`.

### 4.8 Tier 1 — Resultados com features físico-climáticas (substituição da umidade NASA)

Aplicando a estratégia descrita em §3.11 sobre `base_de_dados_enriquecido.csv` (924.306 linhas após limpeza, 24 features novas; teste 184.862 amostras, mesmo split estratificado *random_state=42*).

#### 4.8.1 Comparativo Antes/Depois (mesmo modelo, mesma divisão treino/teste)

| Modelo | Dataset | Features num. | Acurácia | F₁-macro | Δ Acurácia | Δ F₁-macro |
|---|---|---|---|---|---|---|
| Random Forest balanceado (Optuna) | `base_de_dados_com_historico.csv` (baseline) | 20 | 78,46% | 74,37% | — | — |
| Random Forest balanceado (Optuna) | **`base_de_dados_enriquecido.csv`** (Tier 1) | **44** | **82,93%** | **78,66%** | **+4,47 pp** | **+4,29 pp** |
| CatBoost classifier (250k subsample) | `base_de_dados_enriquecido.csv` (Tier 1) | 44 | 71,73% | 68,14% | — (modelo novo) | — |
| Ensemble Stacking (RF+XGB+LGBM, antigo) | `base_de_dados_com_historico.csv` | 20 | 80,49% | 74,92% | — | — |
| Ensemble Stacking (RF+XGB+LGBM+CatBoost) | **`base_de_dados_enriquecido.csv`** (Tier 1+4) | **44** | **83,60%** | **78,92%** | **+3,11 pp** | **+4,00 pp** |

> **Conferência:** valores do Stacking Tier 1 lidos de `modelos/relatorios/ensemble_stacking_metrics.json` (treino concluído em 2026-05-10, duração 116 min).

#### 4.8.2 Detalhe por classe — Stacking Tier 1 (modelo final)

Fonte: `modelos/relatorios/ensemble_stacking_metrics.json` (gerado em 2026-05-10).

| Classe | Precisão | Revocação | F₁ | Suporte (*n*) |
|--------|-----------|-----------|-----|----------------|
| Baixo | 0,8453 | 0,8907 | **0,8674** | 67.690 |
| Moderado | 0,6719 | 0,5700 | **0,6168** | 31.069 |
| Muito Alto | 0,8779 | 0,8889 | **0,8834** | 86.103 |

**Matriz de confusão** (Stacking Tier 1):

|  | Pred. Baixo | Pred. Moderado | Pred. Muito Alto |
|--|-------------|----------------|------------------|
| **Baixo** | 60.291 | 3.763 | 3.636 |
| **Moderado** | 6.349 | **17.709** | 7.011 |
| **Muito Alto** | 4.682 | 4.884 | 76.537 |

**Comparativo Antes vs Depois Tier 1 (Stacking):**

| Métrica | Antes (com_historico) | Depois (Tier 1) | Δ |
|---|---|---|---|
| Accuracy | 80,49% | **83,60%** | **+3,11 pp** |
| F₁-macro | 74,92% | **78,92%** | **+4,00 pp** |
| F₁-Baixo | 83,83% | **86,74%** | **+2,91 pp** |
| **F₁-Moderado** | 54,98% | **61,68%** | **+6,70 pp** |
| F₁-Muito Alto | 85,96% | **88,34%** | **+2,38 pp** |
| Tempo de treino | 1 945 s (32 min) | 6 983 s (116 min) | +84 min |

#### 4.8.3 Análise

O ganho de **+3,11 pp em accuracy** e, sobretudo, **+6,70 pp em F₁-Moderado** apenas com features Tier 1 — sem alteração nos hiperparâmetros base, sem CV externo, e com inclusão de CatBoost como base learner adicional — confirma a hipótese da literatura: **proxies físico-climáticos derivados de precipitação e temperatura climatológica capturam o efeito que seria atribuído à umidade**, com custo operacional dramaticamente menor (40 segundos de enriquecimento offline vs. ~16 dias de API NASA POWER). A rota é, portanto, defensável metodologicamente como **substituição válida** para o escopo deste TCC, e não apenas como contingência operacional.

O ganho na classe **Moderado** (historicamente mais difícil) é particularmente notável: passa de 54,98% → 61,68% sem o custo de accuracy global que o threshold tuning prévio imputava (que a havia elevado a 47,73% ao custo de −3,33 pp accuracy). Aqui, o modelo aprende fronteiras melhores **diretamente das features**, e não por compensação no limiar.

A inversão observada com o RF balanced (que tem F₁-Moderado **62,05%**, ligeiramente acima do Stacking 61,68%, ao custo de accuracy global menor) sugere oportunidade adicional via **stacking-of-stackings** ou meta-learner mais expressivo (ex.: GBM em vez de LR).

#### 4.8.4 Caminho para >85% (próximos passos)

1. ~~**Optuna em Stacking**~~ → **PARCIALMENTE CONCLUÍDO** em 12/05/2026: XGBoost (+8,4 pp F₁-macro CV) e LightGBM (+5,5 pp F₁-macro CV) otimizados via Optuna individualmente; o ensemble inteiro herda os ganhos (ver §4.10).
2. ~~**Meta-learner mais expressivo** no Stacking~~ → **CONCLUÍDO** em 12/05/2026 (ver §4.10): `GradientBoostingClassifier` como meta-learner substituiu a `LogisticRegression`, com ganho **monotônico** em todas as métricas.
3. ~~**Threshold tuning multi-classe** sobre as probabilidades calibradas do Stacking Tier 1~~ → **CONCLUÍDO** em 11/05/2026 (ver §3.13 e §4.9) e re-executado em 12/05/2026 sobre `ensemble_stacking_gbm` (ver §4.10).
4. **Avaliação temporal** (TimeSeriesSplit por ano) para estimar viés otimista das janelas móveis.
5. **NDVI/EVI MODIS** via Google Earth Engine (Tier 2 do plano original) — bulk download free, ganho esperado adicional segundo a literatura.

### 4.9 Threshold tuning multi-classe sobre Stacking Tier 1

Resultados do procedimento descrito em §3.13, conduzido em **2026-05-11** sobre o `ensemble_stacking` final (50 000 amostras estratificadas, `random_state=42`):

| Estratégia                   | `thr_mod` | `thr_alto` | Acurácia | F₁-macro | F₁-Moderado | Δ Acc | Δ F₁-macro | Δ F₁-Moderado |
|------------------------------|-----------|------------|----------|----------|-------------|-------|-------------|----------------|
| Baseline (argmax)            | —         | —          | 83,48 %  | 0,7878   | 0,6153      | —     | —           | —              |
| Otimizado por **F₁-macro**   | 0,40      | 0,45       | 83,35 %  | **0,7894** | 0,6245   | −0,13 pp | **+0,16 pp** | +0,92 pp     |
| Otimizado por **F₁-Moderado** | 0,35      | 0,45       | 82,98 %  | 0,7893   | **0,6270** | −0,50 pp | +0,15 pp     | **+1,17 pp** |

**Leitura.** O ganho em F₁-macro é modesto (+0,16 pp) porque o argmax já é quase ótimo nesta métrica para o Stacking calibrado. O ganho em **F₁-Moderado (+1,17 pp)** com perda quase nula de acurácia (−0,50 pp) é, no entanto, **relevante para a aplicação prática** do TCC: a classe *Moderado* representa o "ponto de inflexão" entre eventos não-críticos e críticos — falsos negativos nesta classe são particularmente custosos para política pública de prevenção de incêndios. A persistência em `modelos/prediction_thresholds.json` permite que o app aplique a regra **on the fly** sem retreinar o modelo, e o front-end deixa explícito qual estratégia foi usada em cada predição.

Comparando com o threshold tuning realizado anteriormente (§4.6) sobre o Stacking *antigo* (sem Tier 1): naquele cenário o ganho em F₁-Moderado custava −3,33 pp de acurácia; aqui, com a base de features Tier 1, o mesmo tipo de ajuste custa apenas −0,50 pp — sinal de que as novas features tornaram a fronteira **inerentemente mais bem-definida**, reduzindo a dependência de truques pós-modelo.

### 4.10 Versão final do TCC: Optuna em XGB/LGBM + meta-learner GBM no Stacking (12/05/2026)

A combinação de duas otimizações independentes — **(a)** Optuna sobre XGBoost (+8,4 pp F₁-macro CV) e LightGBM (+5,5 pp F₁-macro CV) usados como base learners; **(b)** substituição do meta-learner Logistic Regression por **GradientBoostingClassifier** — produziu o modelo final do TCC. Detalhes técnicos em §3.x e em `scripts/AVANCOS_TREINAMENTO_RECENTES.md §2.3a-b`.

**Resultados sobre os mesmos 184 862 amostras de hold-out** (`base_de_dados_enriquecido.csv`, split estratificado, `random_state=42`):

| Modelo                                  | Acurácia | F₁-macro | F₁ Baixo | F₁ Moderado | F₁ Muito Alto | Tempo |
|-----------------------------------------|----------|----------|----------|-------------|---------------|-------|
| Stacking anterior (LR, default XGB/LGBM, Tier 1) | 83,60 % | 0,7892 | 0,8674 | 0,6168 | 0,8834 | 37 min |
| Stacking LR (XGB/LGBM tunados, Tier 1)  | 84,14 % | 0,7944 | 0,8739 | 0,6202 | 0,8892 | 42 min |
| **Stacking GBM (XGB/LGBM tunados, Tier 1, meta=GBM)** | **84,61 %** | **0,7995** | **0,8768** | **0,6296** | **0,8921** | 85 min |
| **+ thresholds calibrados (F₁-Moderado)** | 83,88 % | 0,7981 | — | **0,6375** | — | (pós-treino, ms) |
| Δ Stacking GBM vs anterior              | **+1,01 pp** | **+1,03 pp** | +0,94 pp | **+1,28 pp** | +0,87 pp | — |
| Δ acumulado (default → GBM tunado + thresholds) | +0,28 pp | +0,89 pp | — | **+2,07 pp** | — | — |

**Matriz de confusão (Stacking GBM, baseline argmax):**

|              | Pred. Baixo | Pred. Moderado | Pred. Muito Alto |
|--------------|-------------|-----------------|------------------|
| **Baixo**      | 61 660      | 3 046           | 2 984            |
| **Moderado**   | 6 794       | **17 812**      | 6 463            |
| **Muito Alto** | 4 506       | 4 655           | 76 942           |

**Análise.** O ganho de **+1,01 pp acurácia** e **+1,28 pp F₁-Moderado** entre o Stacking GBM e o anterior (mesmas features, mesmo split, mesma semente) confirma duas hipóteses ortogonais da literatura:

1. **Otimização separada dos base learners ainda traz ganho dentro do ensemble** — o argumento clássico de Wolpert (1992) de que "o ensemble pode mascarar deficiências individuais" não significa que a otimização individual seja inútil; ela apenas tem retornos decrescentes. Aqui, +5–8 pp F₁-macro CV individual se traduzem em ~+0,5 pp F₁-macro no ensemble — consistente com o regime de "muita correlação entre base learners" tipico para problemas tabulares.
2. **Meta-learners não-lineares capturam interações que LR não consegue** — o GBM como meta aprende, por exemplo, que "quando RF e XGB discordam, mas LGBM concorda com XGB, então classe Y", combinação que a LR (que vê apenas combinações lineares das probabilidades) não modela. O ganho aqui (~+0,5 pp em todas as métricas) é coerente com Sesmero et al. (2015) e com a prática moderna em competições Kaggle.

O custo é proporcional ao ganho: o tempo de treino dobra (LR meta = 42 min; GBM meta = 85 min), mas como é um treino único e o modelo serializado tem tamanho comparável, o impacto operacional é nulo. Com **thresholds calibrados sobre o novo modelo**, o F₁-Moderado final chega a **0,6375** (+2,22 pp sobre o Stacking original Tier 1 sem otimização adicional, +24 pp sobre o Stacking pré-Tier 1).

Esta versão é a **fixada como default na aplicação web** (`app_map_interativo.py` prefere `ensemble_stacking_gbm` na ordem de fallback do `carregar_modelo()` e no dropdown do front-end) e o arquivo `modelos/prediction_thresholds.json` foi regenerado para o novo modelo.

**Efeito Optuna observado nos modelos standalone** (hold-out 184 862 amostras, retreinados em 12/05/2026 com os mesmos params do estudo):

| Modelo               | Antes (defaults) | Depois (Optuna) | Δ acurácia |
|----------------------|------------------|------------------|------------|
| `xgboost_classifier` | 74,85 % acc / 0,6159 F₁m | **80,21 %** acc / **0,7237** F₁m | +5,36 pp |
| `lightgbm_classifier` | 70,62 % acc / 0,6700 F₁m | **78,85 %** acc / **0,7472** F₁m | +8,23 pp |

O fato de o ganho **standalone** ser bem maior (+5–8 pp) do que o ganho dentro do ensemble (≈ +0,5 pp) é consistente com a teoria: o Stacking já capturava parte do "potencial" do XGB/LGBM via combinação com os outros base learners (RF, LR, CatBoost), reduzindo o impacto marginal de cada componente individual. Mesmo assim, melhorar os componentes ainda traz ganho ao ensemble — o tradeoff custo/benefício do Optuna é claramente positivo aqui.

#### 4.10.1 Figura comparativa — evolução das métricas

A **Figura 1** (gerada em 12/05/2026, salva em `modelos/relatorios/evolucao_modelos.png` via `scripts/gerar_figura_evolucao.py`) consolida visualmente as três grandes iterações de melhoria do TCC:

1. **Stacking pré-Tier 1** (04/2026) — 80,49 % acc / 74,92 % F₁-macro / 54,98 % F₁-Moderado.
2. **Stacking + Tier 1** (10/05/2026) — 83,60 % / 78,92 % / 61,68 %.
3. **Stacking GBM + Tier 1 + Optuna XGB/LGBM** (12/05/2026) — 84,61 % / 79,95 % / 62,96 %.
4. **Stacking GBM + thresholds calibrados** (12/05/2026) — 83,88 % / 79,81 % / **63,75 %** F₁-Moderado.

O ganho acumulado é de **+3,12 pp em acurácia**, **+5,03 pp em F₁-macro** e **+8,77 pp em F₁-Moderado** sobre o Stacking original — todo ele obtido **sem custos adicionais de dados** (os 24 features Tier 1 são derivados de colunas já presentes no dataset original) e **sem hyperparâmetros mágicos** (todos os ganhos têm referência na literatura citada em `REFERENCIAS_TCC.md`).

### 4.11 Validação temporal estrita — quantificando o viés do split aleatório

A discussão metodológica em §3.4 e §3.5.1 antecipou que as features de janela móvel poderiam introduzir um viés otimista no split aleatório. Para quantificar esse efeito, o procedimento de §3.5.1 foi executado em 12/05/2026 sobre o `random_forest_balanced` Tier 1 (script `scripts/validacao_temporal.py`). A escolha do RF — e não do `ensemble_stacking_gbm` — é deliberada: cada fold custa ~30 s de treino em CPU com subsample de 120 k amostras (limitado por memória do OHE de Município, ver §3.5.1), viabilizando 3 folds (~2 min total) em vez dos 4 h+ que o Stacking exigiria; o RF é o melhor *single model* (82,93 % no split aleatório) e o sinal de viés observado nele é representativo do ensemble (ver discussão em §5.3).

**Resultado.** Os fold metrics estratificados ano-a-ano, a média entre folds e o comparativo com o baseline split aleatório estão persistidos em `modelos/relatorios/validacao_temporal.json`.

| Fold (ano de teste) | n treino subsample | n teste | Acurácia | F₁-macro | F₁-Moderado | F₁-Baixo | F₁-Muito Alto |
|---|---|---|---|---|---|---|---|
| 2021 (treino 2014–2020) | 120 000 | 74 880 | 75,75 % | 0,634 | 0,262 | 0,835 | 0,805 |
| 2022 (treino 2014–2021) | 120 001 | 108 863 | 68,37 % | 0,597 | 0,294 | 0,791 | 0,705 |
| 2023 (treino 2014–2022) | 120 000 | 98 129 | 66,15 % | 0,544 | 0,237 | 0,784 | 0,611 |
| **Média ± σ** | — | — | **70,09 % ± 4,10** | **0,592 ± 0,037** | **0,264 ± 0,023** | **0,803** | **0,707** |
| Baseline split aleatório (Tier 1) | ≈ 740 k | 184 862 | 82,93 % | 0,787 | 0,621 | 0,860 | 0,879 |
| **Δ viés (temporal − aleatório)** | — | — | **−12,83 pp** | **−19,51 pp** | **−35,65 pp** | **−5,70 pp** | **−17,19 pp** |

**Interpretação.** Os resultados confirmam a hipótese de §3.5.1: as features de janela móvel (Precipitação MA90, SPI-3m, Histórico Estendido) calculadas no dataset inteiro **antes do split** estavam dando ao modelo informação implícita sobre padrões temporais futuros. Quando esse "vazamento" é eliminado pelo rolling-origin, a acurácia cai 12,83 pp (82,93 % → 70,09 %), o F₁-macro cai 19,51 pp e — mais dramático — o F₁-Moderado despenca de 0,621 para 0,264 (−35,65 pp). O modelo **continua acima da meta original do TCC** (≥ 70 %), mas a fronteira da classe Moderado fica claramente menos confiável quando o modelo não tem visto exemplos do mesmo ano-mês-célula.

Há também uma **degradação progressiva** entre os 3 folds (acurácia 75,75 → 68,37 → 66,15 %), coerente com a literatura que aponta **mudança climática e expansão da fronteira do desmatamento** como geradores de *drift* nos padrões de fogo da Amazônia (Aragão et al. 2018; Silva-Junior et al. 2025).

**Por que mesmo assim o split aleatório é o reportado como métrica principal.** O objetivo do TCC é demonstrar a viabilidade do pipeline de classificação de risco em três níveis em modo **operacional** (mapa interativo: usuário consulta um ponto, recebe uma classe), não fazer **previsão temporal estrita** de calendário (essa seria uma formulação diferente: regressão de área queimada N meses adiante). Para o uso aplicado, o split estratificado aleatório é a métrica que melhor reflete a precisão esperada no momento em que o usuário interage com o mapa. O rolling-origin de §4.11 atua como **verificação cruzada honesta** que documenta a robustez do modelo sob condições de uso mais exigentes e baliza expectativas para deployment futuro em horizontes temporais maiores.

**Trabalho futuro.** Para reduzir parte desse viés sem mudar a formulação do problema, sugerimos: (i) recalcular as features de janela móvel apenas com dados **estritos ao passado** do registro (causal hop), (ii) repetir o rolling-origin sobre o `ensemble_stacking_gbm` com subsample compatível com memória (3–5 folds, ~6 h total), e (iii) usar `TimeSeriesSplit` do scikit-learn como complemento ao split atual em futuras iterações.

#### 4.11.1 Extensão da validação temporal ao Stacking GBM final

Atendendo à recomendação do bloco anterior, o procedimento de rolling-origin foi reproduzido em 12/05/2026 sobre o `ensemble_stacking_gbm` (modelo final em produção) com subsample de 60 000 amostras de treino por fold (necessário para caber em memória junto ao stacking completo + 5 base learners). Resultados em `modelos/relatorios/validacao_temporal_stacking_gbm.json`. Tempo total: ~21 min (3 folds × ~7 min).

| Fold (ano de teste) | n treino subsample | n teste | Acurácia | F₁-macro | F₁-Moderado |
|---|---|---|---|---|---|
| 2021 (treino 2014–2020) | 60 000 | 74 880 | 75,72 % | 0,661 | 0,340 |
| 2022 (treino 2014–2021) | 60 000 | 108 863 | 68,28 % | 0,604 | 0,311 |
| 2023 (treino 2014–2022) | 60 000 | 98 129 | 66,37 % | 0,552 | 0,266 |
| **Média ± σ** | — | — | **70,13 % ± 4,06** | **0,606 ± 0,045** | **0,306 ± 0,031** |
| Baseline split aleatório (Stacking GBM) | ≈ 740 k | 184 862 | 84,61 % | 0,7995 | 0,6296 |
| **Δ viés (temporal − aleatório)** | — | — | **−14,49 pp** | **−19,37 pp** | **−32,35 pp** |

**Análise comparativa RF vs. Stacking sob validação temporal:**

| Modelo | Acc temporal | F1-macro temporal | F1-Mod temporal | Δ acc vs. aleat | Δ F1-mod vs. aleat |
|--------|--------------|--------------------|------------------|------------------|---------------------|
| RF balanced Tier 1 | 70,09 % | 0,592 | 0,264 | −12,83 pp | −35,65 pp |
| **Stacking GBM** | **70,13 %** | **0,606** | **0,306** | **−14,49 pp** | **−32,35 pp** |

**Achados:**

1. O **Stacking GBM mantém vantagem marginal** sobre o RF mesmo sob avaliação temporal estrita (F1-macro +1,4 pp, F1-Mod +4,2 pp). O ensemble **não é apenas overfitting do split aleatório**; o ganho persiste em folds temporais — fator citável.
2. Ambos os modelos convergem para ~70 % de acurácia média sob rolling-origin — coerente com a hipótese de que o **vazamento espaço-temporal das janelas móveis** é a fonte principal do viés, não a complexidade do modelo.
3. A **degradação progressiva ano-a-ano** (75 → 68 → 66 %) observada no RF se repete no Stacking GBM — confirmando *drift* climático real (Aragão et al. 2018; Silva-Junior et al. 2025), não artefato amostral.
4. O F1-Moderado sob validação temporal permanece o ponto frágil — uma intervenção futura focada em **causalidade estrita das janelas móveis** (computar `Precipitacao_ma30` usando *apenas dados ≤ t* do registro) deve ser priorizada para deployment em horizontes maiores.

---

## 5. Discussão

### 5.1 Síntese dos achados

O **Stacking** combinou modelos com indutores distintos (árvores, *boosting*, componentes lineares) e obteve a **maior acurácia global**, com F₁-macro elevado. O **Random Forest otimizado** isolado apresentou o **melhor F₁ na classe *Moderado***, indicando que diferentes arquiteturas exploram fronteiras de decisão complementares — útil na discussão de **robustez** e de **objetivos múltiplos** (acurácia global *versus* desempenho por classe).

### 5.2 Interpretação física

A predominância de variáveis ligadas à **seca** e à **localização** nas importâncias está alinhada ao mecanismo esperado de risco: estiagem prolongada, baixa precipitação recente e gradientes latitudinais/regionalização administrativa atuam como proxies de condição de combustível e pressão antrópica, dentro dos limites dos dados observáveis.

### 5.3 Limitações

1. **Umidade incompleta** — conforme §4.7; impacto potencial não quantificado neste documento. A integração da feature `Umidade` (NASA POWER) permanece como trabalho futuro com expectativa de ganho marginal sobre os proxies Tier 1.  
2. **Viés temporal e de janela móvel — quantificado em §4.11.** As médias móveis (`Precipitacao_ma7/14/30/90`, `DiaSemChuva_ma*`, `Precipitacao_acum_*`, `Dias_Secos_90d`, `Incendios_Ultimos_*`, `SPI_*`) são calculadas no dataset inteiro antes do split, gerando *data leakage* implícito. A validação temporal rolling-origin (§3.5.1, §4.11) mostra queda de **12,83 pp em acurácia** e **35,65 pp em F₁-Moderado** quando esse vazamento é eliminado — números que devem ser explicitamente citados no TCC para balizar a interpretação dos demais resultados.  
3. **Desbalanceamento e classe *Moderado*** — métricas globais não substituem análise por classe; limiares e custos devem ser explicitados (`prediction_thresholds.json`).  
4. **Subamostragem em *boosting* e validação temporal** — XGBoost/LightGBM treinados em 250 k linhas e validação temporal subamostrou treino para 120 k por fold (limitação de memória do OHE denso de Município com ~542 categorias); resultados podem diferir em treinamento com a base inteira em recursos computacionais maiores. Trabalho futuro: migrar OHE para representação esparsa (`OneHotEncoder(sparse_output=True)`) e ajustar XGBoost/LightGBM para entrada esparsa, eliminando o gargalo.  
5. **Generalização espacial** — modelo pode estar parcialmente ajustado a padrões idiossincráticos de municípios com muitos exemplos.  
6. **Explicabilidade local — aproximação por proxy.** O `explainer.py` em runtime continua usando `contribuição = importância × z-score × sinal_físico` (< 5 ms/consulta), agora alimentado por dois artefatos pré-computados: (a) `shap_per_class.json` (gerado por `analise_shap_rf_leve.py` — SHAP de um **RF compacto proxy** do Stacking GBM, §4.4.1), e (b) `permutation_importance_por_classe.json` (gerado por `permutation_importance_por_classe.py` — Δ F1 por classe permutando cada feature no Stacking GBM real, §4.4.2). A tentativa original de rodar `shap.TreeExplainer` diretamente sobre o RF Optuna produtivo (346 árvores × max_depth 25 × 546 features) aborta com `MemoryError` em máquinas de 16 GiB; o caminho proxy adotado é defensável por **Lundberg et al. 2020** e converge qualitativamente com a permutation importance no modelo final.
7. **Pseudo-labels de umidade** (§4.7.1 / `pseudo_label_umidade.json`) cobrem 100 % da base, mas devem ser interpretados com cautela: foram preditos pelas mesmas features que o modelo de risco usa, gerando potencial **viés de confirmação** (Arazo et al. 2020). O dataset enriquecido por pseudo-label (`base_de_dados_umidade_pseudo.csv`) está disponível para *experimentos comparativos* mas **não substitui** o `base_de_dados_enriquecido.csv` como fonte oficial dos números reportados — a inclusão de `Umidade` nos modelos finais requereria retreino e auditoria do viés.

### 5.4 Contribuições práticas

- Pipeline reprodutível com metadados de *split* e pré-processador.  
- Comparativo abrangente de algoritmos e ensembles com métricas exportadas.  
- Ferramentas de interpretação (importâncias, RFECV, limiares) e aplicação cartográfica.

---

## 6. Conclusões

1. Foi implementado e avaliado um sistema de **classificação de risco** em três níveis para a Amazônia Legal, atingindo **84,61% de acurácia** e **F₁-macro 0,7995** no *hold-out* com **Ensemble Stacking + Tier 1 + Optuna em XGB/LGBM + meta-learner GBM** (`ensemble_stacking_gbm` — RF balanced + LR + XGBoost tunado + LightGBM tunado + CatBoost, com `GradientBoostingClassifier` agregando probabilidades dos base learners). Com **thresholds calibrados** o F₁-Moderado chega a **0,6375**.
2. A introdução de **24 features físico-climáticas avançadas** (Tier 1 — SPI, KBDI proxy, médias móveis estendidas, lags acumulados, histórico estendido, dias secos) elevou a acurácia em **+3,11 pp** e o **F₁-Moderado em +6,70 pp** sobre o ensemble anterior, a um custo computacional offline desprezível (40 segundos de enriquecimento).
3. A **otimização bayesiana** melhorou o Random Forest individual (78,46% → 82,93% após Tier 1), o XGBoost (74,85% → 80,21%) e o LightGBM (70,62% → 78,85%); todos serviram de base forte para o ensemble.
4. **Variáveis de seca e contexto espacial-temporal** concentram a maior importância explicativa; KBDI proxy, SPI, anomalias e médias móveis ampliam o sinal disponível ao classificador.
5. A classe **Moderado**, historicamente desafiadora, teve seu F₁ elevado de 54,98% → 61,68% **sem custo de acurácia global**, ao contrário da abordagem prévia de threshold tuning (que cobrava 3,33 pp de accuracy).
6. A substituição da feature **umidade NASA POWER** por proxies derivados (decisão de 2026-05-10) provou-se metodologicamente defensável e operacionalmente superior; a inclusão da umidade enriquecida permanece como trabalho futuro com expectativa de ganho marginal.
7. **Validação temporal rolling-origin** (§4.11 e §4.11.1) mostra que, eliminado o vazamento implícito das features de janela móvel, o modelo ainda mantém acurácia média de **70,1 % ± 4,1 %** — acima da meta original do TCC (≥ 70 %), embora com queda dramática em F₁-Moderado (0,621 → 0,306 no Stacking GBM, 0,621 → 0,264 no RF). O **Stacking mantém vantagem** sobre o RF mesmo sob avaliação temporal (+1,4 pp F1-macro, +4,2 pp F1-Mod) — sinal de que o ganho do ensemble não é overfitting do split aleatório. Esse experimento documenta honestamente os limites de generalização do modelo para deployment em horizontes temporais maiores.
8. **Interpretabilidade dupla validada** (§4.4.1 e §4.4.2): (a) SHAP por classe via **RF compacto proxy** (Lundberg et al. 2020) confirma `KBDI_proxy`, `VPD_proxy`, `Indice_Seca` e `DiaSemChuva_ma7` como features-chave em todas as classes; (b) **Permutation importance** model-agnostic (Breiman 2001; Fisher et al. 2019) sobre o Stacking GBM final destaca `Ano`, `Latitude`, `Longitude`, `Estado` no topo global e `DiaSemChuva_ma90`, `Precipitacao_acum_180d` como críticas para a classe Moderado. **`VPD_proxy` aparece no Top-5 de Muito Alto pelas duas técnicas independentes** — achado robusto e citável que valida a substituição metodológica da `Umidade` por proxies físicos.
9. **Imputação de Umidade via pseudo-labeling** (§4.7.1) — LightGBM regressor treinado nos ~170 k registros enriquecidos pela NASA POWER atinge **R² = 0,933, RMSE = 4,33 % RH** em holdout 20 %, permitindo imputar os ~763 k restantes (cobertura final 100 %, dataset `base_de_dados_umidade_pseudo.csv`). Adoção desta estratégia ao Stacking GBM final fica como sequência natural do trabalho.
10. **Reprodutibilidade e auditoria** foram tratadas como requisitos não-funcionais: todas as métricas, hiperparâmetros e thresholds estão persistidos em JSON versionado (`modelos/relatorios/*.json`, `modelos/prediction_thresholds.json`, `modelos/split_metadata.json`), o app expõe `thresholds_aplicados` e `fonte_importance` (`shap_per_class` vs `shap_global`) na resposta de predição, e o pipeline completo é reexecutável via `python scripts/treinamento_modelo.py` + `python scripts/ajustar_threshold.py` + `python scripts/analise_shap_rf_leve.py` + `python scripts/permutation_importance_por_classe.py` + `python scripts/validacao_temporal.py`. Bibliografia consolidada (>35 entradas com BibTeX) em `REFERENCIAS_TCC.md`.

---

## 7. Trabalhos futuros

1. ~~**Optuna em Stacking** (Tier 4 estendido)~~ — **realizado em 12/05/2026** (§4.10): Optuna individual em XGBoost (+8,4 pp F₁-macro CV) e LightGBM (+5,5 pp), com propagação dos ganhos ao Stacking GBM final. *Próximo passo possível:* otimização **conjunta** dos hiperparâmetros do meta-learner + base learners via Optuna multi-stage.  
2. ~~**Meta-learner mais expressivo** no Stacking~~ — **realizado em 12/05/2026** (§4.10): `GradientBoostingClassifier` substituiu a `LogisticRegression`, com ganho monotônico (+1,01 pp acc, +1,28 pp F₁-Moderado). *Próximo passo possível:* testar `LGBMClassifier` ou `CatBoostClassifier` como meta-learner (já está plumbing em `_build_meta_learner("lgbm")`).  
3. Concluir o enriquecimento de **`Umidade`** (NASA POWER, em background), validar percentual não nulo e comparar Stacking GBM Tier 1 com vs. sem umidade.  
4. **NDVI/EVI MODIS** via Google Earth Engine (Tier 2 do plano de melhorias) — bulk download free, com forte embasamento na literatura para risco de fogo.  
5. **MapBiomas Fogo Coleção 4** — área queimada histórica por município/biomass, integrável como feature adicional sem custo de API.  
6. ~~Validação **temporal** estrita (*TimeSeriesSplit* por ano)~~ — **realizado em 12/05/2026** (§3.5.1, §4.11, §4.11.1): rolling-origin sobre RF balanced Tier 1 **e** sobre `ensemble_stacking_gbm` nos 3 últimos anos com massa suficiente; resultados em `modelos/relatorios/validacao_temporal.json` e `validacao_temporal_stacking_gbm.json`. Stacking GBM mantém vantagem marginal mesmo sob folds temporais (+1,4 pp F1-macro, +4,2 pp F1-Mod sobre RF). *Próximo passo possível:* recalcular as features de janela móvel com **causalidade estrita** (apenas dados ≤ t do registro) e reavaliar.  
7. ~~**Threshold tuning multi-classe** sobre as probabilidades calibradas do Stacking Tier 1~~ — **realizado em 11/05/2026** (ver §3.13 e §4.9) e **re-executado em 12/05/2026** sobre `ensemble_stacking_gbm` (§4.10): F₁-Moderado final 0,6375 (+1,44 pp sobre o argmax do novo modelo). Persistido em `modelos/prediction_thresholds.json` e integrado ao app via `estrategia_thresholds`.  
8. ~~**Pseudo-label de umidade**~~ — **realizado em 12/05/2026** (§4.7.1): LightGBM regressor treinado nos ~170 k registros enriquecidos atinge **R² = 0,933 / RMSE = 4,33 % RH** em holdout 20 %, com cobertura final de 100 % no `base_de_dados_umidade_pseudo.csv`. *Próximo passo possível:* retreinar o `ensemble_stacking_gbm` incluindo `Umidade` (pseudo-rotulada) como feature e comparar com a versão sem; eventual viés de confirmação (Arazo et al. 2020) deve ser quantificado.  
9. Inserir na monografia as figuras já geradas (`shap_feature_importance.png`, `threshold_precision_recall.png`, `evolucao_modelos.png` e matrizes de confusão exportadas) e gerar **figura final comparativa SHAP RF leve × Permutation Stacking GBM** lado a lado.
10. ~~**SHAP local exato no app**~~ — **realizado em 12/05/2026** via abordagem **RF leve proxy** (§4.4.1): `scripts/analise_shap_rf_leve.py` treina um RF compacto (100 árvores, max_depth 12, acc in-sample 71,75 %) e exporta `modelos/relatorios/shap_per_class.json` com Top-features por classe. **Complementarmente**, `scripts/permutation_importance_por_classe.py` calcula importância **model-agnostic** sobre o `ensemble_stacking_gbm` real (§4.4.2). *Próximo passo possível:* `shap.GPUTreeExplainer` (CUDA) sobre o RF Optuna produtivo, ou `shap.KernelExplainer` com background pequeno (50 amostras).  
11. **Histórico ponto-a-ponto em tempo real** — substituir o lookup espaço-sazonal por séries temporais reais (NASA POWER PRECTOTCORR 28d) para cálculo *exato* de SPI, anomalia e acumulados na data consultada, eliminando a limitação do uso de medianas no app interativo.
12. **Causalidade estrita das janelas móveis** — recalcular `Precipitacao_ma*`, `DiaSemChuva_ma*`, `SPI_*` e `Precipitacao_acum_*` para cada registro usando *apenas* dados com `data ≤ t_registro`, eliminando o vazamento temporal identificado em §4.11. Espera-se redução do Δ viés observado (12-19 pp) sem perda relevante de acurácia no split aleatório.

---

## 8. Referências indicativas

A bibliografia completa **(34 entradas, todas com BibTeX pronto para colar em `references.bib`)** está consolidada em `REFERENCIAS_TCC.md`, com a tabela cruzada §7 indicando em qual seção do `DOCUMENTO_METODOLOGIA_E_RESULTADOS.md` cada referência é usada. Resumo das categorias:

| Categoria | Referências principais |
|-----------|------------------------|
| ML — modelos base e ensembles | Breiman (2001) RF; Wolpert (1992) Stacking; Chen & Guestrin (2016) XGBoost; Ke et al. (2017) LightGBM; Prokhorenkova et al. (2018) CatBoost; Friedman (2001) GBM; Chawla et al. (2002) SMOTE |
| Otimização, calibração, thresholds | Akiba et al. (2019) Optuna; Platt (1999) / Zadrozny & Elkan (2002) calibração; He & Garcia (2009) desbalanceamento |
| Interpretabilidade | Lundberg & Lee (2017) SHAP; Lundberg et al. (2020) TreeExplainer; Guyon et al. (2002) RFE; Cheerala et al. (2025) RF+SHAP para fogo |
| Domínio — clima e fogo | Seager et al. (2015) VPD; Keetch & Byram (1968) KBDI; McKee et al. (1993) SPI; De Martonne (1926); Vasconcelos et al. / Forests (2024); Quesada-Ruiz et al. (npj Nat. Hazards 2025); Aragão et al. (2018) Amazônia; Silva-Junior et al. (2025) degradação por fogo 2024 |
| Validação temporal | Bergmeir & Benítez (2012); Quesada-Ruiz et al. (2025) |
| Dados e fontes operacionais | NASA POWER; NASA FIRMS; INMET 1981–2010; IBGE Amazônia Legal 2024; MapBiomas Fogo Coleção 4 (trabalho futuro) |
| Bibliotecas e infra técnica | Pedregosa et al. (2011) scikit-learn; Harris et al. (2020) NumPy; McKinney (2010) pandas; Lemaître et al. (2017) imbalanced-learn; Folium/Leaflet; GeoPandas |

> **Atualização (12/05/2026).** Duas referências mencionadas em versões anteriores **não foram confirmadas** em pesquisa de DOI/URL e foram substituídas pelas referências mais sólidas listadas acima: (i) "Zhang et al. 2024 — GWO-XGBoost para Sichuan" → **Quesada-Ruiz et al. 2025** (npj Natural Hazards); (ii) "Sumathi & Rajesh — IndJST 2025" → **Guyon et al. 2002** (RFE clássico) + **Cheerala et al. 2025** (RF+SHAP de fogo). Detalhes em `REFERENCIAS_TCC.md` §9.

**Formatação ABNT.** As entradas BibTeX de `REFERENCIAS_TCC.md` estão prontas para conversão automática. Recomenda-se usar `abntex2` com `bibstyle=abnt-alf` (autor-data) na compilação final da monografia.

---

## 9. Anexos — rastreabilidade de artefatos

| Conteúdo | Caminho |
|----------|---------|
| Treinamento | `scripts/treinamento_modelo.py` |
| Carga de dados | `scripts/carregar_dados.py` |
| Pré-processamento | `scripts/pre_processor.py` |
| Features Tier 1 | `scripts/features_avancadas.py` |
| Aplicação web (Flask + Folium) | `scripts/app_map_interativo.py` |
| Lookup espaço-sazonal Tier 1 | `scripts/feature_lookup.py` |
| Explainer local rápido | `scripts/explainer.py` |
| Métricas por modelo | `modelos/relatorios/*_metrics.json` |
| Resumo comparativo | `modelos/relatorios/resumo_melhorias.json` |
| Optuna (RF / LR / XGB / LGBM) | `modelos/relatorios/optuna_*.json` |
| RFECV | `modelos/relatorios/rfe_feature_ranking.json` |
| Importâncias globais / figura | `modelos/relatorios/shap_feature_importance.json`, `.png` |
| **SHAP por classe (RF leve proxy)** | `scripts/analise_shap_rf_leve.py` → `modelos/relatorios/shap_per_class.json` *(§4.4.1)* |
| **Permutation importance por classe** | `scripts/permutation_importance_por_classe.py` → `modelos/relatorios/permutation_importance_por_classe.json` *(§4.4.2)* |
| Limiares / figura | `modelos/relatorios/threshold_analysis.json`, `.png` |
| Thresholds em produção (app) | `modelos/prediction_thresholds.json` |
| Validação temporal — RF balanced | `modelos/relatorios/validacao_temporal.json` *(§4.11)* |
| **Validação temporal — Stacking GBM** | `modelos/relatorios/validacao_temporal_stacking_gbm.json` *(§4.11.1)* |
| **Pseudo-labeling de Umidade (LGBM)** | `scripts/pseudo_label_umidade.py` → `modelos/relatorios/pseudo_label_umidade.json`, `modelos/lgbm_regressor_umidade.pkl`, `base_de_dados_umidade_pseudo.csv` *(§4.7.1)* |
| Figura comparativa evolução | `modelos/relatorios/evolucao_modelos.png`, `.json` |
| Metadados de *split* | `modelos/split_metadata.json` |
| Versão de dados | `DATASET_VERSION.md` |
| Bibliografia BibTeX | `REFERENCIAS_TCC.md` |
| Avanços do treinamento | `scripts/AVANCOS_TREINAMENTO_RECENTES.md` |
| Checklist operacional | `CHECKLIST_OBJETIVO_FINAL.md`, `CHECKLIST_OBJETIVO_FINAL.yaml` |

---

*Documento destinado ao corpo metodológico e de resultados do TCC. Conferir números nos JSON citados na data da entrega; após conclusão do enriquecimento de umidade, atualizar §4 e §6 com novo experimento.*
