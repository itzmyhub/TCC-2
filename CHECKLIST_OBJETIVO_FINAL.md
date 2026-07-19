# Checklist — Objetivo final do TCC (dados + modelos + mapa)

**Companheiro para automação:** `CHECKLIST_OBJETIVO_FINAL.yaml` (fases, IDs `T*`, `done: true/false`).

> **Objetivo:** Base consistente (umidade e features derivadas), modelos robustos (≥70% acurácia alvo), aplicação e mapa alinhados ao pipeline de treino, documentação reprodutível.  
> **Como usar:** Marque `- [ ]` → `- [x]` ao concluir. Tarefas estão ordenadas por dependência lógica; blocos **Paralelo** podem avançar enquanto o enriquecimento NASA roda.

**Convenções para agentes (Cursor / automação):**
- Cada item é uma linha com `- [ ]` ou `- [x]`.
- Coluna **Arquivo(s)** indica onde editar ou executar.
- **Bloqueia:** só iniciar se o pré-requisito estiver `[x]`.

---

## 0. Estado de referência

- [x] Confirmar `requirements.txt` instalado e Python compatível *(validado em 2025-02-12)*
- [x] Dataset principal identificado: `base_de_dados.csv` / `base_de_dados_com_historico.csv`
- [x] Documentar versão do dataset usada no treino (nome do CSV + data) em `DATASET_VERSION.md`

**Arquivo(s):** `requirements.txt`, `DATASET_VERSION.md`

---

## 1. Enriquecimento de dados NASA — Umidade

> **⚠️ STATUS: BACKGROUND, NÃO BLOQUEANTE** *(decisão tomada em 2026-05-10)*  
> Dataset completo: **933.954 linhas** | Progresso após resume: **~755k requisições únicas pendentes** | CSV parcial: **~146k linhas com `Umidade` preenchida (~15,6%)**.  
> Throughput observado com 2 chaves API NASA POWER: **~30–33 req/min** (≈1.920 req/h) → estimativa **~16 dias contínuos** até cobertura total.  
> **Não bloqueia o cronograma**: o ganho da feature `Umidade` foi substituído por proxies físico-climáticos derivados (Tier 1, ver §11). Caso o enriquecimento conclua a tempo, a feature `Umidade` será integrada via §5; caso contrário, fica registrada como "trabalho futuro" no TCC.

- [ ] Manter `enriquecer_dados_umidade.py` rodando até conclusão — comando de retomada abaixo
- [ ] Ao terminar: validar `base_de_dados_com_umidade.csv` (linhas, % NaN em `Umidade` aceitável)
- [ ] Backup do CSV final antes de novos pré-processamentos

**Comando para iniciar/retomar (idempotente — usa cache automaticamente):**
```bash
python scripts/enriquecer_dados_umidade.py --input base_de_dados_com_historico.csv --output base_de_dados_com_umidade.csv --max_workers 4 --salvar_cache_periodicamente 100
```

**Rodar fora do Cursor (sobrevive ao fechar o IDE):** na raiz do projeto, execute **`rodar_enriquecimento_umidade.bat`** (duplo clique ou Explorer). Ele abre uma **nova janela minimizada** com o Python; fechar o Cursor **não** encerra esse processo. Logs continuam em `logs/umidade_enriquecimento*.log`.

**Verificar log:**
```bash
# Stdout/stderr redirecionados para:
logs/umidade_enriquecimento.log
logs/umidade_enriquecimento_err.log
```

**Arquivo(s):** `scripts/enriquecer_dados_umidade.py`, `scripts/config.py` (chaves NASA), `.cache_umidade/`

**Correções realizadas em 13/04/2026:**
- Resume automático: carrega output parcial se tiver mesmo nº de linhas do input; caso contrário usa apenas cache
- Removido `input()` interativo — pode rodar headless/background sem intervenção
- Save periódico de CSV+cache a cada 100 requisições (era somente no final/erro)
- Handler SIGINT/SIGTERM: salva progresso antes de encerrar
- `--no-resume` flag adicionada para forçar início do zero quando necessário

---

## 2. Features sem API externa ✅ CONCLUÍDO

- [x] Executar ou validar pipeline de histórico de incêndios na base disponível  
  **Arquivo(s):** `scripts/adicionar_historico_incendios.py`
- [x] Inclusão automática das colunas de histórico em `carregar_dados.py` quando presentes no CSV
- [x] Retreinar modelos na **base completa** com `base_de_dados_com_historico.csv` e registrar métricas *(2026-04-07: RF 71,52%, LR 64,58%, voting 73,83%, stacking 76,23%; `--no-cv`)*  
  **Arquivo(s):** `scripts/carregar_dados.py`, `scripts/treinamento_modelo.py`, `modelos/relatorios/`

---

## 3. Melhoria dos modelos ✅ CONCLUÍDO (base sem umidade)

- [x] Implementar **VotingClassifier** e **StackingClassifier**  
  **Arquivo(s):** `scripts/treinamento_modelo.py`
- [x] Artefatos `.pkl` e `preprocessor_metadata.json` gerados ao treinar
- [x] `split_metadata.json` com `random_state`, `test_size`, `stratify` e nota de reprodutibilidade
- [x] Criado `scripts/otimizar_hiperparametros.py` com Optuna (RF/LR, 3-fold CV)

**Métricas registradas (base com histórico — 184.862 amostras de teste):**

| Modelo | Accuracy | F1-Macro | F1 Baixo | F1 Moderado | F1 Muito Alto |
|---|---|---|---|---|---|
| **Ensemble Stacking Atualizado** | **80,49%** | **74,92%** | 83,83% | **54,98%** | 85,96% |
| RF Otimizado (Optuna) | 78,46% | 74,37% | 82,83% | 55,67% | 84,62% |
| Ensemble Stacking (antigo) | 76,23% | 67,76% | 80,10% | 40,06% | 83,12% |
| XGBoost | 74,85% | 61,59% | 79,31% | 23,46% | 81,99% |
| Ensemble Voting Soft | 73,83% | 66,69% | 78,36% | 40,38% | 81,33% |
| LightGBM | 70,62% | 67,00% | 77,53% | 45,10% | 78,38% |
| Random Forest Balanced (ant.) | 71,52% | 67,69% | 77,42% | 46,55% | 79,09% |
| Random Forest SMOTE | 69,78% | 66,08% | 75,96% | 45,35% | 76,93% |
| Logistic Regression Balanced | 64,58% | 60,44% | 71,42% | 36,43% | 73,46% |
| SGD Classifier | 62,43% | 58,64% | 67,75% | 36,13% | 72,03% |

> **Destaque:** Ensemble Stacking atualizado (RF otimizado + XGB + LGBM) alcança **80,49% accuracy** e **F1-Moderado de 54,98%** — superior ao anterior (76,23% / 40,06%).  
> Tempo de treino: ~32 minutos (1.945 segundos).

---

## 4. Features temporais avançadas ✅ CONCLUÍDO

- [x] Médias móveis 7 dias (`Precipitacao_ma7`, `DiaSemChuva_ma7`) por (lat, lon) em `carregar_dados.py`
- [x] Retreinar na base completa; limitação de janelas temporais documentada em `DATASET_VERSION.md`

---

## 5. Integração pós-umidade — BLOQUEADO (aguarda §1)

> **Pré-requisito:** §1 concluído (CSV com umidade aceitável).

- [ ] Incluir `Umidade` em `NUM_FEATURES` em `carregar_dados.py` e validar imputação/outliers  
  *(Nota: `carregar_dados.py` já acrescenta `Umidade` aos numéricos quando há valores não nulos — falta dados suficientes no CSV.)*
- [ ] Retreino completo com: umidade + histórico + features temporais + Ensemble Stacking
- [ ] Atualizar `preprocessor_metadata.json` e modelos em `modelos/`
- [ ] Registrar acurácia/F1/confusion matrix final em `modelos/relatorios/`

**Roteiro quando o enriquecimento NASA estabilizar (critério de cobertura a definir):**

1. Rodar `python scripts/_check_umidade.py` (ou inspeção pandas) sobre `base_de_dados_com_umidade.csv` para % não nulo e estatísticas.
2. Definir estratégia para lacunas residuais (ex.: imputação por mediana regional/mês ou exclusão temporária da feature se cobertura < meta).
3. Treinar com `treinamento_modelo.py --models ensemble_stacking random_forest_balanced --dataset base_de_dados_com_umidade.csv --no-cv` (ajustar lista conforme objetivo).
4. Atualizar `DATASET_VERSION.md`, `scripts/_gerar_resumo.py` → `modelos/relatorios/resumo_melhorias.json`.

**Arquivo(s):** `scripts/carregar_dados.py`, `scripts/pre_processor.py`, `scripts/treinamento_modelo.py`

---

## 6. Clima adicional NASA — OPCIONAL (aguarda §1)

- [ ] Implementar `enriquecer_dados_climaticos.py` (vento, T min/max) reutilizando padrão do script de umidade
- [ ] Acrescentar colunas ao carregamento e retreinar

---

## 7. Otimização de Hiperparâmetros com Optuna ✅ CONCLUÍDO (RF)

> **15 trials** concluídos em 16/04/2026. Resultado em `modelos/relatorios/optuna_random_forest_balanced.json`.  
> **Melhor CV accuracy: 77.99%** — Trial 14. RF retreinado em 16/04/2026 com params ótimos: **78.46% test accuracy**.

- [x] **RF concluído**: n_estimators=346, max_depth=25, min_samples_split=3, min_samples_leaf=1, max_features='sqrt'
  - Trial 14: **77.99%** (MELHOR) | Trial 13: 76.42% | Trial 12: 75.29% | Trial 1: 73.39%
  - `treinamento_modelo.py` atualizado com os hiperparâmetros ótimos
- [x] Resultado salvo em `modelos/relatorios/optuna_random_forest_balanced.json`
- [x] **RF otimizado retreinado**: accuracy=**78.46%** | F1-macro=74.37% | F1-Moderado=**55.67%** | F1-Muito Alto=84.62%
- [x] Rodar Optuna para `logistic_regression_balanced` *(15 trials concluídos em 21/04/2026 — best CV accuracy 64,56 %, params `C=12.43`, `max_iter=4768` aplicados em `treinamento_modelo.py`. Ganho marginal vs default (~0,02 pp); registrado em `modelos/relatorios/optuna_logistic_regression_balanced.json`.)*
- [x] Retreinar Ensemble Stacking com RF otimizado como base learner — **CONCLUÍDO: 80,49% accuracy, F1-Moderado: 54,98%**

**Arquivo(s):** `scripts/otimizar_hiperparametros.py`, `modelos/relatorios/optuna_*.json`

---

## 8. Melhorias identificadas por pesquisa (literatura 2024–2025)

> Fontes: MDPI Forests 2024, arXiv 2511.11680, IndJST 2025, ECMWF Nature Comms 2025.

### 8.1 XGBoost / LightGBM — ✅ CONCLUÍDO

> Treinados em 16/04/2026 com 250K amostras (subsampling para memória).  
> Resultados em `modelos/relatorios/xgboost_classifier_metrics.json` e `lightgbm_classifier_metrics.json`.

- [x] Adicionar `XGBoostClassifier` e `LightGBMClassifier` ao `treinamento_modelo.py` (com `_LabelEncodingWrapper`)
- [x] XGBoost treinado: **accuracy=74.85%** | F1-macro=61.59% | F1-Moderado=23.46% (pobre sem class_weight)
- [x] LightGBM treinado: **accuracy=70.62%** | F1-macro=67.00% | F1-Moderado=**45.10%** (melhor Moderado standalone)
- [x] Incluir XGBoost e LightGBM como base learners nos ensembles (Voting e Stacking atualizados)
- [x] Retreinar Ensemble Stacking com RF otimizado + XGBoost + LightGBM — **CONCLUÍDO: 80,49% accuracy, F1-macro: 74,92%**

**Referência:** Quesada-Ruiz et al. (npj Natural Hazards 2025) — modelos híbridos com Random Forest e SPI prevêem anomalia de área queimada um mês adiante em ~68 % das áreas queimáveis. *(A citação anterior "GWO-XGBoost / Zhang et al. 2024 para Sichuan" não foi confirmada e foi substituída — ver `REFERENCIAS_TCC.md` §9.)*

### 8.2 Análise de importância de features com SHAP — ✅ CONCLUÍDO

> Script `scripts/analise_shap.py` criado e executado em 16/04/2026.  
> Resultado em `modelos/relatorios/shap_feature_importance.json` e `shap_feature_importance.png`.

- [x] Instalar `shap` — instalado (0.51.0)
- [x] Criar `scripts/analise_shap.py` (Feature Importance Gini/MDI para RF, com fallback SHAP)
- [x] Resultado: **Top features por importância** (RF balanced, 546 features pós-OHE):
  1. `Indice_Seca` (10.04%) — índice derivado de DiaSemChuva/Precipitacao
  2. `DiaSemChuva` (9.33%) — dias sem chuva
  3. `DiaSemChuva_ma7` (8.22%) — média móvel 7 dias
  4. `Ano` (6.33%) — tendência temporal
  5. `Latitude` (5.89%) — localização geográfica
  6. `Precipitacao` (5.66%), `Precipitacao_ma7` (5.53%), `Longitude` (5.37%)
  7. `FRP` (3.53%), `Dias_Desde_Ultimo_Incendio` (3.20%)
- [x] Gráfico salvo em `modelos/relatorios/shap_feature_importance.png`
- [x] Documentar no TCC: condições de seca e localização geográfica são preditores primários

**Referência:** Cheerala et al. (arXiv 2511.11680) — RF+SHAP para susceptibilidade a incêndios, AUC 0.997

### 8.3 Recursive Feature Elimination (RFE) — ✅ CONCLUÍDO

> Script `scripts/analise_features.py` (RFECV) executado em 16/04/2026.  
> Resultado em `modelos/relatorios/rfe_feature_ranking.json`.

- [x] Implementar RFECV com cross-validation em `scripts/analise_features.py` (step adaptativo)
- [x] **Resultado**: Número ótimo de features = **16** | Best CV accuracy = 67.22%
  - Features selecionadas: DiaSemChuva, Precipitacao, Latitude, Longitude, FRP, Ano, Mes, Dia,
    Mes_sin, Mes_cos, Indice_Seca, Precipitacao_ma7, DiaSemChuva_ma7, Incendios_Ultimos_30_Dias,
    Dias_Desde_Ultimo_Incendio + Estado_AMAZONAS
  - **Insight**: Com apenas 16 features (vs 546 após OHE), accuracy = 67.22%; OHE municípios
    adiciona +9% accuracy mas aumenta 34× a dimensionalidade
- [x] Resultado salvo em `modelos/relatorios/rfe_feature_ranking.json`

**Referência:** Guyon et al. (2002) — formulação clássica do RFE (eliminação recursiva de variáveis), aplicada aqui com `RFECV` para selecionar o subconjunto de 16 features ótimo. Comparativo metodológico com Cheerala et al. (arXiv 2511.11680, 2025), que ranqueia features de risco de fogo via RF+SHAP em vez de RFE. *(A citação anterior "Sumathi & Rajesh IndJST 2025" não foi confirmada e foi substituída — ver `REFERENCIAS_TCC.md` §9.)*

### 8.4 Features derivadas adicionais — ✅ CONCLUÍDO (principais)

- [x] **Estação do Ano** (`Estacao`) — confirmado em `cat_features_final` no pipeline ✅
- [x] **Índice de Seca** (`Indice_Seca = DiaSemChuva / (Precipitacao + 0.1)`) — em `num_features_final` ✅
- [x] **Sazonalidade cíclica** (`Mes_sin`, `Mes_cos`) — confirmado no pipeline ✅
- [x] **Período Crítico** (`Periodo_Critico` jul-set) e **Período do Dia** (`Periodo_Dia`) — confirmados ✅
- [ ] **FRP histórico por município/estado** — opcional, requer re-run de `adicionar_historico_incendios.py`
- [ ] **NDVI** via Copernicus — complexidade elevada, postergado

**Referência:** ECMWF (Nature Comms 2025) — incorporar vegetação e umidade do combustível melhora acurácia em até 30%

### 8.5 Ajuste de threshold por classe — ✅ CONCLUÍDO

> Executado em 16/04/2026. Resultado em `modelos/relatorios/threshold_analysis.json`.  
> Gráfico em `modelos/relatorios/threshold_precision_recall.png`.

- [x] Script `scripts/ajustar_threshold.py` criado com análise de thresholds 0.25–0.50
- [x] Calibração já implementada (CalibratedClassifierCV com método isotônico no treino)
- [x] **Resultado**: threshold_moderado=0.25 melhora F1-Moderado de **39.75% → 47.73%** (custo: -3.33% accuracy)
  - Baseline (argmax): accuracy=76.19% | F1-macro=0.6765 | F1-Moderado=0.3975
  - Melhor: accuracy=72.86% | F1-macro=0.6887 | F1-Moderado=0.4773 | prec=0.3946 | rec=0.6040
- [x] Documentar decisão de threshold no TCC — ver `modelos/relatorios/threshold_analysis.json`

### 8.6 Threshold tuning sobre Stacking Tier 1 — ✅ CONCLUÍDO *(11/05/2026)*

> Re-executado sobre o **`ensemble_stacking` Tier 1** (50 k amostras estratificadas do `base_de_dados_enriquecido.csv`). Resultados persistidos em `modelos/prediction_thresholds.json`.

- [x] `scripts/ajustar_threshold.py` estendido: reporta tanto o melhor por **F1-macro** quanto por **F1-Moderado** e persiste ambos num JSON dedicado.
- [x] Workaround `_LabelEncodingWrapper` adicionado para permitir `joblib.load` em script `__main__`.
- [x] **Baseline (argmax)**: accuracy=83,48% | F1-macro=0,7878 | F1-Moderado=0,6153
- [x] **Melhor F1-macro**: thr_moderado=0,40 | thr_muito_alto=0,45 → acc=83,35% | F1-macro=**0,7894** (+0,16 pp) | F1-Moderado=0,6245
- [x] **Melhor F1-Moderado**: thr_moderado=0,35 | thr_muito_alto=0,45 → acc=82,98% | F1-macro=0,7893 | F1-Moderado=**0,6270 (+1,17 pp)**
- [x] App carrega `prediction_thresholds.json` em runtime; `/api/predict` aceita `estrategia_thresholds ∈ {f1_macro, f1_moderado, argmax}` (default `f1_macro`) e retorna o objeto `thresholds_aplicados` para auditoria.
- [x] Card *Previsão* no front-end ganhou uma linha extra explicando se a decisão veio do argmax ou de thresholds calibrados.

**Arquivo(s):** `scripts/ajustar_threshold.py`, `modelos/prediction_thresholds.json`, `scripts/app_map_interativo.py`, `scripts/AVANCOS_TREINAMENTO_RECENTES.md §2.3`

### 8.7 Optuna XGB + LightGBM + meta-learner GBM no Stacking — ✅ CONCLUÍDO *(12/05/2026)*

> Combinação de duas melhorias independentes que produziram o **modelo final do TCC**: `ensemble_stacking_gbm`.

- [x] `scripts/otimizar_hiperparametros.py` estendido para suportar `xgboost_classifier` e `lightgbm_classifier` (com `--scoring` configurável e `--subsample`).
- [x] **Optuna XGBoost** (15 trials, 3-fold CV, F1-macro, subsample 150k): F1-macro 0,616 → **0,7005** (+8,4 pp). Best params em `modelos/relatorios/optuna_xgboost_classifier.json`.
- [x] **Optuna LightGBM** (15 trials, 3-fold CV, F1-macro, subsample 150k): F1-macro 0,670 → **0,7256** (+5,5 pp). Best params em `modelos/relatorios/optuna_lightgbm_classifier.json`.
- [x] `treinamento_modelo.py`: parâmetros otimizados aplicados em `MODEL_LIBRARY_IMPROVED` e em `_estimador_xgb_opcional` / `_estimador_lgbm_opcional` (usados nos ensembles).
- [x] Novo `_build_meta_learner(kind)` aceita `"lr" | "gbm" | "lgbm"`; modelos `ensemble_stacking_gbm` e `ensemble_stacking_lgbm` registrados.
- [x] **Retreino Stacking LR (XGB/LGBM tunados)**: accuracy=**84,14 %** | F1-macro=**0,7944** | F1-Moderado=**0,6202** (vs 83,60 % / 0,7892 / 0,6168 antes; +0,55 pp / +0,52 pp / +0,34 pp) | tempo=42 min.
- [x] **Retreino Stacking GBM (meta-learner GBM, XGB/LGBM tunados)** ⭐: accuracy=**84,61 %** | F1-macro=**0,7995** | F1-Moderado=**0,6296** (vs anterior: +1,01 pp / +1,03 pp / +1,28 pp) | tempo=85 min.
- [x] Re-threshold tuning sobre o `ensemble_stacking_gbm`: best F1-macro = best F1-Moderado = (thr_mod=0,35; thr_alto=0,45) → acc=83,88 % | F1m=**0,7981** | F1-Moderado=**0,6375** (+1,44 pp sobre argmax do mesmo modelo). `prediction_thresholds.json` regenerado para o novo modelo.
- [x] `app_map_interativo.py`: `carregar_modelo()` agora prefere `ensemble_stacking_gbm` quando nenhum modelo é especificado; front-end também coloca `ensemble_stacking_gbm` no topo do dropdown.

**Arquivo(s):** `scripts/otimizar_hiperparametros.py`, `scripts/treinamento_modelo.py`, `modelos/ensemble_stacking_gbm.pkl`, `modelos/relatorios/ensemble_stacking_gbm_metrics.json`, `modelos/relatorios/optuna_xgboost_classifier.json`, `modelos/relatorios/optuna_lightgbm_classifier.json`, `modelos/prediction_thresholds.json`, `scripts/app_map_interativo.py`, `scripts/AVANCOS_TREINAMENTO_RECENTES.md §2.3a/§2.3b`, `DOCUMENTO_METODOLOGIA_E_RESULTADOS.md §4.10`.

---

## 9. Mapa e aplicação interativa ✅ CONCLUÍDO

- [x] Alinhar `preparar_dados_previsao` com novas features (médias móveis, histórico neutro em tempo real)
- [x] Melhorar UX: loading, erros de API, legenda de risco, exibir probabilidade se modelo calibrado
- [x] Evoluir `gerar_mapa.py` (ensembles preferidos, enriquecimento de features, legenda, probabilidades no popup)
- [x] Validar artefatos do modelo (carga de `ensemble_stacking.pkl`)

**Ver:** `README_APP.md`

---

## 11. Tier 1 — Features físico-climáticas (alternativa à umidade NASA) ✅ CONCLUÍDO

> **Motivação:** o enriquecimento NASA POWER (umidade) levaria ~16 dias contínuos. A literatura recente (Seager et al. 2015; npj Natural Hazards 2025; Forests 2024 para Cerrado-Amazônia) mostra que SPI, KBDI e VPD prevêem risco de fogo tão bem ou melhor que umidade isolada — e podem ser computados localmente a partir das colunas já presentes no dataset.

> **Implementação:** `scripts/features_avancadas.py` enriquece o dataset em ~40s. Integração transparente via `carregar_dados.py` (idempotente; pula recomputo se já presente).

- [x] `scripts/features_avancadas.py` criado com 7 camadas de features:
  1. **Médias móveis estendidas** (14, 30, 90 dias) por célula 0,25°
  2. **Precipitação acumulada** (30, 90, 180, 365 dias)
  3. **SPI** (Standardized Precipitation Index) 1, 3, 6 meses por célula
  4. **Anomalia de precipitação** vs. climatologia (município × mês)
  5. **KBDI proxy / De Martonne / VPD proxy** com temperatura climatológica INMET 1981-2010 (9 estados × 12 meses)
  6. **Histórico estendido de incêndios** (90, 180, 365 dias) por célula + média FRP célula 30d
  7. **Estação seca acumulada** (dias com P<5mm em janela 90d)
- [x] Total de **24 novas features numéricas** adicionadas
- [x] CSV enriquecido salvo em `base_de_dados_enriquecido.csv` (933.954 linhas, 46 colunas)
- [x] Integração em `carregar_dados.py` (detecção automática + recomputo idempotente)
- [x] `scripts/treinamento_modelo.py`: adicionado `catboost_classifier` standalone + base learner dos ensembles (Tier 4)
- [x] **Retreino RF balanced**: accuracy **78,46% → 82,93%** (+4,47 pp), F1-macro **74,37% → 78,66%** (+4,29 pp)
- [x] CatBoost standalone: 71,73% acc / 68,14% F1-macro (subsample 250k; com OHE perde vantagem nativa, mas serve como base learner)
- [x] **Retreino Ensemble Stacking (RF+XGB+LGBM+CatBoost) com Tier 1**: accuracy **80,49% → 83,60%** (+3,11 pp), F1-macro **74,92% → 78,92%** (+4,00 pp), **F1-Moderado 54,98% → 61,68% (+6,70 pp)** — meta superada
- [x] Comparativo final atualizado em `modelos/relatorios/resumo_melhorias.json` (gerado por `scripts/_gerar_resumo.py`)

**Embasamento na literatura:**
- Seager et al. (AMS 2015): VPD e prior-year cold-season precipitation como preditores fortes
- npj Nat. Hazards 2025: SPI prevê área queimada em ~68% das áreas até 1 mês antes
- Forests 2024 (Cerrado-Amazônia): KBDI entre os melhores índices em Canaã dos Carajás
- Sci. Reports 2025 (Gangwon, Germany): NDVI + sazonalidade dominam em modelos year-round

**Arquivo(s):** `scripts/features_avancadas.py`, `scripts/carregar_dados.py`, `scripts/treinamento_modelo.py`, `base_de_dados_enriquecido.csv`, `DATASET_VERSION.md`, `DOCUMENTO_METODOLOGIA_E_RESULTADOS.md`

---

## 12. App interativo explicável (UI/UX) ✅ CONCLUÍDO *(11/05/2026)*

> **Motivação:** com Tier 1 no pipeline, o app precisava gerar as 24 features na predição pontual (caso contrário modelo enriquecido receberia NaNs). Aproveitamos para tornar a previsão **defensável** ao usuário leigo, mostrando o porquê de cada previsão.

- [x] `scripts/feature_lookup.py` — pré-cálculo das medianas das 24 features Tier 1 em 3 níveis: célula 0,25° → (Estado, Mes) → Mes global; lookup em cascata; features físicas (Temp_Climatologica, KBDI_proxy, VPD_proxy, Aridez, Anomalia) recalculadas em tempo real com clima NASA POWER atual
- [x] `scripts/explainer.py` — explicabilidade local *rápida* (< 5 ms/consulta): contrib = importância_global × z-score × sinal_físico. Importâncias padrão Tier 1 calibradas pela literatura (KBDI=0.075, SPI=0.045–0.060, VPD=0.060 etc.)
- [x] `scripts/app_map_interativo.py` refatorado:
  - `preparar_dados_previsao` adiciona as 24 features Tier 1 via lookup
  - `POST /api/predict` retorna agora `dados_usados` (39 campos), `contexto_historico` (climatologia local + granularidade do lookup), `explicacao` (top-6 features ranqueadas)
  - `POST /api/explain` — endpoint dedicado pra refazer a explicação isoladamente
  - `GET /api/models` retorna acurácia/F1 de cada modelo
- [x] **Front-end redesenhado**: painel lateral 420 px com 6 cards (Previsão, *Por que esse risco?*, Dados climáticos atuais, Índices de seca, Acumulados + anomalia, Histórico de focos), cores semânticas, responsividade `< 900 px`, comunicação iframe ↔ documento pai via `postMessage`
- [x] Smoke-test validado: ponto AM-3.5,-62.5 → Baixo 82,3% (célula úmida); ponto PA-8.5,-50.5 (Sta Maria das Barreiras) → **Muito Alto 92,9%** com SPI-1m=-0,57 / KBDI=418 / T=27,5°C **destacadas na explicação local**

**Arquivo(s):** `scripts/app_map_interativo.py`, `scripts/feature_lookup.py`, `scripts/explainer.py`, `scripts/AVANCOS_TREINAMENTO_RECENTES.md §2.2`, `DOCUMENTO_METODOLOGIA_E_RESULTADOS.md §3.12`

**Limitações documentadas:**
- Lookup retorna *medianas* da célula histórica → captura comportamento típico, não anomalia exata da data (limitação aceitável para UI; trabalho futuro = série temporal ponto-a-ponto em tempo real via NASA POWER 28d).
- Explicação é aproximação SHAP-like, não SHAP local exato. Trabalho futuro = pré-computar SHAP por classe sobre o Stacking Tier 1.

---

## 13. Rigor metodológico adicional — SHAP per-class e validação temporal *(12/05/2026)*

> **Motivação:** dois itens listados como "trabalho futuro" no `DOCUMENTO_METODOLOGIA_E_RESULTADOS.md` §7 puderam ser executados ainda dentro do TCC, fechando a discussão de (i) explicabilidade local fiel e (ii) viés do split aleatório.

### 13.1 SHAP global por classe — *resolvido via RF leve proxy (12/05/2026)*

- [x] `scripts/analise_shap_local.py` mantido como tentativa original sobre o RF Optuna produtivo.
- [x] `scripts/explainer.py` integrado para carregar `modelos/relatorios/shap_per_class.json` quando ele existir (substitui a aproximação `importância_global × z-score × sinal_físico` por importâncias específicas da classe predita).
- [x] **`scripts/analise_shap_rf_leve.py` NOVO** — solução adotada: treina um Random Forest compacto (`n_estimators=100`, `max_depth=12`, 60 000 amostras estratificadas — acc in-sample 71,75 %) como **modelo-proxy interpretativo** e roda `shap.TreeExplainer` em 500 amostras. Saída em `modelos/relatorios/shap_per_class.json`. Defensável por Lundberg et al. 2020 (Nat. Mach. Intell.); converge qualitativamente com a Permutation Importance (§13.3).
- [x] **Resultados — Top-3 |SHAP| por classe:** Baixo = (Ano, KBDI_proxy, VPD_proxy); Moderado = (KBDI_proxy, VPD_proxy, Indice_Seca); Muito Alto = (KBDI_proxy, Ano, FRP). `KBDI_proxy` no Top-3 das três classes confirma Forests 2024 (Canaã dos Carajás); `VPD_proxy` confirma Seager et al. 2015.
- [x] Documentação: `DOCUMENTO_METODOLOGIA_E_RESULTADOS.md` §4.4.1 com Top-8 por classe em tabela; integração com app via `fonte_importance` no JSON de resposta.

**Referências:** Lundberg & Lee (NeurIPS 2017) — SHAP; Lundberg et al. (Nature Machine Intelligence 2020) — TreeExplainer.

### 13.2 Validação temporal rolling-origin *(estendida ao Stacking GBM em 12/05/2026)*

- [x] `scripts/validacao_temporal.py` implementa **rolling-origin por ano** (treino = `Ano < Y`, teste = `Ano == Y`, 3 últimos anos com massa suficiente; subsample estratificado por fold por limitação de memória do OHE denso de Município). Suporta `--modelo random_forest_balanced` ou `ensemble_stacking_gbm`.
- [x] **RF balanced Tier 1** (3 folds, 120 k subsample): acc=**70,09 % ± 4,10**, F1-macro=**0,592 ± 0,037**, F1-Moderado=**0,264 ± 0,023**. Saída: `modelos/relatorios/validacao_temporal.json`.
- [x] **Ensemble Stacking GBM** (3 folds, 60 k subsample): acc=**70,13 % ± 4,06**, F1-macro=**0,606 ± 0,045**, F1-Moderado=**0,306 ± 0,031**. Saída: `modelos/relatorios/validacao_temporal_stacking_gbm.json`.
- [x] **Δ viés (temporal − aleatório) Stacking GBM:** acurácia **−14,49 pp** (84,61 → 70,13 %), F1-macro **−19,37 pp**, **F1-Moderado −32,35 pp** (0,630 → 0,306).
- [x] **Stacking mantém vantagem sobre RF mesmo sob folds temporais** (+1,4 pp F1-macro, +4,2 pp F1-Mod) — o ganho do ensemble não é overfitting do split aleatório.
- [x] **Degradação progressiva entre folds** (75,72 → 68,28 → 66,37 %) — drift climático real (Aragão 2018; Silva-Junior 2025).
- [x] Documentação: `DOCUMENTO_METODOLOGIA_E_RESULTADOS.md` §3.5.1, §4.11 e **§4.11.1 (NOVO)**.

**Referência:** Bergmeir & Benítez (*Information Sciences* 2012); Quesada-Ruiz et al. (npj Natural Hazards 2025).

### 13.3 Permutation Importance estratificada por classe *(NOVO — 12/05/2026)*

- [x] `scripts/permutation_importance_por_classe.py` aplica Breiman (2001) / Fisher, Rudin & Dominici (2019) **diretamente sobre o `ensemble_stacking_gbm`** (5 000 amostras de teste, 3 repetições, seed=42). Saída: `modelos/relatorios/permutation_importance_por_classe.json`.
- [x] **Top-10 Δ F1-macro global:** Ano (0,0351), Latitude (0,0318), Longitude (0,0258), Estado (0,0179), DiaSemChuva_ma90 (0,0104), FRP (0,0100), Municipio (0,0099), DiaSemChuva_ma30 (0,0096), Precipitacao_ma90 (0,0093), Precipitacao_acum_180d (0,0078).
- [x] **Por classe (Top-3):** Baixo → Ano, Latitude, Longitude; **Moderado** → Ano, Latitude, **Estado** (+0,0319) e *DiaSemChuva_ma90* (+0,0222) — confirma que o regime hídrico estacional é determinante para essa fronteira; Muito Alto → Latitude, Longitude, Ano, **VPD_proxy** (+0,0070) — confirmação independente do SHAP.
- [x] **Achado robusto inter-técnicas:** `VPD_proxy` aparece no Top-5 de Muito Alto pelas **duas técnicas independentes** (SHAP RF leve § 13.1 + Permutation Stacking GBM § 13.3) — validação cruzada citável da substituição da `Umidade` por proxies físicos.
- [x] Documentação: `DOCUMENTO_METODOLOGIA_E_RESULTADOS.md` §4.4.2.

**Referências:** Breiman (2001); Fisher, Rudin & Dominici (JMLR 2019); Strobl et al. (BMC Bioinformatics 2007); Molnar (2022, §8.5).

### 13.4 Pseudo-labeling de Umidade *(NOVO — 12/05/2026)*

- [x] `scripts/pseudo_label_umidade.py` treina LightGBM regressor sobre os ~170 k registros já enriquecidos pela NASA POWER (15 features: Precipitacao, DiaSemChuva, lat/lon, Mes_sin/cos, histórico de fogo), com early stopping em holdout 20 %.
- [x] **Métricas holdout 20 %:** RMSE = **4,33 % RH**, MAE = **3,19 % RH**, **R² = 0,933** (n_val = 34 093). Justifica a abordagem.
- [x] Refit no 100 % dos labels reais, imputação dos ~763 k registros sem label; clip em [5, 100] %.
- [x] Dataset enriquecido salvo em `base_de_dados_umidade_pseudo.csv` com coluna auxiliar **`Umidade_origem`** ∈ {`nasa_power`, `pseudo_label_lgbm`} para auditoria. Cobertura final 100 %.
- [x] Relatório em `modelos/relatorios/pseudo_label_umidade.json`; regressor salvo em `modelos/lgbm_regressor_umidade.pkl`.
- [ ] **Próximo passo (trabalho futuro):** retreinar o `ensemble_stacking_gbm` com `Umidade` (pseudo-rotulada) e comparar com a versão sem; quantificar viés de confirmação (Arazo et al. 2020).
- [x] Documentação: `DOCUMENTO_METODOLOGIA_E_RESULTADOS.md` §4.7.1 (NOVO).

**Referências:** Lee (ICML Workshop 2013) — *Pseudo-Label*; Arazo et al. (IJCNN 2020) — *Pseudo-Labeling and Confirmation Bias*; Lopez-Garcia et al. (Remote Sensing 2024) — imputação de SMAP em grids tropicais.

**Arquivo(s) afetados:** `scripts/analise_shap_rf_leve.py`, `scripts/permutation_importance_por_classe.py`, `scripts/pseudo_label_umidade.py`, `scripts/validacao_temporal.py`, `modelos/relatorios/shap_per_class.json`, `modelos/relatorios/permutation_importance_por_classe.json`, `modelos/relatorios/validacao_temporal.json`, `modelos/relatorios/validacao_temporal_stacking_gbm.json`, `modelos/relatorios/pseudo_label_umidade.json`, `modelos/lgbm_regressor_umidade.pkl`, `base_de_dados_umidade_pseudo.csv`, `DOCUMENTO_METODOLOGIA_E_RESULTADOS.md` §3.5.1, §3.12.2, §4.4.1, §4.4.2, §4.7.1, §4.11, §4.11.1.

---

## 10. Documentação e fechamento TCC — PARCIALMENTE CONCLUÍDO

- [x] Atualizar `README_APP.md` com novas capacidades
- [x] Referenciar limites NASA / uso de chaves
- [x] Marcar `scripts/CHECKLIST_MELHORIAS.md` como legado
- [x] Resultado do Optuna registrado (RF: 78,46% test accuracy, Optuna best CV: 77,99%) 
- [x] Análise SHAP/Feature Importance concluída — figura em `modelos/relatorios/shap_feature_importance.png`
- [x] Resumo comparativo de todos os modelos em `modelos/relatorios/resumo_melhorias.json` (gerado 16/04/2026)
- [x] **Relatório metodológico** (planejamento, materiais e métodos, resultados, discussão, limitações, trabalhos futuros): `DOCUMENTO_METODOLOGIA_E_RESULTADOS.md`
- [x] **App interativo explicável** (Tier 1 + SHAP-like local + 6 cards informativos) — documentado em `scripts/AVANCOS_TREINAMENTO_RECENTES.md §2.2` e em `DOCUMENTO_METODOLOGIA_E_RESULTADOS.md §3.12` *(11/05/2026)*
- [ ] Aguardar retreino com umidade (§5) → registrar métricas finais com feature Umidade
- [ ] **Copiar métricas finais para o texto do TCC** (tabela abaixo, matrizes de confusão, F1 por classe, limitações) — conteúdo base já consolidado no documento acima

**Tabela pronta para o TCC (modelos avaliados em 184.862 amostras de teste; dataset enriquecido com Tier 1 a partir de 10/05/2026, dataset com histórico de 16/04/2026):**

| Modelo | Dataset | Acurácia | F1-Macro | F1 Baixo | F1 Moderado | F1 Muito Alto |
|---|---|---|---|---|---|---|
| **Ensemble Stacking GBM (XGB+LGBM Optuna + Tier 1, meta=GBM)** ⭐ | enriquecido | **84,61%** | **79,95%** | **87,68%** | **62,96%** | **89,21%** |
| + thresholds calibrados F1-Moderado (thr_mod=0,35; thr_alto=0,45) | enriquecido | 83,88% | 79,81% | — | **63,75%** | — |
| Ensemble Stacking LR (XGB+LGBM Optuna + Tier 1) | enriquecido | 84,14% | 79,44% | 87,39% | 62,02% | 88,92% |
| Ensemble Stacking (RF+XGB+LGBM+CatBoost + Tier 1, defaults) | enriquecido | 83,60% | 78,92% | 86,74% | 61,68% | 88,34% |
| Random Forest balanceado (Optuna + Tier 1) | enriquecido | 82,93% | 78,66% | 86,03% | 62,05% | 87,89% |
| XGBoost (Optuna + Tier 1, 250k subsample) | enriquecido | **80,21%** | 72,37% | — | 46,70% | — |
| LightGBM (Optuna + Tier 1, 250k subsample) | enriquecido | 78,85% | 74,72% | — | 55,33% | — |
| Ensemble Stacking (RF opt+XGB+LGBM, sem Tier 1) | com_historico | 80,49% | 74,92% | 83,83% | 54,98% | 85,96% |
| Random Forest Otimizado (Optuna, sem Tier 1) | com_historico | 78,46% | 74,37% | 82,83% | 55,67% | 84,62% |
| XGBoost (defaults antigos) | com_historico | 74,85% | 61,59% | 79,31% | 23,46% | 81,99% |
| Ensemble Voting Soft | com_historico | 73,83% | 66,69% | 78,36% | 40,38% | 81,33% |
| CatBoost classifier (250k subsample) | enriquecido | 71,73% | 68,14% | — | 46,51% | — |
| LightGBM (defaults antigos) | com_historico | 70,62% | 67,00% | 77,53% | 45,10% | 78,38% |
| Random Forest SMOTE | com_historico | 69,78% | 66,08% | 75,96% | 45,35% | 76,93% |
| Regressão Logística | com_historico | 64,58% | 60,44% | 71,42% | 36,43% | 73,46% |
| SGD Classifier | com_historico | 62,43% | 58,64% | 67,75% | 36,13% | 72,03% |

> **Δ Tier 1 — Stacking (mesmo split, mesmas seeds, +24 features):** +3,11 pp accuracy, +4,00 pp F1-macro, **+6,70 pp F1-Moderado** (a classe historicamente mais difícil — saltou de 54,98% → 61,68%, sem custo de accuracy). Tempo de treino Stacking Tier 1: 37 min. Detalhes em §11 e em `DOCUMENTO_METODOLOGIA_E_RESULTADOS.md` §3.11/§4.8.

> **Δ acumulado (Stacking GBM com Optuna + thresholds, vs Stacking original sem Tier 1):** +4,12 pp acurácia (80,49% → 84,61%), +5,03 pp F1-macro (74,92% → 79,95%) e **+8,77 pp F1-Moderado** (54,98% → 63,75% com thresholds), confirmando o impacto cumulativo das melhorias documentadas em §11 (Tier 1), §8.1-8.5 (Optuna RF/XGB/LGBM, threshold) e §8.7 (meta-learner GBM, novo).

**Features mais importantes (Gini/MDI, Random Forest Balanced):**
1. Índice de Seca (10,04%) — razão DiaSemChuva / Precipitação
2. Dias Sem Chuva (9,33%)
3. Dias Sem Chuva — Média 7d (8,22%)
4. Ano (6,33%) — captura tendência de aquecimento
5. Latitude (5,89%), Precipitação (5,66%), Precipitação MA7 (5,53%), Longitude (5,37%)

**Decisão de threshold (§8.5):** usar threshold_moderado=0.25 melhora F1-Moderado de 39,75% → 47,73% ao custo de -3,33% accuracy (trade-off documentado em `modelos/relatorios/threshold_analysis.json`).

---

## Resumo de status — visão rápida

| Fase | Descrição | Status |
|---|---|---|
| §0 | Estado de referência | ✅ Concluído |
| §1 | Enriquecimento umidade NASA | 🔄 Rodando, **NÃO bloqueante** (~16 dias restantes, ~32 req/min) |
| §2 | Features histórico de incêndios | ✅ Concluído |
| §3 | Modelos ensemble + avaliação | ✅ Concluído — Stacking GBM + Tier 1: **84,61%** acc (hold-out); baseline sem Tier 1 **80,49%** |
| §4 | Features temporais (médias móveis 7d) | ✅ Concluído |
| §5 | Retreino com umidade | 🔒 Bloqueado (depende de §1) — **substituído por §11 enquanto §1 não fecha** |
| §6 | Clima extra NASA (vento, temperatura) | ⏳ Opcional / não iniciado |
| §7 | Optuna RF + Retreino RF otimizado | ✅ Concluído — **78.46% test** (F1-Moderado: 55.67%) |
| §8.1 | XGBoost + LightGBM | ✅ Concluído — XGB: 74.85%, LGBM: 70.62% |
| §8.1 | Ensemble Stacking (RF+XGB+LGBM, sem Tier 1) | ✅ Concluído — **80,49%** acc, F1-Moderado: 54,98% |
| §8.2 | Feature Importance (SHAP/Gini) | ✅ Concluído — Top: Indice_Seca, DiaSemChuva |
| §8.3 | RFE (RFECV) | ✅ Concluído — 16 features ótimas, 67.22% CV |
| §8.4 | Features derivadas (Estacao, Indice_Seca, etc.) | ✅ Concluído |
| §8.5 | Ajuste de threshold Moderado (Stacking sem Tier 1) | ✅ Concluído — F1-Moderado: 39.75%→47.73% |
| §8.6 | Threshold tuning Stacking Tier 1 + persistência | ✅ Concluído *(11/05/2026)* — F1-macro 0,7878→0,7894 / F1-Moderado 0,6153→0,6270 |
| §8.7 | Optuna XGB/LGBM + meta-learner GBM no Stacking | ✅ Concluído *(12/05/2026)* — **84,61%** acc / F1m 0,7995 / F1-Moderado 0,6296 (+1,28 pp); thresholds: F1-Moderado **0,6375** |
| §9 | Mapa e aplicação interativa | ✅ Concluído (v1 abril/2026) |
| §10 | Documentação e fechamento TCC | 🔄 Parcial — copiar tabelas/matrizes para o LaTeX; retreino com \texttt{Umidade} (§5 ou pseudo §13.4) opcional |
| §11 | Tier 1 — Features físico-climáticas | ✅ Concluído — RF: **82,93%** (+4,47 pp) · Stacking: **83,60%** (+3,11 pp acc / +6,70 pp F1-Moderado) |
| §12 | App interativo explicável (UI/UX) | ✅ Concluído *(11/05/2026)* — lookup Tier 1 + explainer SHAP-like + 6 cards informativos |
| §13.1 | SHAP global por classe | ✅ Concluído *(12/05/2026)* — **RF compacto proxy** (`analise_shap_rf_leve.py` → `shap_per_class.json`); SHAP exato no RF Optuna produtivo permanece **trabalho futuro** (memória >1 GiB/classe) |
| §13.2 | Validação temporal rolling-origin | ✅ Concluído *(12/05/2026)* — RF **70,09 % ± 4,10** acc; Stacking GBM **70,13 % ± 4,06**; Δ viés vs aleatório até **−14,49 pp** acc (GBM). Ver `validacao_temporal*.json` |
| §13.3 | Figura evolução + Permutation importance | ✅ Concluído — `evolucao_modelos.png/.json` (`gerar_figura_evolucao.py`); `permutation_importance_por_classe.json` (`permutation_importance_por_classe.py`) |

---

## Dependências rápidas (ordem sugerida)

```
§0 → §2 e §3 (paralelo a §1) → §4 → §7 (Optuna RF+LR) → §8 (XGBoost/LightGBM Optuna, SHAP, RFE, threshold, meta-learner GBM)
→ §11 (Tier 1, substitui §1 no curto prazo) → §12 (app explicável) → §13 (SHAP per-class + validação temporal)
→ §5 (após §1, opcional) → §6 (opcional) → §9 contínuo → §10
```

---

## Bloco máquina (JSON — parsing opcional)

```json
{
  "project": "TCC-2",
  "checklist_file": "CHECKLIST_OBJETIVO_FINAL.md",
  "last_updated": "2026-05-14",
  "phases": [
    {"id": "0", "name": "estado_referencia", "done": true},
    {"id": "1", "name": "enriquecimento_umidade", "done": false, "note": "rodando background, ~892511 req pendentes, ~314 req/min, ~47h restantes, reiniciado 16/04/2026"},
    {"id": "2", "name": "features_historico_incendios", "done": true},
    {"id": "3", "name": "modelos_ensemble_avaliacao", "done": true, "best_accuracy": 0.8049, "best_model": "ensemble_stacking_atualizado", "note": "Stacking atualizado (RF opt+XGB+LGBM): 80.49% acc, F1-macro 74.92%, F1-Moderado 54.98%"},
    {"id": "4", "name": "features_temporais", "done": true},
    {"id": "5", "name": "integracao_pos_umidade", "done": false, "depends_on": ["1"]},
    {"id": "6", "name": "clima_nasa_extra", "done": false, "optional": true, "depends_on": ["1"]},
    {"id": "7", "name": "optuna_hiperparametros", "done": true, "note": "15 trials, RF otimizado retreinado 16/04/2026: 78.46% test, F1-macro 74.37%, F1-Moderado 55.67%"},
    {"id": "8.1", "name": "xgboost_lightgbm", "done": true, "note": "XGB 74.85% / LGBM 70.62%; Ensemble Stacking atualizado: 80.49% acc, F1-macro 74.92%, treino ~32min"},
    {"id": "8.2", "name": "shap_analise", "done": true, "note": "Top: Indice_Seca 10.04%, DiaSemChuva 9.33%, DiaSemChuva_ma7 8.22%"},
    {"id": "8.3", "name": "rfe_analise", "done": true, "note": "16 features otimas, 67.22% CV accuracy"},
    {"id": "8.4", "name": "features_derivadas", "done": true, "note": "Estacao, Indice_Seca, Mes_sin/cos, Periodo_Critico, Periodo_Dia todos confirmados no pipeline"},
    {"id": "8.5", "name": "threshold_moderado", "done": true, "note": "F1-Moderado: 39.75%→47.73% com thr=0.25, custo -3.33% accuracy (Stacking sem Tier 1)"},
    {"id": "8.6", "name": "threshold_tuning_stacking_tier1", "done": true, "note": "Stacking Tier 1: best F1-macro thr=(0.40,0.45) -> F1m 0.7894 (+0.16pp); best F1-Moderado thr=(0.35,0.45) -> F1mod 0.6270 (+1.17pp). Persistido em modelos/prediction_thresholds.json e integrado ao app via /api/predict estrategia_thresholds (default f1_macro)."},
    {"id": "8.7", "name": "optuna_xgb_lgbm_gbm_meta", "done": true, "note": "Optuna XGB +8.4pp F1m CV; Optuna LGBM +5.5pp F1m CV; Stacking GBM (meta=GradientBoosting, base XGB/LGBM tunados): acc 84.61%, F1m 0.7995, F1-Moderado 0.6296. Com thresholds calibrados: F1-Moderado 0.6375. Δ acumulado vs Stacking original: +4.12pp acc / +5.03pp F1m / +8.77pp F1-Moderado."},
    {"id": "9", "name": "mapa_app", "done": true},
    {"id": "10", "name": "documentacao_tcc", "done": false, "note": "copiar métricas/matrizes para LaTeX; SHAP per classe via RF proxy já feito (§13.1); retreino com Umidade opcional"},
    {"id": "11", "name": "tier1_fisico_climaticas", "done": true, "note": "RF 82.93% / Stacking 83.60% / F1-Moderado 61.68%"},
    {"id": "12", "name": "app_interativo_explicavel", "done": true, "note": "Tier 1 lookup + SHAP-like local + thresholds calibrados integrados"},
    {"id": "13.1", "name": "shap_global_per_class", "done": true, "note": "SHAP por classe via RF compacto proxy (analise_shap_rf_leve.py → shap_per_class.json); explainer.py integrado. TreeExplainer no RF Optuna produtivo: memória insuficiente — trabalho futuro GPU ou subsampling."},
    {"id": "13.2", "name": "validacao_temporal_rolling_origin", "done": true, "note": "RF balanced Tier 1, 3 folds 2021–2023 com subsample 120k. Média: acc=70.09%, F1m=0.592, F1-Mod=0.264. Δ viés vs aleatório: -12.83pp acc, -19.51pp F1m, -35.65pp F1-Mod. Documentado em DOCUMENTO §3.5.1 + §4.11."},
    {"id": "13.3", "name": "figura_evolucao_modelos", "done": true, "note": "scripts/gerar_figura_evolucao.py — barras agrupadas das 4 iterações: Stacking pré-Tier 1 / +Tier 1 / +Optuna XGB-LGBM-GBM / +thresholds. Saída: modelos/relatorios/evolucao_modelos.png (.json)."}
  ]
}
```
