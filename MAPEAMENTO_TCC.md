# Mapeamento — Integração dos avanços técnicos no TCC

> **Documento base avaliado:** `Projeto_de_graduação_II_do_curso_de_ciência_da_computação.pdf` (14/02/2024, 56 páginas, banca de defesa marcada).
> **Estado atual do projeto:** versão de produção 12/05/2026 — Stacking GBM 84,61 % acurácia, 24 features Tier 1, app Flask interativo explicável.
> **Objetivo deste documento:** mostrar, **seção a seção**, o que o TCC já possui, o que precisa ser **atualizado** e o que precisa ser **adicionado** para alinhá-lo ao estado atual do código e à literatura 2024-2025.

---

## 0. Diagnóstico — versão do TCC vs estado atual

| Aspecto | TCC II (Fev/2024) | Estado atual (Mai/2026) |
|---------|-------------------|--------------------------|
| **Modelo final** | `SGDClassifier` (mencionado no §4.4) e `DecisionTreeClassifier` (Cap. 5) | `ensemble_stacking_gbm` (Stacking de RF + LR + XGB Optuna + LGBM Optuna + CatBoost + meta-learner GBM) |
| **Features** | 9 numéricas (DiaSemChuva, Precipitacao, Lat, Lon, FRP, Ano, Mes, Dia, Hora) + 2 categóricas (Estado, Municipio) | **44 numéricas** (incluindo 24 Tier 1: SPI 1m/3m/6m, KBDI proxy, VPD proxy, MAs estendidas, lags acumulados, anomalias, histórico estendido, dias secos) + 4 categóricas |
| **Dataset** | ~933 954 linhas (2014-2024) | 924 306 linhas após limpeza + 170 465 com `Umidade` real NASA POWER + 763 489 com Umidade pseudo-rotulada (R² = 0,933) |
| **Pré-processamento** | Pipeline simples (StandardScaler, OneHotEncoder, SimpleImputer) | Idem, mais 24 features físico-climáticas, calibração isotônica (`CalibratedClassifierCV`), histórico de incêndios |
| **Otimização** | Não havia | **Optuna** (15 trials, 3-fold CV) em RF, LR, XGBoost, LightGBM |
| **Imbalance** | Não tratado | `class_weight='balanced'`, SMOTE, threshold tuning multi-classe |
| **Interpretabilidade** | Não havia | **SHAP por classe** (RF leve proxy), **Permutation Importance** (model-agnostic no Stacking GBM), Gini, **RFECV** |
| **Validação** | Holdout 80/20 estratificado | Holdout + **K-Fold CV** (3-fold) + **rolling-origin temporal** (3 folds 2021-2023) |
| **Cap. 5 — Resultados** | **Vazio**, apenas texto sobre Decision Tree | Tabela com 8 modelos, métricas por classe, matriz de confusão, validação temporal, SHAP, permutation, evolução |
| **Aplicação** | Mapa estático Folium (mapa_risco_amazonia_com_previsoes.html) | App **Flask** interativo + explicável (6 cards), predição em tempo real via clique no mapa, integração NASA POWER + NASA FIRMS, explainer local |
| **Referências** | 27 entradas, das quais várias são clássicas (Goodfellow, Russell & Norvig, IBGE, INPE) | **41 entradas BibTeX** consolidadas em `REFERENCIAS_TCC.md`, incluindo literatura 2024-2025 (Aragão 2018, Silva-Junior 2025, Forests 2024, npj Natural Hazards 2025, Cheerala 2025) |

**Lacunas estruturais do TCC original que precisam ser preenchidas:**
1. Cap. 5 (Resultados) **inteiro** está incompleto.
2. Cap. 6 (Conclusão) **antecipa** que o trabalho está em fase inicial — precisa ser completamente reescrito.
3. Cap. 4.4 (Treinamento) cita apenas SGDClassifier — precisa documentar os 8+ modelos avaliados e o ensemble final.
4. Cap. 4.5 (Avaliação) não menciona threshold tuning, SHAP, RFECV nem validação temporal.
5. Cap. 4.6 (Mapa) descreve a primeira versão (mapa estático). O **app interativo Flask** com explicações locais precisa entrar como nova seção (ou substituir).
6. Cap. 3 (Fundamentação Teórica) não cobre XGBoost, LightGBM, CatBoost, SHAP, ensembles, Optuna, índices físico-climáticos (KBDI, SPI, VPD).

---

## 1. Capítulo 1 — Introdução

### 1.0 Resumo / Abstract — versão atualizada sugerida

**Resumo atual** (linhas 80-100 do PDF): correto na motivação, mas pobre na descrição metodológica e sem números finais.

**Resumo proposto** (cola substituindo o atual):

> O crescimento da temperatura global é um fato alarmante. Segundo a *World Meteorological Organization* (WMO, 2021), o ano de 2020 atingiu uma média histórica de 1,2°C acima da era pré-industrial. **Aragão et al. (2018)** documentam que, desde o início do século 21, o fogo associado à seca passou a *neutralizar* os ganhos obtidos com a redução do desmatamento na Amazônia; **Silva-Junior et al. (2025)** mostram que 2024 marcou a pior degradação por fogo em mais de duas décadas — 3,3 Mha queimados (+400 % em relação aos 2 anos anteriores), com emissão estimada de 791 Mt CO₂. A preservação das áreas verdes é crucial para mitigar este cenário e ferramentas de previsão automatizadas, baseadas em aprendizado de máquina, representam uma abordagem promissora.
>
> Este projeto propõe um sistema de classificação supervisionada de risco de incêndio na Amazônia Legal em três níveis (Baixo, Moderado, Muito Alto), treinado sobre ~924 mil registros do Programa Queimadas (2014-2023). O pipeline integra **24 features físico-climáticas Tier 1** (SPI, KBDI proxy, VPD proxy, anomalias de precipitação, médias móveis estendidas, histórico estendido de fogo), substituindo metodologicamente a *umidade relativa* — cuja integração via NASA POWER demandaria 16 dias contínuos de enriquecimento — por proxies amparados pela literatura recente (Seager *et al.*, 2015; *Forests* 2024 — Cerrado-Amazônia; *npj Natural Hazards* 2025 — ECMWF/UKMO). Foram comparados oito modelos de aprendizado supervisionado (Regressão Logística, SGD, Random Forest com SMOTE, RF balanceado otimizado por **Optuna**, XGBoost otimizado, LightGBM otimizado, CatBoost, ensembles *Voting* e *Stacking*), com calibração isotônica e ajuste multi-classe de limiares de decisão.
>
> O melhor modelo — **Ensemble Stacking** com meta-classificador *Gradient Boosting* (`ensemble_stacking_gbm`) — atingiu **84,61 %** de acurácia, **0,7995** de F₁-macro e **0,6375** de F₁-Moderado (classe historicamente mais difícil) em hold-out estratificado de 184 862 amostras, um ganho de **+4,12 pontos percentuais (pp)** em acurácia e **+8,77 pp em F₁-Moderado** sobre o ensemble *baseline*. A interpretabilidade é abordada em dupla técnica: **SHAP por classe** via RF compacto proxy (Lundberg *et al.* 2020) e **Permutation Importance** *model-agnostic* (Breiman 2001; Fisher *et al.* 2019) sobre o Stacking GBM final — ambas confirmam `KBDI_proxy`, `VPD_proxy`, `Indice_Seca`, `DiaSemChuva_ma90` e `Precipitacao_acum_180d` como variáveis dominantes. Uma **validação temporal estrita** (*rolling-origin* por ano, 2021-2023) quantifica o viés do split aleatório frente a uma avaliação causal (queda de 14,49 pp em acurácia, 32,35 pp em F₁-Moderado). O modelo final é exposto via aplicação web **Flask + Folium** com explicação local e integração de clima em tempo real (NASA POWER) e detecção ativa de focos (NASA FIRMS).
>
> **Palavras-chave:** Aprendizado de Máquina; Previsão; Queimadas; Amazônia Legal; Ensemble Stacking; XGBoost; SHAP; Interpretabilidade.

### 1.1 Justificativa
A justificativa atual (linhas 261-289) está adequada. **Sugestão de complemento ao final**: adicionar parágrafo citando Aragão *et al.* (2018) e Silva-Junior *et al.* (2025) para ancorar o trabalho na literatura 2024-2025.

### 1.2 Objetivos

**Objetivo geral**: manter (linhas 295-298).

**Objetivos específicos** — substituir/expandir os 6 itens originais:

1. ~~Revisar bibliografia sobre ML/classificação~~ → manter.
2. ~~Mapear queimadas e causas~~ → manter.
3. ~~Obter dados climáticos, topológicos, históricos~~ → expandir: "...incluindo enriquecimento com **NASA POWER** (umidade relativa RH2M) e **NASA FIRMS** (FRP em tempo real), além de criação de **24 features físico-climáticas avançadas** (Tier 1: SPI, KBDI proxy, VPD proxy, anomalias de precipitação, histórico estendido de fogo)."
4. ~~Desenvolver algoritmo com diferentes modelos~~ → expandir: "...comparando 8 abordagens (LR, SGD, RF com SMOTE, RF balanceado otimizado por **Optuna**, XGBoost, LightGBM, CatBoost, e ensembles *Voting Soft* e *Stacking*), com **calibração isotônica** e **ajuste multi-classe de limiares de decisão**."
5. ~~Avaliar eficácia via métricas~~ → expandir: "...com métricas de desempenho por classe, **validação cruzada estratificada (K-Fold)** e **validação temporal estrita (*rolling-origin* por ano)** para quantificar viés. Análise de interpretabilidade via **SHAP** (Lundberg *et al.* 2020) e **Permutation Importance** (Fisher *et al.* 2019)."
6. ~~Desenhar mapa~~ → substituir: "Desenvolver uma **aplicação web interativa explicável** (Flask + Folium) que, ao receber um clique do usuário, classifica o risco do ponto consultado e exibe explicação local das *features* responsáveis pela decisão."

### 1.3 Estrutura da Monografia
Atualizar para refletir a nova organização dos capítulos 4-6.

---

## 2. Capítulo 2 — Estado da Arte

A seção atual (§2.1 Trabalhos Correlatos) está bem escrita e cobre adequadamente trabalhos de 2010-2023. **Sugestões de complemento**:

### 2.1.1 Adicionar parágrafos sobre literatura 2024-2025
Inserir após o último parágrafo da §2.1, antes do "Em resumo":

> **Trabalhos recentes em interpretabilidade.** O uso de SHAP (*SHapley Additive exPlanations*) para suscetibilidade a incêndios florestais consolidou-se como prática essencial. **Cheerala et al. (2025)** aplicaram Random Forest + SHAP em Califórnia, obtendo AUC de 0,996 (grasslands) e 0,997 (forests), com identificação clara das *features* dominantes (NDVI, EVI, histórico de fogo, umidade do combustível). **Forests 2024** (Cerrado-Amazônia, *MDPI*) identificou o índice KBDI (*Keetch-Byram Drought Index*) como o melhor preditor de fogo em Canaã dos Carajás (Pará). **Quesada-Ruiz et al. (2025, npj Natural Hazards)** propuseram um modelo híbrido (dinâmico ECMWF + Random Forest) que utiliza SPI observado para prever anomalia de área queimada com até um mês de antecedência em ~68 % das áreas queimáveis brasileiras.
>
> **Trabalhos em validação temporal e drift climático.** **Bergmeir & Benítez (2012)** advertem que o uso de *cross-validation* estratificada padrão em séries temporais introduz viés otimista — necessário usar *rolling-origin* ou *TimeSeriesSplit*. Esta recomendação é tomada como princípio metodológico aqui (ver §4.x e §5.x). **Silva-Junior et al. (2025, Biogeosciences)** documentam o pior ano de degradação por fogo na Amazônia em mais de duas décadas (3,3 Mha em 2024), justificando o uso de janelas temporais recentes (2021-2023) como teste mais desafiador.
>
> **Trabalhos em pseudo-labeling para imputação climática.** **Lopez-Garcia et al. (2024, Remote Sensing)** usaram pseudo-labeling para imputar SMAP em grids tropicais com regressor supervisionado, atingindo R² > 0,90. A mesma estratégia foi adotada neste projeto (§4.x) para imputar *Umidade* nos 763 489 registros não enriquecidos pela NASA POWER.

---

## 3. Capítulo 3 — Fundamentação Teórica

### Mudanças propostas

| Subseção | Status | Sugestão |
|----------|--------|----------|
| 3.1 Bioma Amazônia | Manter | — |
| 3.2 Queimadas e Incêndios | Manter | — |
| 3.3 Programa Queimadas | Manter | Atualizar a Figura 1 com captura recente (2026); citar também NASA FIRMS como complemento em tempo real. |
| 3.4 Sensoriamento Remoto | Manter | Adicionar parágrafo curto sobre NASA POWER (reanálise climática) ao final. |
| 3.5 Base de Dados | Manter | Adicionar parágrafo sobre estruturas tabulares serem complementadas por **APIs REST** (NASA POWER, NASA FIRMS) para enriquecimento dinâmico. |
| 3.6 Pré-processamento (ruídos/outliers) | Manter | — |
| 3.7 Imagens Digitais e Espectrais | **Reavaliar** | Como o pipeline final é **tabular** (não usa imagens), considerar mover esta subseção para "trabalhos correlatos" ou enxugar. Caso queira manter, mencionar que é abordagem complementar não adotada neste projeto. |
| 3.8 Inteligência Artificial | Manter | — |
| 3.8.1 Aprendizado de Máquina | Manter | — |
| 3.8.1.1 Regressão Logística | Manter | — |
| 3.8.1.2 Random Forest | Manter | — |
| 3.8.1.3 SVM | **Remover ou reduzir** | SVM não é usado no pipeline final. Substituir por subseções abaixo. |
| **NOVA 3.8.1.4 XGBoost** | Adicionar | Chen & Guestrin (2016). Algoritmo de *gradient boosting* otimizado, com regularização L1/L2, *early stopping*, paralelização. Vencedor consistente em competições Kaggle. |
| **NOVA 3.8.1.5 LightGBM** | Adicionar | Ke et al. (2017). Otimização do GBM via *histogram-based* + *leaf-wise growth*. Mais rápido que XGBoost em datasets grandes. |
| **NOVA 3.8.1.6 CatBoost** | Adicionar | Prokhorenkova et al. (2018). Boosting nativo para features categóricas; `auto_class_weights='Balanced'`. |
| **NOVA 3.8.1.7 Ensembles** | Adicionar | Wolpert (1992) Stacked Generalization; Breiman (1996) Bagging; conceitos de *Voting* (soft/hard) e *Stacking* (meta-learner aprende sobre as probabilidades das bases). |
| 3.8.2 Reconhecimento de padrões | Manter | — |
| 3.8.3 Aprendizado Profundo | Manter | Como não é usado, reduzir tamanho e indicar que é tratado como trabalho futuro. |
| 3.8.4 Métricas | Manter | Adicionar: F1-Macro (média não ponderada entre classes — apropriada para datasets desbalanceados), Matriz de Confusão, Calibration Curve. |
| **NOVA 3.8.5 Otimização de Hiperparâmetros** | Adicionar | Akiba et al. (2019) Optuna — *Tree-structured Parzen Estimator*. Validação cruzada estratificada K-Fold. |
| **NOVA 3.8.6 Interpretabilidade de Modelos** | Adicionar | Lundberg & Lee (2017) SHAP; Lundberg et al. (2020) TreeExplainer; Fisher, Rudin & Dominici (2019) Permutation Importance; Molnar (2022) Interpretable ML. |
| **NOVA 3.8.7 Tratamento de Desbalanceamento** | Adicionar | Chawla et al. (2002) SMOTE; He & Garcia (2009) survey; `class_weight='balanced'` (scikit-learn). |
| **NOVA 3.9 Calibração de Probabilidades** | Adicionar | Platt (1999) sigmoid; Zadrozny & Elkan (2002) isotonic; `CalibratedClassifierCV` no scikit-learn. |
| **NOVA 3.10 Índices Físico-Climáticos de Fogo** | Adicionar | KBDI (Keetch & Byram 1968); SPI (McKee et al. 1993); VPD (Seager et al. 2015); De Martonne (1926); abordagem alternativa à umidade relativa direta. |
| **NOVA 3.11 Validação Temporal de Modelos** | Adicionar | Bergmeir & Benítez (2012); diferença entre split aleatório estratificado e *rolling-origin* / *TimeSeriesSplit*; relevância em séries com *features* de janela móvel. |

### Textos prontos para as novas subseções

#### 3.8.1.4 XGBoost — modelo de *Gradient Boosting* otimizado

> O *XGBoost* (eXtreme Gradient Boosting; Chen & Guestrin, 2016) é uma implementação otimizada do *Gradient Boosting Machine* (Friedman, 2001) que se popularizou em competições de aprendizado de máquina pela sua eficiência computacional, suporte nativo a regularização L1 e L2, *early stopping* e paralelização. Em essência, o algoritmo treina iterativamente novas árvores de decisão para corrigir os erros residuais das árvores anteriores, com uma função objetivo que combina perda + regularização. Cada nova árvore é construída ponderando observações que o modelo atual erra mais, processo conhecido como *boosting*. Em datasets tabulares com forte sinal não-linear, XGBoost tipicamente supera Random Forest puro em acurácia (Quesada-Ruiz *et al.*, 2025).

#### 3.8.1.5 LightGBM — *boosting* baseado em histograma

> O *LightGBM* (Ke *et al.*, 2017) implementa duas otimizações sobre o GBM tradicional: (i) *histogram-based learning*, em que valores contínuos são pré-discretizados em bins, reduzindo significativamente o custo de busca por *splits* ótimos; (ii) *leaf-wise tree growth* (em oposição ao *level-wise* tradicional), em que cada iteração expande o nó folha que mais reduz a perda, gerando árvores mais profundas e específicas. O resultado prático é treino 5 a 10 vezes mais rápido que XGBoost em datasets grandes, com acurácia equivalente ou superior, com a contrapartida de maior risco de *overfitting* em datasets pequenos (mitigado por `min_child_samples` e `num_leaves` adequados).

#### 3.8.1.6 CatBoost — *boosting* nativo para features categóricas

> O *CatBoost* (Prokhorenkova *et al.*, 2018) endereça uma limitação dos GBMs anteriores: a necessidade de pré-codificação das *features* categóricas via *one-hot encoding* ou similar, que infla a dimensionalidade e gera viés de *target leakage* quando o encoding usa estatísticas do alvo. O CatBoost incorpora codificação categórica baseada em *ordered target statistics*, que evita esse leakage. Também oferece `auto_class_weights='Balanced'` nativo para datasets desbalanceados, o que o torna conveniente como *base learner* em pipelines de risco de fogo.

#### 3.8.1.7 Ensembles — Voting e Stacking

> O conceito de combinar múltiplos modelos em um *ensemble* foi formalizado por **Wolpert (1992)** em *Stacked Generalization*. A ideia central é que diferentes algoritmos exploram diferentes regiões do espaço de hipóteses; combinar suas saídas reduz a variância e o viés do classificador final. Duas estratégias são adotadas neste trabalho:
>
> - **Voting Soft**: cada modelo base produz probabilidades para cada classe; a predição final é a classe com a maior probabilidade média entre os modelos. Funciona melhor que *hard voting* (média de classes preditas) porque preserva a incerteza dos modelos individuais.
> - **Stacking**: as probabilidades dos modelos base servem como features para um **meta-classificador** que aprende a melhor combinação. Diferente do *Voting*, o Stacking pode aprender que certos modelos são mais confiáveis para certas classes. O meta-classificador costuma ser um modelo mais simples (Regressão Logística) para evitar *overfitting*, mas modelos mais expressivos como **Gradient Boosting** podem ser usados quando há volume suficiente de dados — escolha adotada neste trabalho com ganho de 1 pp de acurácia.

#### 3.8.5 Otimização de Hiperparâmetros com Optuna

> Modelos de aprendizado de máquina possuem hiperparâmetros — escolhas estruturais que não são aprendidas dos dados (e.g., `n_estimators`, `max_depth`, `learning_rate`). A escolha manual desses valores ("*ajuste por intuição*") é subótima; **otimização bayesiana** explora o espaço de hiperparâmetros de forma sistemática.
>
> O *Optuna* (Akiba *et al.*, 2019) implementa **TPE (*Tree-structured Parzen Estimator*)**, um algoritmo de otimização bayesiana que modela `P(hiperparâmetros | resultado bom)` e `P(hiperparâmetros | resultado ruim)` separadamente, sugerindo o próximo conjunto de hiperparâmetros que maximiza a razão entre essas duas distribuições. Neste trabalho, Optuna foi aplicado em 4 modelos (RF, LR, XGBoost, LightGBM), cada um com 15 *trials* e validação cruzada estratificada em 3 *folds*, otimizando F1-macro. Os ganhos foram substanciais: XGBoost +8,4 pp F1-macro CV, LightGBM +5,5 pp, RF +4,5 pp.

#### 3.8.6 Interpretabilidade de Modelos

> Modelos de *ensemble* e *boosting* funcionam como *black boxes*: têm alta acurácia mas baixa transparência. Para uso em decisões de gestão ambiental (objetivo deste TCC), é fundamental que o sistema **explique por que** classificou um ponto como de risco "Muito Alto". Duas técnicas modernas são adotadas:
>
> **SHAP** (*SHapley Additive exPlanations*; Lundberg & Lee, 2017): baseado em teoria dos jogos cooperativos (valores de Shapley), atribui a cada *feature* uma contribuição numérica para a predição individual, de tal forma que a soma das contribuições reconstitui a probabilidade predita. Para modelos baseados em árvores, há uma implementação exata em tempo polinomial (`TreeExplainer`; Lundberg *et al.*, 2020). SHAP por classe permite identificar quais *features* dirigem cada uma das 3 classes de risco separadamente.
>
> **Permutation Importance** (Breiman, 2001; Fisher, Rudin & Dominici, 2019): formaliza a importância de uma *feature* como a degradação esperada da métrica (F1-macro, F1 por classe) quando os valores da *feature* são permutados aleatoriamente, mantendo a distribuição marginal. É *model-agnostic* (funciona com qualquer modelo) e não enviesada para *features* de alta cardinalidade — propriedade relevante quando há OHE de variáveis como `Municipio` (542 categorias) ou `Estado` (9 categorias).

#### 3.8.7 Tratamento de Desbalanceamento

> Em datasets reais, classes raras (e.g., risco "Moderado" em janelas climáticas de transição) são tipicamente minoritárias, e modelos treinados sem mitigação tendem a privilegiar as classes majoritárias (He & Garcia, 2009). Três estratégias foram avaliadas neste trabalho:
>
> 1. `class_weight='balanced'`: cada classe contribui para a função de perda com peso inversamente proporcional à sua frequência. Não cria amostras sintéticas; apenas re-pondera. É a estratégia adotada no `random_forest_balanced` e no `logistic_regression_balanced` finais.
> 2. **SMOTE** (*Synthetic Minority Over-sampling Technique*; Chawla *et al.*, 2002): gera amostras sintéticas da classe minoritária interpolando vizinhos. Aumenta o suporte de classes raras, mas pode criar exemplos não-realistas em datasets com forte estrutura espacial-temporal.
> 3. **Threshold tuning multi-classe**: em vez de aplicar `argmax` sobre as probabilidades calibradas, ajustam-se limiares `(thr_Moderado, thr_Muito_Alto)` que privilegiam o *recall* da classe minoritária. Maximiza F1-Moderado sem custo de F1-macro global.

#### 3.10 Índices Físico-Climáticos de Fogo

> A literatura de risco de incêndio acumulou desde os anos 1960 um conjunto de índices físicos para quantificar **déficit hídrico**, **estresse atmosférico** e **aridez**. Quatro deles são adotados neste trabalho como proxies à variável `Umidade` (cuja integração via NASA POWER é cara em tempo):
>
> - **KBDI** (*Keetch-Byram Drought Index*; Keetch & Byram, 1968): integral acumulada de déficit de evapotranspiração; alta em períodos secos prolongados, baixa após chuvas. Identificado por *Forests* (2024) como o melhor preditor de fogo em Canaã dos Carajás.
> - **SPI** (*Standardized Precipitation Index*; McKee *et al.*, 1993): padroniza precipitação acumulada em janelas (1, 3, 6, 12 meses) sobre a climatologia local. SPI < -1 = seca; SPI > +1 = chuvoso. Quesada-Ruiz *et al.* (2025) usaram SPI observado para prever anomalia de área queimada com 1 mês de antecedência.
> - **VPD proxy** (*Vapor Pressure Deficit*; Seager *et al.*, 2015): mede a capacidade absoluta da atmosfera de extrair água da superfície. Seager *et al.* mostram que VPD é superior a umidade relativa isolada como métrica de fogo.
> - **Aridez De Martonne** (De Martonne, 1926): índice clássico `P / (T + 10)` baseado em precipitação anual e temperatura média; permite estratificação regional do risco.

#### 3.11 Validação Temporal de Modelos

> Em datasets com componente temporal, a divisão clássica `train_test_split(stratify=y)` mistura observações de todos os anos entre treino e teste. Quando o pipeline computa *features* de janela móvel (e.g., `Precipitacao_ma90`, `SPI_3m`, `Incendios_Ultimos_180_Dias`) sobre o **conjunto inteiro antes do split**, há risco de **vazamento temporal**: linhas de teste podem ter recebido contribuição estatística de linhas próximas no espaço-tempo que estão no treino. Esse efeito **superestima** a capacidade preditiva real do modelo (Bergmeir & Benítez, 2012).
>
> A solução é o procedimento **rolling-origin por ano**: para cada ano `Y` ∈ {3 últimos com massa suficiente}, treina-se em `Ano < Y` e testa-se em `Ano == Y`. Simula um uso real ("*treine com tudo até hoje, prediga o ano seguinte*"). Aplicada ao modelo final neste trabalho, revela uma queda de 14,49 pp em acurácia e 32,35 pp em F1-Moderado — número que deve ser explicitamente reportado para balizar a interpretação das métricas do *split* aleatório.

---

## 4. Capítulo 4 — Metodologia

### 4.1 Coleta de Dados

**Manter §4.1.1 e §4.1.2 (Programa Queimadas)**. Adicionar nova subseção:

#### 4.1.3 Enriquecimento via APIs externas (NASA POWER e NASA FIRMS)

> Além do Programa Queimadas, duas APIs externas são consultadas para enriquecimento adicional:
>
> - **NASA POWER** (*Prediction of Worldwide Energy Resources*): API REST que serve reanálise climática global em resolução de 0,5° × 0,625° com latência diária. Foi utilizada para extrair **RH2M** (umidade relativa a 2 m) — variável climática reconhecida na literatura como preditor relevante de fogo. A integração foi feita via `scripts/enriquecer_dados_umidade.py` com cache em disco, retomada após interrupção e respeito ao limite de taxa (~32 req/min com duas chaves API). Pela alta latência de enriquecimento total (~16 dias para 755 mil requisições únicas), o projeto adotou uma estratégia complementar: 170 465 registros foram efetivamente enriquecidos (18,3 % da base) e os 763 489 restantes foram imputados via **pseudo-labeling** com LightGBM regressor (R² = 0,933 em holdout — ver §4.7.1).
> - **NASA FIRMS** (*Fire Information for Resource Management System*): API REST que serve detecções ativas de fogo em quase tempo real, com FRP (*Fire Radiative Power*) por píxel para os sensores VIIRS e MODIS. É consultada **em tempo de inferência** pelo app interativo (§4.6) para incrementar a feature `FRP` com observações dos últimos 7 dias na vizinhança do ponto consultado.

### 4.2 Tecnologias, Bibliotecas e Ferramentas

A lista atual (linhas 1299-1348) está completa para o pipeline de 02/2024 (Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, Joblib, Geopy). **Adicionar**:

> Além das bibliotecas já listadas, o pipeline atual incorpora:
>
> - **XGBoost (`xgboost`)**, **LightGBM (`lightgbm`)** e **CatBoost (`catboost==1.2.10`)** — modelos de *gradient boosting* usados como classificadores standalone e como *base learners* dos ensembles.
> - **Imbalanced-learn (`imblearn`)** — implementação do SMOTE para a variante `random_forest_smote`.
> - **Optuna (`optuna`)** — otimização bayesiana de hiperparâmetros via TPE.
> - **SHAP (`shap`)** — interpretabilidade local e global via `TreeExplainer`.
> - **Flask (`flask`)** — servidor web para a aplicação interativa.
> - **Folium (`folium`)** — visualização cartográfica (Leaflet sob o capô) integrada à *render* do Flask.
> - **Geopandas (`geopandas`) e Shapely (`shapely`)** — manipulação de geometrias e *shapefile* da Amazônia Legal (IBGE 2024).
> - **Requests (`requests`)** — chamadas HTTP às APIs NASA POWER e NASA FIRMS.

### 4.3 Pré-processamento

A descrição atual (linhas 1349-1429) está correta para a versão de Fev/2024 (9 features numéricas + 2 categóricas). **Adicionar**:

#### 4.3.7 Engenharia de *features* físico-climáticas (Tier 1)

> Em complemento às 9 features originais do Programa Queimadas, foram derivadas **24 features físico-climáticas avançadas** (Tier 1), agrupadas em 7 camadas (implementação em `scripts/features_avancadas.py`):
>
> | # | Camada | Features | Justificativa |
> |---|---|---|---|
> | 1 | Médias móveis estendidas | `Precipitacao_ma14/30/90`, `DiaSemChuva_ma14/30/90` | *Fuel moisture lag* (Seager *et al.* 2015) |
> | 2 | Precipitação acumulada | `Precipitacao_acum_30/90/180/365d` | *Prior-year cold-season precipitation* (Seager *et al.* 2015) |
> | 3 | SPI | `SPI_1m`, `SPI_3m`, `SPI_6m` | npj Natural Hazards 2025 |
> | 4 | Anomalia de precipitação | `Anomalia_Precipitacao`, `Anomalia_Precipitacao_rel` | *Forests* 2024 |
> | 5 | KBDI proxy / VPD proxy / De Martonne | `Temp_Climatologica`, `KBDI_proxy`, `VPD_proxy`, `Aridez_DeMartonne` | KBDI = melhor índice em *Forests* 2024 (Canaã dos Carajás) |
> | 6 | Histórico estendido de incêndios | `Incendios_Ultimos_90/180/365_Dias`, `Media_FRP_Celula_30d` | Cheerala *et al.* 2025 — histórico é Top-3 *feature* |
> | 7 | Estação seca acumulada | `Dias_Secos_90d` | Proxy do *Canadian Drought Code* (FWI) |
>
> **Granularidade espacial:** camadas 1, 2, 6, 7 e SPI usam **células de 0,25°** (≈ 27 km), evitando colapso de janelas temporais que ocorreria com agrupamento por coordenadas literais (cada foco é um ponto único). A **climatologia de temperatura** vem de uma tabela estática de médias mensais por estado, aproximada a partir das normais INMET 1981-2010 das capitais.
>
> O CSV final `base_de_dados_enriquecido.csv` contém 924 306 linhas e 46 colunas (geração em ~40 s sobre 933 954 registros brutos).

#### 4.3.8 Calibração de probabilidades

> A predição padrão dos modelos retorna *scores* não-calibrados — em particular, RF e GBMs tendem a produzir probabilidades subestimadas próximas a 0 e superestimadas próximas a 1 (Niculescu-Mizil & Caruana, 2005). Para que a saída do modelo seja interpretável como "P(classe | features) = 0,73", aplicou-se `CalibratedClassifierCV(method='isotonic', cv=3)` aos modelos do ensemble final. A calibração isotônica (Zadrozny & Elkan, 2002) ajusta uma função monotônica não-paramétrica entre scores brutos e probabilidades empíricas, sem assumir forma sigmóide (que seria a calibração de Platt).

### 4.4 Treinamento dos Modelos de Previsão

**Substituir totalmente** a §4.4 atual (que cita apenas SGDClassifier). Texto sugerido:

> #### 4.4.1 Divisão dos dados e construção do pipeline
>
> O conjunto de dados foi dividido em treino (80 %) e teste (20 %) utilizando `train_test_split(stratify=y, random_state=42)` — preservando a distribuição das 3 classes em ambos os subconjuntos. Os metadados do split (random_state, test_size, estratificação) ficam persistidos em `modelos/split_metadata.json` para garantir reprodutibilidade. O conjunto de teste contém 184 862 observações.
>
> O pipeline de processamento foi implementado via `sklearn.pipeline.Pipeline`, composto por:
>
> 1. **`ColumnTransformer`** — aplica `SimpleImputer(strategy='median') + StandardScaler` às *features* numéricas e `SimpleImputer(strategy='most_frequent') + OneHotEncoder` às *features* categóricas.
> 2. **`CalibratedClassifierCV(method='isotonic', cv=3)`** — wrap do classificador final para garantir calibração das probabilidades.
> 3. **Estimador** — variável conforme o modelo avaliado.
>
> #### 4.4.2 Modelos avaliados
>
> Oito abordagens foram comparadas no mesmo conjunto de treino/teste:
>
> 1. **Regressão Logística balanceada** — `class_weight='balanced'`, hiperparâmetros otimizados via Optuna (`C=12.43, max_iter=4768`).
> 2. **SGDClassifier** — perda logística, *online learning* — baseline simples original do projeto.
> 3. **Random Forest com SMOTE** — variante para tratamento de desbalanceamento amostral.
> 4. **Random Forest balanceado otimizado** — `class_weight='balanced'`, Optuna (15 *trials*, 3-fold CV, F1-macro): `n_estimators=346, max_depth=25, min_samples_split=3, max_features='sqrt'`.
> 5. **XGBoost otimizado** — Optuna (15 *trials*, 3-fold CV, F1-macro, subsample 150k): `n_estimators=346, learning_rate=0.073, max_depth=14, subsample=0.74, colsample_bytree=0.93, gamma=0.03`.
> 6. **LightGBM otimizado** — Optuna (15 *trials*, F1-macro): `n_estimators=683, learning_rate=0.066, num_leaves=204, min_child_samples=24, subsample=0.78, colsample_bytree=0.77`.
> 7. **CatBoost** — `iterations=1000, learning_rate=0.05, depth=8, auto_class_weights='Balanced'`.
> 8. **Ensemble Voting Soft** — média das probabilidades calibradas de RF + LR + XGBoost + LightGBM + CatBoost.
> 9. **Ensemble Stacking** — RF + LR + XGB + LGBM + CatBoost como *base learners*; **`GradientBoostingClassifier`** como meta-classificador (`n_estimators=200, max_depth=3, learning_rate=0.1`). Esta é a configuração final adotada (`ensemble_stacking_gbm`).
>
> Todos os modelos foram salvos via `joblib.dump()` em arquivos `.pkl` no diretório `modelos/`. Os hiperparâmetros otimizados de cada modelo via Optuna foram persistidos em `modelos/relatorios/optuna_*.json`.
>
> #### 4.4.3 Tempo de treinamento (atualizar Tabela 1)
>
> | Modelo | Tempo de treino (segundos) |
> |--------|-----------------------------|
> | Regressão Logística balanceada | ~ 25 |
> | SGDClassifier | ~ 8 |
> | Random Forest + SMOTE | ~ 480 |
> | Random Forest balanceado (Optuna) | ~ 1100 |
> | XGBoost (Optuna, 250k subsample) | ~ 540 |
> | LightGBM (Optuna, 250k subsample) | ~ 290 |
> | CatBoost | ~ 410 |
> | Ensemble Voting Soft | ~ 2360 (soma dos bases + agregação) |
> | **Ensemble Stacking GBM** | **~ 5100 (≈ 85 min)** |

### 4.5 Avaliação dos Modelos

Reescrever §4.5.1 (Métricas) para acomodar as novas técnicas:

> #### 4.5.1 Métricas utilizadas
>
> - **Acurácia global** — `accuracy_score`, fração de predições corretas.
> - **F1 por classe e F1-macro** — `f1_score(average=None)` e `f1_score(average='macro')`. F1-macro é a média não ponderada entre as três classes, métrica primária por ser robusta a desbalanceamento.
> - **Precisão e recall por classe** — via `classification_report`.
> - **Matriz de confusão normalizada** — `confusion_matrix(normalize='true')`.
> - **Confiança média e distribuição** — média e histograma do máximo de `predict_proba(X_test)`.
> - **Curva de calibração** — `calibration_curve` para confirmar que a probabilidade predita corresponde à frequência empírica.
>
> #### 4.5.2 Análise de interpretabilidade (NOVA)
>
> Duas técnicas complementares são aplicadas para identificar as *features* mais relevantes:
>
> - **SHAP por classe (Lundberg *et al.*, 2020)** — `TreeExplainer` aplicado a um Random Forest *compacto proxy* (100 árvores, max_depth=12, 60 000 amostras de treino) com `class_weight='balanced'`. O modelo proxy é necessário porque o Stacking GBM final e o RF Optuna produtivo não suportam `TreeExplainer` em CPU (alocação >2 GiB por classe pós-OHE). Saída em `modelos/relatorios/shap_per_class.json`.
> - **Permutation Importance (Breiman 2001; Fisher *et al.* 2019)** — aplicada diretamente sobre o `ensemble_stacking_gbm` em 5 000 amostras de teste, 3 repetições, seed=42. Para cada *feature*, calcula-se a queda em F1 por classe e em F1-macro quando os valores são permutados. Saída em `modelos/relatorios/permutation_importance_por_classe.json`.
>
> #### 4.5.3 Ajuste multi-classe de limiares de decisão (NOVA)
>
> A predição padrão usa `argmax` sobre as probabilidades calibradas. Embora maximize P(classe | x), penaliza a classe `Moderado` — fronteira ambígua entre `Baixo` e `Muito Alto`. O procedimento de `scripts/ajustar_threshold.py` varre uma grade de limiares `(thr_Moderado, thr_Muito_Alto) ∈ {0.30, 0.35, ..., 0.55}` × idem (36 combinações), com critério primário F1-macro e secundário F1-Moderado. Os limiares finais são persistidos em `modelos/prediction_thresholds.json` e aplicados em runtime no app web via parâmetro `estrategia_thresholds` da API `/api/predict`.
>
> A lógica de decisão multi-classe em runtime é:
>
> ```
> se      P(Moderado)   ≥ thr_Moderado    →  classe = "Moderado"
> senão se P(Muito Alto) ≥ thr_Muito_Alto →  classe = "Muito Alto"
> senão                                   →  classe = "Baixo"
> ```
>
> #### 4.5.4 Validação temporal (rolling-origin)
>
> Em complemento à validação no *split* aleatório, foi implementado em `scripts/validacao_temporal.py` um procedimento de **rolling-origin por ano**. Para cada ano `Y ∈ {2021, 2022, 2023}` (3 últimos com massa suficiente), o modelo é treinado em `Ano < Y` (subsample estratificado de 60 000 amostras por fold, limite de memória do OHE denso de Município) e avaliado em `Ano == Y`. Cobre o uso operacional realista: "treine com o passado, prediga o ano seguinte". O resultado é apresentado em §5.5.

### 4.6 Geração do Mapa de Risco → **Substituir por: Aplicação Web Interativa Explicável**

A §4.6 original (mapa estático Folium) deve ser **substituída** por uma seção mais robusta que reflete o app Flask. Texto sugerido:

> #### 4.6 Aplicação web interativa explicável
>
> Para tornar o modelo final utilizável e auditável por gestores ambientais, foi desenvolvida uma aplicação web (`scripts/app_map_interativo.py`) baseada em **Flask + Folium + Leaflet** que serve como interface explicável sobre o pipeline. O usuário clica em qualquer ponto da Amazônia Legal e o sistema responde, em ~ 10-20 s, com a predição classificada, acompanhada das 24 *features* Tier 1 calculadas para o ponto, da explicação local das *features* mais influentes e do contexto histórico comparativo.
>
> **4.6.1 Pipeline de predição pontual.** Como o app prediz **um único ponto isolado**, sem histórico temporal local, as *features* Tier 1 que dependem de séries (SPI, acumulados longos, histórico estendido de focos) não podem ser calculadas em tempo real. Adotamos a estratégia de **lookup espaço-sazonal** (`scripts/feature_lookup.py`):
> 1. Pré-cálculo das medianas das 24 *features* Tier 1 do dataset enriquecido, agregadas em três níveis de granularidade decrescente: `(LatBin 0,25°, LonBin 0,25°, Mes)`, depois `(Estado, Mes)`, e por fim `Mes` global.
> 2. Para o ponto consultado, lookup em cascata (do mais fino ao mais grosso) com tolerância de ±1 célula.
> 3. *Features* físicas independentes de série temporal (`Temp_Climatologica`, `KBDI_proxy`, `VPD_proxy`, `Aridez_DeMartonne`, `Anomalia_Precipitacao`) são recalculadas em tempo real com o clima NASA POWER atual e a climatologia local da precipitação — o lookup fornece apenas os "blocos de construção" históricos imutáveis.
>
> **4.6.2 Integração de dados em tempo real.**
> - **Geocodificação reversa** via Nominatim/OpenStreetMap para extrair Estado e Município do clique.
> - **Clima atual** via NASA POWER (precipitação, dias sem chuva, RH2M dos últimos 28 dias).
> - **Focos ativos** via NASA FIRMS para `FRP` dos últimos 7 dias no raio de 10 km do ponto.
>
> **4.6.3 Explicabilidade local.** Em runtime, `scripts/explainer.py` aproxima a contribuição de cada *feature* como
>
> `contrib(f) = importância(f) × z-score(f) × sinal_físico(f)`,
>
> onde `importância(f)` é carregada de `shap_per_class.json` (importância por classe predita, quando disponível) ou de `shap_feature_importance.json` (Gini global como fallback). `z-score(f)` mede a anormalidade do valor no ponto em relação à distribuição global do dataset, e `sinal_físico(f) ∈ {-1, +1}` codifica o efeito esperado (e.g., `+1` para `Indice_Seca`, `-1` para `Precipitacao_ma90`). A predição é classificada como `AUMENTA RISCO`, `REDUZ RISCO` ou `neutro`. O top-6 *features* por |contribuição| é exibido no painel lateral.
>
> **4.6.4 Interface.** O front-end é organizado em **6 *cards*** dentro de um painel lateral de 420 px:
> 1. **Previsão** — badge com a classe, barras de probabilidade por classe, métricas do modelo, fonte do clima, granularidade do lookup, e trace da decisão (argmax vs thresholds calibrados).
> 2. **Por que esse risco?** — top-6 *features* com valor, faixa típica (p10-p90 da distribuição global), z-score e direção do impacto.
> 3. **Dados climáticos atuais** — precipitação, dias sem chuva, umidade NASA, temperatura climatológica, estação, período do dia.
> 4. **Índices de seca e aridez** — Índice de Seca, SPI-1m/3m/6m, KBDI proxy, Aridez De Martonne, VPD proxy, Dias_Secos_90d.
> 5. **Precipitação acumulada e anomalia** — acumulados 30/90/180/365 dias com narrativa textual comparando precipitação atual vs média histórica do município/mês.
> 6. **Histórico de focos** — focos NASA FIRMS em 7/30/90/180/365 dias, dias desde último foco, FRP médio/máximo 7d, FRP no ponto agora.
>
> O **mapa estático** (versão original da §4.6 do TCC II) permanece disponível como artefato batch (`scripts/gerar_mapa.py`).

### Nova subseção 4.7 — Pseudo-labeling de Umidade (NOVA)

> **4.7 Imputação de Umidade via pseudo-labeling**
>
> Como discutido em §4.1.3, a integração via NASA POWER produziu 170 465 linhas com `Umidade` real (18,3 % da base). Para mitigar essa cobertura parcial sem aguardar mais 13 dias de enriquecimento, foi implementada a estratégia de **pseudo-labeling** descrita em Lee (2013) e Arazo *et al.* (2020). O script `scripts/pseudo_label_umidade.py`:
>
> 1. Treina um **LightGBM regressor** sobre os ~170 k registros com `Umidade` real, usando 15 *features* (precipitação, dias sem chuva, lat/lon, sazonalidade cíclica `Mes_sin/cos`, histórico de fogo) — *features* que existem para todas as linhas, com ou sem `Umidade` real.
> 2. Avalia o erro real em holdout 20 %: **RMSE = 4,33 % RH, MAE = 3,19 % RH, R² = 0,933**. O R² alto é cientificamente justificável: a umidade tem alta autocorrelação espaço-sazonal com as *features* usadas — lat/lon definem o regime de monção, mês define a estação seca/chuvosa, precipitação recente define o estado higroscópico do ar.
> 3. Refita no 100 % dos *labels* reais e imputa os 763 489 registros sem label (clip [5, 100] %).
> 4. Salva o dataset enriquecido em `base_de_dados_umidade_pseudo.csv` com coluna auxiliar `Umidade_origem ∈ {nasa_power, pseudo_label_lgbm}` para auditoria.
>
> **Status no pipeline final.** O dataset `base_de_dados_umidade_pseudo.csv` está disponível mas **não é usado nos números finais reportados** em §5. Razão: incluir `Umidade` (pseudo-rotulada ou real) exigiria retreino completo do `ensemble_stacking_gbm` (~85 min CPU) + revalidação das *thresholds* + revalidação temporal — fora do orçamento computacional desta entrega. O artefato fica registrado como **etapa preparatória** para a sequência natural do trabalho. Cautela registrada: pseudo-labels devem ser interpretados considerando potencial **viés de confirmação** (Arazo *et al.*, 2020).

---

## 5. Capítulo 5 — Resultados (estava VAZIO)

Este capítulo precisa ser **completamente reescrito**. Já há um texto pronto consolidado em `DOCUMENTO_METODOLOGIA_E_RESULTADOS.md` §4 — **basta importar para o TCC**. Estrutura sugerida:

| Subseção | Conteúdo | Fonte no projeto |
|----------|----------|------------------|
| 5.1 Visão geral dos experimentos | Resumo executivo: dataset, split, modelos avaliados, melhor resultado | `DOCUMENTO_METODOLOGIA_E_RESULTADOS.md` §4 |
| 5.2 Comparativo de modelos | **Tabela com 8+ modelos** (acurácia, F1-macro, F1 por classe), análise quantitativa | `DOCUMENTO_METODOLOGIA_E_RESULTADOS.md` §4.1, §4.8.1 |
| 5.3 Desempenho do modelo final | Stacking GBM detalhado: classification_report, matriz de confusão, curva de calibração | `DOCUMENTO_METODOLOGIA_E_RESULTADOS.md` §4.10 + `modelos/relatorios/ensemble_stacking_gbm_metrics.json` |
| 5.4 Análise de interpretabilidade | SHAP por classe (RF leve proxy) + Permutation Importance + Top-features | `DOCUMENTO_METODOLOGIA_E_RESULTADOS.md` §4.4.1 e §4.4.2 |
| 5.5 Validação temporal estrita | rolling-origin RF + Stacking GBM, Δ viés vs split aleatório | `DOCUMENTO_METODOLOGIA_E_RESULTADOS.md` §4.11 e §4.11.1 |
| 5.6 Threshold tuning | Ganho de F1-Moderado, tabela de configurações testadas | `DOCUMENTO_METODOLOGIA_E_RESULTADOS.md` §3.13, §4.9, §4.10 |
| 5.7 Evolução incremental | Figura comparativa: baseline → +Tier 1 → +Optuna+Meta-GBM → +thresholds | `modelos/relatorios/evolucao_modelos.png` + .json |
| 5.8 Aplicação web — validação por consultas | Smoke-tests AM x PA (centro úmido vs sul seco), respostas do app, screenshots | `DOCUMENTO_METODOLOGIA_E_RESULTADOS.md` §3.12.4 |

**Tabelas e figuras prontas para o TCC:**

1. **Tabela 2** — Comparativo de 8+ modelos (do `DOCUMENTO_METODOLOGIA_E_RESULTADOS.md` §4.10 → `CHECKLIST_OBJETIVO_FINAL.md` linha 369-385).
2. **Tabela 3** — Métricas detalhadas do Stacking GBM por classe.
3. **Tabela 4** — SHAP Top-8 por classe.
4. **Tabela 5** — Permutation Importance Top-10 global e por classe.
5. **Tabela 6** — Validação temporal: RF vs Stacking GBM, 3 folds + média + Δ viés.
6. **Figura 9** (nova) — Evolução das métricas (já gerada em `modelos/relatorios/evolucao_modelos.png`).
7. **Figura 10** (nova) — SHAP Top-features por classe (matplotlib, gerar via `analise_shap_rf_leve.py` com `--plot`).
8. **Figura 11** (nova) — Matriz de confusão normalizada do Stacking GBM.
9. **Figura 12** (nova) — Screenshot do app interativo mostrando um clique em zona de alto risco (PA Sul, set/2024).
10. **Figura 13** (nova) — Captura do *Painel do Fogo* (CENSIPAM) com o evento `6668060` (Pium-TO, 12/05/2026), referenciada na §5.9 (ver abaixo).

### 5.9 Validação cruzada com fonte externa independente — caso Pium-TO

Adicionado em 12/05/2026 após investigação que partiu de uma consulta do usuário no aplicativo:

| Subseção | Conteúdo | Fonte no projeto |
|----------|----------|------------------|
| 5.9.1 O evento | Foco `6668060` em Pium-TO (Tocantins), duração 48,3 h, 12 focos | *Painel do Fogo* / CENSIPAM (independente do INPE) |
| 5.9.2 Resposta do sistema | 5 janelas temporais consultadas, todas previram *Baixo* 81-88% | `DOCUMENTO_ESTUDO_CASO_PIUM_TO.md` §5 |
| 5.9.3 Diagnóstico em 3 causas | (C1) NASA FIRMS 400, (C2) Tier 1 degradando precoce, (C3) descompasso clima de grade vs realidade local | `DOCUMENTO_ESTUDO_CASO_PIUM_TO.md` §7 |
| 5.9.4 Correções incorporadas | C1+C2 aplicadas no código; C3 documentada como limite estrutural | `tcc_tex/07_estudo_caso_pium_to.tex` |
| 5.9.5 Contrafactual | Mesmo ponto em 15/05/2024 → Muito Alto (47,3%) com KBDI=148; em 12/05/2026 → Baixo com KBDI=29,8 | Tabela `tab:pium_contrafactual` |
| 5.9.6 Discussão + implementação (M1--M4) + síntese | Limites do paradigma de grade; auditoria JSONL; fusão INMET (tabela `tab:pium_inmet`); OOD (tabelas `tab:pium_ood_*`); **M4** extensões API/auditoria (§5.10) | `tcc_tex/07_estudo_caso_pium_to.tex` |

**Arquivo a inserir no LaTeX**: `tcc_tex/07_estudo_caso_pium_to.tex` (Seção 5.9 sugerida — ver `INSTRUCOES_INSERCAO.md` §2.8).

**Documento reproduzível complementar**: `DOCUMENTO_ESTUDO_CASO_PIUM_TO.md` na raiz do projeto — contém timeline da investigação, comandos executados, saídas brutas, *diff* das três correções e referências às fontes externas.

**Por que esta seção é importante para o TCC**:
- Validação por **fonte externa independente** (CENSIPAM ≠ INPE) — fortalece a defesa.
- Confronta o sistema com um **evento real**, não com hold-out sintético.
- Documenta um **limite estrutural** honestamente (granularidade ~50 km da reanálise MERRA-2 em zonas de transição Cerrado-Amazônia), reforçando a maturidade científica do trabalho.
- Gera **trabalhos futuros concretos** já listados na Conclusão (fusão MERRA-2 + INMET, oversampling OOD, auditoria operacional contínua).

### 5.10 Extensões alinhadas ao estado da arte (motivação para o TCC)

**Implementadas após a revisão bibliográfica 2024--2026** (código em produção; **não** alteram o vetor de *features* do modelo já treinado — evitam *train/serve skew* até novo retreino documentado).

| Tema na literatura | O que foi feito no projeto | Por que entra no TCC |
|--------------------|---------------------------|----------------------|
| Fusão multi-fonte (reanálise + in situ) | Busca INMET **hierárquica**: raio preferencial `INMET_MAX_DISTANCE_KM` (50 km), depois `INMET_EXTENDED_MAX_DISTANCE_KM` (180 km), com metadados `inmet_representatividade` e `inmet_busca_raio_km` | Transparência metodológica: em Pium--TO passa a existir fusão via estação sinótica distante (ex.: FORMOSO) com rótulo **baixa** representatividade — honestidade científica superior a omitir INMET. |
| *Fire weather* (vento, seca, aridez) | NASA POWER passa a solicitar **WS2M** e **T2M_MAX**; WIS2 agrega **vento médio 7d** na mesma estação da chuva; *proxy* escalar `FWI_fire_weather_proxy` na resposta da API | Conecta o Cap.~3 (índices físicos) à literatura que combina vento + umidade + seca; o *proxy* é interpretável e **separado** do classificador até ablação futura. |
| Incerteza / decisão sob ambiguidade | Módulo `operational_uncertainty.py`: `gap` top-2, `set_sugerido`, `incerteza_alta` expostos em `incerteza_operacional` no JSON de `/api/predict` | Alinha-se a *uncertainty quantification* em observação da Terra sem exigir conjunto de calibração conformal pré-computado; útil na discussão de limitações e de painéis de decisão. |
| Métricas orientadas a alerta | `auditar_predicoes.py` reporta **FNR** e **FPR** no binário alerta vs.~foco FIRMS | Literatura de sistemas de alerta enfatiza o custo de **falso negativo** (fogo não alertado); FNR nomeia isso explicitamente no relatório operacional. |

**Arquivos**: `scripts/config.py`, `scripts/inmet_api.py`, `scripts/nasa_power_realtime.py`, `scripts/climate_api.py`, `scripts/operational_uncertainty.py`, `scripts/app_map_interativo.py`, `scripts/auditar_predicoes.py`.

**Onde escrever no texto do TCC**: §4 (Metodologia) — subseção curta “Dados em tempo real e metadados de fusão”; §5 (Resultados ou Discussão) — parágrafo “Indicadores complementares de *fire weather* e incerteza”; estudo de caso `07_estudo_caso_pium_to.tex` — **subsubseção** M4 (ver `INSTRUCOES_INSERCAO.md` §2.8).

---

## 6. Capítulo 6 — Conclusão (precisa ser reescrita completamente)

O texto atual (linhas 1631-1654) reconhece que o projeto estava em **fase inicial** em fev/2024. Agora a conclusão deve ser **substancial** e baseada nos resultados reais. Texto sugerido:

> # 6 CONCLUSÃO
>
> Este trabalho desenvolveu um sistema completo de classificação supervisionada de risco de incêndio em três níveis (Baixo, Moderado, Muito Alto) para a Amazônia Legal, integrando dados do Programa Queimadas/INPE (2014-2023), reanálise climática NASA POWER, detecções em tempo real NASA FIRMS, e o *shapefile* IBGE 2024. Os principais resultados são:
>
> 1. **Acurácia operacional de 84,61 %** com F1-macro de 0,7995 e F1-Moderado de 0,6375 (após *threshold tuning*) — superando em quase 15 pontos percentuais a meta acadêmica inicial de 70 %. O modelo final é um **Ensemble Stacking** de Random Forest balanceado (Optuna), Logistic Regression balanceada (Optuna), XGBoost (Optuna), LightGBM (Optuna) e CatBoost, com meta-classificador *Gradient Boosting* — escolha que demonstrou ganho monotônico em todas as etapas de evolução incremental (+4,12 pp em acurácia, +5,03 pp em F1-macro e +8,77 pp em F1-Moderado sobre o ensemble *baseline* sem Tier 1).
>
> 2. **A engenharia de *features* físico-climáticas é o fator dominante de ganho.** A introdução das 24 *features* Tier 1 (SPI, KBDI proxy, VPD proxy, médias móveis estendidas, lags acumulados, histórico estendido, dias secos) — substituindo a variável `Umidade` por proxies amparados pela literatura recente (Seager *et al.* 2015; *Forests* 2024; *npj Natural Hazards* 2025) — entregou +3,11 pp em acurácia e +6,70 pp em F1-Moderado, ao custo de apenas 40 segundos de geração offline. Esse achado reforça a tese de que **engenharia de variáveis bem embasada na literatura é tão importante quanto a escolha do algoritmo** em problemas de domínio.
>
> 3. **Interpretabilidade por duas técnicas independentes converge na mesma narrativa física.** Análise SHAP via Random Forest *compacto proxy* (Lundberg *et al.* 2020) e *Permutation Importance* model-agnostic (Fisher *et al.* 2019) sobre o Stacking GBM final identificam coerentemente `KBDI_proxy`, `VPD_proxy`, `Indice_Seca`, `DiaSemChuva_ma90` e `Precipitacao_acum_180d` como *features* dominantes. `VPD_proxy` aparece no Top-5 da classe **Muito Alto** por ambas as técnicas independentes — validação cruzada que sustenta a escolha metodológica de substituir `Umidade` por proxies físicos.
>
> 4. **Validação temporal estrita revela o viés do split aleatório**, e ainda assim o modelo mantém **acurácia acima da meta acadêmica original** (≥ 70 %) sob folds *rolling-origin*. A acurácia média em 3 folds (2021-2023) é de 70,13 % ± 4,06 (queda de 14,49 pp em relação ao split aleatório). O Stacking GBM mantém vantagem de +1,4 pp F1-macro e +4,2 pp F1-Moderado sobre o RF mesmo nesse regime mais severo — sinal de que o ganho do ensemble não é mero *overfitting*. A queda em F1-Moderado (-32,35 pp) sob validação temporal indica que **as fronteiras de transição entre regimes** são intrinsecamente difíceis em horizontes maiores, abrindo espaço para trabalhos futuros em causalidade estrita das janelas móveis.
>
> 5. **A aplicação web interativa** (Flask + Folium + Leaflet) demonstra a viabilidade operacional do sistema: ao receber um clique do usuário em qualquer ponto da Amazônia Legal, o servidor compõe em ~ 10-20 segundos um vetor de 44 *features* (incluindo 24 Tier 1 via lookup espaço-sazonal + 5 calculadas em tempo real a partir do clima atual + 9 originais + histórico FIRMS), classifica o risco com *thresholds* calibrados, e exibe explicação local das *features* responsáveis. Smoke-tests em zonas opostas (centro úmido do Amazonas no inverno × Sul do Pará em pleno setembro) confirmam que o modelo discrimina corretamente, com explicações fisicamente defensáveis.
>
> 6. **Reprodutibilidade tratada como requisito não-funcional.** Todas as métricas, hiperparâmetros, *thresholds* e metadados de *split* estão persistidos em JSON versionado (`modelos/relatorios/*.json`); o pipeline completo é reexecutável via scripts modulares; o app expõe `thresholds_aplicados` e `fonte_importance` na resposta de cada predição para auditoria. A bibliografia consolidada em `REFERENCIAS_TCC.md` totaliza 41 entradas BibTeX cobrindo as referências fundamentais usadas em cada seção.
>
> ## Limitações reconhecidas
>
> - **Viés temporal de janelas móveis** quantificado em §5.5 (queda de 14,49 pp em acurácia sob validação temporal estrita). Para deployment em horizontes maiores, sugere-se reimplementar as *features* com causalidade estrita (computar `Precipitacao_ma30` usando apenas dados ≤ t do registro).
> - **Cobertura parcial da `Umidade` real (18,3 %)** — mitigada via pseudo-labeling (R² = 0,933 em holdout) com cobertura final de 100 %, mas o retreino do Stacking incluindo `Umidade` fica como trabalho futuro imediato.
> - **SHAP exato sobre o Stacking GBM** não é viável em CPU por requisitos de memória; SHAP via *RF compacto proxy* é defensável (Lundberg *et al.* 2020) mas tem perda qualitativa em relação ao modelo de produção.
>
> ## Trabalhos futuros
>
> 1. Retreinar `ensemble_stacking_gbm` incluindo `Umidade` (pseudo-rotulada e real) e quantificar ganho marginal.
> 2. Causalidade estrita das *features* de janela móvel para reduzir o viés temporal.
> 3. Integração de NDVI/EVI MODIS via Google Earth Engine (Tier 2 do plano de melhorias).
> 4. Integração de MapBiomas Fogo Coleção 4 como *feature* adicional de histórico de área queimada.
> 5. Migração do SHAP para `shap.GPUTreeExplainer` (CUDA) para análise exata sobre o RF Optuna produtivo.
> 6. Substituição do *lookup* espaço-sazonal por séries temporais reais (NASA POWER PRECTOTCORR 28d) no app para cálculo *exato* de SPI, anomalia e acumulados na data consultada.

---

## 7. Referências do TCC — atualização

A bibliografia atual (linhas 1657-1721) possui 27 entradas, das quais ~10 podem ser mantidas. **Substituições e adições**:

| Ação | Referência | Onde será usada |
|------|------------|-----------------|
| Manter | WMO (2021), Aragão (citado em §1.1) | §1 |
| Manter | INMET (2023), INPE (2024), IBGE (2004) | §1, §3.3 |
| Manter | Novo & Ponzoni (2001), Liu (2007), Fitz (2020) | §3.4 |
| Manter | Goodfellow, Bengio & Courville (2016), Russell & Norvig (2020) | §3.8 |
| Manter | Cerri & Carvalho (2017) | §3.8.1 |
| **Adicionar** | Breiman (2001) — Random Forest | §3.8.1.2 |
| **Adicionar** | Wolpert (1992) — Stacked Generalization | §3.8.1.7 |
| **Adicionar** | Chen & Guestrin (2016) — XGBoost | §3.8.1.4 |
| **Adicionar** | Ke et al. (2017) — LightGBM | §3.8.1.5 |
| **Adicionar** | Prokhorenkova et al. (2018) — CatBoost | §3.8.1.6 |
| **Adicionar** | Friedman (2001) — Gradient Boosting | §3.8.1.4, §3.8.1.7 |
| **Adicionar** | Chawla et al. (2002) — SMOTE | §3.8.7 |
| **Adicionar** | Akiba et al. (2019) — Optuna | §3.8.5 |
| **Adicionar** | Platt (1999) e Zadrozny & Elkan (2002) — Calibração | §3.9 |
| **Adicionar** | He & Garcia (2009) — Desbalanceamento | §3.8.7 |
| **Adicionar** | Lundberg & Lee (2017) — SHAP | §3.8.6 |
| **Adicionar** | Lundberg et al. (2020) — TreeExplainer | §3.8.6, §4.5.2 |
| **Adicionar** | Fisher, Rudin & Dominici (2019) — Permutation Importance | §3.8.6, §4.5.2 |
| **Adicionar** | Strobl et al. (2007) — viés do MDI | §4.5.2 |
| **Adicionar** | Molnar (2022) — Interpretable ML | §3.8.6 |
| **Adicionar** | Guyon et al. (2002) — RFE | §3.8.6 |
| **Adicionar** | Cheerala et al. (2025) — RF+SHAP fogo Califórnia | §2.1.1 |
| **Adicionar** | Seager et al. (2015) — VPD | §3.10 |
| **Adicionar** | Keetch & Byram (1968) — KBDI | §3.10 |
| **Adicionar** | McKee et al. (1993) — SPI | §3.10 |
| **Adicionar** | De Martonne (1926) — Aridez | §3.10 |
| **Adicionar** | Forests 2024 — Cerrado-Amazônia | §2.1.1, §3.10 |
| **Adicionar** | Quesada-Ruiz et al. (2025) — npj Natural Hazards | §2.1.1, §3.10 |
| **Adicionar** | Aragão et al. (2018) — Nature Communications | §1.1 |
| **Adicionar** | Silva-Junior et al. (2025) — Biogeosciences | §1.1, §2.1.1 |
| **Adicionar** | Bergmeir & Benítez (2012) — validação temporal | §3.11, §4.5.4, §5.5 |
| **Adicionar** | Lee (2013) — Pseudo-Label | §4.7 |
| **Adicionar** | Arazo et al. (2020) — Confirmation bias | §4.7 |
| **Adicionar** | Lopez-Garcia et al. (2024) — Imputação SMAP | §4.7 |
| **Adicionar** | NASA POWER (Stackhouse et al., 2018) | §3.4, §4.1.3 |
| **Adicionar** | NASA FIRMS (Schroeder et al., 2014) | §4.1.3, §4.6.2 |
| **Adicionar** | Pedregosa et al. (2011) — scikit-learn | §4.2 |
| **Adicionar** | Lemaître et al. (2017) — imbalanced-learn | §4.2 |

Todos os BibTeX correspondentes já estão em **`REFERENCIAS_TCC.md`** (41 entradas, pronto para colar em `references.bib`). A tabela cruzada em `REFERENCIAS_TCC.md` §7 mostra em qual seção do TCC cada referência será citada.

---

## 8. Cronograma sugerido de integração no TCC LaTeX

Recomendo executar a integração em **três fases** para não comprometer a coerência:

### Fase 1 — Resultados e Conclusão (essencial)
Sem isso o TCC fica incompleto.
1. Capítulo 5 — preencher inteiro usando §5.1 a §5.8 desta carta (~ 8-12 páginas de texto + tabelas + figuras).
2. Capítulo 6 — reescrever conclusão (~ 2 páginas).
3. Atualizar Resumo / Abstract.

### Fase 2 — Fundamentação Teórica e Metodologia (alto valor)
Para que os resultados façam sentido, adicionar as novas subseções teóricas:
1. Capítulo 3 — novas subseções 3.8.1.4 (XGBoost), 3.8.1.5 (LightGBM), 3.8.1.6 (CatBoost), 3.8.1.7 (Ensembles), 3.8.5 (Optuna), 3.8.6 (Interpretabilidade), 3.8.7 (Desbalanceamento), 3.9 (Calibração), 3.10 (Índices físico-climáticos), 3.11 (Validação temporal).
2. Capítulo 4 — substituir §4.4, expandir §4.5, substituir §4.6, adicionar §4.7.

### Fase 3 — Polimento e referências
1. Bibliografia: atualizar `references.bib` com as 41 entradas de `REFERENCIAS_TCC.md`.
2. Estado da Arte — adicionar parágrafos da §2.1.1.
3. Introdução — adicionar Aragão 2018 e Silva-Junior 2025 na §1.1.
4. Revisão final: garantir que as referências antigas que **não são mais usadas** (SVM em §3.8.1.3) sejam revisadas — ou se mantidas, indicar que foram consideradas mas não adotadas.

---

## 9. Recursos de apoio já disponíveis no projeto

Para facilitar a redação, **tudo já está consolidado**:

| Arquivo | Conteúdo | Uso no TCC |
|---------|----------|------------|
| `DOCUMENTO_METODOLOGIA_E_RESULTADOS.md` | Materiais e métodos + resultados completos (~10 mil linhas, formato dissertação) | Texto pronto para colar nos Cap. 4 e 5 |
| `REFERENCIAS_TCC.md` | 41 entradas BibTeX + tabela cruzada com §do TCC | Bibliografia pronta para `references.bib` |
| `CHECKLIST_OBJETIVO_FINAL.md` | Estado de tarefas e tabela final de modelos | Tabela 2 do TCC + roteiro de redação |
| `scripts/AVANCOS_TREINAMENTO_RECENTES.md` | Resumo executivo das melhorias por iteração | Apoio para §5.7 (evolução) |
| `README_APP.md` | Documentação técnica do app web | Apoio para §4.6 |
| `modelos/relatorios/*.json` | Métricas exatas para verificação final | Auditoria dos números colados |
| `modelos/relatorios/evolucao_modelos.png` | Figura comparativa pronta | Figura 9 do TCC |
| `modelos/relatorios/shap_feature_importance.png` | Figura SHAP global | Figura 10 do TCC |
| `modelos/relatorios/threshold_precision_recall.png` | Curva precision-recall por threshold | Figura para §5.6 |

---

## 10. Observação sobre o título do TCC

O título atual — *"Aprendizado de Máquina e Sensoriamento Remoto: Uma Ferramenta de Mapeamento e Previsão da Suscetibilidade de Incêndios Florestais na Amazônia Legal"* — está adequado para o trabalho.

**Sugestão opcional de subtítulo** caso queira destacar a contribuição metodológica:

> *Aprendizado de Máquina e Sensoriamento Remoto: Uma Ferramenta Interativa Explicável de Mapeamento e Previsão da Suscetibilidade de Incêndios Florestais na Amazônia Legal*

(adiciona apenas a palavra "Interativa Explicável" — destaca os diferenciais técnicos da §4.6 e §4.5.2).

---

**Próximos passos sugeridos**:
1. Ao ler este mapa, identifique quais seções têm **maior urgência** para sua banca de defesa (provavelmente Cap. 5 e 6).
2. Posso gerar agora um documento `TEXTO_CAPITULO_5.md` com o capítulo 5 inteiro pronto para colar no LaTeX, formatado em ABNT, com tabelas tabuladas e citações no formato `\cite{chave_bibtex}`.
3. Quando você quiser, podemos também produzir as **figuras adicionais** (matriz de confusão, SHAP por classe, Permutation Importance) em alta resolução para colar no documento.
