# Estado da arte e propostas de melhoria — TCC "Previsão de risco de incêndio na Amazônia Legal com aprendizado de máquina explicável"

> **Elaborado em:** 01/06/2026
> **Objetivo:** (i) diagnosticar criticamente o estado atual do projeto; (ii) sintetizar o estado da arte em previsão e monitoramento de incêndios (2022–2025); (iii) propor melhorias priorizadas, cada uma fundamentada em literatura científica.
> **Documentos-base:** `DOCUMENTO_METODOLOGIA_E_RESULTADOS.md`, `DOCUMENTO_ESTUDO_CASO_PIUM_TO.md`, `REFERENCIAS_TCC.md`, código em `scripts/`.
> Este documento complementa o `REFERENCIAS_TCC.md` (41 entradas existentes) com **13 novas referências** (§7), prontas para `references.bib`.

---

## 1. Sumário executivo

O projeto possui **engenharia de ML de alto nível para um TCC**: ensembles otimizados (Stacking GBM, 83,6% acurácia / 78,9% F1-macro), 24 features físico-climáticas fundamentadas, calibração, ajuste de limiares, interpretabilidade (SHAP/permutation/RFECV), app interativo e estudo de caso real (Pium-TO). Contudo, a **tese científica** está exposta por quatro lacunas de fundo (§2). A mais importante: **o alvo do modelo é a discretização do índice Risco de Fogo do INPE, não a ocorrência observada de incêndio** — o modelo aprende a reproduzir um índice, e nunca é confrontado com fogo real.

As três ações de maior retorno, em ordem, são: **(P1)** rotular por fogo observado; **(P2)** introduzir horizonte de previsão com validação espaço-temporal em bloco; **(P3)** benchmark contra os índices físicos (FWI / INPE RF). Juntas, elevam o trabalho de "reprodução de um índice" para "previsão de incêndio validada contra a realidade" — o que define o estado da arte (§3, §5, §6).

---

## 2. Diagnóstico crítico do estado atual

### 2.1 Lacuna 1 (crítica) — O alvo é a saída de outro modelo, e a base é *presence-only*

Em `scripts/carregar_dados.py:299-330` (`classificar_risco`), o rótulo `RiscoClassificado` é apenas uma discretização em três faixas (limiares semânticos 0,50 e 0,80) da coluna `RiscoFogo`. Essa coluna provém dos arquivos de focos do INPE (`focos_qmd_inpe_*.csv`) e corresponde ao **índice operacional de Risco de Fogo do INPE** (escala 0–1), calculado a partir de dias-sem-chuva, precipitação e umidade — **as mesmas variáveis que estão entre as features do modelo** (ver §2.4).

**Evidência empírica direta (CSVs brutos do INPE).** A inspeção dos `focos_qmd_inpe_*.csv` confirma duas coisas:

1. **`RiscoFogo` é o índice do INPE, não um valor calculado pelo pipeline.** Vem pronto na coluna (escala 0,0–1,0; `-999` = ausente), ao lado de `DiaSemChuva` e `Precipitacao`, também fornecidos pelo INPE. Cabeçalho: `DataHora, Satelite, Pais, Estado, Municipio, Bioma, DiaSemChuva, Precipitacao, RiscoFogo, Latitude, Longitude, FRP` — formato padrão de exportação de **focos (detecções de fogo ativo)** do Programa Queimadas (satélite `AQUA_M-T`).
2. **A base é *presence-only*: toda linha é um foco real.** As ~934 mil linhas do `base_de_dados*.csv` são concatenação direta dos focos do INPE (`adicionar_historico_incendios.py`, `features_avancadas.py` apenas agregam features; nenhum script gera pontos de não-fogo, *background points* ou pseudo-ausências). **Não existe classe negativa.** As três classes são bins do índice *em locais que efetivamente queimaram*: "Baixo" significa "um foco real ocorreu onde o índice do INPE marcava baixo" — não "local sem risco".

Num único arquivo (2014–2015, 82.678 focos), **8.288 detecções (~10%) ocorreram com `RiscoFogo` ∈ {0,0; 0,1}** — fogo real em locais que o índice do INPE classificava como risco baixíssimo. É o caso Pium-TO generalizado a ~10% dos dados: prova quantitativa de que o rótulo é sistematicamente falho no extremo baixo.

Consequências:
- **Circularidade / vazamento parcial de rótulo.** O modelo re-deriva uma fórmula determinística cujos insumos são suas próprias features. Isso explica por que `Indice_Seca` e `DiaSemChuva` lideram a importância (são os componentes do índice). A acurácia de 83,6% mede sobretudo "quão bem reproduzo o índice do INPE", não "quão bem preveio incêndio".
- **O modelo não distingue fogo de não-fogo.** Como nunca viu um local sem foco, não pode responder "haverá incêndio aqui?". Porém o app consulta pontos arbitrários do mapa — **extrapolação para fora do suporte de treino** (toda a distribuição de treino é de áreas queimadas). A saída "Baixo" não é "baixo risco de fogo"; é "se houvesse fogo aqui, o índice do INPE seria baixo".
- **Herança de pontos cegos.** O caso Pium-TO (12/05/2026) é a prova qualitativa: fogo real confirmado pelo CENSIPAM, índice indicando "Baixo" porque a reanálise NASA POWER (~50 km) não capturou a seca local. Um modelo que imita o índice **não pode corrigir** esse erro.
- **Ausência de validação contra a verdade de campo.** O sistema nunca é avaliado contra ocorrência observada de fogo (focos VIIRS/FIRMS, área queimada MODIS/MapBiomas Fogo).

### 2.2 Lacuna 2 — É *nowcast* (diagnóstico do presente), não previsão

O modelo classifica condições no instante consultado; não há horizonte (t+1, t+7, t+30). Sistemas operacionais de referência (INPE RF; GEFF/EFFIS) entregam **previsão com antecedência** dirigida por modelos numéricos de tempo (§3.1). O termo "previsão" exige um horizonte temporal explícito.

### 2.3 Lacuna 3 — Vazamento temporal nas features de janela (já documentado pelos autores)

Médias móveis, SPI e acumulados são calculados sobre o dataset completo antes do split (`carregar_dados.py`, `features_avancadas.py`). A validação *rolling-origin* dos próprios autores quantifica o viés: acurácia **82,9% → 70,1%** e F1-Moderado **60% → 28%**. O número honesto é ~70%.

### 2.4 Lacuna 4 — Sem estrutura espacial; memorização de região

RF/GBM tratam cada ponto isoladamente. A *permutation importance* sobre o Stacking aponta `Ano`, `Latitude`, `Longitude`, `Estado` como mais influentes — indício de **memorização de regiões/anos** em vez de dinâmica transferível, coerente com a fraca generalização fora-de-domínio já notada.

---

## 3. Estado da arte (2022–2025)

### 3.1 Sistemas operacionais físicos (a baseline obrigatória)
- **INPE — Risco de Fogo (RF v9):** índice de probabilidade de fogo a partir de tempo seco/precipitação/umidade, com versões **observada, prevista (dias) e futura (semanas)** via modelos numéricos (ETA 15 km, BRAMS, T213). É a referência nacional — e o que o projeto atualmente imita [inpe-rf].
- **Copernicus EFFIS / GEFF (ECMWF):** roda operacionalmente o **Fire Weather Index canadense** (+ NFDRS e McArthur), previsão determinística ~8 km e sazonal (SEAS5). Referência internacional [digiuseppe2024geff].

**Implicação:** o índice físico (FWI/RF) é a baseline contra a qual o ML precisa demonstrar ganho — avaliado sobre fogo observado.

### 3.2 Deep learning espaço-temporal (a fronteira metodológica)
- **Next Day Wildfire Spread (Huot et al., 2022):** benchmark que prevê a **máscara de fogo observada do dia seguinte**; consolidou CNN/U-Net e o paradigma "prever fogo observado, não índice" [huot2022nextday].
- **ConvLSTM / U-Net + atenção / Transformers:** dominam previsão de propagação e perigo, capturando dependência espaço-temporal conjunta; revisões recentes mostram ganho sobre baselines tabulares [chen2024review; zhou2025cnntransformer].
- **FireCastNet (2025):** "Earth-as-a-Graph" (codificação 3D convolucional + GNN tipo GraphCast); prevê **área queimada global até 6 meses à frente** sobre o **SeasFire cube** (59 variáveis, 0,25°, 2001–2021). Estado da arte sazonal [karasante2025firecastnet; alonso2025seasfire].

### 3.3 Específico da Amazônia/Brasil
- **LSTM + GRU para focos ativos na Amazônia (2024):** prevê acumulados mensais de **focos observados** (AQUA_M-T) — alvo observado [deandrade2024lstmgru].
- **ANN + MODIS NDVI (Amazônia):** ~90% de acurácia detectando áreas de alto risco a partir de vegetação observada.
- **Suscetibilidade com ML no Xingu/PA (2025):** confirma RF/SHAP como abordagem corrente regional.

### 3.4 Incerteza e modelos de fundação
- **Conformal prediction** em observação da Terra: conjuntos de predição com garantia estatística, agnósticos ao modelo; recomendados para fogo, onde o erro de omissão custa mais [mortier2024conformal]. Trabalho de 2025 padroniza métricas probabilísticas de fogo: **ECE, Brier, NLL** [spatialuq2025].
- **Prithvi-EO-2.0 (NASA/IBM, 2024, com colaboradores brasileiros):** *foundation model* geoespacial (ViT/HLS) que supera U-Net em **mapeamento de cicatriz de queimada** — referência para o lado de *monitoramento* [prithvi2024].

---

## 4. Mapeamento lacuna → estado da arte → proposta

| Lacuna atual | O que o estado da arte faz | Proposta |
|---|---|---|
| L1: alvo é índice do INPE; base *presence-only* | Prevê fogo observado (presença vs ausência amostrada) [huot2022; karasante2025; deandrade2024] | **P1** |
| L2: nowcast, sem horizonte | Previsão com lead time (NWP/sazonal) [digiuseppe2024; inpe-rf] | **P2** |
| L3: vazamento temporal | CV espaço-temporal em bloco [roberts2017] | **P2 / P3** |
| L4: sem estrutura espacial | ConvLSTM/U-Net/GNN [chen2024; karasante2025] | **P4** |
| Temperatura/clima estáticos | Vegetação e clima observados (NDVI, SMAP, ERA5) [alonso2025] | **P5** |
| Incerteza ad-hoc (gap top-2) | Conformal prediction; ECE/Brier [mortier2024; spatialuq2025] | **P6** |
| "Monitoramento" pouco desenvolvido | Segmentação por foundation model [prithvi2024] | **P7** |

---

## 5. Propostas detalhadas

### Prioridade ALTA

**P1 — Redefinir o alvo para fogo observado e construir a classe negativa** *(resolve L1)*
- **Fundamento:** todo o estado da arte prevê fenômeno observado [huot2022nextday; karasante2025firecastnet; deandrade2024lstmgru]. Como a base atual é *presence-only* (§2.1), o passo indispensável é gerar **ausências** — sem elas não existe problema de classificação fogo/não-fogo bem-posto.
- **Como:** (a) rotular presença por ocorrência observada — focos VIIRS/FIRMS (375 m) ou área queimada (MODIS MCD64A1 / **MapBiomas Fogo** [mapbiomasfogo], já no roadmap); (b) **gerar ausências** amostrando células/datas sem foco no mesmo domínio espaço-temporal (*background/pseudo-absence sampling*, padrão em modelagem de distribuição e em [huot2022nextday]), controlando o *prevalence ratio*. Tarefa-alvo: P(fogo na célula c no horizonte [t, t+h]).
- **Impacto:** transforma "surrogate do INPE" em "preditor de incêndio" que de fato distingue fogo de não-fogo; permite medir acerto contra a realidade e explicar casos como Pium-TO. Elimina a extrapolação fora-de-suporte que o app hoje faz ao consultar pontos arbitrários.

**P2 — Horizonte de previsão + validação espaço-temporal em bloco** *(resolve L2 e L3)*
- **Fundamento:** *blocked spatio-temporal CV* é padrão para evitar otimismo em dados geográficos [roberts2017cv; bergmeir2012cv].
- **Como:** features apenas com dados ≤ t; prever t+1/t+7/t+30; folds bloqueados no espaço **e** no tempo. Reaproveitar `validacao_temporal.py` como protocolo padrão e recomputar features causalmente.
- **Impacto:** elimina o viés de +12,8 pp; entrega previsão de fato.

**P3 — Benchmark contra o índice físico (FWI / INPE RF)** *(fecha o argumento científico)*
- **Fundamento:** GEFF/EFFIS tratam o FWI como baseline de referência [digiuseppe2024geff].
- **Como:** calcular o FWI canadense (T, UR, vento, chuva às 12h — já disponíveis via INMET/NASA POWER) e comparar ML vs FWI vs INPE RF **sobre fogo observado**.
- **Impacto:** demonstra (ou refuta) o valor agregado do ML — resultado mais citável do TCC.

### Prioridade MÉDIA

**P4 — Adicionar estrutura espacial** *(resolve L4)*
- **Fundamento:** ConvLSTM/U-Net/GNN superam baselines tabulares [chen2024review; karasante2025firecastnet].
- **Como (escalonável):** (a) leve — features de contexto de vizinhança (agregados das 8 células adjacentes) em RF/GBM; (b) completa — reamostrar para grade 0,25° e treinar ConvLSTM/U-Net, usando a organização do **SeasFire cube** [alonso2025seasfire] como referência.
- **Impacto:** reduz memorização de região; melhora generalização OOD.

**P5 — Variáveis dinâmicas de vegetação/combustível e clima real** *(Tier 2 do roadmap)*
- **Fundamento:** vegetação observada é preditor forte (ANN+MODIS Amazônia ~90%); SeasFire usa 59 variáveis incluindo vegetação [alonso2025seasfire].
- **Como:** NDVI/EVI (MODIS via GEE), umidade do solo (SMAP), **VPD e temperatura reais via ERA5**, substituindo a climatologia estática mensal por estado.
- **Impacto:** remove a aproximação de temperatura constante; agrega sinal de combustível.

**P6 — Incerteza calibrada com conformal prediction**
- **Fundamento:** [mortier2024conformal]; métricas ECE/Brier/NLL [spatialuq2025].
- **Como:** substituir a heurística de gap top-2 por **conjuntos conformais** com cobertura garantida; reportar ECE e Brier.
- **Impacto:** incerteza com garantia estatística, relevante onde omitir fogo custa mais que falso alarme.

### Prioridade BAIXA / exploratória

**P7 — Componente de monitoramento com foundation model**
- **Fundamento:** Prithvi-EO-2.0 supera U-Net em cicatriz de queimada [prithvi2024].
- **Como:** módulo de mapeamento near-real-time de área queimada (Sentinel-2/HLS) por segmentação, complementando o app preditivo e cobrindo o lado "monitoramento" do título.

---

## 6. Roadmap priorizado

| # | Proposta | Resolve | Esforço | Impacto | Dependências |
|---|---|---|---|---|---|
| P1 | Alvo = fogo observado | L1 | Médio | **Muito alto** | FIRMS/MapBiomas Fogo |
| P2 | Horizonte + CV em bloco | L2, L3 | Médio | **Muito alto** | P1 (ideal); `validacao_temporal.py` |
| P3 | Benchmark FWI/INPE RF ✅ implementado | tese | Baixo | Alto | P1 |
| P4 | Estrutura espacial | L4 | Alto | Alto | grade 0,25° |
| P5 | Clima gridado real + FWI ⚠️ testado: ganho nulo (mensal) | clima estático | Médio | ~nulo (mensal) | NASA POWER (rate-limited) |
| P6 | Conformal prediction | incerteza | Baixo | Médio | — |
| P7 | Monitoramento (Prithvi) | escopo | Alto | Médio | HLS/Sentinel-2 |

**Caminho mínimo recomendado para o TCC:** P1 → P2 → P3 (reposiciona cientificamente o trabalho com esforço médio), seguido de P5 e P6 (incrementos de alto valor e baixo/médio esforço).

---

## 6.1 Implementação realizada (01/06/2026) — P1 + P2 + P6

Foi construído um **pipeline novo e independente** (`scripts/previsao_ocorrencia/`, ver `README.md` da pasta) que implementa as três melhorias prioritárias, **sem remover** o pipeline original (mantido como *baseline* de comparação). Resumo do que mudou e dos resultados **reais** obtidos sobre os focos do INPE (2015–2023; 450.144 cell-months de células *fire-prone* 0,25°; prevalência do alvo 24,7%):

- **P1 — alvo = fogo observado + classe negativa.** O alvo deixou de ser a discretização do índice do INPE e passou a ser **ocorrência observada de foco no mês `t+1`**. Cell-months sem foco viram a **classe negativa** que faltava (a base deixou de ser *presence-only*). `RiscoFogo` virou apenas covariável.
- **P2 — previsão causal + validação em bloco.** Todas as features usam só dados ≤ t (lags de focos, somas móveis com `shift`, sazonalidade, climatologia da célula). Validação espaço-temporal em bloco [roberts2017cv]:
  - **Temporal (rolling-origin 2021–23):** PR-AUC **0,777** / ROC-AUC **0,904**, superando os baselines de **persistência** (0,449) e **climatologia** (0,715).
  - **Espacial (leave-region-out):** PR-AUC **0,759**, enquanto a climatologia colapsa (0,17–0,34) em regiões não vistas — evidência de generalização (contraria a Lacuna 4).
- **P6 — conformal prediction.** Conjuntos de predição classe-condicionais (LAC; Sadinle 2019) com **cobertura empírica 88,8%** (alvo 90%), 17,1% de predições sinalizadas como ambíguas; ECE 0,099, Brier 0,123. Substitui a heurística *gap top-2*.
- **P3 — benchmark contra índices físicos.** `fwi.py` implementa o **FWI canadense** completo [vanwagner1985fwi], validado contra os valores de referência publicados. Na tarefa de fogo observado em `t+1`, o **modelo de ML supera os índices físicos**: ML PR-AUC **0,777** vs índice INPE RiscoFogo 0,294 (dataset completo); e numa amostra de 100 células com **FWI real** (clima diário NASA POWER), ML 0,858 > **FWI 0,608** > RiscoFogo 0,398 (mesmas linhas). O FWI separa fisicamente bem (média 6,0 em meses com fogo vs 1,5 sem). É a demonstração de que o ML "agrega valor" sobre o índice que o pipeline original apenas reproduzia.

- **P5 — clima gridado real + FWI (testado; ganho nulo).** Coletamos clima diário real do NASA POWER por célula e computamos o FWI dia-a-dia (594/4.168 células — o NASA POWER aplicou *rate-limit*). Resultado, em validação em bloco no subconjunto com clima real: ML-base (só histórico de fogo + sazonalidade) PR-AUC **0,617** vs ML-gridded (+ clima real + FWI) PR-AUC **0,617** → **ganho +0,001 (nulo)**. **Achado:** na escala mensal/0,25° a ocorrência de fogo é dominada por **persistência + sazonalidade**; o histórico de fogo já absorve o sinal meteorológico. Redireciona o esforço para **granularidade sub-mensal** e **drivers antrópicos de ignição**.

- **Drivers antrópicos (testados; desmatamento é o único que agrega).** Após o P5 nulo, testamos pressão humana: (i) distância a cidades (estático) → ganho nulo; (ii) **desmatamento PRODES por célula/ano** (dinâmico, do WFS TerraBrasilis [prodes-terrabrasilis]) → ganho geral ainda pequeno (+0,003), **mas +0,025 PR-AUC no estrato de NOVA IGNIÇÃO** (células sem foco recente; base 0,206 → 0,231), onde a persistência é inútil. Confirma o mecanismo causal: o desmatamento precede a primeira queimada [aragao2018]. É o **único** dos três drivers testados (clima, acessibilidade, desmatamento) que agrega valor não-trivial — e exatamente no regime operacionalmente crítico.

**Síntese (limitação + direção, honesta):** na escala mensal/0,25° a ocorrência de fogo é dominada por **persistência + sazonalidade**; o histórico de fogo torna redundantes o clima (P5) e a acessibilidade estática. O problema difícil e relevante é a **nova ignição** (PR-AUC ~0,2), e ali o sinal útil é o **desmatamento recente** — não a meteorologia. Direções de maior retorno restantes: (a) desmatamento em granularidade fina/sub-mensal (DETER) focado na nova ignição; (b) granularidade temporal sub-mensal geral. Detalhes e tabelas em `scripts/previsao_ocorrencia/README.md`.

---

## 7. Novas referências (BibTeX) — para `references.bib`

Marcadores conforme `REFERENCIAS_TCC.md`: 📌 fundamental · 🔧 ferramenta · 🌱 domínio · 🧠 ML/explicabilidade.

### 🌱 INPE — Risco de Fogo (metodologia) [inpe-rf]
Usar em: §2.1 (definição do alvo), discussão da circularidade.
```bibtex
@techreport{inpe-rf,
  title       = {Risco de Fogo: Metodologia do C{\'a}lculo --- Descri{\c{c}}{\~a}o sucinta da Vers{\~a}o 9},
  author      = {{Instituto Nacional de Pesquisas Espaciais (INPE)}},
  institution = {Programa Queimadas, INPE},
  year        = {2013},
  note        = {Dispon{\'i}vel em: https://dataserver-coids.inpe.br/queimadas/}
}
```

### 📌🌱 Huot et al. (2022) — Next Day Wildfire Spread [huot2022nextday]
Usar em: §3.2, P1 (alvo = fogo observado).
```bibtex
@article{huot2022nextday,
  title   = {Next Day Wildfire Spread: A Machine Learning Dataset to Predict Wildfire Spreading From Remote-Sensing Data},
  author  = {Huot, Fantine and Hu, R. Lily and Goyal, Nita and Sankar, Tharun and Ihme, Matthias and Chen, Yi-Fan},
  journal = {IEEE Transactions on Geoscience and Remote Sensing},
  volume  = {60},
  pages   = {1--13},
  year    = {2022},
  doi     = {10.1109/TGRS.2022.3192974}
}
```

### 📌🌱🧠 Karasante et al. (2025) — FireCastNet [karasante2025firecastnet]
Usar em: §3.2, P4 (estrutura espacial / GNN).
```bibtex
@article{karasante2025firecastnet,
  title   = {FireCastNet: earth-as-a-graph for seasonal fire prediction},
  author  = {Karasante, Ioannis and Prapas, Ioannis and Kondylatos, Spyros and others},
  journal = {Scientific Reports},
  volume  = {15},
  year    = {2025},
  doi     = {10.1038/s41598-025-30645-7}
}
```

### 🌱🔧 Alonso et al. (2025) — SeasFire cube [alonso2025seasfire]
Usar em: §3.2, P4, P5 (variáveis dinâmicas; datacube de referência).
```bibtex
@article{alonso2025seasfire,
  title   = {SeasFire cube: a multivariate dataset for global wildfire modeling},
  author  = {Alonso, Lazaro and Prapas, Ioannis and Kondylatos, Spyros and others},
  journal = {Scientific Data},
  volume  = {12},
  year    = {2025},
  doi     = {10.1038/s41597-025-04546-3}
}
```

### 📌🌱 Di Giuseppe et al. (2024) — Global seasonal prediction of fire danger (GEFF/FWI) [digiuseppe2024geff]
Usar em: §3.1, P3 (baseline FWI).
```bibtex
@article{digiuseppe2024geff,
  title   = {Global seasonal prediction of fire danger},
  author  = {Di Giuseppe, Francesca and Vitolo, Claudia and others},
  journal = {Scientific Data},
  volume  = {11},
  year    = {2024},
  doi     = {10.1038/s41597-024-02948-3}
}
```

### 🌱🧠 de Andrade et al. (2024) — LSTM+GRU para focos na Amazônia [deandrade2024lstmgru]
Usar em: §3.3, P1 (alvo observado), P2 (séries temporais).
```bibtex
@misc{deandrade2024lstmgru,
  title         = {Neural Networks with LSTM and GRU in Modeling Active Fires in the Amazon},
  author        = {de Andrade, Ramon Tavares and others},
  year          = {2024},
  eprint        = {2409.02681},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG}
}
```

### 🧠 Roberts et al. (2017) — CV espaço-temporal em bloco [roberts2017cv]
Usar em: §2.3, P2 (validação honesta).
```bibtex
@article{roberts2017cv,
  title   = {Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure},
  author  = {Roberts, David R. and Bahn, Volker and Ciuti, Simone and others},
  journal = {Ecography},
  volume  = {40},
  number  = {8},
  pages   = {913--929},
  year    = {2017},
  doi     = {10.1111/ecog.02881}
}
```

### 🧠 Mortier et al. (2024) — Conformal prediction em observação da Terra [mortier2024conformal]
Usar em: §3.4, P6 (incerteza calibrada).
```bibtex
@article{mortier2024conformal,
  title   = {Uncertainty quantification for probabilistic machine learning in earth observation using conformal prediction},
  author  = {Mortier, Thomas and others},
  journal = {Scientific Reports},
  volume  = {14},
  year    = {2024},
  doi     = {10.1038/s41598-024-65954-w}
}
```

### 🌱🧠 Spatial UQ in Wildfire Forecasting (2025) [spatialuq2025]
Usar em: §3.4, P6 (métricas ECE/Brier/NLL).
```bibtex
@misc{spatialuq2025,
  title         = {Spatial Uncertainty Quantification in Wildfire Forecasting for Climate-Resilient Emergency Planning},
  author        = {Anonymous},
  year          = {2025},
  eprint        = {2510.09666},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG}
}
```

### 🔧🧠 Prithvi-EO-2.0 (2024) — Foundation model geoespacial [prithvi2024]
Usar em: §3.4, P7 (monitoramento por segmentação).
```bibtex
@misc{prithvi2024,
  title         = {Prithvi-EO-2.0: A Versatile Multi-Temporal Foundation Model for Earth Observation Applications},
  author        = {{NASA-IBM Prithvi-EO Team}},
  year          = {2024},
  eprint        = {2412.02732},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV}
}
```

### 🧠 Sadinle, Lei & Wasserman (2019) — LAC / conjuntos classe-condicionais [sadinle2019lac]
Usar em: §6.1, P6 (implementação conformal).
```bibtex
@article{sadinle2019lac,
  title   = {Least Ambiguous Set-Valued Classifiers With Bounded Error Levels},
  author  = {Sadinle, Mauricio and Lei, Jing and Wasserman, Larry},
  journal = {Journal of the American Statistical Association},
  volume  = {114},
  number  = {525},
  pages   = {223--234},
  year    = {2019},
  doi     = {10.1080/01621459.2017.1395341}
}
```

### 🌱 Van Wagner & Pickett (1985) — Sistema FWI canadense [vanwagner1985fwi]
Usar em: §3.1, §6.1, P3 (implementação do FWI em `fwi.py`).
```bibtex
@techreport{vanwagner1985fwi,
  title       = {Equations and FORTRAN Program for the Canadian Forest Fire Weather Index System},
  author      = {Van Wagner, C. E. and Pickett, T. L.},
  institution = {Canadian Forestry Service},
  type        = {Forestry Technical Report},
  number      = {33},
  year        = {1985},
  address     = {Ottawa}
}
```

### 🌱🧠 Chen et al. (2024) — Review ML/DL para propagação de incêndio [chen2024review]
Usar em: §3.2, P4.
```bibtex
@article{chen2024review,
  title   = {Machine Learning and Deep Learning for Wildfire Spread Prediction: A Review},
  author  = {Chen, and others},
  journal = {Fire},
  volume  = {7},
  number  = {12},
  pages   = {482},
  year    = {2024},
  doi     = {10.3390/fire7120482}
}
```

### 🌱🧠 Zhou et al. (2025) — CNN vs Transformer em propagação [zhou2025cnntransformer]
Usar em: §3.2.
```bibtex
@article{zhou2025cnntransformer,
  title   = {Comparative and Interpretative Analysis of CNN and Transformer Models in Predicting Wildfire Spread Using Remote Sensing Data},
  author  = {Zhou, and others},
  journal = {Journal of Geophysical Research: Machine Learning and Computation},
  year    = {2025},
  doi     = {10.1029/2024JH000409}
}
```

### 🌱 PRODES / TerraBrasilis — desmatamento Amazônia Legal [prodes-terrabrasilis]
Usar em: §6.1, driver antrópico de desmatamento (`coletar_desmatamento.py`).
```bibtex
@misc{prodes-terrabrasilis,
  title        = {PRODES --- Monitoramento do Desmatamento da Floresta Amazônica Brasileira por Satélite},
  author       = {{Instituto Nacional de Pesquisas Espaciais (INPE)}},
  howpublished = {TerraBrasilis. http://terrabrasilis.dpi.inpe.br/},
  note         = {Camada WFS prodes-legal-amz:yearly_deforestation (incrementos anuais)}
}
```

### 🌱 MapBiomas Fogo (coleção) [mapbiomasfogo]
Usar em: P1 (rótulo de área queimada).
```bibtex
@misc{mapbiomasfogo,
  title        = {MapBiomas Fogo --- Mapeamento de {\'a}rea queimada e cicatrizes de fogo no Brasil},
  author       = {{Projeto MapBiomas}},
  howpublished = {https://brasil.mapbiomas.org/},
  note         = {Cole{\c{c}}{\~a}o de {\'a}rea queimada (1985--presente)}
}
```

---

> **Notas de verificação:** DOIs de periódicos Nature (`10.1038/<id>`), IEEE TGRS e Ecography foram derivados dos identificadores das publicações; confirmar autoria completa e número de volume/página de `chen2024review`, `zhou2025cnntransformer` e `spatialuq2025` (pré-prints/em edição) antes da submissão final. As referências já existentes no projeto (Seager 2015, Forests 2024, Quesada-Ruiz 2025, etc.) permanecem válidas e complementam esta seção.
