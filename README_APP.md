# Aplicação Interativa de Previsão de Risco de Incêndio - Amazônia Legal

Aplicação web Flask para visualização interativa de previsões de risco de incêndio na Amazônia Legal.

## Requisitos

- Python 3.8+
- Dependências listadas em `requirements.txt`

## Instalação

1. Instalar dependências:
```bash
pip install -r requirements.txt
```

2. Verificar se os modelos treinados estão disponíveis em `modelos/`

3. Verificar se os shapefiles da Amazônia Legal estão disponíveis:
   - `Limites_Amazonia_Legal_2024_shp/Limites_Amazonia_Legal_2024.shp` (preferencial)
   - OU `brazilian_legal_amazon/brazilian_legal_amazon.shp` (alternativa)

## Configuração

A chave API do HG Weather já está configurada no arquivo `scripts/config.py` como `14068910`.

Se precisar alterar, edite o arquivo `scripts/config.py` ou defina a variável de ambiente:
```bash
export HG_WEATHER_API_KEY="sua_chave_aqui"
```

## Uso

1. Iniciar o servidor Flask:
```bash
cd scripts
python app_map_interativo.py
```

Ou de qualquer diretório:
```bash
python scripts/app_map_interativo.py
```

2. Abrir no navegador:
```
http://localhost:5000
```

3. Usar o mapa:
   - Clique em qualquer ponto no mapa dentro da Amazônia Legal
   - Aguarde o processamento da previsão
   - Veja o resultado com marcador colorido:
     - Verde: Risco Baixo
     - Laranja: Risco Moderado
     - Vermelho: Risco Muito Alto
   - Selecione um modelo diferente no dropdown (canto superior direito)

## Funcionalidades

- **Mapa interativo**: Visualização da Amazônia Legal com limites oficiais
- **Clique para prever**: Clique em qualquer coordenada para obter previsão
- **Dados climáticos em tempo real**: Integração com HG Weather e NASA POWER (umidade, precipitação, temperatura)
- **Múltiplos modelos**: Selecione entre modelos treinados disponíveis (default: `ensemble_stacking_gbm`)
- **Geocodificação reversa**: Identificação automática de Estado/Município
- **Fallback histórico**: Usa lookup espaço-sazonal (Tier 1) se API falhar
- **Probabilidades por classe**: Exibe confiança da classe predita e distribuição das classes
- **Legenda visual de risco**: Cores no painel (baixo/moderado/muito alto)
- **Feedback de status e erro**: Indicador de processamento e mensagens de falha no painel
- **Painel lateral com 6 cards**: Previsão, Por que esse risco? (top-6 features), Dados climáticos atuais, Índices de seca (SPI/KBDI/VPD), Acumulados + anomalia, Histórico de focos
- **Thresholds calibrados**: opção `estrategia_thresholds` para usar `f1_macro` (default), `f1_moderado` ou `argmax`; UI deixa explícita qual regra foi aplicada

## Endpoints da API

- `GET /`: Página principal com mapa interativo
- `GET /api/models`: Lista modelos disponíveis (inclui `metricas` com acurácia / F1-macro de cada modelo)
- `POST /api/climate`: Obtém dados climáticos para coordenadas
- `POST /api/predict`: Faz previsão de risco de incêndio. Aceita `estrategia_thresholds` ∈ `{f1_macro, f1_moderado, argmax}` (default `f1_macro`). A resposta inclui `dados_usados` (clima efetivo + Tier 1 + FRP + campos de apoio: vento ma7 + fonte, Tmax MERRA-7d, proxy `FWI_fire_weather_proxy`, flags de incerteza), `contexto_historico` (climatologia + **metadados INMET**: estação, distância, cobertura, `inmet_representatividade`, `inmet_busca_raio_km`, espelhos MERRA de precip/dias secos, ventos MERRA/INMET), `incerteza_operacional` (ex.: gap entre as duas classes mais prováveis), `explicacao` (top-6 features), `thresholds_aplicados` e `fonte_clima_detalhe` / `log_prediction` enriquecidos para auditoria.
- `POST /api/explain`: Endpoint dedicado para gerar a explicação local a partir de um vetor de features (útil para testes e usos isolados).

> **Modelo default**: a aplicação carrega `ensemble_stacking_gbm` (Stacking com XGB/LGBM otimizados via Optuna + meta-learner GradientBoosting + features Tier 1). Acurácia hold-out **84,61 %**, F1-macro **0,7995**, F1-Moderado **0,6296** (ver `DOCUMENTO_METODOLOGIA_E_RESULTADOS.md` §4.10).

## Mapa estático (HTML para relatório)

Gera um arquivo HTML com Folium usando o mesmo pré-processador dos modelos atuais:

```bash
cd scripts
python gerar_mapa.py --amostra-dataset ../base_de_dados_com_historico.csv --amostra 500 --modelo ensemble_stacking
```

- `--modelo` opcional; se omitido, usa `ensemble_stacking` ou o próximo da lista de preferência.
- `--saida` para definir o caminho do HTML (padrão: `scripts/mapa_risco_amazonia_com_previsoes.html`).
- Documentação dos avanços: `scripts/AVANCOS_TREINAMENTO_RECENTES.md`.

## Estrutura

- `scripts/app_map_interativo.py`: Aplicação Flask principal (UI explicável + endpoints `/api/predict`, `/api/explain`, `/api/models`)
- `scripts/feature_lookup.py`: Lookup espaço-sazonal das 24 features Tier 1 (medianas por célula 0,25° / Estado-Mês / Mês global)
- `scripts/explainer.py`: Explicabilidade local rápida (contribuição = importância × z-score × sinal físico)
- `scripts/climate_api.py`: NASA POWER (real-time) + fusão INMET (WIS2/ZIP; modo hierárquico opcional)
- `scripts/inmet_api.py`: Cliente INMET (ZIP histórico + WIS2) usado pela fusão
- `scripts/operational_uncertainty.py`: Heurísticas de incerteza operacional e proxy de *fire weather* (painel/API)
- `scripts/frp_api.py`: Integração com NASA FIRMS (FRP em tempo quase-real)
- `scripts/features_avancadas.py`: 24 features físico-climáticas Tier 1 (SPI, KBDI, anomalias, lags) — usado offline
- `scripts/config.py`: Configurações da aplicação
- `scripts/pre_processor.py`: Pré-processador (`ColumnTransformer` com imputação, normalização e OHE)
- `modelos/prediction_thresholds.json`: Thresholds calibrados (gerados por `scripts/ajustar_threshold.py`)
- `modelos/relatorios/shap_per_class.json`: Importâncias SHAP médias por classe (Baixo/Moderado/Muito Alto)
- `modelos/relatorios/validacao_temporal.json`: Métricas rolling-origin (validação temporal estrita)
- `requirements.txt`: Dependências do projeto (inclui `shap`, `optuna`, `xgboost`, `lightgbm`, `catboost`, `imbalanced-learn`)

## Notas

- Enriquecimento de umidade via NASA POWER e uso de múltiplas chaves: ver `scripts/AVISO_MULTIPLAS_CHAVES_API.md` e `scripts/GUIA_CHAVE_API_NASA.md`.
- Resumo do último treino e métricas: `DATASET_VERSION.md` e `modelos/relatorios/`.

- A aplicação roda localmente na porta 5000 por padrão
- O shapefile da Amazônia Legal é carregado automaticamente se disponível
- Se geopandas não estiver instalado, usa bounding box aproximada
- Dados climáticos são obtidos em tempo real via HG Weather API
- Em caso de falha da API, usa dados históricos do dataset

## Troubleshooting

**Erro ao carregar shapefile:**
- Instale geopandas: `pip install geopandas`
- Ou verifique se os shapefiles estão nos caminhos corretos

**Erro ao obter dados climáticos:**
- Verifique sua conexão com a internet
- A API usa fallback para dados históricos automaticamente

**Modelo não encontrado:**
- Execute o treinamento primeiro: `python scripts/treinamento_modelo.py`

**Treinar ensembles (recomendado):**
- `python scripts/treinamento_modelo.py --models ensemble_voting_soft ensemble_stacking --dataset base_de_dados_com_historico.csv`
- Opcional para incluir XGBoost nos ensembles: `pip install xgboost`

**Porta já em uso:**
- Altere a porta no final de `app_map_interativo.py`: `app.run(port=5001)`


