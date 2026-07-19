# Versão do dataset (rastreio de treino)

| Campo | Valor |
|--------|--------|
| **Dataset recomendado (Tier 1, features físico-climáticas)** | `base_de_dados_enriquecido.csv` (gerado por `scripts/features_avancadas.py` a partir do CSV com histórico) |
| **Dataset intermediário (features de histórico)** | `base_de_dados_com_historico.csv` (gerado por `scripts/adicionar_historico_incendios.py`) |
| **Dataset com umidade (pós-NASA, opcional)** | `base_de_dados_com_umidade.csv` *(parcial — ver §1 do `CHECKLIST_OBJETIVO_FINAL.md`)* |
| **Dataset bruto padrão no código** | `base_de_dados.csv` |
| **Última atualização deste registro** | 2026-05-10 |

## Umidade relativa (NASA POWER) e pseudo-label

Números canônicos (gerados em `modelos/relatorios/pseudo_label_umidade.json` por `scripts/pseudo_label_umidade.py`):

| Métrica | Valor |
|--------|--------|
| Registros com RH **observada** (NASA POWER) | 170 465 |
| Registros **sem** label antes do pseudo | 763 489 |
| Universo considerado (soma) | **933 954** linhas |
| Fração com umidade real | **18,25 %** (0,18252) |
| Após pseudo-label LGBM | cobertura **100 %** em `base_de_dados_umidade_pseudo.csv` |
| Coluna de auditoria | `Umidade_origem` ∈ {`nasa_power`, `pseudo_label_lgbm`} |

O modelo de classificação final (`ensemble_stacking_gbm`) documentado neste repositório foi treinado **sem** usar a coluna `Umidade` como *feature*; o CSV acima existe para retreinos comparativos e para o texto do TCC.

## Último treino registrado — Tier 1 (features físico-climáticas)

| Campo | Valor |
|--------|--------|
| **Data** | 2026-05-10 |
| **Dataset** | `base_de_dados_enriquecido.csv` (~924 306 linhas após limpeza, 46 colunas brutas) |
| **Decisão metodológica** | Substituição da feature `Umidade` (NASA POWER, ~16 dias de enriquecimento) por proxies derivados — ver `DOCUMENTO_METODOLOGIA_E_RESULTADOS.md` |
| **Features adicionadas (24)** | MAs estendidas (14/30/90d), precipitação acumulada (30/90/180/365d), SPI-1/3/6, anomalia de precipitação, KBDI proxy, temp climatológica, aridez de De Martonne, VPD proxy, histórico de fogo (90/180/365d), média FRP célula 30d, dias secos 90d |
| **Comando** | `python scripts/treinamento_modelo.py --models random_forest_balanced catboost_classifier --no-cv --dataset base_de_dados_enriquecido.csv` |
| **Resultados (RF + CatBoost + Stacking)** | `random_forest_balanced` **82,93%** acc / 78,66% F1-macro · `catboost_classifier` 71,73% acc / 68,14% F1-macro · **`ensemble_stacking` 83,60% acc / 78,92% F1-macro / 61,68% F1-Moderado** |
| **Ganho RF vs treino anterior** | **+4,47 pp accuracy** · **+4,29 pp F1-macro** (78,46% → 82,93%) |
| **Ganho Stacking vs treino anterior** | **+3,11 pp accuracy** · **+4,00 pp F1-macro** · **+6,70 pp F1-Moderado** (80,49% → 83,60%) |
| **Tempo de treino Stacking Tier 1** | 116 minutos (≈ 2 h) |
| **Validação cruzada** | Desligada (`--no-cv`); métricas no hold-out 20% |
| **Artefatos** | `modelos/random_forest_balanced.pkl`, `modelos/catboost_classifier.pkl`, `modelos/relatorios/*_metrics.json` |

### Treino anterior (referência — sem Tier 1)

| Campo | Valor |
|--------|--------|
| **Data** | 2026-04-16 |
| **Dataset** | `base_de_dados_com_historico.csv` (~924 306 linhas) |
| **Acurácia (teste)** | `random_forest_balanced` 78,46% · `ensemble_stacking` **80,49%** · `xgboost_classifier` 74,85% · `lightgbm_classifier` 70,62% |

### Limitação (médias móveis e features rolling)

As features `Precipitacao_ma7/14/30/90`, `DiaSemChuva_ma7/14/30/90`, `Precipitacao_acum_*`, `Dias_Secos_90d` e `Incendios_Ultimos_*_Dias` usam janela deslizante **no mesmo conjunto** antes do split, com agrupamento por célula espacial 0,25° (≈27 km). Para o TCC, registrar como possível **viés otimista** frente a previsão estritamente causal. Para rigor maior, calcular janelas só com dados anteriores ao instante do registro e/ou usar validação temporal (TimeSeriesSplit). O mesmo aplica-se a SPI e à climatologia in-sample da anomalia de precipitação.

## Como gerar e treinar com o dataset Tier 1

```bash
# 1) Gerar CSV enriquecido (40 segundos sobre 933 954 linhas):
python scripts/features_avancadas.py \
    --input base_de_dados_com_historico.csv \
    --output base_de_dados_enriquecido.csv

# 2) Treinar modelos sobre o CSV enriquecido:
python scripts/treinamento_modelo.py \
    --models random_forest_balanced catboost_classifier ensemble_stacking \
    --no-cv \
    --dataset base_de_dados_enriquecido.csv
```

### Reverter para o pipeline antigo

```bash
python scripts/treinamento_modelo.py --dataset base_de_dados_com_historico.csv --no-cv --models ensemble_stacking
```

`carregar_dados.py` detecta automaticamente se o CSV já contém as 24 features avançadas e pula a recomputação. Caso contrário, as features são geradas em memória (custo ~40s extra por treino).

## Notas

- Colunas de histórico (`Incendios_Ultimos_*`) e features avançadas (`SPI_*`, `KBDI_proxy`, `Precipitacao_acum_*`, etc.) são incluídas automaticamente em `carregar_dados.py` quando presentes no CSV.
- Registrar aqui a data e o nome do CSV após cada geração importante do dataset.
