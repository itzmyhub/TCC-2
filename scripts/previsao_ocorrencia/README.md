# Pipeline de previsão de OCORRÊNCIA de fogo (P1 + P2 + P6)

> **Implementado em:** 01/06/2026. Fundamentação e contexto: `../../ESTADO_DA_ARTE_E_PROPOSTAS_MELHORIA.md`.

Pipeline **novo e independente**, que implementa as três melhorias prioritárias do
documento de estado da arte. **Não substitui** o pipeline original (classificação do
índice RiscoFogo) — ele permanece como *baseline* para comparação no TCC. A diferença
de fundo: aqui o alvo é **fogo observado**, com **classe negativa real**, em um problema
de **previsão** (horizonte explícito) validado **sem vazamento**.

## O que cada melhoria resolve

| | Pipeline original | Pipeline novo |
|---|---|---|
| **Alvo** | discretização do índice `RiscoFogo` do INPE | **ocorrência observada** de foco no mês `t+H` (P1) |
| **Classe negativa** | inexistente (*presence-only*) | cell-months sem foco = **ausência** (P1) |
| **Natureza** | *nowcast* (presente) | **previsão** com horizonte (P2) |
| **Features** | janelas calculadas sobre o dataset todo (vazamento) | **causais**, só dados ≤ t (P2) |
| **Validação** | split aleatório (otimista) | **blocos espaço-temporais** [roberts2017cv] (P2) |
| **Incerteza** | heurística *gap top-2* | **conformal** com cobertura garantida (P6) [mortier2024] |

## Como rodar

```bash
# 1) Constrói o dataset de ocorrência (célula 0,25° × mês) a partir dos focos do INPE
python scripts/previsao_ocorrencia/construir_ocorrencia.py --horizonte 1

# 2) Treino + validação espaço-temporal em bloco + baselines (persistência, climatologia)
python scripts/previsao_ocorrencia/treinar_validar.py

# 3) Conformal prediction + métricas calibradas (ECE, Brier)
python scripts/previsao_ocorrencia/conformal.py --alpha 0.1
```

Saídas em `modelos/relatorios/`: `previsao_ocorrencia_validacao.json`,
`previsao_ocorrencia_conformal.json`. Dataset: `dataset_ocorrencia_mensal.csv` (raiz).

## Resultados obtidos (dados reais 2015–2023, 450.144 cell-months, prevalência 24,7%)

**P2 — Validação em bloco** (modelo: `HistGradientBoostingClassifier`):

| Validação | Modelo PR-AUC | Modelo ROC-AUC | Baseline persistência | Baseline climatologia |
|---|---|---|---|---|
| Temporal (rolling-origin 2021–23) | **0,777** | 0,904 | 0,449 | 0,715 |
| Espacial (leave-region-out, 5 blocos) | **0,759** | 0,895 | — | 0,167–0,335 |

- O modelo **supera os dois baselines** — condição necessária para "agregar valor" sobre o índice físico/climatológico (argumento central do TCC).
- Na validação **espacial**, a climatologia colapsa (PR-AUC 0,17–0,34) ao prever regiões nunca vistas, enquanto o modelo mantém 0,759 — evidência de que aprende **dinâmica transferível** (histórico de fogo + sazonalidade), não memorização de taxa-base local. Contrasta diretamente com a Lacuna 4 (memorização de região) do pipeline original.

**P6 — Conformal prediction** (cobertura alvo 90%, calib 2022 → teste 2023):

| Métrica | Valor |
|---|---|
| Cobertura marginal | 0,888 |
| Cobertura por classe (não-fogo / fogo) | 0,886 / 0,896 |
| Tamanho médio do conjunto | 1,171 |
| Singletons (confiantes) / ambíguos `{0,1}` / vazios | 0,829 / 0,171 / 0,000 |
| ECE / Brier | 0,099 / 0,123 |

- A cobertura empírica (~88,8%) fica próxima do alvo (90%); a leve subcobertura é esperada sob deriva temporal (conformal pressupõe permutabilidade — calibramos no ano imediatamente anterior para mitigar).
- 17,1% das predições são marcadas como **ambíguas** com base estatística — substituto principiado para o *gap top-2*.

## P3 — Benchmark contra índices físicos (FWI canadense e Risco de Fogo INPE)

`fwi.py` implementa o **FWI canadense** completo (FFMC/DMC/DC/ISI/BUI/FWI) de
van Wagner & Pickett (1985), validado por `self_test()` contra os valores de
referência publicados (FFMC 87,69 · DMC 8,55 · DC 19,01 · ISI 10,85 · BUI 8,49 ·
FWI 10,10). `benchmark_fwi.py` compara, na tarefa de fogo observado em `t+1`:

```bash
python scripts/previsao_ocorrencia/fwi.py                       # auto-teste
python scripts/previsao_ocorrencia/benchmark_fwi.py --amostra_fwi 100 --ano_fwi 2023
```

| Preditor | PR-AUC | ROC-AUC | Observação |
|---|---|---|---|
| **Modelo ML** (dataset completo, blocos temporais) | **0,777** | 0,904 | |
| Índice INPE RiscoFogo (como forecast t+1) | 0,294 | 0,553 | handicap: clima das ausências é climatologia |
| **Amostra 100 células, 2023 — ML** | **0,858** | — | mesmas linhas que abaixo |
| → **FWI canadense REAL** (NASA POWER diário) | 0,608 | 0,781 | computado dia-a-dia, justo p/ todas as células |
| → Índice INPE RiscoFogo | 0,398 | — | |

- **O modelo de ML supera o índice físico** (FWI e RiscoFogo) na previsão de fogo
  observado — o resultado central do TCC: o ML "agrega valor" sobre o índice que o
  pipeline original apenas reproduzia.
- O **FWI real é fisicamente coerente**: média **6,0** em meses com fogo vs **1,5**
  sem fogo, e PR-AUC 0,608 (bem acima da prevalência), validando a implementação.
- **Caveat de justiça:** na Parte A o RiscoFogo das ausências é climatologia (o que
  o penaliza); a Parte B computa FWI fresco do NASA POWER para *todas* as células
  amostradas (fogo e não-fogo), sendo a comparação justa — e ainda assim o ML vence.
- O FWI real para o dataset completo exige clima diário gridado para todas as
  ~4.168 células (melhoria **P5**); aqui demonstramos em amostra (cache em
  `.cache_umidade/nasa_daily/`).

## P5 — Clima gridado real + FWI (resultado: ganho nulo — achado relevante)

`coletar_clima_gridded.py` busca o clima diário real (NASA POWER) por célula e
computa o FWI dia-a-dia, agregando ao mês; `montar_dataset_gridded.py` funde no
dataset; `validar_gridded.py` compara ML-base vs ML-com-clima-real.

```bash
python scripts/previsao_ocorrencia/coletar_clima_gridded.py --workers 3   # resumível
python scripts/previsao_ocorrencia/montar_dataset_gridded.py
python scripts/previsao_ocorrencia/validar_gridded.py
```

**Resultado (subconjunto de 594 células com clima real, 64.152 cell-months,
prevalência 0,327; validação rolling-origin 2021–23):**

| Preditor (mesmas linhas) | PR-AUC | ROC-AUC |
|---|---|---|
| ML **base** (só histórico de fogo + sazonalidade) | 0,617 | 0,798 |
| ML **gridded** (+ clima real + FWI) | 0,617 | 0,796 |
| FWI real sozinho | 0,340 | 0,560 |
| INPE RiscoFogo | 0,290 | — |

- **Ganho do clima real + FWI sobre o histórico-só: +0,001 PR-AUC (nulo).**
- **Interpretação (achado científico):** na escala **mensal / 0,25°**, a ocorrência
  de fogo é governada por **persistência espaço-temporal + sazonalidade**; o
  histórico de fogo defasado já absorve implicitamente o sinal meteorológico
  (clima → fogo → histórico). Adicionar reanálise climática e o FWI operacional
  **não melhora** a previsão mensal. Isso redireciona o esforço: o ganho deve vir
  de (a) **granularidade sub-mensal** (onde a dinâmica do tempo importa) e/ou
  (b) **drivers antrópicos de ignição** (desmatamento, uso do solo).

**Caveats (honestos):**
1. **Cobertura parcial:** o NASA POWER aplicou *rate-limit* (HTTP 429) sob alta
   concorrência; coletamos 594/4.168 células (~14%). O coletor agora é gentil
   (3 workers + backoff respeitando `Retry-After`) e **resumível** — pode completar
   a grade ao longo do tempo, mas dado o ganho nulo não é prioritário.
2. **Subconjunto enviesado** para células de alta atividade (prevalência 0,327 vs
   0,247 do dataset completo) — em células marginais o clima poderia discriminar
   mais; testar isso exige a grade completa.
3. **Agregação mensal** do FWI pode diluir picos diários — daí a hipótese de que a
   granularidade sub-mensal seja onde o clima passa a importar.

## Drivers antrópicos — proxy leve (distância a cidades) + achado-chave

`drivers_antropicos.py` adiciona `dist_cidade_km` (distância à cidade mais próxima,
Natural Earth, proxy de acessibilidade/ignição humana — download estático único,
sem rate-limit). `validar_antropico.py` mede o ganho (temporal + espacial em bloco).

```bash
python scripts/previsao_ocorrencia/drivers_antropicos.py
python scripts/previsao_ocorrencia/validar_antropico.py
```

**Resultado:** ganho do proxy antrópico sobre o histórico = **nulo** (temporal
Δ=+0,001; espacial Δ=−0,002). Mesma causa do P5: o histórico de fogo já absorve o
padrão espacial de pressão humana.

**Achado-chave (análise estratificada, teste 2023):**

| Estrato | n | prevalência | base PR-AUC | ROC-AUC |
|---|---|---|---|---|
| **Com** histórico recente (foco nos últimos 12 m) | 39.549 | 0,316 | **0,805** | 0,891 |
| **Sem** histórico recente — **nova ignição** | 10.467 | 0,054 | **0,230** | 0,857 |

- O modelo é **excelente onde o fogo recorre** (persistência) e **fraco onde o fogo
  é novo** (PR-AUC 0,23) — e a nova ignição é justamente o caso crítico para alerta
  precoce. O proxy antrópico coarse deu só +0,006 nesse estrato, mas na direção certa.
- **Conclusão metodológica:** na escala mensal, persistência domina; o valor de
  qualquer covariável (clima, acessibilidade) só pode aparecer no regime de **nova
  ignição**, e proxies estáticos coarse não bastam. O teste decisivo é a fonte
  **dinâmica** — desmatamento recente (PRODES/DETER) e uso do solo (MapBiomas) —,
  que precede a *primeira* queimada de uma célula e não é redundante com o histórico.

## Driver antrópico DINÂMICO — desmatamento PRODES (o experimento decisivo)

`coletar_desmatamento.py` baixa do WFS do TerraBrasilis (`prodes-legal-amz:yearly_deforestation`)
os polígonos anuais de desmatamento, computa o centróide → célula 0,25° e soma a área
por (célula, ano) — 87.118 km² em 2014–2023, ~4 min, ~80 requisições (sem rate-limit).
`validar_desmatamento.py` cria features causais (`defor_lag1/lag2/cum3` = desmatamento
nos anos *anteriores*, estritamente ≤ t) e mede o ganho.

```bash
python scripts/previsao_ocorrencia/coletar_desmatamento.py
python scripts/previsao_ocorrencia/validar_desmatamento.py
```

**Resultado (validação temporal em bloco, média 2021–23):**

| Estrato | base PR-AUC | +desmatamento | Δ |
|---|---|---|---|
| Geral | 0,777 | 0,781 | +0,003 |
| Recorrente (foco nos últimos 12 m) | 0,791 | 0,794 | +0,003 |
| **Nova ignição (sem foco recente)** | **0,206** | **0,231** | **+0,025** |

- **O desmatamento é o ÚNICO driver, entre todos os testados (clima/P5, distância-a-cidade,
  desmatamento), que agrega valor não-trivial** — e o faz **exatamente no regime de nova
  ignição** (até +0,040 em 2022), onde a persistência do histórico de fogo é inútil.
- Confirma o **mecanismo causal**: o desmatamento **precede a primeira queimada** de uma
  célula (limpeza de área recém-desmatada) — informação que o histórico de fogo, por
  definição, não tem. Coerente com o fogo amazônico ser majoritariamente antrópico
  [aragao2018]. Fonte: PRODES/TerraBrasilis [prodes-terrabrasilis].
- **Magnitude honesta:** o ganho é modesto em valor absoluto porque a nova ignição é
  rara (prevalência 3–5%) e intrinsecamente difícil; mas +0,025 PR-AUC (~+12% relativo)
  no estrato decisão-relevante é real e consistente entre anos.

### Síntese dos três experimentos de covariáveis (P5 + antrópicos)

| Driver | Tipo | Ganho geral | Ganho nova ignição | Veredito |
|---|---|---|---|---|
| Clima real + FWI (P5) | dinâmico, meteorológico | +0,001 | — | redundante c/ histórico |
| Distância a cidade | estático, antrópico | +0,001 | ~+0,006 | redundante / coarse |
| **Desmatamento PRODES** | **dinâmico, antrópico** | +0,003 | **+0,025** | **agrega na nova ignição** |

**Conclusão científica:** na escala mensal/0,25°, a ocorrência de fogo é dominada por
**persistência + sazonalidade**; o problema difícil e operacionalmente relevante é a
**nova ignição**, e o sinal que ajuda ali não é meteorológico nem de acessibilidade
estática, mas o **desmatamento recente** — a causa proximal antrópica.

### Robustez: DETER mensal confirma o PRODES anual

`coletar_deter.py` + `validar_deter.py` repetem o teste com o **DETER** (alertas de
desmatamento com data → resolução mensal; classe `CICATRIZ_DE_QUEIMADA` excluída para
não vazar o alvo; 94.435 km², 2016–2023). Resultado na nova ignição: **+0,026 PR-AUC**
(0,206→0,233) — praticamente idêntico ao PRODES anual (+0,025). Ou seja: (i) o sinal
desmatamento→nova-ignição é **robusto a duas fontes independentes**; (ii) a resolução
**mensal não amplifica** o ganho — o que importa é *ter sido desmatado recentemente*
(escala anual/sazonal), não o mês exato. Relatório: `modelos/relatorios/driver_deter.json`.

## Robustez a famílias de modelo (IC 95%)

`comparar_familias.py` compara famílias no conjunto base+DETER (validação rolling-origin,
IC 95% por cluster bootstrap por célula; `modelos/relatorios/comparacao_familias.json`):

| Família | PR-AUC [IC 95%] | Brier | Nova ignição [IC 95%] |
|---|---|---|---|
| HistGradientBoosting | 0,776 [0,769–0,783] | 0,123 | 0,232 [0,209–0,258] |
| RandomForest | 0,769 [0,762–0,776] | **0,113** | 0,204 [0,185–0,232] |
| ExtraTrees | 0,761 [0,754–0,768] | 0,129 | 0,191 [0,174–0,217] |
| LogisticRegression | 0,699 [0,690–0,707] | 0,143 | 0,117 [0,108–0,131] |

Ensembles de árvore **estatisticamente equivalentes** (ICs sobrepostos); regressão
logística **significativamente inferior**. Os achados não dependem do algoritmo. RF
tem o melhor Brier (calibração) — alternativa válida ao HGB.

## Validação do rótulo contra área queimada (MapBiomas Fogo)

`validar_rotulo_mapbiomas.py` confronta o rótulo (focos INPE) com a área queimada
independente do MapBiomas Fogo (estatística oficial estado/ano, Coleção 3; o raster
exigiria GEE/rasterio). Resultado: **Spearman interanual médio intra-estado = 0,814**
(9/9 estados positivos, 0,69–0,94) — a dinâmica do rótulo é fortemente corroborada.
A correlação entre estados é fraca (focos ≠ magnitude em hectares), mas como o rótulo é
**ocorrência binária**, isso não o afeta. Relatório: `modelos/relatorios/validacao_rotulo_mapbiomas.json`.

**Nível célula × ano** (`validar_rotulo_celular.py`): lê os rasters anuais do MapBiomas
(COGs públicos no GCS, 2015–2020) via `/vsicurl/` + overview com `Resampling.average`
(sem Earth Engine, sem baixar GB), binariza presença de cicatriz por célula 0,25° e
cruza com presença de foco. Em 23.982 célula-anos: **concordância 78%, Cohen's κ=0,33**;
tratando a cicatriz como referência, os focos têm **revocação 0,90** e precisão 0,83
(F1 0,86). Os focos capturam ~90% dos célula-anos queimados; discordâncias = fogos sem
cicatriz (pequenos/nuvem) e cicatrizes sem foco (omissão da detecção ativa). κ moderado
reflete a alta prevalência do universo fire-prone. Relatório:
`modelos/relatorios/validacao_rotulo_celular.json`.

## Avaliação operacional (priorização de alertas) e horizonte

`avaliar_operacional.py` traduz a PR-AUC em valor de decisão: alertando as top-K%
células de maior risco, quantos focos do mês seguinte se captura (recall@K), vs o
índice INPE. `avaliar_horizonte.py` mede o skill por horizonte (t+1/t+2/t+3).

```bash
python scripts/previsao_ocorrencia/avaliar_operacional.py   # recall@K + figura
python scripts/previsao_ocorrencia/avaliar_horizonte.py     # skill vs horizonte + figura
```

**Priorização de alertas (recall@K, ML vs INPE):**

| Orçamento | Global ML | Global INPE | Nova ignição ML | Nova ignição INPE |
|---|---|---|---|---|
| 10% | **0,360** (prec. 0,88) | 0,158 | **0,492** (lift 4,9×) | 0,096 (lift ~1) |
| 20% | 0,611 | 0,282 | 0,721 | 0,206 |
| 30% | 0,775 | 0,372 | 0,845 | 0,305 |

- Alertando 10% das células, o ML captura **2,3× mais** focos que o índice INPE.
- Na **nova ignição**, o índice INPE tem lift ≈1 (**não supera o acaso**); o ML tem
  lift 4,9× — o desmatamento recente é o que faz a diferença.

**Horizonte:** PR-AUC estável (0,779 / 0,779 / 0,781 em t+1/t+2/t+3) — **até um
trimestre de antecedência sem perda de skill**, pois o sinal é sazonal/persistente,
não meteorológico de curto prazo. Figuras e JSON em `modelos/relatorios/`
(`avaliacao_operacional_*.png/json`, `avaliacao_horizonte.*`).

## Integração ao produto (modelo serializado + app de forecast)

`gerar_forecast_ocorrencia.py` treina o modelo final (base + DETER), serializa-o e
gera os artefatos operacionais; `app_forecast.py` os serve (Flask), independente do
app original (que classifica o índice do INPE).

```bash
python scripts/previsao_ocorrencia/gerar_forecast_ocorrencia.py   # gera artefatos + mapa
python scripts/previsao_ocorrencia/app_forecast.py                # serve em :5001
```

Artefatos: `modelos/modelo_ocorrencia.pkl`, `modelos/ocorrencia_conformal.json`
(cobertura calibrada **0,900**), `dataset_forecast_celulas.csv` (previsão das 4.168
células), `mapa_forecast_ocorrencia.html` (mapa interativo). Rotas: `/` (mapa),
`POST /api/forecast` ({lat,lon} → P(fogo no próximo mês) + rótulo conformal + drivers),
`GET /api/forecast_top`. Distribuição conformal do forecast (mês de ref. 2023-12):
182 células "fogo", 799 "incerto", 3.187 "não-fogo". Sanidade: Roraima (0,75,−60,25)
→ P=92% "fogo" (alta atividade + desmatamento); sul do PA em dez → P=22% "não-fogo"
(estação de fogo do sul é jul–out — correto).

### Operação em tempo real (dados abertos do INPE)

Os dados brutos (`focos_qmd_inpe_*.csv`, `dataset_ocorrencia_*.csv`) e o `.pkl` não são
versionados; são reconstruídos a partir das fontes públicas:

```bash
# 1x: baixa focos 2014→mês corrente (AQUA_M-T, bioma Amazônia) e o DETER 2016→hoje
python scripts/previsao_ocorrencia/coletar_focos_inpe.py
python scripts/previsao_ocorrencia/coletar_deter.py --ano_fim 2026 --saida scripts/previsao_ocorrencia/deter_celula_mes_atual.csv
python scripts/previsao_ocorrencia/atualizar_forecast.py --sem_download --retreinar

# rotina: atualiza o ano corrente (focos + DETER) e prevê com o modelo salvo
python scripts/previsao_ocorrencia/atualizar_forecast.py
python scripts/previsao_ocorrencia/app_forecast.py   # recarrega o forecast sozinho; GET /api/status
```

`atualizar_forecast.py` usa as features até o último mês **completo** *t* e prevê *t*+1;
`--retreinar` (recomendado 1x/ano, quando o INPE publica o anual consolidado) recalibra o
conformal no último ano completo. `deter_celula_mes.csv` (2016–2023) é mantido intacto para
reproduzir os números do TCC; a operação usa `deter_celula_mes_atual.csv`.

**Reprodução verificada (2026-09-24).** Reconstruindo a base 2014-01–2024-01 a partir dos dados
abertos: 450.684 cell-months (vs 450.144; revisões do INPE), prevalência 0,249 (vs 0,247),
PR-AUC temporal **0,779** (vs 0,777) e espacial **0,757** (vs 0,759). Atenção: após a
atualização, `dataset_ocorrencia_mensal.csv` cobre até o mês corrente — para reproduzir os
experimentos do TCC, reconstrua-o só com os focos de 2014–2023 + jan/2024.

**Primeiro forecast operacional (set/2026, modelo treinado até 2026-07, calib. 2025: cobertura
0,900).** Conferido contra os focos de set/2026 observados até 27/09: PR-AUC 0,748, ROC-AUC
0,827; das células rotuladas "fogo", 69% já queimaram; das "não-fogo", 5%.

**Calibração isotônica (2026-09-28).** `class_weight='balanced'` inflava P(fogo) (média 0,62 no
forecast de set/2026). O `.pkl` agora guarda um `calibrador` (isotônica ajustada no ano de
calibração) e `dataset_forecast_celulas.csv` traz `p_fogo` (calibrada) e `p_fogo_bruta`; o rótulo
conformal segue sobre a bruta. `avaliar_calibracao.py` (treino<2022, calib 2022, teste 2023):
ECE 0,087 → 0,015, Brier 0,123 → 0,111, PR-AUC 0,791 → 0,784 (empates da isotônica). No
forecast de set/2026 a média calibrada foi 0,30 contra 0,38 observado até 27/09: o calibrador
herda a taxa-base de 2025, ano de pouco fogo.

`atualizar_forecast.py` segue com o DETER já salvo se o TerraBrasilis falhar (504 em 28/09).

## Features (todas causais, ≤ t)

Histórico de fogo defasado (`focos_lag1-3`, `fogo_lag1-3`, somas móveis 3/6/12 m com `shift(1)`),
`meses_desde_fogo`, sazonalidade (`mes_sin/cos`, `estacao_seca`), localização (`LatBin/LonBin`),
`FRP_lag1`, e clima do mês corrente (`DiaSemChuva`, `Precipitacao`, `RiscoFogo_inpe` — agora apenas
**covariável**, não mais o alvo).

## Limitações (honestas — e próximos passos)

1. **Clima das ausências vem de climatologia.** Os CSVs do INPE são *presence-only*: só há clima
   mês-a-mês onde houve foco. Cell-months de ausência recebem clima da **climatologia da célula
   por mês-calendário**. O sinal preditivo, portanto, vem majoritariamente do **histórico de fogo
   defasado** + sazonalidade. A solução plena é a melhoria **P5**: reanálise gridada (NASA POWER /
   ERA5) por célula-mês para *todas* as células. Sem ela, o ganho de variáveis meteorológicas
   dinâmicas está subaproveitado.
2. **Domínio restrito a células *fire-prone*** (≥1 foco em 2014–2023). O modelo responde
   "dado que a célula é propensa, haverá fogo em t+1?", não cobre o interior de floresta densa
   que nunca queimou. Ampliar exige a grade meteorológica completa (P5) + recorte por shapefile.
3. **Calibração probabilística** (ECE≈0,10) pode melhorar com calibração isotônica/Platt; a
   validade do conformal **não** depende disso (cobertura garantida independe da calibração).

## Referências

`roberts2017cv` (CV espaço-temporal em bloco), `mortier2024conformal` e `spatialuq2025`
(conformal/UQ em observação da Terra), `deandrade2024lstmgru` (focos mensais na Amazônia),
`alonso2025seasfire`/`karasante2025firecastnet` (paradigma de ocorrência sobre grade 0,25°).
BibTeX em `../../ESTADO_DA_ARTE_E_PROPOSTAS_MELHORIA.md` §7.
