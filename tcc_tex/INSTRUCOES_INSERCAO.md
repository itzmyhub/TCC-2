# Instruções de Inserção — Passo a Passo

> **Objetivo:** integrar os `.tex` da pasta `tcc_tex/` no documento `.tex` original do TCC, com referência cruzada às páginas do PDF de fevereiro de 2024 e à seção correspondente.
> **Tempo estimado:** 90--120 min de edição assistida + 1 compilação completa.

---

## 1. Pré-requisitos no `.tex` principal

**UFTeX / `tcc-II.tex` da UFT:** a classe já carrega `babel` com `[english,brazil]`. **Não** acrescente `\usepackage[portuguese]{babel}` nem `\usepackage[brazil]{babel}` no preâmbulo; isso gera *Option clash for package babel* (muitas vezes o erro aparece na linha do próximo `\usepackage`, por exemplo `quoting`).

Garanta os seguintes pacotes no preâmbulo (modelo genérico; **omitir** `babel` se a sua classe já o carregar):

```latex
% \usepackage[brazil]{babel}  % só se a classe NÃO carregar babel
\usepackage[utf8]{inputenc}
\usepackage{siunitx}
\sisetup{output-decimal-marker={,}, group-separator={.}, group-minimum-digits=4}
\usepackage{booktabs}
\usepackage{graphicx}
\usepackage{mhchem}
\usepackage[alf,abnt-emphasize=bf]{abntex2cite}
```

Copie `references.bib` para a pasta do `.tex` principal (substituindo o `.bib` antigo) ou inclua um `\addbibresource{references.bib}` se usar `biblatex`.

---

## 2. Capítulo a capítulo

### 2.1 Pré-textual — Resumo e Abstract

| PDF original | Pg. 7--8 |
| --- | --- |
| Arquivo a usar | `00_resumo_abstract.tex` |
| Ação | Substituir o conteúdo dos ambientes `\begin{resumo}...\end{resumo}` e `\begin{abstract}...\end{abstract}` pelo arquivo. |
| Alterações principais | • Novos parágrafos com dados de 2024--2025 (Aragão 2018, Silva-Junior 2025). • Detalhamento das 24 \emph{features} Tier 1, 8 modelos comparados, Stacking GBM como modelo final, SHAP por classe + Permutation Importance, validação temporal estrita. • Métricas atualizadas: 84,61% / 0,7995 / 0,6375. |

---

### 2.2 Capítulo 1 — Introdução

| PDF original | Pgs. 11--17 |
| --- | --- |
| Arquivo | `01_capitulo1_complementos.tex` |

**Três ações:**

#### (A) Justificativa — adicionar parágrafo de âncora científica

Localize na Seção 1.1 ("Justificativa", Pg. 12) o parágrafo final que começa com:

> "Dada a magnitude e complexidade das queimadas..."

Insira **ANTES** desse parágrafo o **Bloco A** de `01_capitulo1_complementos.tex` (parágrafo iniciando "A urgência do tema é corroborada por publicações recentes...").

#### (B) Objetivos específicos — substituir os 6 itens

Localize na Seção 1.2.2 ("Objetivos específicos", Pg. 15) a lista numerada de 6 objetivos. Substitua a **lista inteira** pelo **Bloco B** de `01_capitulo1_complementos.tex` (mantém os 6 objetivos, mas atualizados com Tier 1, Optuna, SHAP, validação temporal e aplicação web).

#### (C) Estrutura da monografia — substituir Seção 1.3

Localize a Seção 1.3 ("Estrutura da Monografia", Pgs. 16--17) e substitua o conteúdo pelo **Bloco C** de `01_capitulo1_complementos.tex`. A nova redação descreve Capítulos 5 (Resultados) e 6 (Conclusão), antes vazios.

---

### 2.3 Capítulo 2 — Estado da Arte

| PDF original | Pgs. 18--29 |
| --- | --- |
| Arquivo | `02_capitulo2_estado_arte_complemento.tex` |
| Ação | Inserir o arquivo inteiro como **nova subseção** no final da Seção 2.1 ("Trabalhos Correlatos"), **ANTES** do parágrafo de fechamento que começa em "Em resumo, a bibliografia discutida destaca o amplo espectro..." (Pgs. 27--28). |
| Conteúdo | Nova subseção "Literatura recente (2024--2025)" com 3 parágrafos: Interpretabilidade via SHAP/Permutation; Validação temporal e \emph{drift} climático; \emph{Pseudo-labeling} para imputação climática. |

---

### 2.4 Capítulo 3 — Fundamentação Teórica

| PDF original | Pgs. 30--42 |
| --- | --- |
| Arquivo | `03_capitulo3_subsecoes_novas.tex` |

**Três blocos:**

#### Bloco A — Modelos modernos de boosting e ensembles

Localize a Seção 3.8.1 ("Aprendizado Supervisionado") e suas três sub-seções existentes:

- 3.8.1.1 Regressão Linear (Pg. 36)
- 3.8.1.2 Decision Tree (Pg. 37)
- 3.8.1.3 Support Vector Machine (Pg. 38)

Insira **DEPOIS** da Seção 3.8.1.3 (e antes do encerramento da Seção 3.8.1 ou início da 3.8.2):

- 3.8.1.4 XGBoost
- 3.8.1.5 LightGBM
- 3.8.1.6 CatBoost
- 3.8.1.7 Ensembles --- Voting e Stacking

#### Bloco B — Otimização, Interpretabilidade, Desbalanceamento

Insira ao **final da Seção 3.8** ("Inteligência Artificial e Aprendizado de Máquina", Pg. 42), antes do início da Metodologia (Cap. 4):

- 3.9 Otimização de hiperparâmetros com Optuna
- 3.10 Interpretabilidade de modelos
- 3.11 Tratamento de desbalanceamento de classes

#### Bloco C — Calibração, Índices físico-climáticos, Validação temporal

Continue como novas seções principais:

- 3.12 Calibração de probabilidades
- 3.13 Índices físico-climáticos de fogo
- 3.14 Validação temporal de modelos

**Importante:** se o documento original numera seções automaticamente, basta usar `\section{}` e `\subsection{}` --- a numeração se ajustará. Se preferir preservar a numeração original do TCC II, use `\section*{}` / `\subsection*{}` (sem numeração) e ajuste manualmente o índice (TOC).

---

### 2.5 Capítulo 4 — Metodologia

| PDF original | Pgs. 43--52 |
| --- | --- |
| Arquivo | `04_capitulo4_metodologia.tex` |

**Sete ações:**

| Ação | Onde no PDF | Bloco a usar |
|---|---|---|
| (A) Adicionar subseção "Enriquecimento via APIs externas" | Final da Seção 4.1 (Coleta de Dados, Pg. 44) | Bloco (A) |
| (B) Complementar lista de bibliotecas | Final da Seção 4.2 (Pgs. 46--47) | Bloco (B) |
| (C) Adicionar subseções 4.3.7 (Tier 1) e 4.3.8 (Calibração) | Final da Seção 4.3 (Pgs. 48--49) | Bloco (C) |
| (D) **SUBSTITUIR INTEGRALMENTE** Seção 4.4 (Treinamento) | Pgs. 50--51 | Bloco (D) |
| (E) **SUBSTITUIR INTEGRALMENTE** Seção 4.5 (Avaliação) | Pg. 51--52 | Bloco (E) |
| (F) **SUBSTITUIR INTEGRALMENTE** Seção 4.6 (Geração do Mapa) | Pg. 52 | Bloco (F) |
| (G) Adicionar nova Seção 4.7 (Pseudo-labeling de Umidade) | Final do Cap. 4 | Bloco (G) |

---

### 2.6 Capítulo 5 — Resultados

| PDF original | Pg. 53 (ESTÁ VAZIO) |
| --- | --- |
| Arquivo | `05_capitulo5_resultados.tex` |
| Ação | **Substituir todo o Capítulo 5** pelo conteúdo do arquivo. |
| Conteúdo | 9 seções (5.1 Dataset, 5.2 Comparativo, 5.3 Modelo final, 5.4 Interpretabilidade, 5.5 Validação temporal, 5.6 Threshold tuning, 5.7 Evolução, 5.8 App), 11 tabelas e 6 figuras. |

**Importante:** o arquivo usa `\ref{sec:metodologia_pre_processamento}` --- ajustar essa chave (e outras `\ref{...}`) para corresponder aos `\label{}` efetivamente usados no Capítulo 4 original. O arquivo deixa uma nota de rodapé indicando os pontos a ajustar.

---

### 2.7 Capítulo 6 — Conclusão

| PDF original | Pg. 54 |
| --- | --- |
| Arquivo | `06_capitulo6_conclusao.tex` |
| Ação | **Substituir todo o Capítulo 6** pelo conteúdo do arquivo. |
| Conteúdo | 8 parágrafos numerados de achados + Limitações reconhecidas (6 itens) + Trabalhos futuros (7 itens) + Considerações finais. |

---

### 2.8 Capítulo 5 — adendo: estudo de caso operacional Pium-TO

| Arquivo | `07_estudo_caso_pium_to.tex` |
| --- | --- |
| Origem | Investigação iniciada em 12/05/2026 a partir de evento real reportado pelo *Painel do Fogo* / CENSIPAM (evento `ID 6668060`, Pium-TO, duração 48,3 h). |
| Ação | Inserir como **nova seção 5.9** dentro do Capítulo 5, logo após a Seção 5.8 ("Validação operacional via aplicação web"). Não substitui nada — é conteúdo adicional. |
| Conteúdo | Subseção principal (evento, resposta, diagnóstico, correções, discussão, síntese) + subsubseções **M1--M4** em ``Implementação das três melhorias\ldots'': M1 auditoria JSONL; M2 fusão INMET (WIS2/ZIP) + tabela `tab:pium_inmet`; M3 retreino OOD; **M4** extensões pós-literatura (busca INMET hierárquica até 180 km com `inmet_representatividade`, vento WIS2 alinhado, NASA POWER `WS2M`/`T2M_MAX` na API, proxy FWI + `incerteza_operacional`, FNR/FPR na auditoria) --- **motivo**: amarrar implementação ao argumento do Cap.~4/5 sem misturar com o vetor de treino até retreino documentado. Tabelas: `tab:pium_evento`, `tab:pium_respostas_pos_fix`, `tab:pium_tier1_pre_pos`, `tab:pium_contrafactual`, `tab:pium_correcoes`, `tab:pium_inmet`, `tab:pium_ood_alvos`, `tab:pium_ood_metricas`. |
| Por quê | (i) Validação por fonte externa independente (CENSIPAM ≠ INPE); (ii) Confronto com evento real; (iii) Diagnóstico honesto de bugs corrigidos e limites estruturais; (iv) Gera trabalhos futuros concretos para a Conclusão; **(v)** M4 documenta *o quê* e *por quê* das extensões de painel/auditoria alinhadas à literatura 2024--2026 para colagem no texto do TCC. |

**Cuidado com referências cruzadas**:

- A nova Seção 5.9 referencia `\ref{cap:metodologia}`, `\ref{subsec:metod_features_tier1}`, `\ref{sec:res_app}`, `\ref{cap:conclusao}`. Verificar que esses `\label{}` já estejam definidos nos capítulos respectivos (devem estar, pois vêm dos outros arquivos desta pasta).
- A seção cita `\cite{inmetnormais}` (já listado em `references.bib` — usado também no Cap. 4).
- A seção cita `\cite{nasapower}` e `\cite{seager2015climatology}` — já em `references.bib`.

**Documento complementar reproduzível**: `DOCUMENTO_ESTUDO_CASO_PIUM_TO.md` na raiz do projeto. Contém:
- Timeline da investigação (12/05/2026, 18:15--18:40 UTC-3).
- Comandos `python -c "..."` executáveis para reproduzir as 5 janelas + contrafactual.
- Saídas brutas dos diagnósticos (NASA FIRMS HTTP 400, validação de MAP_KEY etc.).
- *Diff* das três correções (`scripts/frp_api.py`, `scripts/feature_lookup.py`, `scripts/app_map_interativo.py`).
- 5 propostas concretas de trabalhos futuros decorrentes do caso.

Usar esse documento como **apêndice técnico** ou apenas como referência para a banca, conforme a preferência do orientador.

---

## 3. Diff de palavras-chave a corrigir no texto antigo (capítulos 1--4)

O texto original do TCC II foi escrito assumindo um único modelo (SGDClassifier ou Decision Tree). Recomenda-se substituir as ocorrências abaixo nos capítulos que **não vão ser inteiramente reescritos** (ou seja, principalmente no Capítulo 1):

| Termo antigo | Termo novo |
|---|---|
| "modelo SGDClassifier" / "modelo Decision Tree" | "modelo final — Ensemble Stacking com meta-classificador Gradient Boosting" |
| "algoritmo escolhido" | "pipeline de oito modelos comparados (Tabela X), com Ensemble Stacking como modelo final" |
| "acurácia próxima de 60%" / "70%" | "acurácia de 84,61% (split aleatório) / 70,13% (validação temporal estrita)" |
| "70% (meta)" | "meta acadêmica original ($\geq$ 70%) superada com margem confortável" |
| "umidade do ar" (como variável-chave) | "proxies físico-climáticos (KBDI, VPD, Índice de Seca) que substituem a umidade direta" |
| "shapefile do INPE" | "shapefile oficial do IBGE (Amazônia Legal 2024)~\cite{ibgeamazonia2024}" |
| "(Aragão et al., 2018)" / "(Aragão, 2018)" | "~\cite{aragao2018amazon}" |
| "(WMO, 2021)" | "~\cite{wmo2021}" |
| "(INPE, 2024)" / "(BDQueimadas)" | "~\cite{INPE24}" |
| "(Novo, 2010)" | "~\cite{novo10}" |
| "(Russell e Norvig, 2020)" | "~\cite{russel20}" |
| "(Goodfellow et al., 2016)" | "~\cite{goodfellow16}" |

---

## 4. Lista de chaves BibTeX usadas pelos arquivos `.tex`

Para conferência rápida durante a compilação:

```text
% Capítulo 0 (Resumo/Abstract)
wmo2021, aragao2018amazon, silvajunior2025amazon, seager2015climatology,
forests2024cerradoamazon, npjhazards2025fire, akiba2019optuna, chen2016xgboost,
ke2017lightgbm, prokhorenkova2018catboost, wolpert1992stacked, zadrozny2002transforming,
friedman2001greedy, lundberg2017unified, lundberg2020local, breiman2001random,
fisher2019all, bergmeir2012cv, folium, leafletjs, nasapower, nasafirms

% Capítulo 1 (Introdução)
aragao2018amazon, silvajunior2025amazon, seager2015climatology, forests2024cerradoamazon,
npjhazards2025fire, akiba2019optuna, chen2016xgboost, ke2017lightgbm, prokhorenkova2018catboost,
wolpert1992stacked, zadrozny2002transforming, bergmeir2012cv, lundberg2017unified,
lundberg2020local, breiman2001random, fisher2019all, folium, leafletjs

% Capítulo 2 (Estado da Arte — complemento)
cheerala2025probabilistic, lundberg2017unified, forests2024cerradoamazon,
keetch1968drought, npjhazards2025fire, mckee1993spi, seager2015climatology,
bergmeir2012cv, silvajunior2025amazon, lopezgarcia2024spatiotemporal,
lee2013pseudo, arazo2020pseudo

% Capítulo 3 (Fundamentação Teórica)
chen2016xgboost, friedman2001greedy, npjhazards2025fire, ke2017lightgbm,
prokhorenkova2018catboost, wolpert1992stacked, akiba2019optuna, lundberg2017unified,
lundberg2020local, breiman2001random, fisher2019all, strobl2007bias,
hooker2021unrestricted, molnar2022interpretable, he2009learning,
chawla2002smote, lemaitre2017imbalanced, platt1999probabilistic,
zadrozny2002transforming, pedregosa2011scikit, keetch1968drought,
mckee1993spi, seager2015climatology, demartonne1926aridity,
forests2024cerradoamazon, bergmeir2012cv

% Capítulo 4 (Metodologia)
nasapower, nasafirms, mckinney2010data, harris2020numpy, pedregosa2011scikit,
chen2016xgboost, ke2017lightgbm, prokhorenkova2018catboost, lemaitre2017imbalanced,
akiba2019optuna, lundberg2017unified, lundberg2020local, folium, leafletjs,
geopandas, ibgeamazonia2024, seager2015climatology, forests2024cerradoamazon,
keetch1968drought, mckee1993spi, demartonne1926aridity, inmetnormais,
cheerala2025probabilistic, zadrozny2002transforming, breiman2001random,
wolpert1992stacked, friedman2001greedy, fisher2019all, strobl2007bias,
he2009learning, bergmeir2012cv, lee2013pseudo, lopezgarcia2024spatiotemporal,
arazo2020pseudo

% Capítulo 5 (Resultados)
he2009learning, pedregosa2011scikit, akiba2019optuna, breiman2001random,
wolpert1992stacked, chen2016xgboost, ke2017lightgbm, prokhorenkova2018catboost,
friedman2001greedy, zadrozny2002transforming, lundberg2017unified, lundberg2020local,
fisher2019all, strobl2007bias, molnar2022interpretable, hooker2021unrestricted,
bergmeir2012cv, aragao2018amazon, silvajunior2025amazon, folium, leafletjs

% Capítulo 6 (Conclusão)
friedman2001greedy, wolpert1992stacked, mckee1993spi, keetch1968drought,
seager2015climatology, forests2024cerradoamazon, npjhazards2025fire,
lundberg2017unified, lundberg2020local, breiman2001random, fisher2019all,
bergmeir2012cv, zadrozny2002transforming, folium, leafletjs, lee2013pseudo,
lopezgarcia2024spatiotemporal, molnar2022interpretable, arazo2020pseudo,
INPE24, nasapower, nasafirms, ibgeamazonia2024, cheerala2025probabilistic,
mapbiomasfogo

% Capítulo 5 — Seção 5.9 (estudo de caso Pium-TO)
nasapower, nasafirms, seager2015climatology, inmetnormais
```

Total: 42 chaves no `references.bib`. As 4 chaves usadas na Seção 5.9 já estão todas presentes (compartilhadas com os capítulos 4 e 6).

---

## 5. Ordem de inserção sugerida (95--130 min)

1. **Copiar `references.bib`** para a pasta do `.tex` original (3 min).
2. **Adicionar pacotes** ao preâmbulo (`siunitx`, `booktabs`, `graphicx`, `mhchem`, `abntex2cite`) (5 min).
3. **Resumo/Abstract** (`00_resumo_abstract.tex`) (10 min).
4. **Capítulo 6** — Conclusão (substituir integral) (10 min).
5. **Capítulo 5** — Resultados (substituir integral) (25 min).
6. **Capítulo 5 — Seção 5.9** — Estudo de caso Pium-TO (adendo) (10 min).
7. **Capítulo 4** — Metodologia (7 inserções/substituições) (20 min).
8. **Capítulo 3** — Fundamentação (10 novas subseções) (15 min).
9. **Capítulo 1** — Introdução (3 blocos) (10 min).
10. **Capítulo 2** — Estado da Arte (1 subseção nova) (5 min).
11. **Diff de palavras-chave** nos capítulos 1--4 (12 min).
12. **Compilar e verificar referências** (10 min).

---

## 6. Verificação final

Após compilação, conferir:

- [ ] Sumário (TOC) reflete a nova estrutura (Caps. 5 e 6 com seções).
- [ ] Todas as citações `\cite{...}` resolvidas (não há `[?]` no PDF).
- [ ] Todas as figuras aparecem com legendas (`fig:res_distribuicao`, `fig:res_matriz_confusao`, `fig:res_shap_per_classe`, `fig:res_perm_importance`, `fig:res_temporal_barras`, `fig:res_evolucao`, `fig:res_calibracao`).
- [ ] Todas as tabelas aparecem com legendas e numeração consistente.
- [ ] Lista de tabelas e Lista de figuras (se presentes no original) também são populadas automaticamente pelos `\caption{}`.
- [ ] Página inicial do Resumo/Abstract não está duplicada.

---

## 7. Em caso de erros de compilação

| Erro | Causa provável | Solução |
|---|---|---|
| `\num` not defined | falta `\usepackage{siunitx}` | adicionar ao preâmbulo |
| `\toprule` not defined | falta `\usepackage{booktabs}` | adicionar ao preâmbulo |
| `\ce` not defined | falta `\usepackage{mhchem}` | adicionar ao preâmbulo |
| `Citation undefined` | falta `bibtex` ou chave inexistente | rodar `bibtex tcc.aux` e verificar grafia |
| Caracteres mal renderizados | encoding | confirmar `\usepackage[utf8]{inputenc}` |
| `Reference 'sec:xxx' on page Y undefined` | `\label{}` referenciado não existe | criar o `\label{}` correspondente no capítulo original, ou ajustar o `\ref{}` no arquivo novo |
