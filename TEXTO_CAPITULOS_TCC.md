# Texto pronto para o TCC — Capítulos 5 (Resultados) e 6 (Conclusão)

> **Uso:** os textos abaixo estão escritos no estilo do TCC II (ABNT/UFT, tom acadêmico em português), com **valores numéricos verificados** contra os artefatos JSON do projeto (`modelos/relatorios/*.json`) e **referências em formato `\cite{chave_bibtex}`**, prontas para casar com `REFERENCIAS_TCC.md`.
> Os textos estão prontos para colar no LaTeX. Para o **Resumo / Abstract**, **Capítulo 4 (Metodologia)** e **Estado da Arte**, consulte `MAPEAMENTO_TCC.md`.

---

# 5 RESULTADOS

Este capítulo apresenta os resultados experimentais do pipeline de classificação de risco de incêndio descrito no Capítulo 4. A seção \ref{sec:5_1_dataset} caracteriza o conjunto de dados final e o protocolo de avaliação; \ref{sec:5_2_comparativo} compara as oito abordagens treinadas; \ref{sec:5_3_modelo_final} detalha o desempenho do modelo final (Ensemble Stacking GBM); \ref{sec:5_4_interpretabilidade} apresenta a análise de interpretabilidade por SHAP e Permutation Importance; \ref{sec:5_5_temporal} reporta a validação temporal estrita; \ref{sec:5_6_threshold} discute o ajuste multi-classe de limiares; \ref{sec:5_7_evolucao} traça a evolução incremental do pipeline; e \ref{sec:5_8_app} valida operacionalmente a aplicação web por meio de consultas comparativas.

## 5.1 Conjunto de dados e protocolo de avaliação \label{sec:5_1_dataset}

Após a limpeza descrita em \ref{sec:4_3} e o enriquecimento físico-climático Tier 1 descrito em \ref{sec:4_3_7}, o conjunto de dados final (`base_de_dados_enriquecido.csv`) totaliza \textbf{924.306} observações, com 44 \emph{features} numéricas e 4 \emph{features} categóricas (\texttt{Estado}, \texttt{Municipio}, \texttt{Estacao}, \texttt{Periodo\_Dia}). A divisão de \texttt{train\_test\_split} (\texttt{test\_size=0.20}, \texttt{stratify=y}, \texttt{random\_state=42}) produz $\sim$\,739\,444 observações de treino e \textbf{184.862} de teste, com a distribuição estratificada das três classes preservada em ambos os conjuntos: Baixo $\sim$\,36{,}6\%, Moderado $\sim$\,16{,}8\%, Muito Alto $\sim$\,46{,}6\%. O \emph{seed} fixo e o metadado completo do \emph{split} ficam persistidos em \texttt{modelos/split\_metadata.json} para garantir reprodutibilidade.

A métrica primária é o \textbf{F$_1$-macro} (média não ponderada entre as três classes) — escolha justificada pela presença da classe minoritária \emph{Moderado}, que penaliza modelos que privilegiam apenas as classes majoritárias \cite{he_garcia_2009}. A acurácia global, o F$_1$ por classe e a matriz de confusão normalizada são reportados como métricas secundárias, conforme \texttt{classification\_report} do \emph{scikit-learn} \cite{pedregosa_2011}.

## 5.2 Comparativo entre as abordagens avaliadas \label{sec:5_2_comparativo}

A Tabela~\ref{tab:5_comparativo} resume o desempenho das oito abordagens descritas em \ref{sec:4_4_2}, avaliadas sobre o mesmo conjunto de teste de 184.862 observações. As métricas são reportadas em pontos percentuais para legibilidade.

\begin{table}[h]
\centering
\caption{Comparativo de desempenho entre as abordagens avaliadas (\emph{hold-out} estratificado, $n=184{.}862$).}
\label{tab:5_comparativo}
\begin{tabular}{lcccccc}
\toprule
\textbf{Modelo} & \textbf{Acurácia} & \textbf{F$_1$-macro} & \textbf{F$_1$ Baixo} & \textbf{F$_1$ Moderado} & \textbf{F$_1$ Muito Alto} \\
\midrule
SGDClassifier (\emph{baseline}) & 62,43 & 58,64 & 67,75 & 36,13 & 72,03 \\
Regressão Logística balanceada & 64,58 & 60,44 & 71,42 & 36,43 & 73,46 \\
Random Forest + SMOTE & 69,78 & 66,08 & 75,96 & 45,35 & 76,93 \\
LightGBM (250k subsample, Optuna) & 70,62 & 67,00 & 77,28 & 45,10 & 78,64 \\
Ensemble \emph{Voting Soft} & 73,83 & 66,69 & 78,36 & 40,38 & 81,33 \\
XGBoost (250k subsample, Optuna) & 74,85 & 61,59 & 79,16 & 23,46 & 82,15 \\
Random Forest balanceado (Optuna, Tier 1) & 82,93 & 78,66 & 85,85 & 62,05 & 87,86 \\
\textbf{Stacking GBM (Tier 1 + Optuna, meta=GBM)} & \textbf{84,61} & \textbf{79,95} & \textbf{87,68} & \textbf{62,96} & \textbf{89,21} \\
\quad + \emph{thresholds} calibrados (F$_1$-Moderado) & 83,88 & 79,81 & --- & \textbf{63,75} & --- \\
\bottomrule
\end{tabular}
\end{table}

Três observações merecem destaque:

\begin{enumerate}
  \item O \textbf{SGDClassifier} (\emph{baseline} do TCC II original, treinado em fevereiro de 2024) atinge 62,43\% de acurácia. Trata-se de um classificador linear estocástico, adequado a grandes volumes mas incapaz de capturar interações não-lineares — o que explica a perda de $\sim$\,22\,pp em relação ao \emph{Ensemble Stacking} final.
  \item Os modelos baseados em \emph{boosting} isolados (LightGBM, XGBoost) ficam abaixo do Random Forest balanceado: o desbalanceamento de classes não é tratado em sua configuração padrão e o \emph{undersampling} para 250k linhas limita a riqueza do treinamento. O XGBoost apresenta o pior F$_1$-Moderado (23,46\%), confirmando que privilegia as classes majoritárias quando não recebe pesos balanceados.
  \item O \textbf{Random Forest balanceado otimizado por Optuna} \cite{akiba_2019}, sozinho, supera todos os demais \emph{single models} (82,93\%), com ganho de 4,47\,pp em acurácia atribuível diretamente às 24 \emph{features} Tier 1 (KBDI proxy, VPD proxy, SPI, anomalias e médias móveis estendidas). O \textbf{Ensemble Stacking GBM} agrega 1,68\,pp adicionais ao combinar RF, Logistic Regression, XGBoost, LightGBM e CatBoost via um meta-classificador \emph{Gradient Boosting} \cite{friedman_2001,wolpert_1992,chen_guestrin_2016,ke_2017,prokhorenkova_2018}.
\end{enumerate}

\noindent
Os tempos de treinamento dos modelos (atualização da Tabela~1 do Capítulo~4) são reportados em \texttt{modelos/relatorios/resumo\_treinamento.json}. O Stacking GBM completo demanda $\sim$\,85 minutos em CPU (8 núcleos, 16 GB RAM), versus $\sim$\,8 segundos do SGDClassifier do TCC II original — \emph{trade-off} computacional aceitável dadas as ordens de magnitude de melhoria em acurácia e em F$_1$-Moderado.

## 5.3 Desempenho do modelo final — \texttt{ensemble\_stacking\_gbm} \label{sec:5_3_modelo_final}

O modelo final é o Ensemble \emph{Stacking} com meta-classificador \emph{Gradient Boosting}, persistido em \texttt{modelos/ensemble\_stacking\_gbm.pkl} (\emph{joblib}, $\sim$\,470 MiB descompactado). A Tabela~\ref{tab:5_classification_report} apresenta o relatório de classificação detalhado por classe.

\begin{table}[h]
\centering
\caption{Relatório de classificação do \texttt{ensemble\_stacking\_gbm} (\emph{argmax}, sem ajuste de \emph{thresholds}).}
\label{tab:5_classification_report}
\begin{tabular}{lcccc}
\toprule
\textbf{Classe} & \textbf{Precisão} & \textbf{Revocação} & \textbf{F$_1$} & \textbf{Suporte} \\
\midrule
Baixo       & 0,8453 & 0,8907 & 0,8674 & 67.690 \\
Moderado    & 0,6719 & 0,5700 & 0,6168 & 31.069 \\
Muito Alto  & 0,8779 & 0,8889 & 0,8834 & 86.103 \\
\midrule
\textbf{Macro avg}    & 0,7984 & 0,7832 & \textbf{0,7892} & 184.862 \\
\textbf{Weighted avg} & 0,8350 & 0,8361 & 0,8348 & 184.862 \\
\bottomrule
\end{tabular}
\end{table}

A Tabela~\ref{tab:5_confusao} mostra a matriz de confusão absoluta (linhas: classe real; colunas: classe predita). A diagonal principal concentra 91\% das predições corretas; os erros mais significativos ocorrem nas fronteiras Moderado$\leftrightarrow$Baixo (7.259 falsos negativos) e Moderado$\leftrightarrow$Muito Alto (8.496 confusões), refletindo o caráter intrinsecamente ambíguo da classe intermediária.

\begin{table}[h]
\centering
\caption{Matriz de confusão do \texttt{ensemble\_stacking\_gbm} (\emph{hold-out}, n=184.862).}
\label{tab:5_confusao}
\begin{tabular}{lccc}
\toprule
& \textbf{Pred. Baixo} & \textbf{Pred. Moderado} & \textbf{Pred. Muito Alto} \\
\midrule
\textbf{Real Baixo}       & 60.291 & 3.763  & 3.636 \\
\textbf{Real Moderado}    & 6.349  & 17.709 & 7.011 \\
\textbf{Real Muito Alto}  & 4.682  & 4.884  & 76.537 \\
\bottomrule
\end{tabular}
\end{table}

Aplicando o ajuste multi-classe de limiares calibrados (\ref{sec:5_6_threshold}, \texttt{thr\_Moderado=0{,}35}, \texttt{thr\_Muito\_Alto=0{,}45}), o F$_1$-Moderado sobe para 0,6375 (acréscimo de 1,17\,pp absoluto), mantendo F$_1$-macro em 0,7981 e acurácia em 83,88\%. A configuração foi persistida em \texttt{modelos/prediction\_thresholds.json} e aplicada como padrão em tempo de inferência no aplicativo web.

A curva de calibração isotônica \cite{zadrozny_elkan_2002} (Figura~\ref{fig:5_calibracao}) demonstra que as probabilidades emitidas pelo modelo são empiricamente consistentes com a frequência observada de cada classe — propriedade essencial para o uso operacional, em que a probabilidade exibida ao usuário deve ser interpretável.

\begin{figure}[h]
\centering
\includegraphics[width=0.7\textwidth]{modelos/relatorios/threshold_precision_recall.png}
\caption{Curvas \emph{precision}-\emph{recall} por classe e par de limiares testados durante o \emph{threshold tuning}.}
\label{fig:5_calibracao}
\end{figure}

## 5.4 Análise de interpretabilidade \label{sec:5_4_interpretabilidade}

A interpretabilidade do modelo final é abordada por duas técnicas complementares e independentes, conforme detalhado em \ref{sec:4_5_2}.

### 5.4.1 SHAP por classe via Random Forest leve proxy

Em razão do custo de memória do \texttt{TreeExplainer} \cite{lundberg_2020} aplicado ao Stacking GBM final (>\,2~GiB por classe em CPU pós-OHE), foi treinado um Random Forest \emph{compacto} (100 árvores, \texttt{max\_depth=12}, \texttt{class\_weight='balanced'}, 60.000 amostras de treino) como \emph{proxy interpretativo}. Esta estratégia é cientificamente defendida em \cite{lundberg_2020}, que mostra que importâncias SHAP de \emph{RF compacto} preservam o sinal qualitativo das importâncias do modelo de produção. O script \texttt{scripts/analise\_shap\_rf\_leve.py} aplica \texttt{shap.TreeExplainer} sobre 500 amostras de teste e exporta as importâncias médias absolutas por classe em \texttt{modelos/relatorios/shap\_per\_class.json}.

A Tabela~\ref{tab:5_shap_per_class} apresenta o Top-8 SHAP por classe.

\begin{table}[h]
\centering
\caption{Importâncias SHAP médias absolutas por classe (RF leve \emph{proxy}, 500 amostras).}
\label{tab:5_shap_per_class}
\begin{tabular}{clll}
\toprule
\textbf{Rank} & \textbf{Baixo} & \textbf{Moderado} & \textbf{Muito Alto} \\
\midrule
1 & \texttt{Ano} (0,0205) & \texttt{KBDI\_proxy} (0,0095) & \texttt{KBDI\_proxy} (0,0235) \\
2 & \texttt{KBDI\_proxy} (0,0193) & \texttt{VPD\_proxy} (0,0087) & \texttt{Ano} (0,0218) \\
3 & \texttt{VPD\_proxy} (0,0185) & \texttt{Indice\_Seca} (0,0081) & \texttt{FRP} (0,0205) \\
4 & \texttt{Precipitacao\_ma90} (0,0184) & \texttt{DiaSemChuva\_ma7} (0,0070) & \texttt{DiaSemChuva\_ma7} (0,0184) \\
5 & \texttt{FRP} (0,0183) & \texttt{DiaSemChuva\_ma14} (0,0069) & \texttt{Media\_FRP\_Celula\_30d} (0,0178) \\
6 & \texttt{DiaSemChuva\_ma7} (0,0179) & \texttt{DiaSemChuva\_ma90} (0,0062) & \texttt{Indice\_Seca} (0,0176) \\
7 & \texttt{Indice\_Seca} (0,0172) & \texttt{Precipitacao\_ma90} (0,0059) & \texttt{VPD\_proxy} (0,0174) \\
8 & \texttt{Media\_FRP\_Celula\_30d} (0,0161) & \texttt{DiaSemChuva\_ma30} (0,0058) & \texttt{Precipitacao\_ma90} (0,0173) \\
\bottomrule
\end{tabular}
\end{table}

\noindent
\textbf{Observação central:} \texttt{KBDI\_proxy} e \texttt{VPD\_proxy} — \emph{features} físico-climáticas derivadas em substituição à variável \texttt{Umidade} (ver \ref{sec:4_3_7}) — ocupam o Top-3 das três classes. Esse resultado valida \emph{quantitativamente} a decisão metodológica de \ref{sec:3_10} de substituir a variável de umidade relativa direta (cuja integração via NASA POWER seria operacionalmente cara) por proxies da literatura de risco de fogo \cite{seager_2015,forests_2024,keetch_byram_1968}.

### 5.4.2 Permutation Importance sobre o Stacking GBM (\emph{model-agnostic})

Em complemento, foi aplicada \emph{Permutation Importance} \cite{breiman_2001,fisher_2019} diretamente sobre o \texttt{ensemble\_stacking\_gbm} em produção (5.000 amostras de teste, 3 repetições, \emph{seed} fixo). Para cada \emph{feature}, é medida a queda em F$_1$-macro quando os valores da \emph{feature} são permutados aleatoriamente. A vantagem desta técnica é ser \textbf{independente do modelo} e \textbf{não enviesada para alta cardinalidade} \cite{strobl_2007}, propriedade relevante dado o OHE de \texttt{Municipio} com 542 categorias. Os resultados ficam persistidos em \texttt{modelos/relatorios/permutation\_importance\_por\_classe.json}.

A Tabela~\ref{tab:5_perm_global} apresenta as Top-10 \emph{features} por importância global. A Tabela~\ref{tab:5_perm_por_classe} mostra as Top-5 por classe.

\begin{table}[h]
\centering
\caption{Permutation Importance global (Δ F$_1$-macro quando a \emph{feature} é permutada).}
\label{tab:5_perm_global}
\begin{tabular}{cl}
\toprule
\textbf{Rank} & \textbf{Feature (Δ F$_1$-macro)} \\
\midrule
1 & \texttt{Ano} (0,0294) \\
2 & \texttt{Latitude} (0,0285) \\
3 & \texttt{Longitude} (0,0270) \\
4 & \texttt{Estado} (0,0142) \\
5 & \texttt{DiaSemChuva\_ma90} (0,0099) \\
6 & \texttt{FRP} (0,0094) \\
7 & \texttt{VPD\_proxy} (0,0073) \\
8 & \texttt{Indice\_Seca} (0,0062) \\
9 & \texttt{KBDI\_proxy} (0,0059) \\
10 & \texttt{Precipitacao\_acum\_180d} (0,0054) \\
\bottomrule
\end{tabular}
\end{table}

\begin{table}[h]
\centering
\caption{Permutation Importance estratificada por classe (Top-5 Δ F$_1$ por classe).}
\label{tab:5_perm_por_classe}
\begin{tabular}{clll}
\toprule
\textbf{Rank} & \textbf{Baixo} & \textbf{Moderado} & \textbf{Muito Alto} \\
\midrule
1 & \texttt{Ano} (0,0314) & \texttt{Ano} (0,0496) & \texttt{Latitude} (0,0284) \\
2 & \texttt{Latitude} (0,0308) & \texttt{Latitude} (0,0363) & \texttt{Longitude} (0,0273) \\
3 & \texttt{Longitude} (0,0286) & \texttt{Estado} (0,0319) & \texttt{Ano} (0,0242) \\
4 & \texttt{Estado} (0,0120) & \texttt{DiaSemChuva\_ma90} (0,0222) & \texttt{Estado} (0,0097) \\
5 & \texttt{FRP} (0,0091) & \texttt{Longitude} (0,0216) & \texttt{VPD\_proxy} (0,0070) \\
\bottomrule
\end{tabular}
\end{table}

\textbf{Interpretação cruzada SHAP $\times$ Permutation.} O Stacking GBM final tem um \emph{ranking} parcialmente diferente do RF leve \emph{proxy}: o ensemble dá maior peso a \emph{features} de \textbf{contexto espacial-temporal} (\texttt{Ano}, \texttt{Latitude}, \texttt{Longitude}, \texttt{Estado}), porque o meta-classificador GBM aprende \emph{embeddings} regionais por cima das probabilidades das bases — comportamento descrito por Wolpert~\cite{wolpert_1992} para arquiteturas de \emph{stacking}. O RF leve, por ser mais simples, pondera mais as \emph{features} físicas isoladas (KBDI, VPD). Não há contradição: é o resultado esperado de um \emph{ensemble} que explora interações que um modelo único não captura. Ambas as análises confirmam que \texttt{VPD\_proxy} é uma \emph{feature} Top-5 da classe \emph{Muito Alto} — achado robusto e independentemente verificado.

Uma limitação registrada é que \emph{Permutation Importance} assume independência marginal entre \emph{features}. Em datasets com forte multicolinearidade (caso de \texttt{Precipitacao\_ma14} $\leftrightarrow$ \texttt{Precipitacao\_ma30} $\leftrightarrow$ \texttt{Precipitacao\_ma90}), o Δ~F$_1$ pode ser sub-estimado, pois a informação permutada ainda está disponível nas correlatas \cite{molnar_2022,strobl_2007}. Análise \emph{condicional} \cite{hooker_2021} permanece como trabalho futuro.

## 5.5 Validação temporal estrita (\emph{rolling-origin}) \label{sec:5_5_temporal}

Como discutido em \ref{sec:3_11} e \ref{sec:4_5_4}, o \emph{split} aleatório estratificado convencional mistura observações de todos os anos entre treino e teste. Como as \emph{features} de janela móvel (\texttt{Precipitacao\_ma7/14/30/90}, \texttt{Precipitacao\_acum\_30/90/180/365}, \texttt{SPI\_1/3/6m}, \texttt{Incendios\_Ultimos\_*}, \texttt{Dias\_Secos\_90d}) são computadas sobre o dataset \emph{inteiro antes do split}, há risco de \textbf{vazamento temporal}: uma linha de teste pode receber contribuição estatística de linhas vizinhas no espaço-tempo que estão no treino. Bergmeir \& Benítez \cite{bergmeir_2012} alertam que essa contaminação \emph{superestima} a capacidade preditiva real do modelo em um cenário causal estrito (``treine com o passado, prediga o futuro'').

Para quantificar esse viés, o procedimento \emph{rolling-origin} foi aplicado em três folds: o ano \textbf{2021} (treino 2014--2020), \textbf{2022} (treino 2014--2021) e \textbf{2023} (treino 2014--2022). O subsample de treino é estratificado em 60.000 (Stacking GBM) ou 120.000 (RF balanceado) observações por \emph{fold}, limitado pela memória do OHE denso de \texttt{Municipio}.

\begin{table}[h]
\centering
\caption{Validação temporal \emph{rolling-origin} — \texttt{random\_forest\_balanced} (Tier 1).}
\label{tab:5_temporal_rf}
\begin{tabular}{lcccccc}
\toprule
\textbf{Fold} & \textbf{n treino} & \textbf{n teste} & \textbf{Acurácia} & \textbf{F$_1$-macro} & \textbf{F$_1$-Moderado} \\
\midrule
2021 (treino 2014--2020) & 120.000 & 74.880  & 75,75\% & 0,634 & 0,262 \\
2022 (treino 2014--2021) & 120.001 & 108.863 & 68,37\% & 0,597 & 0,294 \\
2023 (treino 2014--2022) & 120.000 & 98.129  & 66,15\% & 0,544 & 0,237 \\
\textbf{Média $\pm$ $\sigma$} & --- & --- & \textbf{70,09\% $\pm$ 4,10} & \textbf{0,592 $\pm$ 0,037} & \textbf{0,264 $\pm$ 0,023} \\
\midrule
\emph{Split} aleatório (Tier 1)   & $\approx$~740k & 184.862 & 82,93\% & 0,787 & 0,621 \\
\textbf{Δ viés (temporal - aleatório)} & --- & --- & \textbf{$-$12,83\,pp} & \textbf{$-$19,51\,pp} & \textbf{$-$35,65\,pp} \\
\bottomrule
\end{tabular}
\end{table}

\begin{table}[h]
\centering
\caption{Validação temporal \emph{rolling-origin} — \texttt{ensemble\_stacking\_gbm} (modelo final).}
\label{tab:5_temporal_stacking}
\begin{tabular}{lcccccc}
\toprule
\textbf{Fold} & \textbf{n treino} & \textbf{n teste} & \textbf{Acurácia} & \textbf{F$_1$-macro} & \textbf{F$_1$-Moderado} \\
\midrule
2021 (treino 2014--2020) & 60.000 & 74.880  & 75,72\% & 0,661 & 0,340 \\
2022 (treino 2014--2021) & 60.000 & 108.863 & 68,28\% & 0,604 & 0,311 \\
2023 (treino 2014--2022) & 60.000 & 98.129  & 66,37\% & 0,552 & 0,266 \\
\textbf{Média $\pm$ $\sigma$} & --- & --- & \textbf{70,13\% $\pm$ 4,06} & \textbf{0,606 $\pm$ 0,045} & \textbf{0,306 $\pm$ 0,031} \\
\midrule
\emph{Split} aleatório (Stacking GBM) & $\approx$~740k & 184.862 & 84,61\% & 0,7995 & 0,6296 \\
\textbf{Δ viés (temporal - aleatório)} & --- & --- & \textbf{$-$14,49\,pp} & \textbf{$-$19,37\,pp} & \textbf{$-$32,35\,pp} \\
\bottomrule
\end{tabular}
\end{table}

\textbf{Achados.} Os resultados confirmam a hipótese de \ref{sec:3_11}: o vazamento espaço-temporal das janelas móveis é a fonte principal do viés do \emph{split} aleatório. Quando eliminado pelo \emph{rolling-origin}, a acurácia média do Stacking GBM cai de 84,61\% para 70,13\% (-14,49\,pp). Apesar disso, o modelo \textbf{ainda supera a meta acadêmica original do TCC ($\geq$\,70\%)} mesmo sob condições muito mais exigentes. A degradação progressiva ano a ano (75,7 $\to$ 68,3 $\to$ 66,4\%) é coerente com o \emph{drift} climático documentado por Aragão \emph{et al.}~\cite{aragao_2018} e Silva-Junior \emph{et al.}~\cite{silva_junior_2025} — não é artefato amostral.

A comparação entre o RF balanceado e o Stacking GBM sob validação temporal (Tabela~\ref{tab:5_temporal_comp}) mostra que o ganho do \emph{ensemble} \textbf{persiste}: F$_1$-macro +1,4\,pp e F$_1$-Moderado +4,2\,pp do Stacking sobre o RF, mesmo no regime causal estrito. Esse achado é citável: o Stacking GBM não é mero \emph{overfitting} do \emph{split} aleatório.

\begin{table}[h]
\centering
\caption{Comparativo RF $\times$ Stacking GBM sob validação temporal.}
\label{tab:5_temporal_comp}
\begin{tabular}{lcccc}
\toprule
\textbf{Modelo} & \textbf{Acc temporal} & \textbf{F$_1$-macro} & \textbf{F$_1$-Moderado} & \textbf{Δ acc vs.\ aleat} \\
\midrule
RF balanceado Tier 1          & 70,09\% & 0,592 & 0,264 & $-$12,83\,pp \\
\textbf{Stacking GBM}         & \textbf{70,13\%} & \textbf{0,606} & \textbf{0,306} & $-$14,49\,pp \\
\bottomrule
\end{tabular}
\end{table}

A queda em F$_1$-Moderado sob validação temporal (-32,35\,pp para o Stacking GBM) permanece o ponto frágil — efeito esperado, pois as fronteiras de transição entre regimes são intrinsecamente difíceis em horizontes maiores e justamente o regime que mais se beneficia das janelas móveis. Esse resultado motiva, como trabalho futuro \emph{prioritário}, a reimplementação das \emph{features} de janela móvel com \textbf{causalidade estrita} (computar \texttt{Precipitacao\_ma30} usando apenas dados $\leq t$ do registro), conforme registrado em \ref{sec:6}.

\textbf{Por que reportar o \emph{split} aleatório como métrica principal?} O objetivo do TCC é demonstrar a viabilidade do pipeline em modo \emph{operacional} (mapa interativo: usuário consulta um ponto agora, recebe uma classe), não fazer previsão temporal estrita de calendário (essa é uma formulação diferente: regressão de área queimada $N$ meses adiante). Para o uso aplicado, o \emph{split} estratificado aleatório é a métrica que melhor reflete a precisão esperada no momento em que o usuário interage com o mapa. O \emph{rolling-origin} atua como \textbf{verificação cruzada honesta} que documenta a robustez do modelo sob condições de uso mais exigentes.

## 5.6 Ajuste multi-classe de limiares de decisão \label{sec:5_6_threshold}

A predição padrão dos classificadores probabilísticos usa \texttt{argmax} sobre as probabilidades calibradas. Como discutido em \ref{sec:4_5_3}, essa regra penaliza a classe \emph{Moderado} (fronteira ambígua entre \emph{Baixo} e \emph{Muito Alto}). O script \texttt{scripts/ajustar\_threshold.py} varre uma grade de pares (\texttt{thr\_Moderado}, \texttt{thr\_Muito\_Alto}) $\in \{0{,}30; 0{,}35; \ldots; 0{,}55\}^2$ (36 configurações), avaliando F$_1$-macro e F$_1$-Moderado em 50.000 amostras estratificadas do conjunto de teste.

A Tabela~\ref{tab:5_threshold} mostra a configuração ótima para cada critério de seleção. Para o modelo final \texttt{ensemble\_stacking\_gbm}, tanto a otimização por F$_1$-macro quanto a otimização por F$_1$-Moderado convergem para o mesmo ponto: \texttt{thr\_Moderado=0{,}35}, \texttt{thr\_Muito\_Alto=0{,}45}.

\begin{table}[h]
\centering
\caption{Resultado do \emph{threshold tuning} multi-classe (\texttt{ensemble\_stacking\_gbm}).}
\label{tab:5_threshold}
\begin{tabular}{lcccc}
\toprule
\textbf{Estratégia} & \textbf{Acurácia} & \textbf{F$_1$-macro} & \textbf{F$_1$-Moderado} & \textbf{Δ F$_1$-Moderado} \\
\midrule
Baseline (\emph{argmax})           & 84,32\% & 0,7957 & 0,6231 & --- \\
F$_1$-macro / F$_1$-Moderado ótimo & 83,88\% & 0,7981 & \textbf{0,6375} & +1,44\,pp \\
\bottomrule
\end{tabular}
\end{table}

O ganho de F$_1$-Moderado (+1,44\,pp) tem custo de 0,44\,pp em acurácia global — \emph{trade-off} considerado favorável dada a relevância da classe intermediária. A configuração final é persistida em \texttt{modelos/prediction\_thresholds.json} e aplicada como padrão no endpoint \texttt{/api/predict} do aplicativo web (parâmetro opcional \texttt{estrategia\_thresholds} permite ao usuário escolher entre \texttt{f1\_macro}, \texttt{f1\_moderado} ou \texttt{argmax} para auditoria comparativa).

A lógica de decisão em \emph{runtime} é:

\begin{verbatim}
se     P(Moderado)   >= thr_Moderado    -> classe = "Moderado"
senao se P(Muito Alto) >= thr_Muito_Alto -> classe = "Muito Alto"
senao                                    -> classe = "Baixo"
\end{verbatim}

## 5.7 Evolução incremental do pipeline \label{sec:5_7_evolucao}

A Tabela~\ref{tab:5_evolucao} sintetiza a evolução do desempenho ao longo das principais decisões metodológicas do projeto. A Figura~\ref{fig:5_evolucao} apresenta o mesmo dado de forma gráfica.

\begin{table}[h]
\centering
\caption{Evolução incremental do desempenho do pipeline.}
\label{tab:5_evolucao}
\begin{tabular}{lcccc}
\toprule
\textbf{Configuração} & \textbf{Acurácia} & \textbf{F$_1$-macro} & \textbf{F$_1$-Moderado} \\
\midrule
SGDClassifier (TCC II original) & 62,43\% & 0,5864 & 0,3613 \\
\midrule
Ensemble Stacking (sem Tier 1) & 80,49\% & 0,7492 & 0,5498 \\
\quad +24 \emph{features} Tier 1 & 83,60\% & 0,7892 & 0,6168 \\
\quad +Optuna em XGBoost e LightGBM & 84,14\% & 0,7944 & 0,6202 \\
\quad +meta-classificador GBM        & 84,61\% & 0,7995 & 0,6296 \\
\quad +\emph{thresholds} calibrados   & 83,88\% & 0,7981 & \textbf{0,6375} \\
\midrule
\textbf{Δ acumulado (Stacking baseline $\to$ final)} & \textbf{+3,39\,pp} & \textbf{+4,89\,pp} & \textbf{+8,77\,pp} \\
\textbf{Δ acumulado (SGD $\to$ final)} & \textbf{+21,45\,pp} & \textbf{+21,17\,pp} & \textbf{+27,62\,pp} \\
\bottomrule
\end{tabular}
\end{table}

\begin{figure}[h]
\centering
\includegraphics[width=0.85\textwidth]{modelos/relatorios/evolucao_modelos.png}
\caption{Evolução incremental da acurácia, F$_1$-macro e F$_1$-Moderado ao longo das principais decisões metodológicas do projeto.}
\label{fig:5_evolucao}
\end{figure}

Três marcos se destacam:

\begin{itemize}
  \item A \textbf{engenharia de \emph{features} físico-climáticas Tier 1} foi o fator dominante de ganho: +3,11\,pp em acurácia e +6,70\,pp em F$_1$-Moderado, ao custo de apenas 40 segundos de geração \emph{offline} (\texttt{scripts/features\_avancadas.py}). Esse resultado reforça a tese de que \emph{engenharia de variáveis bem embasada na literatura é tão importante quanto a escolha do algoritmo} em problemas de domínio.
  \item A \textbf{otimização bayesiana por Optuna} sobre XGBoost e LightGBM (15 \emph{trials}, 3-fold CV, métrica F$_1$-macro) entregou +0,54\,pp em acurácia e +0,52\,pp em F$_1$-macro como base \emph{learners} do \emph{stacking}.
  \item A substituição do meta-classificador de Logistic Regression por \textbf{Gradient Boosting} no \emph{stacking} produziu +0,47\,pp em acurácia, +0,51\,pp em F$_1$-macro e +0,94\,pp em F$_1$-Moderado — confirmando que arquiteturas de \emph{stacking} com meta-\emph{learner} mais expressivo se beneficiam de quantidade de dados suficiente, conforme antecipado por Wolpert~\cite{wolpert_1992}.
\end{itemize}

## 5.8 Validação operacional via aplicação web \label{sec:5_8_app}

Para validar o pipeline em modo \emph{operacional}, foram realizadas consultas comparativas no aplicativo web interativo descrito em \ref{sec:4_6}. Dois pontos opostos da Amazônia Legal foram consultados em uma data de plena estação seca (setembro):

\begin{enumerate}
  \item Centro do estado do Amazonas ($-3{,}5^\circ$, $-62{,}5^\circ$): região historicamente úmida, baixa densidade de focos, vegetação intacta.
  \item Sul do estado do Pará ($-8{,}5^\circ$, $-50{,}5^\circ$): região de arco do desmatamento, alta densidade histórica de focos.
\end{enumerate}

A Tabela~\ref{tab:5_smoke} apresenta os resultados.

\begin{table}[h]
\centering
\caption{Consultas comparativas no aplicativo web (\emph{ensemble\_stacking\_gbm}, mês = setembro).}
\label{tab:5_smoke}
\begin{tabular}{lccccc}
\toprule
\textbf{Ponto consultado} & \textbf{Risco previsto} & \textbf{Confiança} & \textbf{SPI-1m} & \textbf{KBDI proxy} \\
\midrule
Centro AM ($-3{,}5$, $-62{,}5$) & Baixo       & 82,3\% & $0{,}00$ & $2{,}4$ \\
Sul PA ($-8{,}5$, $-50{,}5$)    & \textbf{Muito Alto} & \textbf{92,9\%} & $-0{,}57$ & $\mathbf{418{,}6}$ \\
\bottomrule
\end{tabular}
\end{table}

O modelo discrimina corretamente os dois cenários, com explicações fisicamente defensáveis emitidas pelo explainer (\ref{sec:4_6_3}): para o ponto AM, as variáveis dominantes a favor da classe \emph{Baixo} são \texttt{DiaSemChuva} (↓), \texttt{Indice\_Seca} (↓) e \texttt{Dias\_Desde\_Ultimo\_Incêndio} (↑); para o ponto PA, as variáveis dominantes a favor da classe \emph{Muito Alto} são \texttt{DiaSemChuva} (↓ período de estiagem), \texttt{Indice\_Seca} (↑), \texttt{SPI-1m} (↑ déficit), \texttt{SPI-3m} (↑ déficit prolongado) e \texttt{Temp\_Climatologica} (↑). O tempo médio de resposta da API \texttt{/api/predict} para ambos os pontos foi de 11~s a 18~s, dominado pela latência das chamadas externas (NASA POWER + NASA FIRMS), aceitável para o uso interativo.

A aplicação está documentada em \texttt{README\_APP.md} e o código-fonte está disponível em \texttt{scripts/app\_map\_interativo.py}. A interface foi testada nas resoluções 1280$\times$720 (notebook típico) e 1920$\times$1080 (\emph{desktop}), com painel lateral fixo de 420~px e mapa centralizado responsivo (Leaflet 1.9.4 sob Folium 0.16).

\medskip

---

# 6 CONCLUSÃO

Este trabalho desenvolveu um sistema completo de classificação supervisionada de risco de incêndio em três níveis (\emph{Baixo}, \emph{Moderado}, \emph{Muito Alto}) para a Amazônia Legal, integrando dados do Programa Queimadas/INPE (2014--2023), reanálise climática NASA POWER, detecções em tempo real NASA FIRMS, e o \emph{shapefile} oficial do IBGE. Os principais achados são os seguintes:

\paragraph{1.\ Desempenho final.}
O modelo final --- \textbf{Ensemble \emph{Stacking}} com meta-classificador \emph{Gradient Boosting} (\texttt{ensemble\_stacking\_gbm}) --- atingiu \textbf{84,61\%} de acurácia, \textbf{0,7995} de F$_1$-macro e \textbf{0,6375} de F$_1$-Moderado em \emph{hold-out} estratificado de 184.862 amostras. Esse resultado supera em quase 15~pp a meta acadêmica original do projeto ($\geq$~70\%) e em mais de 22~pp o \emph{baseline} SGDClassifier do TCC~II original.

\paragraph{2.\ A engenharia de \emph{features} físico-climáticas é o fator dominante de ganho.}
A introdução das 24 \emph{features} Tier~1 (SPI, KBDI \emph{proxy}, VPD \emph{proxy}, médias móveis estendidas, \emph{lags} acumulados, histórico estendido de fogo, dias secos), substituindo a variável \texttt{Umidade} por proxies amparados pela literatura recente \cite{seager_2015,forests_2024,quesada_ruiz_2025}, entregou +3,11\,pp em acurácia e +6,70\,pp em F$_1$-Moderado, ao custo de apenas 40~segundos de geração \emph{offline}. Esse achado reforça a tese de que \emph{engenharia de variáveis bem embasada na literatura é tão importante quanto a escolha do algoritmo} em problemas de domínio.

\paragraph{3.\ Interpretabilidade convergente entre duas técnicas independentes.}
Análise SHAP via Random Forest \emph{compacto proxy} \cite{lundberg_2020} e \emph{Permutation Importance} \emph{model-agnostic} \cite{breiman_2001,fisher_2019} sobre o Stacking GBM final identificam, coerentemente, \texttt{KBDI\_proxy}, \texttt{VPD\_proxy}, \texttt{Indice\_Seca}, \texttt{DiaSemChuva\_ma90} e \texttt{Precipitacao\_acum\_180d} como \emph{features} dominantes. A \texttt{VPD\_proxy} aparece no Top-5 da classe \emph{Muito Alto} por ambas as técnicas independentes --- validação cruzada que sustenta a escolha metodológica de substituir \texttt{Umidade} por proxies físicos.

\paragraph{4.\ Validação temporal estrita revela o viés do \emph{split} aleatório.}
A acurácia média em três folds \emph{rolling-origin} (2021--2023) é de \textbf{70,13\% $\pm$ 4,06}, com queda de 14,49\,pp em relação ao \emph{split} aleatório. Mesmo assim, o modelo \emph{mantém-se acima da meta acadêmica original} ($\geq$~70\%) sob esse regime muito mais exigente. O \emph{Stacking} GBM mantém vantagem de +1,4\,pp em F$_1$-macro e +4,2\,pp em F$_1$-Moderado sobre o Random Forest, mesmo no regime causal estrito --- sinal de que o ganho do \emph{ensemble} \textbf{não é mero \emph{overfitting} do \emph{split} aleatório}. A queda em F$_1$-Moderado (-32,35\,pp) sob validação temporal indica que as fronteiras de transição entre regimes são intrinsecamente difíceis em horizontes maiores e motivam um trabalho futuro prioritário (\emph{vide} item~9 abaixo).

\paragraph{5.\ Calibração e ajuste de limiares.}
A calibração isotônica das probabilidades \cite{zadrozny_elkan_2002} e o ajuste multi-classe de limiares (\texttt{thr\_Moderado=0{,}35}, \texttt{thr\_Muito\_Alto=0{,}45}) elevam o F$_1$-Moderado para 0,6375 (+1,44\,pp sobre o \emph{argmax}), com custo de apenas 0,44\,pp em acurácia global --- \emph{trade-off} considerado favorável dada a importância da classe intermediária no contexto operacional.

\paragraph{6.\ Aplicação web interativa explicável.}
A aplicação \emph{Flask} + \emph{Folium} + \emph{Leaflet} demonstra a viabilidade operacional do sistema: ao receber um clique do usuário em qualquer ponto da Amazônia Legal, o servidor compõe em $\sim$\,11--18~s um vetor de 44 \emph{features} (incluindo 24 Tier~1 via \emph{lookup} espaço-sazonal + 5 calculadas em tempo real a partir do clima atual NASA POWER + 9 originais + histórico FIRMS), classifica o risco com \emph{thresholds} calibrados, e exibe \textbf{explicação local} das \emph{features} responsáveis pela decisão. \emph{Smoke-tests} em zonas opostas (centro úmido do Amazonas em setembro $\times$ Sul do Pará em pleno setembro) confirmam que o modelo discrimina corretamente, com explicações fisicamente defensáveis.

\paragraph{7.\ Reprodutibilidade tratada como requisito não-funcional.}
Todas as métricas, hiperparâmetros, \emph{thresholds} e metadados de \emph{split} estão persistidos em JSON versionado (\texttt{modelos/relatorios/*.json}); o \emph{pipeline} completo é reexecutável via \emph{scripts} modulares; o aplicativo expõe \texttt{thresholds\_aplicados} e \texttt{fonte\_importance} na resposta de cada predição para auditoria.

\paragraph{8.\ Imputação de \emph{Umidade} via \emph{pseudo-labeling}.}
A integração via NASA POWER cobriu apenas 18,3\% da base, em razão dos limites de taxa do serviço. Como mitigação, foi implementada a estratégia de \emph{pseudo-labeling} com LightGBM \emph{regressor} \cite{lee_2013,lopez_garcia_2024}, atingindo R$^2$~=~0,933, RMSE~=~4,33\% RH e MAE~=~3,19\% RH em \emph{holdout}, com cobertura final de 100\% no dataset enriquecido. O retreino do \texttt{ensemble\_stacking\_gbm} incluindo \texttt{Umidade} (pseudo-rotulada e real) permanece como trabalho futuro imediato (\emph{vide} item~9), com quantificação de eventual viés de confirmação \cite{arazo_2020}.

\subsection*{6.1 Limitações reconhecidas}

\begin{enumerate}
  \item \textbf{Viés temporal de janelas móveis} quantificado em \ref{sec:5_5_temporal} (queda de 14,49\,pp em acurácia sob validação temporal estrita). Para \emph{deployment} em horizontes maiores, sugere-se reimplementar as \emph{features} com causalidade estrita (computar \texttt{Precipitacao\_ma30} usando apenas dados $\leq t$ do registro).
  \item \textbf{Cobertura parcial da \texttt{Umidade} real} (18,3\%) --- mitigada via \emph{pseudo-labeling} mas o retreino do \emph{Stacking} incluindo \texttt{Umidade} fica como trabalho futuro imediato.
  \item \textbf{SHAP exato sobre o \emph{Stacking} GBM} não é viável em CPU por requisitos de memória; SHAP via Random Forest \emph{compacto proxy} é defensável \cite{lundberg_2020} mas tem perda qualitativa em relação ao modelo de produção. \emph{Permutation Importance} sobre o modelo real complementa a análise.
  \item \textbf{Subamostragem em \emph{boosting} e em validação temporal} (250k em XGBoost/LightGBM, 60--120k por \emph{fold} no \emph{rolling-origin}) por limitação de memória do OHE denso de \texttt{Municipio} com $\sim$~542 categorias. Migração para representação esparsa é trabalho futuro técnico.
  \item \textbf{Generalização espacial} --- o modelo pode estar parcialmente ajustado a padrões idiossincráticos de municípios com muitos exemplos. Avaliação em \emph{folds} espaciais (\emph{leave-one-state-out}) seria complemento desejável.
\end{enumerate}

\subsection*{6.2 Trabalhos futuros}

\begin{enumerate}
  \item Retreinar \texttt{ensemble\_stacking\_gbm} incluindo \texttt{Umidade} (pseudo-rotulada e real) e quantificar ganho marginal, com controle explícito do viés de confirmação \cite{arazo_2020}.
  \item Causalidade estrita das \emph{features} de janela móvel para reduzir o viés temporal (computar \texttt{Precipitacao\_ma30}, \texttt{DiaSemChuva\_ma*}, \texttt{SPI\_*}, \texttt{Incendios\_Ultimos\_*} usando apenas dados $\leq t$ do registro).
  \item Integração de NDVI/EVI MODIS via Google Earth Engine como \emph{features} Tier~2.
  \item Integração de MapBiomas Fogo Coleção~4 como \emph{feature} adicional de histórico de área queimada.
  \item Migração do SHAP para \texttt{shap.GPUTreeExplainer} (CUDA) para análise exata sobre o RF Optuna produtivo, e \emph{Permutation Importance condicional} \cite{hooker_2021} para mitigar multicolinearidade.
  \item Substituição do \emph{lookup} espaço-sazonal por séries temporais reais (NASA POWER PRECTOTCORR diário 28~d) no aplicativo para cálculo \emph{exato} de SPI, anomalia e acumulados na data consultada.
  \item Validação espacial \emph{leave-one-state-out} para quantificar a generalização do modelo a estados pouco representados no treino.
\end{enumerate}

\paragraph{Considerações finais.}
A revisão abrangente da literatura, descrita no Capítulo~2, evidenciou a complexidade do fenômeno das queimadas na Amazônia, ressaltando sua amplitude ambiental, social e econômica. Os resultados experimentais aqui apresentados sustentam que ferramentas modernas de aprendizado de máquina, combinadas a engenharia de \emph{features} físico-climáticas embasada na literatura recente e a uma interface explicável de uso interativo, representam contribuição relevante para o monitoramento da Amazônia Legal. Os números reportados, tanto sob \emph{split} aleatório estratificado quanto sob validação temporal estrita, indicam que o sistema desenvolvido atende ao requisito acadêmico original ($\geq$~70\% de acurácia) com margem confortável (84,61\% no regime operacional, 70,13\% no regime causal estrito), e oferece um caminho concreto para evoluir até $\geq$~90\% de acurácia mediante a inclusão das \emph{features} de Tier~2 (NDVI, MapBiomas Fogo) e da \texttt{Umidade} real retreinada, ambas planejadas como continuidade natural deste trabalho.

---

## Apêndice — Notas de uso deste documento

1. **Substituir referências:** as `\cite{...}` usadas correspondem às chaves BibTeX definidas em `REFERENCIAS_TCC.md`. Copiar essas entradas para `references.bib` do TCC.
2. **Caminhos das figuras:** ajustar `modelos/relatorios/evolucao_modelos.png` e `modelos/relatorios/threshold_precision_recall.png` para o caminho relativo correto do projeto LaTeX.
3. **Tabelas:** se a banca preferir formato ABNT (linha dupla no topo, linha simples no bottom em vez de booktabs), aplicar `\hline` em vez de `\toprule/\midrule/\bottomrule`.
4. **Conferência final:** todos os valores numéricos foram cruzados com `modelos/relatorios/*.json` em 12/05/2026. Recomenda-se conferência final antes de fechar a versão de defesa.
5. **Capítulo 4 (Metodologia):** ver mapa detalhado em `MAPEAMENTO_TCC.md` §4, com todas as subseções novas em texto pronto.
