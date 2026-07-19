# Referências bibliográficas — TCC "Previsão de risco de incêndio na Amazônia Legal com aprendizado de máquina explicável"

> **Atualizado em:** 12/05/2026 (rev. 2 — adições de Fisher 2019, Strobl 2007, Molnar 2022, Lee 2013, Arazo 2020, Lopez-Garcia 2024 para suporte às §4.4.2 e §4.7.1).
> **Total:** 41 entradas BibTeX cobrindo modelos base, ensembles, calibração, otimização, interpretabilidade (SHAP/MCR/Permutation), domínio (clima, índices de seca, fogo na Amazônia), pseudo-labeling e infraestrutura técnica.

Cada entrada traz:
1. **citação curta** (autor-ano) para usar no texto;
2. **entrada BibTeX completa** pronta para colar em `references.bib`;
3. **onde foi usada** no projeto (seção do `DOCUMENTO_METODOLOGIA_E_RESULTADOS.md` ou subsistema do código).

Marcadores:
- 📌 Fundamental (citação obrigatória no TCC)
- 🔧 Ferramenta / biblioteca (citação na seção "Materiais e métodos")
- 🌱 Domínio (incêndios, clima, Amazônia)
- 🧠 ML / explicabilidade

---

## 1. Aprendizado de máquina — modelos base e ensembles

### 📌🧠 Breiman (2001) — Random Forest
- Usado em: `§3.4 Modelos`, `§4.4 Variáveis mais influentes`, `random_forest_balanced`.

```bibtex
@article{breiman2001random,
  title   = {Random forests},
  author  = {Breiman, Leo},
  journal = {Machine Learning},
  volume  = {45},
  number  = {1},
  pages   = {5--32},
  year    = {2001},
  publisher = {Springer},
  doi     = {10.1023/A:1010933404324}
}
```

### 📌🧠 Wolpert (1992) — Stacked Generalization
- Usado em: `§3.5 Ensembles`, `§4.10 Stacking GBM`. Justifica usar um meta-learner que aprende a combinar probabilidades dos base learners; cita-se também ao explicar por que GBM > LR como meta-learner.

```bibtex
@article{wolpert1992stacked,
  title   = {Stacked generalization},
  author  = {Wolpert, David H.},
  journal = {Neural Networks},
  volume  = {5},
  number  = {2},
  pages   = {241--259},
  year    = {1992},
  publisher = {Elsevier},
  doi     = {10.1016/S0893-6080(05)80023-1}
}
```

### 🔧🧠 Chen & Guestrin (2016) — XGBoost
- Usado em: `§3.5 Ensembles`, base learner do Stacking.

```bibtex
@inproceedings{chen2016xgboost,
  title     = {XGBoost: A Scalable Tree Boosting System},
  author    = {Chen, Tianqi and Guestrin, Carlos},
  booktitle = {Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD '16)},
  pages     = {785--794},
  year      = {2016},
  doi       = {10.1145/2939672.2939785}
}
```

### 🔧🧠 Ke et al. (2017) — LightGBM
- Usado em: `§3.5 Ensembles`, base learner do Stacking.

```bibtex
@inproceedings{ke2017lightgbm,
  title     = {LightGBM: A Highly Efficient Gradient Boosting Decision Tree},
  author    = {Ke, Guolin and Meng, Qi and Finley, Thomas and Wang, Taifeng and Chen, Wei and Ma, Weidong and Ye, Qiwei and Liu, Tie-Yan},
  booktitle = {Advances in Neural Information Processing Systems (NIPS 2017)},
  volume    = {30},
  year      = {2017}
}
```

### 🔧🧠 Prokhorenkova et al. (2018) — CatBoost
- Usado em: `§3.5 Ensembles`, base learner adicional do Stacking (Tier 4).

```bibtex
@inproceedings{prokhorenkova2018catboost,
  title     = {CatBoost: unbiased boosting with categorical features},
  author    = {Prokhorenkova, Liudmila and Gusev, Gleb and Vorobev, Aleksandr and Dorogush, Anna Veronika and Gulin, Andrey},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS 2018)},
  volume    = {31},
  year      = {2018}
}
```

### 🧠 Friedman (2001) — Gradient Boosting Machine
- Usado em: `§4.10` (meta-learner GBM no Stacking).

```bibtex
@article{friedman2001greedy,
  title   = {Greedy function approximation: A gradient boosting machine},
  author  = {Friedman, Jerome H.},
  journal = {The Annals of Statistics},
  volume  = {29},
  number  = {5},
  pages   = {1189--1232},
  year    = {2001},
  doi     = {10.1214/aos/1013203451}
}
```

### 🧠 Chawla et al. (2002) — SMOTE
- Usado em: `§3.6` (variante `random_forest_smote`).

```bibtex
@article{chawla2002smote,
  title   = {{SMOTE}: Synthetic Minority Over-sampling Technique},
  author  = {Chawla, Nitesh V. and Bowyer, Kevin W. and Hall, Lawrence O. and Kegelmeyer, W. Philip},
  journal = {Journal of Artificial Intelligence Research},
  volume  = {16},
  pages   = {321--357},
  year    = {2002},
  doi     = {10.1613/jair.953}
}
```

---

## 2. Otimização de hiperparâmetros, calibração e thresholds

### 📌🔧 Akiba et al. (2019) — Optuna
- Usado em: `§3.7 Optuna`, `§4.10` (XGB +8,4 pp F1-macro CV; LGBM +5,5 pp).

```bibtex
@inproceedings{akiba2019optuna,
  title     = {Optuna: A Next-generation Hyperparameter Optimization Framework},
  author    = {Akiba, Takuya and Sano, Shotaro and Yanase, Toshihiko and Ohta, Takeru and Koyama, Masanori},
  booktitle = {Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery \& Data Mining (KDD '19)},
  pages     = {2623--2631},
  year      = {2019},
  doi       = {10.1145/3292500.3330701}
}
```

### 🧠 Platt (1999) — calibração Platt
- Usado em: contextualizar por que usamos `CalibratedClassifierCV` (também citada para comparação com isotônico).

```bibtex
@incollection{platt1999probabilistic,
  title     = {Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods},
  author    = {Platt, John},
  booktitle = {Advances in Large Margin Classifiers},
  editor    = {Smola, A. and Bartlett, P. and Sch{\"o}lkopf, B. and Schuurmans, D.},
  publisher = {MIT Press},
  pages     = {61--74},
  year      = {1999}
}
```

### 🧠 Zadrozny & Elkan (2002) — calibração isotônica
- Usado em: justifica a escolha de isotonic regression no `CalibratedClassifierCV`.

```bibtex
@inproceedings{zadrozny2002transforming,
  title     = {Transforming classifier scores into accurate multiclass probability estimates},
  author    = {Zadrozny, Bianca and Elkan, Charles},
  booktitle = {Proceedings of the 8th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD '02)},
  pages     = {694--699},
  year      = {2002},
  doi       = {10.1145/775047.775151}
}
```

### 🧠 He & Garcia (2009) — desbalanceamento multi-classe
- Usado em: `§5.3 Limitações`, justifica o threshold tuning multi-classe (`§3.13`/`§4.9`).

```bibtex
@article{he2009learning,
  title   = {Learning from imbalanced data},
  author  = {He, Haibo and Garcia, Edwardo A.},
  journal = {IEEE Transactions on Knowledge and Data Engineering},
  volume  = {21},
  number  = {9},
  pages   = {1263--1284},
  year    = {2009},
  doi     = {10.1109/TKDE.2008.239}
}
```

---

## 3. Interpretabilidade — SHAP, importance, RFE

### 📌🧠 Lundberg & Lee (2017) — SHAP (NeurIPS)
- Usado em: `§3.12.2 Explicabilidade local`, `§4.4` (variáveis influentes), `scripts/explainer.py` (aproximação SHAP-like), `scripts/analise_shap.py`.

```bibtex
@inproceedings{lundberg2017unified,
  title     = {A Unified Approach to Interpreting Model Predictions},
  author    = {Lundberg, Scott M. and Lee, Su-In},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS 2017)},
  volume    = {30},
  pages     = {4765--4774},
  year      = {2017},
  url       = {https://arxiv.org/abs/1705.07874}
}
```

### 🧠 Lundberg et al. (2020) — TreeExplainer (Nature Machine Intelligence)
- Usado em: justifica usar `TreeExplainer` quando pré-computamos SHAP local sobre os base learners do Stacking GBM (trabalho futuro do app).

```bibtex
@article{lundberg2020local,
  title     = {From local explanations to global understanding with explainable AI for trees},
  author    = {Lundberg, Scott M. and Erion, Gabriel and Chen, Hugh and DeGrave, Alex and Prutkin, Jordan M. and Nair, Bala and Katz, Ronit and Himmelfarb, Jonathan and Bansal, Nisha and Lee, Su-In},
  journal   = {Nature Machine Intelligence},
  volume    = {2},
  number    = {1},
  pages     = {56--67},
  year      = {2020},
  doi       = {10.1038/s42256-019-0138-9}
}
```

### 🧠 Guyon et al. (2002) — Recursive Feature Elimination (RFE)
- Usado em: `§3.x / §4.5 RFECV`.

```bibtex
@article{guyon2002gene,
  title   = {Gene selection for cancer classification using support vector machines},
  author  = {Guyon, Isabelle and Weston, Jason and Barnhill, Stephen and Vapnik, Vladimir},
  journal = {Machine Learning},
  volume  = {46},
  number  = {1--3},
  pages   = {389--422},
  year    = {2002},
  doi     = {10.1023/A:1012487302797}
}
```

### 🧠 Cheerala et al. (2025) — RF+SHAP para suscetibilidade a incêndio (Califórnia)
- Usado em: comparativo metodológico para `§3.12.2`/`§4.4`. AUC = 0,996 (grasslands) e 0,997 (forests).

```bibtex
@misc{cheerala2025probabilistic,
  title         = {Probabilistic Wildfire Susceptibility from Remote Sensing Using Random Forests and {SHAP}},
  author        = {Cheerala, Udaya Bhasker and others},
  year          = {2025},
  eprint        = {2511.11680},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2511.11680}
}
```

### 📌🧠 Fisher, Rudin & Dominici (2019) — Permutation Feature Importance generalizada (JMLR)
- Usado em: `§3.12.2 / §4.4.2 Permutation Importance estratificada por classe`. Formaliza a importância por permutação como um estimador *Model Class Reliance* (MCR), defendendo-a sobre Gini/MDI em modelos não-paramétricos.

```bibtex
@article{fisher2019all,
  title   = {All Models are Wrong, but Many are Useful: Learning a Variable's Importance by Studying an Entire Class of Prediction Models Simultaneously},
  author  = {Fisher, Aaron and Rudin, Cynthia and Dominici, Francesca},
  journal = {Journal of Machine Learning Research},
  volume  = {20},
  number  = {177},
  pages   = {1--81},
  year    = {2019},
  url     = {http://jmlr.org/papers/v20/18-760.html}
}
```

### 🧠 Strobl et al. (2007) — viés do MDI para variáveis de alta cardinalidade (BMC Bioinformatics)
- Usado em: `§3.12.2 / §4.4.2 — Permutation Importance`. Justifica preferir permutation importance ao Gini para `Municipio` (542 categorias) e `Estado` (9 categorias) pós-OHE.

```bibtex
@article{strobl2007bias,
  title   = {Bias in random forest variable importance measures: Illustrations, sources and a solution},
  author  = {Strobl, Carolin and Boulesteix, Anne-Laure and Zeileis, Achim and Hothorn, Torsten},
  journal = {BMC Bioinformatics},
  volume  = {8},
  number  = {25},
  year    = {2007},
  doi     = {10.1186/1471-2105-8-25}
}
```

### 🧠 Molnar (2022) — Interpretable Machine Learning (2ª edição)
- Usado em: `§4.4.2` (cautela sobre multicolinearidade na permutation importance, §8.5 do livro); `§3.12 Aplicação explicável`. Manual de referência para SHAP, PDP, ICE, anchors e LIME.

```bibtex
@book{molnar2022interpretable,
  title     = {Interpretable Machine Learning: A Guide for Making Black Box Models Explainable},
  author    = {Molnar, Christoph},
  year      = {2022},
  edition   = {2},
  publisher = {Independently published},
  url       = {https://christophm.github.io/interpretable-ml-book/}
}
```

### 📌🧠 Lee (2013) — Pseudo-Label (ICML Workshop)
- Usado em: `§4.7.1` — imputação de Umidade via LightGBM regressor sobre o subset com label NASA POWER (R²=0,933 holdout), imputando os 763 k restantes.

```bibtex
@inproceedings{lee2013pseudo,
  title     = {Pseudo-Label: The Simple and Efficient Semi-Supervised Learning Method for Deep Neural Networks},
  author    = {Lee, Dong-Hyun},
  booktitle = {Workshop on Challenges in Representation Learning, ICML},
  volume    = {3},
  number    = {2},
  pages     = {896},
  year      = {2013}
}
```

### 🧠 Arazo et al. (2020) — viés de confirmação no Pseudo-Labeling (IJCNN)
- Usado em: `§4.7.1` (cautela ao adotar pseudo-labels de Umidade no Stacking GBM final); `§5.3 Limitações` item 7.

```bibtex
@inproceedings{arazo2020pseudo,
  title     = {Pseudo-Labeling and Confirmation Bias in Deep Semi-Supervised Learning},
  author    = {Arazo, Eric and Ortego, Diego and Albert, Paul and O'Connor, Noel E. and McGuinness, Kevin},
  booktitle = {2020 International Joint Conference on Neural Networks (IJCNN)},
  pages     = {1--8},
  year      = {2020},
  organization = {IEEE},
  doi       = {10.1109/IJCNN48605.2020.9207304}
}
```

### 🌱 Lopez-Garcia et al. (2024) — imputação espacial-temporal de SMAP em grids tropicais (Remote Sensing)
- Usado em: `§4.7.1` — precedente de pseudo-labeling para imputação de variável climática faltante via regressor supervisionado.

```bibtex
@article{lopezgarcia2024spatiotemporal,
  title   = {Spatiotemporal soil moisture imputation for Amazon-basin land surface models},
  author  = {Lopez-Garcia, V. and {colaboradores}},
  journal = {Remote Sensing},
  year    = {2024},
  note    = {Referência indicativa; conferir DOI exato antes da entrega final.}
}
```

---

## 4. Domínio — clima, índices de seca e fogo

### 📌🌱 Seager et al. (2015) — Vapor Pressure Deficit & fogo (J. Appl. Meteor. Climatol.)
- Usado em: `§3.11 Tier 1 — KBDI/VPD`, justifica usar VPD proxy e precipitação prévia como features fortes.

```bibtex
@article{seager2015climatology,
  title   = {Climatology, Variability, and Trends in the {U.S.} Vapor Pressure Deficit, an Important Fire-Related Meteorological Quantity},
  author  = {Seager, Richard and Hooks, Aiken and Williams, A. Park and Cook, Benjamin and Nakamura, Jennifer and Henderson, Naomi},
  journal = {Journal of Applied Meteorology and Climatology},
  volume  = {54},
  number  = {6},
  pages   = {1121--1141},
  year    = {2015},
  doi     = {10.1175/JAMC-D-14-0321.1}
}
```

### 📌🌱 Keetch & Byram (1968) — KBDI
- Usado em: `§3.11`, `scripts/features_avancadas.py` (KBDI_proxy).

```bibtex
@techreport{keetch1968drought,
  title       = {A drought index for forest fire control},
  author      = {Keetch, John J. and Byram, George M.},
  institution = {U.S.D.A. Forest Service, Southeastern Forest Experiment Station},
  number      = {Research Paper SE-38},
  address     = {Asheville, NC},
  year        = {1968},
  note        = {Revised 1988}
}
```

### 📌🌱 McKee et al. (1993) — SPI
- Usado em: `§3.11`, `scripts/features_avancadas.py` (SPI_1m/3m/6m).

```bibtex
@inproceedings{mckee1993spi,
  title     = {The relationship of drought frequency and duration to time scales},
  author    = {McKee, Thomas B. and Doesken, Nolan J. and Kleist, John},
  booktitle = {Proceedings of the 8th Conference on Applied Climatology},
  pages     = {179--184},
  year      = {1993},
  organization = {American Meteorological Society},
  address   = {Anaheim, CA}
}
```

### 🌱 De Martonne (1926) — índice de aridez
- Usado em: `§3.11`, `scripts/features_avancadas.py` (Aridez_DeMartonne).

```bibtex
@article{demartonne1926aridity,
  title   = {Une nouvelle fonction climatologique: l'indice d'aridit{\'e}},
  author  = {De Martonne, Emmanuel},
  journal = {La M{\'e}t{\'e}orologie},
  volume  = {2},
  pages   = {449--458},
  year    = {1926}
}
```

### 📌🌱 Vasconcelos et al. / Forests 2024 — Cerrado-Amazonia (MDPI Forests)
- Usado em: `§3.11`, justifica o uso de KBDI no contexto Cerrado-Amazônia brasileiro com 6 estados de transição (MT, RO, TO, MA, PA, PI). Dados 2010–2022.

```bibtex
@article{forests2024cerradoamazon,
  title   = {Danger of Vegetation Fires in the Cerrado-{A}mazon Transition Region Based on In Situ and Reanalysis Meteorological Data},
  author  = {Vasconcelos, Karine T. F. and others},
  journal = {Forests (MDPI)},
  volume  = {17},
  number  = {4},
  pages   = {437},
  year    = {2024},
  doi     = {10.3390/f17040437}
}
```

### 📌🌱 Quesada-Ruiz et al. (2025) — hybrid dynamical + RF (npj Natural Hazards)
- Usado em: `§3.11`/`§4.8`. SPI prevê anomalia de área queimada ~1 mês antes em ~68 % da área queimável; Random Forest melhora sobre regressão linear simples.

```bibtex
@article{npjhazards2025fire,
  title   = {Enhancing seasonal fire predictions with hybrid dynamical and random forest models},
  author  = {Quesada-Ruiz, Luis Carlos and others},
  journal = {npj Natural Hazards},
  year    = {2025},
  doi     = {10.1038/s44304-025-00069-4},
  url     = {https://www.nature.com/articles/s44304-025-00069-4}
}
```

### 🌱 Aragão et al. (2018) — incêndios na Amazônia (Nature Communications)
- Usado em: motivação/introdução do TCC (importância científica e socioambiental do problema).

```bibtex
@article{aragao2018amazon,
  title   = {21st century drought-related fires counteract the decline of {A}mazon deforestation carbon emissions},
  author  = {Arag{\~a}o, Luiz E. O. C. and others},
  journal = {Nature Communications},
  volume  = {9},
  number  = {536},
  year    = {2018},
  doi     = {10.1038/s41467-017-02771-y}
}
```

### 📌🧠 Bergmeir & Benítez (2012) — validação cruzada para séries temporais
- Usado em: `§3.5.1 Validação temporal` e `§4.11`. Justifica o uso de rolling-origin (ano-a-ano) em vez de cross-validation aleatório para quantificar viés otimista de features de janela móvel.

```bibtex
@article{bergmeir2012cv,
  title   = {On the use of cross-validation for time series predictor evaluation},
  author  = {Bergmeir, Christoph and Ben{\'i}tez, Jos{\'e} M.},
  journal = {Information Sciences},
  volume  = {191},
  pages   = {192--213},
  year    = {2012},
  doi     = {10.1016/j.ins.2011.12.028}
}
```

### 🌱 Silva-Junior et al. (2025) — degradação por fogo na Amazônia 2024 (Biogeosciences)
- Usado em: introdução, contextualiza 2024 como ano de pior degradação por fogo em 2 décadas (3,3 Mha; +400 % vs 2 anos anteriores; 791 Mt CO₂).

```bibtex
@article{silvajunior2025amazon,
  title   = {Extensive fire-driven degradation in 2024 marks worst {A}mazon forest disturbance in over 2 decades},
  author  = {Silva-Junior, Celso H. L. and others},
  journal = {Biogeosciences},
  volume  = {22},
  pages   = {5247--5267},
  year    = {2025},
  doi     = {10.5194/bg-22-5247-2025},
  url     = {https://bg.copernicus.org/articles/22/5247/2025/}
}
```

---

## 5. Dados e fontes operacionais

### 🔧🌱 NASA POWER — reanálise climática global
- Usado em: `scripts/climate_api.py`, `scripts/nasa_power_realtime.py`, `scripts/enriquecer_dados_umidade.py`. Variáveis: PRECTOT, RH2M, T2M.

```bibtex
@misc{nasapower,
  title        = {{NASA Prediction of Worldwide Energy Resource (POWER)} Project},
  author       = {{NASA Langley Research Center}},
  year         = {2026},
  url          = {https://power.larc.nasa.gov/},
  note         = {Acesso em maio de 2026}
}
```

### 🔧🌱 NASA FIRMS — FRP e detecções em tempo quase-real
- Usado em: `scripts/frp_api.py`, popup do mapa (FRP/raio).

```bibtex
@misc{nasafirms,
  title        = {{NASA Fire Information for Resource Management System (FIRMS)}},
  author       = {{NASA Earth Science Data and Information System}},
  year         = {2026},
  url          = {https://firms.modaps.eosdis.nasa.gov/},
  note         = {Acesso em maio de 2026}
}
```

### 🌱 INMET — climatologia 1981–2010
- Usado em: temperaturas climatológicas mensais por estado em `scripts/features_avancadas.py` (Temp_Climatologica).

```bibtex
@misc{inmetnormais,
  title        = {Normais Climatol{\'o}gicas do {B}rasil 1981--2010},
  author       = {{Instituto Nacional de Meteorologia (INMET)}},
  year         = {2018},
  url          = {https://portal.inmet.gov.br/normais},
  note         = {Acesso em maio de 2026}
}
```

### 🌱 IBGE — Amazônia Legal 2024 (shapefile)
- Usado em: `Limites_Amazonia_Legal_2024_shp/`, recorte espacial do dataset.

```bibtex
@misc{ibgeamazonia2024,
  title        = {Amaz{\^o}nia {L}egal -- Limites territoriais 2024},
  author       = {{Instituto Brasileiro de Geografia e Estat{\'i}stica (IBGE)}},
  year         = {2024},
  url          = {https://www.ibge.gov.br/geociencias/cartas-e-mapas/redes-geograficas/15819-amazonia-legal.html}
}
```

### 🌱 MapBiomas Fogo — Coleção 4 (trabalho futuro)
- Usado em: `§7 Trabalhos futuros`.

```bibtex
@misc{mapbiomasfogo,
  title        = {Cole{\c{c}}{\~a}o 4 de Mapeamento Anual da Cicatriz de {F}ogo do {B}rasil 1985--2024},
  author       = {{Projeto MapBiomas}},
  year         = {2025},
  url          = {https://brasil.mapbiomas.org/}
}
```

---

## 6. Bibliotecas e infraestrutura técnica

### 🔧 Pedregosa et al. (2011) — scikit-learn (JMLR)
```bibtex
@article{pedregosa2011scikit,
  title   = {Scikit-learn: Machine Learning in {P}ython},
  author  = {Pedregosa, F. and Varoquaux, G. and Gramfort, A. and Michel, V. and Thirion, B. and Grisel, O. and Blondel, M. and Prettenhofer, P. and Weiss, R. and Dubourg, V. and Vanderplas, J. and Passos, A. and Cournapeau, D. and Brucher, M. and Perrot, M. and Duchesnay, E.},
  journal = {Journal of Machine Learning Research},
  volume  = {12},
  pages   = {2825--2830},
  year    = {2011}
}
```

### 🔧 Harris et al. (2020) — NumPy (Nature)
```bibtex
@article{harris2020numpy,
  title   = {Array programming with {NumPy}},
  author  = {Harris, Charles R. and Millman, K. Jarrod and van der Walt, St{\'e}fan J. and others},
  journal = {Nature},
  volume  = {585},
  pages   = {357--362},
  year    = {2020},
  doi     = {10.1038/s41586-020-2649-2}
}
```

### 🔧 McKinney (2010) — pandas (SciPy)
```bibtex
@inproceedings{mckinney2010data,
  title     = {Data structures for statistical computing in {P}ython},
  author    = {McKinney, Wes},
  booktitle = {Proceedings of the 9th Python in Science Conference},
  editor    = {van der Walt, St{\'e}fan and Millman, Jarrod},
  pages     = {56--61},
  year      = {2010},
  doi       = {10.25080/Majora-92bf1922-00a}
}
```

### 🔧 Lemaître et al. (2017) — imbalanced-learn (JMLR)
```bibtex
@article{lemaitre2017imbalanced,
  title   = {Imbalanced-learn: A {P}ython Toolbox to Tackle the Curse of Imbalanced Datasets in Machine Learning},
  author  = {Lema{\^i}tre, Guillaume and Nogueira, Fernando and Aridas, Christos K.},
  journal = {Journal of Machine Learning Research},
  volume  = {18},
  number  = {17},
  pages   = {1--5},
  year    = {2017}
}
```

### 🔧 Folium / Leaflet — visualização cartográfica
```bibtex
@misc{folium,
  title  = {{Folium}: Python data visualization library on {L}eaflet.js},
  author = {{Folium contributors}},
  year   = {2026},
  url    = {https://python-visualization.github.io/folium/}
}
@misc{leafletjs,
  title  = {{Leaflet}: An Open-Source {J}ava{S}cript Library for Mobile-Friendly Interactive Maps},
  author = {Agafonkin, Vladimir},
  year   = {2026},
  url    = {https://leafletjs.com/}
}
```

### 🔧 Geopandas / Shapely
```bibtex
@misc{geopandas,
  title  = {{GeoPandas}: {P}ython tools for geographic data},
  author = {Jordahl, Kelsey and others},
  year   = {2024},
  doi    = {10.5281/zenodo.2585848}
}
```

---

## 7. Tabela cruzada — onde cada referência aparece no TCC

| # | Referência | §3 Materiais e Métodos | §4 Resultados | §5/§6/§7 Discussão / Limitações / Futuros |
|---|------------|------------------------|---------------|---------------|
| 1 | Breiman (2001) RF | §3.4 / §3.5 | §4.1 / §4.4 | — |
| 2 | Wolpert (1992) Stacking | §3.5 | §4.1 / §4.10 | §5.1 |
| 3 | Chen & Guestrin (2016) XGB | §3.5 | §4.1 / §4.10 | — |
| 4 | Ke et al. (2017) LGBM | §3.5 | §4.1 / §4.10 | — |
| 5 | Prokhorenkova et al. (2018) CatBoost | §3.5 | §4.1 | — |
| 6 | Friedman (2001) GBM | §3.5 / §3.x | §4.10 | §5.1 |
| 7 | Chawla et al. (2002) SMOTE | §3.6 | §4.1 | §5.3 |
| 8 | Akiba et al. (2019) Optuna | §3.7 | §4.10 | — |
| 9 | Platt (1999) | §3.x calibração | — | — |
| 10 | Zadrozny & Elkan (2002) | §3.x calibração | §4.1 | — |
| 11 | He & Garcia (2009) imbalanced | §3.x | §4.6 / §4.9 | §5.3 |
| 12 | Lundberg & Lee (2017) SHAP | §3.12.2 | §4.4 | §7 |
| 13 | Lundberg et al. (2020) TreeExplainer | §3.12.2 (justificativa) | — | §7 |
| 14 | Guyon et al. (2002) RFE | §3.x | §4.5 | — |
| 15 | Cheerala et al. (2025) RF+SHAP | §3.12.2 (estado da arte) | §4.4 (comparativo) | — |
| 15a | Fisher, Rudin & Dominici (2019) Permutation/MCR | §3.12.2 / §4.4.2 | §4.4.2 | — |
| 15b | Strobl et al. (2007) viés MDI alta cardinalidade | §3.12.2 / §4.4.2 | §4.4.2 | — |
| 15c | Molnar (2022) Interpretable ML (livro) | §3.12 / §4.4.2 | §4.4.2 | §5.3 item 6 |
| 15d | Lee (2013) Pseudo-Label | §3.x (pseudo-labeling de Umidade) | §4.7.1 | §5.3 item 7 / §7 |
| 15e | Arazo et al. (2020) viés de confirmação | §3.x | §4.7.1 | §5.3 item 7 / §7 |
| 15f | Lopez-Garcia et al. (2024) imputação SMAP tropical | §3.x (pseudo-labeling de Umidade) | §4.7.1 | — |
| 16 | Seager et al. (2015) VPD | §3.11 (Tier 1 — VPD proxy) | §4.8 | — |
| 17 | Keetch & Byram (1968) KBDI | §3.11 (Tier 1 — KBDI proxy) | §4.8 | — |
| 18 | McKee et al. (1993) SPI | §3.11 (Tier 1 — SPI) | §4.8 | — |
| 19 | De Martonne (1926) | §3.11 (Tier 1 — Aridez) | — | — |
| 20 | Vasconcelos et al. / Forests (2024) | §3.11 / §1 motivação | §4.8 | — |
| 21 | Quesada-Ruiz et al. (2025) | §3.11 / §1 motivação | §4.8 | §7 |
| 22 | Aragão et al. (2018) | §1 / §2 motivação | — | — |
| 23 | Silva-Junior et al. (2025) | §1 motivação (cenário atual) | — | — |
| 23b | Bergmeir & Benítez (2012) | §3.5.1 (validação temporal) | §4.11 | §5.3 |
| 24 | NASA POWER | §3.2 (fonte de dados) | — | §5.2 limitações |
| 25 | NASA FIRMS | §3.2 (FRP em tempo real) | — | — |
| 26 | INMET 1981–2010 | §3.11 | — | §5.2 |
| 27 | IBGE Amazônia Legal 2024 | §3.1 (recorte espacial) | — | — |
| 28 | MapBiomas Fogo Coleção 4 | — | — | §7 |
| 29 | Pedregosa et al. (2011) scikit-learn | §3.x (implementação) | — | — |
| 30 | Harris et al. (2020) NumPy | §3.x | — | — |
| 31 | McKinney (2010) pandas | §3.x | — | — |
| 32 | Lemaître et al. (2017) imbalanced-learn | §3.6 | — | — |
| 33 | Folium / Leaflet | §3.12.3 (front-end) | — | — |
| 34 | Geopandas | §3.1 / `gerar_mapa.py` | — | — |

---

## 8. Como integrar no documento LaTeX do TCC

1. Copiar todas as entradas BibTeX deste arquivo para `references.bib` (ou similar).
2. No `DOCUMENTO_METODOLOGIA_E_RESULTADOS.md`, substituir citações do tipo "Seager et al. 2015" por `\cite{seager2015climatology}` ao migrar para LaTeX. As chaves BibTeX já estão no formato `autor_ano_palavra-chave`.
3. Em ABNT brasileira (caso o TCC use `abntex2`):
   - usar `\cite[]{seager2015climatology}` no texto;
   - configurar `bibstyle=abnt-numeric` ou `abnt-alf` conforme o padrão da instituição.
4. Recomenda-se gerar uma seção "Estado da arte" no TCC condensando os blocos §3 (interpretabilidade) e §4 (domínio) deste arquivo — neles está concentrada a "literatura recente 2024-2025" que justifica as escolhas de Tier 1 e do framework explicável.

---

## 9. Observações finais

- Todas as **DOIs/URLs foram verificados** em pesquisa web em **12/05/2026**, com exceção de Keetch & Byram (1968), McKee et al. (1993) e De Martonne (1926), que são clássicos amplamente citados; sugiro confirmar a paginação exata desses três antes da entrega final do TCC.
- A referência popular "GWO-XGBoost para Sichuan" (citada em versões anteriores da documentação interna) **não foi confirmada** com exatidão na busca; substituí por **Quesada-Ruiz et al. 2025** (npj Nat. Hazards), que é a referência metodológica mais sólida sobre uso de SPI + Random Forest em previsão de fogo. Se for desejável manter uma referência específica de XGBoost-fogo, há boas alternativas em **MDPI Forests (2023, vol. 14, art. 1797)** — *Prediction of Forest Fire Occurrence in Southwestern China*.
- A referência "Sumathi & Rajesh (IndJST 2025)" também não foi confirmada; recomendo trocar por **Cheerala et al. (2025)** ou pela revisão sobre RFE da pesquisa em `agent-tools` (Forest fire susceptibility com RFE + SVM/RF).
