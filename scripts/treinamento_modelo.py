import json
import logging
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline

from carregar_dados import carregar_dados
from pre_processor import PreProcessor

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# Opcional: SMOTE para balanceamento de classes
try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
    HAS_IMBLEARN = True
except ImportError:
    HAS_IMBLEARN = False
    logger.warning("imblearn não instalado. SMOTE não estará disponível.")

try:
    from xgboost import XGBClassifier

    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    logger.info("xgboost não instalado. Ensembles usarão apenas RF + LR.")

try:
    from lightgbm import LGBMClassifier

    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    logger.info("lightgbm não instalado. Ensembles usarão apenas RF + LR (+ XGBoost).")

try:
    from catboost import CatBoostClassifier

    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False
    logger.info("catboost não instalado. Ensembles não terão CatBoost como base learner.")

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder


class _LabelEncodingWrapper(BaseEstimator, ClassifierMixin):
    """Wrapper que converte labels string→int para modelos que exigem labels numéricos (XGBoost, LightGBM)."""

    def __init__(self, estimator):
        self.estimator = estimator
        self._le = LabelEncoder()

    def fit(self, X, y, **kwargs):
        y_enc = self._le.fit_transform(y)
        self.estimator.fit(X, y_enc, **kwargs)
        self.classes_ = self._le.classes_
        return self

    def predict(self, X):
        return self._le.inverse_transform(self.estimator.predict(X))

    def predict_proba(self, X):
        return self.estimator.predict_proba(X)

    def get_params(self, deep=True):
        return {'estimator': self.estimator}

    def set_params(self, **params):
        if 'estimator' in params:
            self.estimator = params['estimator']
        return self

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR.parent / 'modelos'
METRICS_DIR = MODEL_DIR / 'relatorios'
METRICS_DIR.mkdir(parents=True, exist_ok=True)

PREPROCESSOR_METADATA_PATH = MODEL_DIR / 'preprocessor_metadata.json'
SPLIT_METADATA_PATH = MODEL_DIR / 'split_metadata.json'

# Modelos básicos (versões originais)
MODEL_LIBRARY_BASIC = {
    'sgd_classifier': SGDClassifier(loss='log_loss', random_state=42, max_iter=2000),
    'logistic_regression': LogisticRegression(
        max_iter=2000,
        multi_class='multinomial',
        solver='lbfgs',
    ),
    'random_forest': RandomForestClassifier(
        n_estimators=250,
        max_depth=16,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
    ),
}

# Modelos melhorados (com otimizações para ≥70% acurácia)
MODEL_LIBRARY_IMPROVED = {
    'random_forest_balanced': RandomForestClassifier(
        # Hiperparâmetros otimizados via Optuna (16/04/2026) — CV 77.99%
        n_estimators=346,
        max_depth=25,
        min_samples_split=3,
        min_samples_leaf=1,
        max_features='sqrt',
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
    ),
    'logistic_regression_balanced': LogisticRegression(
        # Hiperparâmetros otimizados via Optuna (15 trials, 3-fold CV, base_de_dados_com_historico.csv) — 21/04/2026
        # Best CV accuracy: 0.6456 (Trial 3). Ganho marginal sobre defaults (~0,01 pp), mas
        # registrado em modelos/relatorios/optuna_logistic_regression_balanced.json para fechar §7.
        max_iter=4768,
        multi_class='multinomial',
        solver='lbfgs',
        class_weight='balanced',
        C=12.425396449863609,
        random_state=42,
    ),
}

# Adicionar versão com SMOTE se disponível
if HAS_IMBLEARN:
    MODEL_LIBRARY_IMPROVED['random_forest_smote'] = RandomForestClassifier(
        # Hiperparâmetros otimizados via Optuna (16/04/2026)
        n_estimators=346,
        max_depth=25,
        min_samples_split=3,
        min_samples_leaf=1,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1,
    )

# Adicionar XGBoost standalone se disponível
if HAS_XGBOOST:
    # Hiperparâmetros otimizados via Optuna (15 trials, 3-fold CV, F1-macro,
    # base_de_dados_enriquecido.csv, subsample=150k) — 2026-05-11.
    # Best CV F1-macro: 0.7005 (Trial 2). Default anterior: ~0.616 F1-macro.
    MODEL_LIBRARY_IMPROVED['xgboost_classifier'] = _LabelEncodingWrapper(
        XGBClassifier(
            n_estimators=455,
            max_depth=8,
            learning_rate=0.18965641873235228,
            subsample=0.9614210448729252,
            colsample_bytree=0.7037231845711419,
            min_child_weight=10,
            reg_alpha=0.289173140462925,
            reg_lambda=0.06433007424652636,
            random_state=42,
            n_jobs=-1,
            eval_metric='mlogloss',
        )
    )

# Adicionar LightGBM standalone se disponível
if HAS_LIGHTGBM:
    # Hiperparâmetros otimizados via Optuna (15 trials, 3-fold CV, F1-macro,
    # base_de_dados_enriquecido.csv, subsample=150k) — 2026-05-11.
    # Best CV F1-macro: 0.7256 (Trial X). Default anterior: ~0.670 F1-macro.
    MODEL_LIBRARY_IMPROVED['lightgbm_classifier'] = _LabelEncodingWrapper(
        LGBMClassifier(
            n_estimators=410,
            max_depth=11,
            num_leaves=149,
            learning_rate=0.08901940879868259,
            subsample=0.8452973377255052,
            colsample_bytree=0.9615655941622185,
            min_child_samples=18,
            reg_alpha=1.9734585140189238,
            reg_lambda=0.009893127818125904,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
    )

# Adicionar CatBoost standalone se disponível.
# CatBoost lida bem com features esparsas pós-OHE e oferece auto_class_weights.
if HAS_CATBOOST:
    MODEL_LIBRARY_IMPROVED['catboost_classifier'] = _LabelEncodingWrapper(
        CatBoostClassifier(
            iterations=400,
            depth=8,
            learning_rate=0.05,
            l2_leaf_reg=3,
            auto_class_weights='Balanced',
            random_state=42,
            thread_count=-1,
            verbose=0,
            allow_writing_files=False,
        )
    )


def _estimador_rf_balanced() -> RandomForestClassifier:
    # Hiperparâmetros otimizados via Optuna (15 trials, 3-fold CV, base_de_dados_com_historico.csv)
    # Melhor CV accuracy: 77.99% — Trial 14 em 16/04/2026
    return RandomForestClassifier(
        n_estimators=346,
        max_depth=25,
        min_samples_split=3,
        min_samples_leaf=1,
        max_features='sqrt',
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
    )


def _estimador_lr_balanced() -> LogisticRegression:
    # Mesmo hiperparâmetros otimizados via Optuna (ver MODEL_LIBRARY_IMPROVED).
    return LogisticRegression(
        max_iter=4768,
        multi_class='multinomial',
        solver='lbfgs',
        class_weight='balanced',
        C=12.425396449863609,
        random_state=42,
    )


def _estimador_xgb_opcional():
    """XGBoost para uso como base learner em ensembles. Parâmetros otimizados
    via Optuna (15 trials, F1-macro, 2026-05-11) — ver MODEL_LIBRARY_IMPROVED.
    Reduzimos n_estimators para 300 (vs 455 standalone) por economia: o
    Stacking treina cada base learner 3× (folds CV interno) — 25% menos
    árvores cabe num cenário onde já há diversidade pelos outros learners."""
    if not HAS_XGBOOST:
        return None
    return _LabelEncodingWrapper(
        XGBClassifier(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.18965641873235228,
            subsample=0.9614210448729252,
            colsample_bytree=0.7037231845711419,
            min_child_weight=10,
            reg_alpha=0.289173140462925,
            reg_lambda=0.06433007424652636,
            random_state=42,
            n_jobs=-1,
            eval_metric='mlogloss',
        )
    )


def _estimador_lgbm_opcional():
    """LightGBM para uso como base learner em ensembles. Parâmetros otimizados
    via Optuna (15 trials, F1-macro, 2026-05-11) — n_estimators reduzido para
    economia no Stacking (250 vs 410 standalone)."""
    if not HAS_LIGHTGBM:
        return None
    return _LabelEncodingWrapper(
        LGBMClassifier(
            n_estimators=250,
            max_depth=11,
            num_leaves=149,
            learning_rate=0.08901940879868259,
            subsample=0.8452973377255052,
            colsample_bytree=0.9615655941622185,
            min_child_samples=18,
            reg_alpha=1.9734585140189238,
            reg_lambda=0.009893127818125904,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
    )


def _estimador_catboost_opcional():
    """CatBoost com auto_class_weights='Balanced'. Mais leve que o standalone
    (200 iterações vs 400) para uso como base learner em ensembles."""
    if not HAS_CATBOOST:
        return None
    return _LabelEncodingWrapper(
        CatBoostClassifier(
            iterations=200,
            depth=6,
            learning_rate=0.1,
            l2_leaf_reg=3,
            auto_class_weights='Balanced',
            random_state=42,
            thread_count=-1,
            verbose=0,
            allow_writing_files=False,
        )
    )


def criar_ensemble_voting() -> VotingClassifier:
    """Voting soft: RF + LR (+ XGBoost + LightGBM + CatBoost se instalados)."""
    estimators = [
        ('rf', _estimador_rf_balanced()),
        ('lr', _estimador_lr_balanced()),
    ]
    xgb = _estimador_xgb_opcional()
    if xgb is not None:
        estimators.append(('xgb', xgb))
    lgbm = _estimador_lgbm_opcional()
    if lgbm is not None:
        estimators.append(('lgbm', lgbm))
    cat = _estimador_catboost_opcional()
    if cat is not None:
        estimators.append(('cat', cat))
    return VotingClassifier(estimators=estimators, voting='soft', n_jobs=1)


def _build_meta_learner(kind: str):
    """Constrói o meta-learner do Stacking.

    kind ∈ {"lr" (padrão), "gbm" (GradientBoostingClassifier), "lgbm" (LightGBM)}.
    A escolha "gbm" / "lgbm" testa a hipótese de que um meta-learner não-linear
    captura interações entre as probabilidades dos base learners que a LR não
    consegue (referência: Wolpert 1992; Sesmero et al. 2015).
    """
    kind = (kind or 'lr').lower()
    if kind == 'gbm':
        from sklearn.ensemble import GradientBoostingClassifier
        return GradientBoostingClassifier(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42,
        )
    if kind == 'lgbm':
        if not HAS_LIGHTGBM:
            raise RuntimeError("LightGBM não instalado: meta_learner='lgbm' indisponível.")
        return _LabelEncodingWrapper(
            LGBMClassifier(
                n_estimators=300,
                num_leaves=63,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.9,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1,
                verbose=-1,
            )
        )
    return LogisticRegression(
        max_iter=3000,
        multi_class='multinomial',
        solver='lbfgs',
        random_state=42,
    )


def criar_ensemble_stacking(meta_learner: str = 'lr') -> StackingClassifier:
    """Stacking com meta-aprendiz configurável sobre probabilidades dos base learners.

    `meta_learner` aceita "lr" (default), "gbm" ou "lgbm". Os base learners
    são sempre RF balanced + LR balanced + (XGB, LGBM, CatBoost) quando
    disponíveis (Tier 4 do plano de melhorias).
    """
    estimators = [
        ('rf', _estimador_rf_balanced()),
        ('lr', _estimador_lr_balanced()),
    ]
    xgb = _estimador_xgb_opcional()
    if xgb is not None:
        estimators.append(('xgb', xgb))
    lgbm = _estimador_lgbm_opcional()
    if lgbm is not None:
        estimators.append(('lgbm', lgbm))
    cat = _estimador_catboost_opcional()
    if cat is not None:
        estimators.append(('cat', cat))
    return StackingClassifier(
        estimators=estimators,
        final_estimator=_build_meta_learner(meta_learner),
        cv=3,
        stack_method='predict_proba',
        n_jobs=1,
    )


MODEL_LIBRARY_IMPROVED['ensemble_voting_soft'] = criar_ensemble_voting()
MODEL_LIBRARY_IMPROVED['ensemble_stacking'] = criar_ensemble_stacking('lr')
MODEL_LIBRARY_IMPROVED['ensemble_stacking_gbm'] = criar_ensemble_stacking('gbm')
if HAS_LIGHTGBM:
    MODEL_LIBRARY_IMPROVED['ensemble_stacking_lgbm'] = criar_ensemble_stacking('lgbm')

# Biblioteca padrão (usa melhorados se disponível)
MODEL_LIBRARY = {**MODEL_LIBRARY_BASIC, **MODEL_LIBRARY_IMPROVED}


def _serializar_confusao(matriz, classes) -> Dict[str, Iterable]:
    return {
        'classes': list(classes),
        'matrix': matriz.tolist(),
    }


def _salvar_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=2)
    logger.info("Arquivo JSON salvo em %s", path)


def _preparar_dados(
    dataset_path: Optional[str] = None,
    preprocessor_metadata_path: Optional[Path] = None,
    split_metadata_path: Optional[Path] = None,
):
    logger.info("Carregando e preparando dados...")
    X, y, cat_features, num_features, df = carregar_dados(
        dataset_path=dataset_path,
        use_saved_thresholds=False, 
        persist_thresholds=True
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    split_metadata = {
        'timestamp': datetime.now(UTC).isoformat(),
        'random_state': 42,
        'test_size': 0.2,
        'stratify': True,
        'train_test_split': 'sklearn.model_selection.train_test_split',
        'nota': 'Mesmo random_state garante split reprodutível com mesma versão do sklearn e mesmas linhas do CSV.',
    }
    split_path = split_metadata_path or SPLIT_METADATA_PATH
    _salvar_json(split_path, split_metadata)

    meta_path = preprocessor_metadata_path or PREPROCESSOR_METADATA_PATH
    metadata_preprocessor = PreProcessor(num_features=num_features, cat_features=cat_features, verbose=False)
    metadata_preprocessor.fit(X_train)
    metadata_preprocessor.save_metadata(str(meta_path))
    logger.info("Metadados do pré-processador salvos em %s", meta_path)

    return X_train, X_test, y_train, y_test, num_features, cat_features


def _avaliar_modelo(nome_modelo: str, y_true, y_pred, cv_scores: Optional[list] = None) -> Dict:
    relatorio = classification_report(y_true, y_pred, output_dict=True)
    f1_macro = f1_score(y_true, y_pred, average='macro')
    acc = accuracy_score(y_true, y_pred)
    matriz = confusion_matrix(y_true, y_pred)
    
    logger.info("Resultados %s - accuracy: %.4f (%.2f%%) | f1_macro: %.4f", 
                nome_modelo, acc, acc*100, f1_macro)
    
    if cv_scores is not None:
        cv_mean = float(np.mean(cv_scores))
        cv_std = float(np.std(cv_scores))
        logger.info("  CV (5-fold): %.4f (%.2f%%) ± %.4f (%.2f%%)", 
                    cv_mean, cv_mean*100, cv_std, cv_std*100)
    
    metricas = {
        'accuracy': acc,
        'f1_macro': f1_macro,
        'classification_report': relatorio,
        'confusion_matrix': _serializar_confusao(matriz, sorted(set(y_true))),
    }
    
    if cv_scores is not None:
        metricas['cv_mean'] = float(np.mean(cv_scores))
        metricas['cv_std'] = float(np.std(cv_scores))
        metricas['cv_scores'] = [float(s) for s in cv_scores]
    
    return metricas


def treinar_modelo(
    nome_modelo: str,
    estimator,
    X_train,
    y_train,
    X_test,
    y_test,
    num_features,
    cat_features,
    use_cv: bool = True,
    use_smote: bool = False,
    calibrate: bool = True,
    max_train_samples: Optional[int] = None,
    artifact_basename: Optional[str] = None,
    extra_metrics: Optional[Dict] = None,
) -> Dict:
    logger.info("Treinando modelo: %s", nome_modelo)

    # Subamostrar se solicitado (útil para modelos com alto consumo de memória)
    if max_train_samples is not None and len(X_train) > max_train_samples:
        logger.info(
            "  Subsampling treino: %d → %d amostras (estratificado)",
            len(X_train), max_train_samples,
        )
        rng = np.random.default_rng(42)
        idx = rng.choice(len(X_train), size=max_train_samples, replace=False)
        X_train = X_train.iloc[idx]
        y_train = y_train.iloc[idx]

    preprocessor_template = PreProcessor(num_features=num_features, cat_features=cat_features)
    
    # Usar pipeline com SMOTE se solicitado e disponível
    if use_smote and HAS_IMBLEARN:
        logger.info("  Usando SMOTE para balanceamento de classes")
        pipeline_steps = [
            ('preprocessor', preprocessor_template.preprocessor),
            ('smote', SMOTE(random_state=42, k_neighbors=3)),
            ('modelo', estimator),
        ]
        pipeline = ImbPipeline(steps=pipeline_steps)
    else:
        if use_smote and not HAS_IMBLEARN:
            logger.warning("  SMOTE solicitado mas imblearn não está instalado. Usando pipeline normal.")
        pipeline_steps = [
            ('preprocessor', clone(preprocessor_template.preprocessor)),
            ('modelo', estimator),
        ]
        pipeline = Pipeline(steps=pipeline_steps)
    
    # Validação cruzada (opcional)
    cv_scores = None
    if use_cv:
        logger.info("  Executando validação cruzada (3-fold)...")
        # 3 folds para reduzir consumo de recursos em Windows
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        # Usar estimador com n_jobs=1 durante CV para evitar multiplicação de processos/threads
        estimator_cv = clone(estimator)
        if hasattr(estimator_cv, 'n_jobs'):
            try:
                estimator_cv.set_params(n_jobs=1)
            except Exception:
                pass
        if use_smote and HAS_IMBLEARN:
            pipeline_cv = ImbPipeline(steps=[
                ('preprocessor', preprocessor_template.preprocessor),
                ('smote', SMOTE(random_state=42, k_neighbors=3)),
                ('modelo', estimator_cv),
            ])
        else:
            pipeline_cv = Pipeline(steps=[
                ('preprocessor', clone(preprocessor_template.preprocessor)),
                ('modelo', estimator_cv),
            ])
        # Rodar CV em single-process para evitar WinError 1450
        cv_scores = cross_val_score(pipeline_cv, X_train, y_train, cv=cv, scoring='accuracy', n_jobs=1)
        logger.info("  CV scores: %s", [f"{s:.4f}" for s in cv_scores])
        logger.info("  CV média: %.4f (%.2f%%) ± %.4f", 
                    np.mean(cv_scores), np.mean(cv_scores)*100, np.std(cv_scores))

    # Treinar no conjunto completo de treino
    inicio = time.time()
    pipeline.fit(X_train, y_train)
    tempo_treino = time.time() - inicio
    
    # Calibrar probabilidades se solicitado e modelo suporta
    if calibrate and hasattr(estimator, 'predict_proba'):
        logger.info("  Calibrando probabilidades do modelo...")
        try:
            if use_smote and HAS_IMBLEARN:
                # Para SMOTE, calibrar após oversampling
                X_train_processed = pipeline.named_steps['preprocessor'].transform(X_train)
                X_train_resampled, y_train_resampled = pipeline.named_steps['smote'].fit_resample(
                    X_train_processed, y_train
                )
                calibrated_estimator = CalibratedClassifierCV(
                    pipeline.named_steps['modelo'],
                    method='isotonic',
                    cv=3
                )
                calibrated_estimator.fit(X_train_resampled, y_train_resampled)
                pipeline.named_steps['modelo'] = calibrated_estimator
            else:
                calibrated_estimator = CalibratedClassifierCV(
                    pipeline.named_steps['modelo'],
                    method='isotonic',
                    cv=3
                )
                X_train_processed = pipeline.named_steps['preprocessor'].transform(X_train)
                calibrated_estimator.fit(X_train_processed, y_train)
                pipeline.named_steps['modelo'] = calibrated_estimator
            logger.info("  ✅ Calibração concluída")
        except Exception as e:
            logger.warning("  ⚠️ Erro na calibração: %s. Continuando sem calibração.", e)

    # Avaliar no conjunto de teste
    y_pred = pipeline.predict(X_test)
    metricas = _avaliar_modelo(nome_modelo, y_test, y_pred, cv_scores)
    metricas['train_time_sec'] = tempo_treino
    if extra_metrics:
        metricas.update(extra_metrics)

    disk_name = artifact_basename or nome_modelo
    model_path = MODEL_DIR / f'{disk_name}.pkl'
    joblib.dump(pipeline, model_path)
    logger.info("Modelo salvo em %s", model_path)

    metricas_path = METRICS_DIR / f'{disk_name}_metrics.json'
    _salvar_json(metricas_path, metricas)
    metricas['model_path'] = str(model_path)
    metricas['metrics_path'] = str(metricas_path)

    return metricas


def main(
    model_list: Optional[Iterable[str]] = None,
    use_improved: bool = True,
    use_cv: bool = True,
    use_smote: bool = False,
    calibrate: bool = True,
    dataset_path: Optional[str] = None,
    save_as: Optional[str] = None,
):
    """
    Treina modelos com melhorias opcionais.
    
    Args:
        model_list: Lista de nomes de modelos para treinar. Se None, treina todos.
        use_improved: Se True, usa versões melhoradas dos modelos quando disponível.
        use_cv: Se True, executa validação cruzada 5-fold.
        use_smote: Se True, usa SMOTE para balanceamento (apenas para modelos compatíveis).
        calibrate: Se True, calibra probabilidades do modelo.
        dataset_path: Caminho para o dataset CSV. Se None, usa o padrão (base_de_dados.csv).
    """
    # Selecionar biblioteca de modelos
    if use_improved:
        library = MODEL_LIBRARY
        logger.info("Usando modelos melhorados (≥70% objetivo)")
    else:
        library = MODEL_LIBRARY_BASIC
        logger.info("Usando modelos básicos (versões originais)")
    
    modelos = library if model_list is None else {k: library[k] for k in model_list if k in library}
    if not modelos:
        raise ValueError("Nenhum modelo válido informado para treinamento.")
    if save_as is not None and len(modelos) != 1:
        raise ValueError("--save-as exige exatamente um modelo em --models.")

    logger.info("Modelos selecionados: %s", list(modelos.keys()))

    pre_meta_path = None
    split_path = None
    if save_as:
        pre_meta_path = MODEL_DIR / f'preprocessor_metadata_{save_as}.json'
        split_path = MODEL_DIR / f'split_metadata_{save_as}.json'

    X_train, X_test, y_train, y_test, num_features, cat_features = _preparar_dados(
        dataset_path,
        preprocessor_metadata_path=pre_meta_path,
        split_metadata_path=split_path,
    )

    # Modelos gradient boosting com OHE denso podem exceder memória em datasets grandes.
    # Limitar a 250K amostras de treino para esses modelos.
    _GB_MODELS = {'xgboost_classifier', 'lightgbm_classifier', 'catboost_classifier'}
    _MAX_SAMPLES_GB = 250_000

    ds_note: Dict = {}
    if dataset_path:
        ds_note['dataset'] = dataset_path
    lp = dataset_path.lower()
    if 'umidade_pseudo' in lp or 'umidade' in lp and 'pseudo' in lp:
        ds_note['umidade_como_feature'] = True
        ds_note['umidade_origem'] = 'NASA POWER + pseudo_label_lgbm (modelos/relatorios/pseudo_label_umidade.json)'

    resultados: Dict[str, Dict] = {}
    for nome, base_estimator in modelos.items():
        # Determinar se usa SMOTE (apenas para modelos específicos)
        use_smote_this = use_smote and 'smote' in nome.lower()

        max_samples = _MAX_SAMPLES_GB if nome in _GB_MODELS else None

        estimator = clone(base_estimator)
        resultado = treinar_modelo(
            nome,
            estimator,
            X_train,
            y_train,
            X_test,
            y_test,
            num_features,
            cat_features,
            use_cv=use_cv,
            use_smote=use_smote_this,
            calibrate=calibrate,
            max_train_samples=max_samples,
            artifact_basename=save_as,
            extra_metrics=ds_note,
        )
        resultados[nome] = resultado
        
        # Verificar se atingiu objetivo de 70%
        if resultado['accuracy'] >= 0.70:
            logger.info("🎯 OBJETIVO ATINGIDO: Acurácia >= 70%% (%.2f%%)", resultado['accuracy']*100)
        else:
            logger.warning("⚠️ Acurácia abaixo do objetivo: %.2f%% < 70%%", resultado['accuracy']*100)

    resumo_path = METRICS_DIR / 'resumo_treinamento.json'
    _salvar_json(resumo_path, resultados)
    logger.info("Resumo de treinamento salvo em %s", resumo_path)
    
    # Resumo final
    logger.info("\n" + "="*70)
    logger.info("RESUMO FINAL")
    logger.info("="*70)
    for nome, metricas in resultados.items():
        acc = metricas['accuracy']
        status = "✅" if acc >= 0.70 else "⚠️"
        cv_info = ""
        if 'cv_mean' in metricas:
            cv_info = f" | CV: {metricas['cv_mean']*100:.2f}% (±{metricas['cv_std']*100:.2f}%)"
        logger.info(
            "%s %s: Teste=%.2f%%%s",
            status, nome, acc*100, cv_info
        )


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Treina modelos de previsão de risco de incêndio')
    parser.add_argument(
        '--models',
        nargs='+',
        help='Modelos: random_forest_balanced, logistic_regression_balanced, ensemble_voting_soft, ensemble_stacking, etc.',
    )
    parser.add_argument('--basic', action='store_true', help='Usa modelos básicos (sem melhorias)')
    parser.add_argument('--no-cv', action='store_true', help='Desabilita validação cruzada')
    parser.add_argument('--smote', action='store_true', help='Usa SMOTE para balanceamento (apenas modelos compatíveis)')
    parser.add_argument('--no-calibrate', action='store_true', help='Desabilita calibração de probabilidades')
    parser.add_argument('--dataset', type=str, default=None, help='Caminho para o dataset CSV (padrão: base_de_dados.csv)')
    parser.add_argument(
        '--save-as',
        type=str,
        default=None,
        metavar='NAME',
        help='Nome base para .pkl, *_metrics.json, preprocessor_metadata_NAME.json e split_metadata_NAME.json '
        '(não sobrescreve artefatos padrão; exige um único modelo em --models).',
    )

    args = parser.parse_args()

    main(
        model_list=args.models,
        use_improved=not args.basic,
        use_cv=not args.no_cv,
        use_smote=args.smote,
        calibrate=not args.no_calibrate,
        dataset_path=args.dataset,
        save_as=args.save_as,
    )
