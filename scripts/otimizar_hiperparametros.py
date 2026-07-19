import argparse
import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

from carregar_dados import carregar_dados
from pre_processor import PreProcessor

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR.parent / 'modelos'
REPORT_DIR = MODEL_DIR / 'relatorios'
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def _build_estimator(model_name: str, trial):
    if model_name == 'random_forest_balanced':
        return RandomForestClassifier(
            n_estimators=trial.suggest_int('n_estimators', 150, 500),
            max_depth=trial.suggest_int('max_depth', 8, 30),
            min_samples_split=trial.suggest_int('min_samples_split', 2, 20),
            min_samples_leaf=trial.suggest_int('min_samples_leaf', 1, 10),
            max_features=trial.suggest_categorical('max_features', ['sqrt', 'log2']),
            class_weight='balanced',
            random_state=42,
            n_jobs=-1,
        )
    if model_name == 'logistic_regression_balanced':
        return LogisticRegression(
            C=trial.suggest_float('C', 1e-2, 20.0, log=True),
            solver='lbfgs',
            max_iter=trial.suggest_int('max_iter', 1000, 5000),
            multi_class='multinomial',
            class_weight='balanced',
            random_state=42,
        )
    if model_name == 'xgboost_classifier':
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise RuntimeError("xgboost não instalado. Execute: pip install xgboost") from exc
        # Importa wrapper de label encoding (necessário no Stacking que usa LR como meta)
        from treinamento_modelo import _LabelEncodingWrapper
        return _LabelEncodingWrapper(
            XGBClassifier(
                n_estimators=trial.suggest_int('n_estimators', 150, 500),
                max_depth=trial.suggest_int('max_depth', 4, 10),
                learning_rate=trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
                subsample=trial.suggest_float('subsample', 0.6, 1.0),
                colsample_bytree=trial.suggest_float('colsample_bytree', 0.6, 1.0),
                min_child_weight=trial.suggest_int('min_child_weight', 1, 10),
                reg_alpha=trial.suggest_float('reg_alpha', 1e-3, 5.0, log=True),
                reg_lambda=trial.suggest_float('reg_lambda', 1e-3, 5.0, log=True),
                random_state=42,
                n_jobs=-1,
                eval_metric='mlogloss',
            )
        )
    if model_name == 'lightgbm_classifier':
        try:
            from lightgbm import LGBMClassifier
        except ImportError as exc:
            raise RuntimeError("lightgbm não instalado. Execute: pip install lightgbm") from exc
        from treinamento_modelo import _LabelEncodingWrapper
        return _LabelEncodingWrapper(
            LGBMClassifier(
                n_estimators=trial.suggest_int('n_estimators', 150, 500),
                max_depth=trial.suggest_int('max_depth', 4, 12),
                num_leaves=trial.suggest_int('num_leaves', 31, 200),
                learning_rate=trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
                subsample=trial.suggest_float('subsample', 0.6, 1.0),
                colsample_bytree=trial.suggest_float('colsample_bytree', 0.6, 1.0),
                min_child_samples=trial.suggest_int('min_child_samples', 5, 50),
                reg_alpha=trial.suggest_float('reg_alpha', 1e-3, 5.0, log=True),
                reg_lambda=trial.suggest_float('reg_lambda', 1e-3, 5.0, log=True),
                class_weight='balanced',
                random_state=42,
                n_jobs=-1,
                verbose=-1,
            )
        )
    raise ValueError(f"Modelo não suportado para otimização: {model_name}")


def otimizar(
    model_name: str,
    n_trials: int,
    dataset_path: Optional[str] = None,
    cv_folds: int = 3,
    subsample: Optional[int] = None,
    scoring: str = 'accuracy',
):
    try:
        import optuna
    except ImportError as exc:
        raise RuntimeError(
            "Optuna não instalado. Execute: pip install optuna"
        ) from exc

    X, y, cat_features, num_features, _ = carregar_dados(
        dataset_path=dataset_path,
        use_saved_thresholds=True,
        persist_thresholds=False,
    )
    logger.info("Dataset carregado: %d linhas", len(X))

    if subsample and len(X) > subsample:
        from sklearn.model_selection import train_test_split
        X, _, y, _ = train_test_split(
            X, y, train_size=subsample, stratify=y, random_state=42
        )
        logger.info("Subsample estratificado: %d linhas (random_state=42)", len(X))

    logger.info("Modelo alvo: %s | Trials: %d | Scoring: %s", model_name, n_trials, scoring)

    preprocessor = PreProcessor(num_features=num_features, cat_features=cat_features, verbose=False)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

    def objective(trial):
        estimator = _build_estimator(model_name, trial)
        pipeline = Pipeline(
            steps=[
                ('preprocessor', preprocessor.preprocessor),
                ('modelo', estimator),
            ]
        )
        scores = cross_val_score(
            pipeline,
            X,
            y,
            cv=cv,
            scoring=scoring,
            n_jobs=1,
        )
        return float(np.mean(scores))

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)

    payload = {
        'model_name': model_name,
        'n_trials': n_trials,
        'scoring': scoring,
        f'best_value_{scoring}': float(study.best_value),
        'best_params': study.best_params,
        'cv_folds': cv_folds,
        'dataset_path': dataset_path or 'default',
        'subsample': subsample,
    }
    out = REPORT_DIR / f'optuna_{model_name}.json'
    with out.open('w', encoding='utf-8') as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=2)
    logger.info("Resultado salvo em %s", out)
    logger.info("Melhor accuracy CV: %.4f", study.best_value)
    logger.info("Melhores parâmetros: %s", study.best_params)


def main():
    parser = argparse.ArgumentParser(description='Otimiza hiperparâmetros com Optuna')
    parser.add_argument(
        '--model',
        type=str,
        default='random_forest_balanced',
        choices=[
            'random_forest_balanced',
            'logistic_regression_balanced',
            'xgboost_classifier',
            'lightgbm_classifier',
        ],
        help='Modelo a otimizar',
    )
    parser.add_argument('--n-trials', type=int, default=30, help='Número de trials do Optuna')
    parser.add_argument('--dataset', type=str, default=None, help='Caminho do dataset CSV')
    parser.add_argument('--cv-folds', type=int, default=3, help='Número de folds da CV')
    parser.add_argument(
        '--subsample', type=int, default=None,
        help='Subamostra estratificada para acelerar (ex.: 100000)'
    )
    parser.add_argument(
        '--scoring', type=str, default='accuracy',
        help='Métrica do CV (accuracy, f1_macro, etc.)'
    )
    args = parser.parse_args()

    otimizar(
        model_name=args.model,
        n_trials=args.n_trials,
        dataset_path=args.dataset,
        cv_folds=args.cv_folds,
        subsample=args.subsample,
        scoring=args.scoring,
    )


if __name__ == '__main__':
    main()
