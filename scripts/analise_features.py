"""
Recursive Feature Elimination (RFE) com cross-validation para identificar
features relevantes no modelo de risco de incêndio.

Referência: Sumathi & Rajesh (IndJST 2025) — RFE identificou temperatura,
velocidade do vento e umidade como mais relevantes; R² 0.92.

Gera:
- modelos/relatorios/rfe_feature_ranking.json: ranking e scores por número de features
- modelos/relatorios/rfe_n_features_vs_accuracy.png: curva de acurácia por nº de features
"""
import json
import logging
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR.parent / 'modelos'
METRICS_DIR = MODEL_DIR / 'relatorios'

sys.path.insert(0, str(BASE_DIR))
from carregar_dados import carregar_dados
from pre_processor import PreProcessor


def executar_rfe(
    n_features_range=None,
    dataset_path: str = None,
    n_cv_folds: int = 3,
    n_amostras: int = 50000,
):
    """
    Executa RFECV e análise de importância de features com Random Forest.

    Args:
        n_features_range: Lista com nº de features a testar. None = todos (RFECV automático).
        dataset_path: Caminho para o CSV. None = padrão.
        n_cv_folds: Número de folds na cross-validation.
        n_amostras: Número de amostras para análise (subsample para velocidade).
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.feature_selection import RFECV
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import Pipeline

    logger.info("Carregando dados...")
    X, y, cat_features, num_features, _ = carregar_dados(
        dataset_path=dataset_path,
        use_saved_thresholds=True,
        persist_thresholds=False,
    )

    # Subsample para velocidade
    rng = np.random.default_rng(42)
    n = min(n_amostras, len(X))
    idx = rng.choice(len(X), size=n, replace=False)
    X_sub = X.iloc[idx].reset_index(drop=True)
    y_sub = y.iloc[idx].reset_index(drop=True)

    logger.info("Usando %d amostras para RFE", n)

    # Preprocessar dados
    logger.info("Pré-processando dados...")
    preprocessor = PreProcessor(num_features=num_features, cat_features=cat_features)
    preprocessor.fit(X_sub)
    X_transformed = preprocessor.preprocessor.transform(X_sub)

    # Nomes das features transformadas
    try:
        feature_names_out = preprocessor.preprocessor.get_feature_names_out()
    except Exception:
        feature_names_out = [f'feature_{i}' for i in range(X_transformed.shape[1])]

    logger.info("Features após pré-processamento: %d", X_transformed.shape[1])

    # RF base para RFE
    rf_rfe = RandomForestClassifier(
        n_estimators=100,
        max_depth=12,
        min_samples_leaf=5,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
    )

    cv = StratifiedKFold(n_splits=n_cv_folds, shuffle=True, random_state=42)

    # step=10 para não avaliar cada feature individualmente (muito lento com 546 features)
    step = max(1, X_transformed.shape[1] // 50)
    logger.info(
        "Executando RFECV com %d folds, step=%d (pode demorar alguns minutos)...",
        n_cv_folds, step,
    )
    rfecv = RFECV(
        estimator=rf_rfe,
        step=step,
        cv=cv,
        scoring='accuracy',
        min_features_to_select=5,
        n_jobs=1,
    )
    rfecv.fit(X_transformed, y_sub)

    n_optimal = rfecv.n_features_
    logger.info("Número ótimo de features (RFECV): %d", n_optimal)

    # Rankings e scores por nº de features
    cv_scores = rfecv.cv_results_['mean_test_score']
    std_scores = rfecv.cv_results_['std_test_score']
    # Reconstruir n_features testadas a partir do step usado pelo RFECV
    _n_max = X_transformed.shape[1]
    _step_used = step
    _n_range_raw = list(range(_n_max, rfecv.min_features_to_select - 1, -_step_used))
    _n_range_raw = sorted(_n_range_raw)
    # Garantir que min_features esteja incluído
    if rfecv.min_features_to_select not in _n_range_raw:
        _n_range_raw.insert(0, rfecv.min_features_to_select)
    # Alinhar com o número de cv_scores (alguns últimos passos podem ser omitidos)
    n_range = _n_range_raw[:len(cv_scores)]

    # Top features selecionadas
    rankings = rfecv.ranking_
    suporte = rfecv.support_

    features_ranking = sorted(
        [(name, int(rank), bool(sel))
         for name, rank, sel in zip(feature_names_out, rankings, suporte)],
        key=lambda x: x[1],
    )

    resultado = {
        'n_features_optimal': int(n_optimal),
        'n_features_total': int(X_transformed.shape[1]),
        'cv_folds': n_cv_folds,
        'n_amostras': int(n),
        'best_cv_accuracy': float(cv_scores[n_optimal - rfecv.min_features_to_select]),
        'features_ranking': [
            {'feature': f.replace('num__', '').replace('cat__', ''),
             'raw_feature': f,
             'ranking': r,
             'selected': s}
            for f, r, s in features_ranking
        ],
        'cv_scores_by_n_features': [
            {'n_features': n_range[i], 'mean_accuracy': float(cv_scores[i]), 'std': float(std_scores[i])}
            for i in range(len(cv_scores))
        ],
    }

    saida = METRICS_DIR / 'rfe_feature_ranking.json'
    saida.parent.mkdir(parents=True, exist_ok=True)
    with saida.open('w', encoding='utf-8') as fp:
        json.dump(resultado, fp, ensure_ascii=False, indent=2)
    logger.info("Ranking RFE salvo em %s", saida)

    # Log top 20 features
    logger.info("Top 20 features por ranking RFE:")
    for entry in features_ranking[:20]:
        f, r, s = entry
        nome = f.replace('num__', '').replace('cat__', '')
        sel = '✓' if s else ' '
        logger.info("  [%s] rank=%2d  %s", sel, r, nome)

    # Gráfico (opcional)
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(n_range, cv_scores, 'b-o', markersize=4, label='Acurácia CV média')
        ax.fill_between(
            n_range,
            [c - s for c, s in zip(cv_scores, std_scores)],
            [c + s for c, s in zip(cv_scores, std_scores)],
            alpha=0.2, color='blue', label='±1 std'
        )
        ax.axvline(x=n_optimal, color='red', linestyle='--', label=f'Ótimo: {n_optimal} features')
        ax.set_xlabel('Número de Features', fontsize=11)
        ax.set_ylabel('Acurácia (CV)', fontsize=11)
        ax.set_title('RFECV — Acurácia por Número de Features\n(Random Forest, Amazônia Legal)', fontsize=12)
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()

        grafico_path = METRICS_DIR / 'rfe_n_features_vs_accuracy.png'
        plt.savefig(grafico_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info("Gráfico RFE salvo em %s", grafico_path)
    except Exception as e:
        logger.warning("Não foi possível gerar gráfico RFE: %s", e)

    return resultado


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Análise RFE de features para modelos de risco de incêndio')
    parser.add_argument(
        '--n_amostras',
        type=int,
        default=50000,
        help='Número de amostras para análise (padrão: 50000)',
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default=None,
        help='Caminho para o dataset CSV (padrão: base_de_dados.csv)',
    )
    parser.add_argument(
        '--cv_folds',
        type=int,
        default=3,
        help='Número de folds na cross-validation (padrão: 3)',
    )
    args = parser.parse_args()

    executar_rfe(
        dataset_path=args.dataset,
        n_cv_folds=args.cv_folds,
        n_amostras=args.n_amostras,
    )
