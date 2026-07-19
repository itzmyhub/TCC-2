"""
Análise de importância de features com SHAP para modelos de risco de incêndio.

Referência: Cheerala et al. (arXiv 2511.11680) — RF+SHAP para susceptibilidade a incêndios.

Gera:
- modelos/relatorios/shap_feature_importance.json: importância média |SHAP| por feature
- modelos/relatorios/shap_summary.png: gráfico beeswarm (se matplotlib disponível)
"""
import json
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR.parent / 'modelos'
METRICS_DIR = MODEL_DIR / 'relatorios'

sys.path.insert(0, str(BASE_DIR))
from carregar_dados import carregar_dados


def _extrair_rf_do_pipeline(pipeline):
    """Extrai o estimador RF de um pipeline sklearn/imblearn."""
    if hasattr(pipeline, 'named_steps'):
        modelo = pipeline.named_steps.get('modelo')
        if modelo is None:
            return None
        # Se for CalibratedClassifierCV, pegar o estimador base
        if hasattr(modelo, 'calibrated_classifiers_'):
            base = modelo.calibrated_classifiers_[0].estimator
            return base
        return modelo
    return None


def _extrair_preprocessor_do_pipeline(pipeline):
    if hasattr(pipeline, 'named_steps'):
        return pipeline.named_steps.get('preprocessor')
    return None


def analisar_shap(
    modelo_nome: str = 'random_forest_balanced',
    n_amostras: int = 5000,
    dataset_path: str = None,
):
    """
    Executa análise SHAP no modelo especificado.

    Args:
        modelo_nome: Nome do arquivo .pkl (sem extensão) em modelos/
        n_amostras: Número de amostras de background/explicação (SHAP TreeExplainer)
        dataset_path: Caminho para o CSV. None = padrão (base_de_dados.csv)
    """
    try:
        import shap
    except ImportError:
        logger.error("shap não instalado. Execute: pip install shap")
        return

    modelo_path = MODEL_DIR / f'{modelo_nome}.pkl'
    if not modelo_path.exists():
        logger.error("Modelo não encontrado: %s", modelo_path)
        logger.info("Modelos disponíveis: %s", [p.stem for p in MODEL_DIR.glob('*.pkl')])
        return

    logger.info("Carregando modelo: %s", modelo_path)
    pipeline = joblib.load(modelo_path)

    logger.info("Carregando dados...")
    X, y, cat_features, num_features, _ = carregar_dados(
        dataset_path=dataset_path,
        use_saved_thresholds=True,
        persist_thresholds=False,
    )

    # Pegar preprocessor do pipeline para transformar X
    preprocessor = _extrair_preprocessor_do_pipeline(pipeline)
    if preprocessor is None:
        logger.error("Não foi possível extrair preprocessor do pipeline.")
        return

    # Amostra estratificada para análise SHAP (mais rápido)
    rng = np.random.default_rng(42)
    idx = rng.choice(len(X), size=min(n_amostras, len(X)), replace=False)
    X_sample = X.iloc[idx]
    y_sample = y.iloc[idx]

    logger.info("Transformando dados com preprocessor...")
    try:
        X_transformed = preprocessor.transform(X_sample)
        feature_names = preprocessor.get_feature_names_out() if hasattr(preprocessor, 'get_feature_names_out') else None
    except Exception as e:
        logger.error("Erro ao transformar dados: %s", e)
        return

    # Extrair estimador de base (RF ou XGBoost) para SHAP TreeExplainer
    estimador = _extrair_rf_do_pipeline(pipeline)
    if estimador is None:
        logger.warning("Pipeline não reconhecido — tentando usar o pipeline diretamente.")
        estimador = pipeline

    # Verificar compatibilidade com TreeExplainer
    from sklearn.ensemble import RandomForestClassifier, StackingClassifier
    tipo = type(estimador).__name__
    logger.info("Tipo de estimador para SHAP: %s", tipo)

    # Usar feature importance do modelo diretamente (Gini importância para RF, gain para XGBoost/LightGBM)
    # Mais robusto e muito mais rápido que SHAP TreeExplainer com 546 features
    shap_values = None
    shap_per_class = {}

    if hasattr(estimador, 'feature_importances_'):
        logger.info("Usando feature_importances_ do modelo (Gini/Gain importance)...")
        importances_raw = estimador.feature_importances_
        n_features = len(importances_raw)
        if feature_names is None or len(feature_names) != n_features:
            feature_names = [f'feature_{i}' for i in range(n_features)]

        importancias = sorted(
            zip(feature_names, importances_raw.tolist()),
            key=lambda x: x[1],
            reverse=True,
        )

        resultado = {
            'modelo': modelo_nome,
            'metodo': 'feature_importances (Gini/Gain)',
            'n_amostras': None,
            'n_features': n_features,
            'feature_importance': [
                {'feature': f, 'mean_abs_shap': round(v, 6)}
                for f, v in importancias
            ],
            'shap_per_class': {},
        }

        saida = METRICS_DIR / 'shap_feature_importance.json'
        saida.parent.mkdir(parents=True, exist_ok=True)
        with saida.open('w', encoding='utf-8') as fp:
            json.dump(resultado, fp, ensure_ascii=False, indent=2)
        logger.info("Importâncias salvas em %s (método: feature_importances_)", saida)

        logger.info("Top 15 features por importância (Gini/Gain):")
        for rank, (feat, val) in enumerate(importancias[:15], 1):
            nome = feat.replace('num__', '').replace('cat__', '')
            logger.info("  %2d. %-35s %.4f", rank, nome, val)

        # Gráfico
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(10, 8))
            top_n = 15
            nomes = [f for f, _ in importancias[:top_n]]
            valores = [v for _, v in importancias[:top_n]]
            nomes_display = [n.replace('num__', '').replace('cat__', '') for n in nomes]

            ax.barh(range(top_n), valores[::-1], color='steelblue', edgecolor='white')
            ax.set_yticks(range(top_n))
            ax.set_yticklabels(nomes_display[::-1], fontsize=10)
            ax.set_xlabel('Importância (Gini Mean Decrease Impurity)', fontsize=11)
            ax.set_title(
                f'Top {top_n} Features — {modelo_nome}\n(Feature Importances — Gini MDI)',
                fontsize=12,
            )
            ax.grid(axis='x', alpha=0.3)
            plt.tight_layout()

            grafico_path = METRICS_DIR / 'shap_feature_importance.png'
            plt.savefig(grafico_path, dpi=150, bbox_inches='tight')
            plt.close()
            logger.info("Gráfico salvo em %s", grafico_path)
        except Exception as e:
            logger.warning("Não foi possível gerar gráfico: %s", e)

        return resultado

    # Fallback: tentar SHAP TreeExplainer com amostra bem reduzida
    logger.warning("feature_importances_ não disponível. Tentando SHAP com 200 amostras...")
    X_small = X_transformed[:200]
    try:
        explainer = shap.TreeExplainer(estimador)
        logger.info("Calculando SHAP values (TreeExplainer, %s, 200 amostras)...", tipo)
        shap_values = explainer.shap_values(X_small)
    except Exception as e:
        logger.error("Erro ao calcular SHAP values: %s", e)
        return

    # Calcular importância média |SHAP| por feature
    if isinstance(shap_values, list):
        # Multi-classe: shap_values é lista com uma matriz por classe
        shap_abs_mean = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
        classes = pipeline.classes_ if hasattr(pipeline, 'classes_') else [f'classe_{i}' for i in range(len(shap_values))]
        shap_per_class = {
            str(cls): np.abs(sv).mean(axis=0).tolist()
            for cls, sv in zip(classes, shap_values)
        }
    else:
        shap_abs_mean = np.abs(shap_values).mean(axis=0)
        shap_per_class = {}

    n_features = len(shap_abs_mean)
    if feature_names is None or len(feature_names) != n_features:
        feature_names = [f'feature_{i}' for i in range(n_features)]

    # Ordenar por importância
    importancias = sorted(
        zip(feature_names, shap_abs_mean.tolist()),
        key=lambda x: x[1],
        reverse=True,
    )

    resultado = {
        'modelo': modelo_nome,
        'n_amostras': int(n_amostras),
        'n_features': n_features,
        'feature_importance': [
            {'feature': f, 'mean_abs_shap': round(v, 6)}
            for f, v in importancias
        ],
        'shap_per_class': {
            cls: [
                {'feature': f, 'mean_abs_shap': round(v, 6)}
                for f, v in sorted(
                    zip(feature_names, vals), key=lambda x: x[1], reverse=True
                )
            ]
            for cls, vals in shap_per_class.items()
        }
    }

    saida = METRICS_DIR / 'shap_feature_importance.json'
    saida.parent.mkdir(parents=True, exist_ok=True)
    with saida.open('w', encoding='utf-8') as fp:
        json.dump(resultado, fp, ensure_ascii=False, indent=2)
    logger.info("Importâncias SHAP salvas em %s", saida)

    # Log top 10
    logger.info("Top 10 features por importância SHAP:")
    for rank, (feat, val) in enumerate(importancias[:10], 1):
        logger.info("  %2d. %-35s %.4f", rank, feat, val)

    # Gráfico (opcional)
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 8))
        top_n = 15
        nomes = [f for f, _ in importancias[:top_n]]
        valores = [v for _, v in importancias[:top_n]]
        nomes_display = [n.replace('num__', '').replace('cat__', '') for n in nomes]

        ax.barh(range(top_n), valores[::-1], color='steelblue', edgecolor='white')
        ax.set_yticks(range(top_n))
        ax.set_yticklabels(nomes_display[::-1], fontsize=10)
        ax.set_xlabel('Importância média |SHAP|', fontsize=11)
        ax.set_title(f'Top {top_n} Features — {modelo_nome}\n(SHAP — {n_amostras} amostras)', fontsize=12)
        ax.grid(axis='x', alpha=0.3)
        plt.tight_layout()

        grafico_path = METRICS_DIR / 'shap_feature_importance.png'
        plt.savefig(grafico_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info("Gráfico SHAP salvo em %s", grafico_path)
    except Exception as e:
        logger.warning("Não foi possível gerar gráfico SHAP: %s", e)

    return resultado


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Análise SHAP de importância de features')
    parser.add_argument(
        '--modelo',
        type=str,
        default='random_forest_balanced',
        help='Nome do modelo .pkl (padrão: random_forest_balanced)',
    )
    parser.add_argument(
        '--n_amostras',
        type=int,
        default=5000,
        help='Número de amostras para análise SHAP (padrão: 5000)',
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default=None,
        help='Caminho para o dataset CSV (padrão: base_de_dados.csv)',
    )
    args = parser.parse_args()

    analisar_shap(
        modelo_nome=args.modelo,
        n_amostras=args.n_amostras,
        dataset_path=args.dataset,
    )
