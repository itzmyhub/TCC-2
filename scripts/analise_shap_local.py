"""Pré-computa SHAP global *por classe* sobre o RF Tier 1 (proxy do Stacking GBM).

Diferente de ``analise_shap.py`` que produz apenas a importância global agregada,
este script salva um JSON com a importância média |SHAP| **separada por classe**
(Baixo / Moderado / Muito Alto). Isso permite que o explainer do app web (visite
``scripts/explainer.py``) ranqueie features de maneira **classe-específica**,
substituindo a aproximação anterior (importância global × z-score × sinal físico).

Saída: ``modelos/relatorios/shap_per_class.json`` com a estrutura

.. code-block:: json

    {
      "modelo": "random_forest_balanced",
      "classes": ["Baixo", "Moderado", "Muito Alto"],
      "n_amostras": 5000,
      "shap_per_class": {
        "Baixo":      [{"feature": "...", "mean_abs_shap": 0.0123}, ...],
        "Moderado":   [{...}, ...],
        "Muito Alto": [{...}, ...]
      },
      "shap_global": [{"feature": "...", "mean_abs_shap": 0.0098}, ...]
    }

Referência: Lundberg & Lee (NeurIPS 2017); Lundberg et al. (Nat. Mach. Intell. 2020).
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Optional

import joblib
import numpy as np

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR.parent / 'modelos'
METRICS_DIR = MODEL_DIR / 'relatorios'

sys.path.insert(0, str(BASE_DIR))
from carregar_dados import carregar_dados  # noqa: E402


def _extrair_rf_e_preproc(pipeline):
    """Retorna (estimador_rf, preprocessor) do pipeline padrão do projeto."""
    if hasattr(pipeline, 'named_steps'):
        modelo = pipeline.named_steps.get('modelo')
        if modelo is not None and hasattr(modelo, 'calibrated_classifiers_'):
            modelo = modelo.calibrated_classifiers_[0].estimator
        return modelo, pipeline.named_steps.get('preprocessor')
    return None, None


def analisar_shap_por_classe(
    modelo_nome: str = 'random_forest_balanced',
    n_amostras: int = 5000,
    dataset_path: Optional[str] = None,
    saida_path: Optional[str] = None,
) -> None:
    try:
        import shap
    except ImportError:
        logger.error("shap não instalado. Execute: pip install shap")
        return

    modelo_path = MODEL_DIR / f'{modelo_nome}.pkl'
    if not modelo_path.exists():
        logger.error("Modelo não encontrado: %s", modelo_path)
        return

    logger.info("Carregando modelo: %s", modelo_path)
    pipeline = joblib.load(modelo_path)

    estimador, preprocessor = _extrair_rf_e_preproc(pipeline)
    if estimador is None or preprocessor is None:
        logger.error("Não foi possível extrair estimador/preprocessor do pipeline.")
        return
    logger.info("Estimador para SHAP: %s", type(estimador).__name__)

    logger.info("Carregando dados...")
    X, y, _cat, _num, _ = carregar_dados(
        dataset_path=dataset_path,
        use_saved_thresholds=True,
        persist_thresholds=False,
    )

    rng = np.random.default_rng(42)
    idx = rng.choice(len(X), size=min(n_amostras, len(X)), replace=False)
    X_sample = X.iloc[idx]
    y_sample = y.iloc[idx]
    logger.info("Amostra de análise: %d linhas (de %d)", len(X_sample), len(X))

    logger.info("Aplicando preprocessor...")
    X_transformed = preprocessor.transform(X_sample)
    if hasattr(X_transformed, 'toarray'):
        X_transformed = X_transformed.toarray()
    feature_names = (
        preprocessor.get_feature_names_out().tolist()
        if hasattr(preprocessor, 'get_feature_names_out')
        else [f'feature_{i}' for i in range(X_transformed.shape[1])]
    )

    logger.info("Inicializando shap.TreeExplainer...")
    explainer = shap.TreeExplainer(estimador)

    # Sub-amostra adicional para SHAP — 5000 é caro com 546 features
    # 500 amostras × 546 features × 346 árvores ≈ 5–10 min em CPU; mais que isso explode tempo
    n_for_shap = min(500, X_transformed.shape[0])
    Xs = X_transformed[:n_for_shap]
    logger.info("Calculando SHAP values em %d amostras (otimizado para tempo)...", n_for_shap)
    shap_values = explainer.shap_values(Xs)

    # Em multi-classe, shap_values é list[ndarray] OU ndarray 3D (n_amostras, n_features, n_classes)
    if isinstance(shap_values, list):
        sv_per_class = shap_values
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        sv_per_class = [shap_values[:, :, k] for k in range(shap_values.shape[2])]
    else:
        sv_per_class = [shap_values]

    classes_modelo = getattr(estimador, 'classes_', None)
    if classes_modelo is None and hasattr(pipeline, 'classes_'):
        classes_modelo = pipeline.classes_
    if classes_modelo is None:
        classes_modelo = [f'classe_{i}' for i in range(len(sv_per_class))]
    classes_str = [str(c) for c in classes_modelo]
    logger.info("Classes: %s", classes_str)

    def _format_top(arr):
        importancias = sorted(
            ((nome, float(val)) for nome, val in zip(feature_names, arr.tolist())),
            key=lambda kv: kv[1],
            reverse=True,
        )
        return [
            {'feature': n, 'mean_abs_shap': round(v, 6)}
            for n, v in importancias
            if v > 0  # filtrar zeros para arquivo menor
        ]

    shap_per_class = {}
    for cls, sv in zip(classes_str, sv_per_class):
        mean_abs = np.abs(sv).mean(axis=0)
        shap_per_class[cls] = _format_top(mean_abs)
        topN = shap_per_class[cls][:8]
        logger.info("Top-8 |SHAP| em %s:", cls)
        for it in topN:
            nome = it['feature'].replace('num__', '').replace('cat__', '')
            logger.info("  %.4f  %s", it['mean_abs_shap'], nome)

    # SHAP global = média entre classes
    sv_global = np.mean([np.abs(sv) for sv in sv_per_class], axis=0).mean(axis=0)
    shap_global = _format_top(sv_global)

    payload = {
        'modelo': modelo_nome,
        'metodo': 'shap.TreeExplainer (RF balanced Tier 1, proxy do Stacking GBM)',
        'classes': classes_str,
        'n_amostras_treinamento': len(X),
        'n_amostras_explicacao': int(n_for_shap),
        'random_state': 42,
        'shap_per_class': shap_per_class,
        'shap_global': shap_global,
    }
    out = Path(saida_path) if saida_path else METRICS_DIR / 'shap_per_class.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w', encoding='utf-8') as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=2)
    logger.info("SHAP por classe salvo em %s", out)


def main():
    import argparse

    p = argparse.ArgumentParser(description='SHAP global por classe (RF Tier 1)')
    p.add_argument('--modelo', type=str, default='random_forest_balanced')
    p.add_argument('--n-amostras', type=int, default=5000)
    p.add_argument('--dataset', type=str, default='base_de_dados_enriquecido.csv')
    p.add_argument('--saida', type=str, default=None)
    args = p.parse_args()

    analisar_shap_por_classe(
        modelo_nome=args.modelo,
        n_amostras=args.n_amostras,
        dataset_path=args.dataset,
        saida_path=args.saida,
    )


if __name__ == '__main__':
    main()
