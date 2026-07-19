"""Permutation Importance **estratificada por classe** sobre o Stacking GBM.

Por que permutation importance?
================================
Diferente da Gini Importance (MDI), permutation importance é:

1. **Model-agnostic** — funciona com qualquer modelo (Stacking, GBM, RF, ...).
2. **Não tendenciosa para alta cardinalidade** — corrigi o viés do MDI a favor
   de features com muitos níveis (problema discutido por Strobl et al. 2007).
3. **Faz sentido também para classes raras** — calculamos a queda no F1 por
   classe quando uma feature é permutada.

Metodologia (Breiman 2001; Fisher, Rudin & Dominici 2019)
---------------------------------------------------------
Para cada feature ``j`` e classe ``c``:

1. Avaliamos F1_c base no conjunto de teste.
2. Permutamos a coluna ``j`` (mantendo distribuição marginal) ``n_repeats`` vezes.
3. Importância(j, c) = F1_c_base - média(F1_c_permutado).

Importância > 0 → feature relevante para identificar a classe.
Importância ≈ 0 → feature irrelevante.

Saída: ``modelos/relatorios/permutation_importance_por_classe.json``.

Referências
-----------
- Breiman, L. (2001). *Random Forests*. Machine Learning.
- Fisher, A., Rudin, C., & Dominici, F. (2019). *All Models are Wrong, but Many
  are Useful: Learning a Variable's Importance by Studying an Entire Class of
  Prediction Models Simultaneously*. JMLR.
- Strobl, C., Boulesteix, A.-L., Zeileis, A., & Hothorn, T. (2007). *Bias in
  random forest variable importance measures*. BMC Bioinformatics.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR.parent / 'modelos'
METRICS_DIR = MODEL_DIR / 'relatorios'

sys.path.insert(0, str(BASE_DIR))
from carregar_dados import carregar_dados  # noqa: E402
from treinamento_modelo import _LabelEncodingWrapper  # noqa: E402,F401  (necess\u00e1rio para joblib.load do Stacking)


def permutation_importance_por_classe(
    modelo_nome: str = 'ensemble_stacking_gbm',
    n_test: int = 8_000,
    n_repeats: int = 5,
    random_state: int = 42,
    dataset_path: Optional[str] = None,
    saida_path: Optional[str] = None,
) -> None:
    modelo_path = MODEL_DIR / f'{modelo_nome}.pkl'
    if not modelo_path.exists():
        logger.error("Modelo não encontrado: %s", modelo_path)
        return

    logger.info("Carregando modelo: %s", modelo_path)
    pipeline = joblib.load(modelo_path)

    logger.info("Carregando dados...")
    X, y, _cat, _num, _ = carregar_dados(
        dataset_path=dataset_path,
        use_saved_thresholds=True,
        persist_thresholds=False,
    )
    logger.info("Total: %d linhas", len(X))

    _X_train, X_test_full, _y_train, y_test_full = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=random_state,
    )

    rng = np.random.default_rng(random_state)
    if len(X_test_full) > n_test:
        idx = rng.choice(len(X_test_full), size=n_test, replace=False)
        X_test = X_test_full.iloc[idx].reset_index(drop=True)
        y_test = y_test_full.iloc[idx].reset_index(drop=True)
    else:
        X_test = X_test_full.reset_index(drop=True)
        y_test = y_test_full.reset_index(drop=True)
    logger.info("Conjunto de avaliação: %d linhas", len(X_test))

    classes = sorted(y_test.unique().tolist())
    logger.info("Classes encontradas: %s", classes)

    logger.info("Calculando F1 baseline por classe...")
    y_pred_base = pipeline.predict(X_test)
    f1_base = f1_score(y_test, y_pred_base, labels=classes, average=None)
    f1_base_dict = {str(c): float(s) for c, s in zip(classes, f1_base)}
    logger.info("F1 baseline: %s", f1_base_dict)

    features = X_test.columns.tolist()
    logger.info("Permutando %d features × %d repetições (~%d predições)",
                len(features), n_repeats, len(features) * n_repeats)

    importancias_por_classe = {str(c): [] for c in classes}
    importancia_global = []

    for j, feat in enumerate(features, start=1):
        f1_repeats = []
        for r in range(n_repeats):
            X_perm = X_test.copy()
            X_perm[feat] = rng.permutation(X_perm[feat].values)
            y_pred = pipeline.predict(X_perm)
            f1_classes = f1_score(y_test, y_pred, labels=classes, average=None)
            f1_repeats.append(f1_classes)
        f1_arr = np.asarray(f1_repeats)
        f1_perm_mean = f1_arr.mean(axis=0)
        delta = f1_base - f1_perm_mean
        delta_global = float(np.mean(delta))
        importancia_global.append({
            'feature': feat,
            'delta_f1_macro': round(delta_global, 6),
        })
        for c, d in zip(classes, delta):
            importancias_por_classe[str(c)].append({
                'feature': feat,
                'delta_f1': round(float(d), 6),
            })
        if j % 5 == 0 or j == len(features):
            logger.info("Progresso: %d/%d features (última: %s, Δf1_macro=%.4f)",
                        j, len(features), feat, delta_global)

    importancia_global.sort(key=lambda d: d['delta_f1_macro'], reverse=True)
    for c in importancias_por_classe:
        importancias_por_classe[c].sort(key=lambda d: d['delta_f1'], reverse=True)

    logger.info("\n=== Top-10 importância global (Δ F1-macro) ===")
    for it in importancia_global[:10]:
        logger.info("  %.5f  %s", it['delta_f1_macro'], it['feature'])
    for c in classes:
        logger.info("\n=== Top-10 importância para a classe '%s' (Δ F1) ===", c)
        for it in importancias_por_classe[str(c)][:10]:
            logger.info("  %.5f  %s", it['delta_f1'], it['feature'])

    payload = {
        'modelo': modelo_nome,
        'metodo': 'Permutation Importance (Breiman 2001; Fisher et al. 2019)',
        'metrica': 'delta_f1_por_classe e delta_f1_macro',
        'n_amostras_teste': int(len(X_test)),
        'n_repeticoes': int(n_repeats),
        'random_state': random_state,
        'classes': [str(c) for c in classes],
        'f1_baseline_por_classe': f1_base_dict,
        'importancia_global': importancia_global,
        'importancia_por_classe': importancias_por_classe,
    }
    out = Path(saida_path) if saida_path else METRICS_DIR / 'permutation_importance_por_classe.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w', encoding='utf-8') as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=2)
    logger.info("Permutation importance salva em %s", out)


def main():
    import argparse
    p = argparse.ArgumentParser(description='Permutation importance por classe')
    p.add_argument('--modelo', type=str, default='ensemble_stacking_gbm')
    p.add_argument('--n-test', type=int, default=8_000)
    p.add_argument('--n-repeats', type=int, default=5)
    p.add_argument('--dataset', type=str, default='base_de_dados_enriquecido.csv')
    p.add_argument('--saida', type=str, default=None)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()

    permutation_importance_por_classe(
        modelo_nome=args.modelo,
        n_test=args.n_test,
        n_repeats=args.n_repeats,
        random_state=args.seed,
        dataset_path=args.dataset,
        saida_path=args.saida,
    )


if __name__ == '__main__':
    main()
