"""Retreina o modelo final aplicando oversampling OOD apenas no treino.

Fluxo:
1. Carrega `base_de_dados_enriquecido.csv` e calcula features+rótulos via
   `carregar_dados` (mesmo pipeline do treino original).
2. Faz `train_test_split` com `random_state=42` (mesmo split do baseline).
3. Aplica oversampling estratificado apenas no TREINO, usando os pares
   identificados em `modelos/relatorios/ood_pares_alvo.json`.
4. Treina o modelo escolhido (default: `ensemble_stacking_gbm`).
5. Avalia no TESTE original (não oversampleado).
6. Compara métricas baseline (de `modelos/relatorios/ensemble_stacking_gbm_metrics.json`)
   com as do modelo OOD e salva relatório.

Saídas:
- `modelos/{nome_modelo}_ood.pkl`
- `modelos/relatorios/{nome_modelo}_ood_metrics.json`
- `modelos/relatorios/comparativo_ood_vs_baseline.json`

Uso:
    # Versão LEVE (RF balanceado Tier 1, ~10 min): para iterar rápido.
    python scripts/retreinar_com_oversample.py --modelo random_forest_balanced

    # Versão COMPLETA (Stacking GBM, ~90 min): produção.
    python scripts/retreinar_com_oversample.py --modelo ensemble_stacking_gbm
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("retreino_ood")

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from carregar_dados import carregar_dados  # noqa: E402
from pre_processor import PreProcessor  # noqa: E402
from treinamento_modelo import (  # noqa: E402
    MODEL_LIBRARY_IMPROVED,
    METRICS_DIR,
    MODEL_DIR,
)

ROOT = _SCRIPTS_DIR.parent
ENRICHED_PATH = ROOT / "base_de_dados_enriquecido.csv"
OOD_ALVOS_PATH = METRICS_DIR / "ood_pares_alvo.json"
COMPARATIVO_PATH = METRICS_DIR / "comparativo_ood_vs_baseline.json"

# Sigmas de jitter (mesmas do oversample_ood_cerrado.py)
SIGMA_LAT = 0.02
SIGMA_LON = 0.02
SIGMA_HORA = 1.0
SIGMA_FRP_REL = 0.05

SEED = 42


def _norm_estado(s) -> str:
    return s.strip().upper() if isinstance(s, str) else ""


def _carregar_alvos() -> List[Dict]:
    if not OOD_ALVOS_PATH.exists():
        raise FileNotFoundError(
            f"Arquivo {OOD_ALVOS_PATH} não existe. "
            "Rode antes: python scripts/analise_ood_cerrado.py"
        )
    with OOD_ALVOS_PATH.open("r", encoding="utf-8") as f:
        return json.load(f).get("pares_alvo", [])


def _aplicar_jitter(X: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    X = X.copy()
    n = len(X)
    if "Latitude" in X.columns:
        X["Latitude"] = X["Latitude"].astype(float) + rng.normal(0.0, SIGMA_LAT, size=n)
    if "Longitude" in X.columns:
        X["Longitude"] = X["Longitude"].astype(float) + rng.normal(0.0, SIGMA_LON, size=n)
    if "Hora" in X.columns:
        h = X["Hora"].astype(float) + rng.normal(0.0, SIGMA_HORA, size=n)
        X["Hora"] = np.clip(np.round(h).astype(int), 0, 23)
    if "FRP" in X.columns:
        frp = X["FRP"].astype(float).values
        X["FRP"] = np.maximum(0.0, frp * (1.0 + rng.normal(0.0, SIGMA_FRP_REL, size=n)))
    return X


def aplicar_oversample_no_treino(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    alvos: List[Dict],
    rng: np.random.Generator,
) -> Tuple[pd.DataFrame, pd.Series, Dict]:
    """Oversampleia X_train/y_train usando os pares alvo. Retorna (X', y', metadados)."""
    if "Estado" not in X_train.columns or "Mes" not in X_train.columns:
        raise ValueError("X_train precisa ter Estado e Mes.")
    estado_norm = X_train["Estado"].map(_norm_estado)

    chunks_X = []
    chunks_y = []
    metadados: Dict = {"pares_aplicados": [], "n_treino_antes": int(len(X_train))}

    for alvo in alvos:
        est = alvo["estado"]
        mes = int(alvo["mes"])
        fator = float(alvo["fator_oversample_sugerido"])
        n_copias = max(1, int(round(fator - 1.0)))
        mask = (estado_norm == est) & (X_train["Mes"] == mes)
        if not mask.any():
            continue
        Xa = X_train.loc[mask]
        ya = y_train.loc[mask]
        for _ in range(n_copias):
            chunks_X.append(_aplicar_jitter(Xa, rng))
            chunks_y.append(ya.copy())
        metadados["pares_aplicados"].append({
            "estado": est, "mes": mes,
            "n_originais": int(len(Xa)),
            "n_copias_extras": n_copias,
        })

    if not chunks_X:
        logger.warning("Nenhum par-alvo casou com o treino — sem oversampling.")
        metadados["n_treino_depois"] = int(len(X_train))
        return X_train, y_train, metadados

    X_extras = pd.concat(chunks_X, ignore_index=True)
    y_extras = pd.concat(chunks_y, ignore_index=True)
    X_final = pd.concat([X_train.reset_index(drop=True), X_extras], ignore_index=True)
    y_final = pd.concat([y_train.reset_index(drop=True), y_extras], ignore_index=True)

    # Shuffle estável
    idx = np.arange(len(X_final))
    rng.shuffle(idx)
    X_final = X_final.iloc[idx].reset_index(drop=True)
    y_final = y_final.iloc[idx].reset_index(drop=True)

    metadados["n_treino_depois"] = int(len(X_final))
    metadados["n_extras_total"] = int(len(X_extras))
    return X_final, y_final, metadados


def _carregar_baseline_metrics(nome_modelo: str) -> Optional[Dict]:
    p = METRICS_DIR / f"{nome_modelo}_metrics.json"
    if not p.exists():
        return None
    try:
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def _f1_per_class(report: Dict) -> Dict[str, float]:
    out = {}
    for classe in ("Baixo", "Moderado", "Muito Alto"):
        info = report.get(classe, {})
        if isinstance(info, dict) and "f1-score" in info:
            out[classe] = float(info["f1-score"])
    return out


def avaliar_e_comparar(
    nome_modelo: str,
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    meta_oversample: Dict,
) -> Dict:
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1m = f1_score(y_test, y_pred, average="macro")
    rep = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred).tolist()
    f1_por_classe = _f1_per_class(rep)
    metricas_ood = {
        "modelo": f"{nome_modelo}_ood",
        "accuracy": acc,
        "f1_macro": f1m,
        "f1_por_classe": f1_por_classe,
        "classification_report": rep,
        "confusion_matrix": cm,
        "meta_oversample": meta_oversample,
        "n_test": int(len(X_test)),
    }

    # Avaliação restrita ao Cerrado (TO, MA, PA, MT, BA, GO, DF) em meses 5/6/7
    cerrado_estados = {"TOCANTINS", "MARANHÃO", "PARÁ", "MATO GROSSO",
                       "BAHIA", "GOIÁS", "DISTRITO FEDERAL", "PIAUÍ"}
    if "Estado" in X_test.columns and "Mes" in X_test.columns:
        est = X_test["Estado"].map(_norm_estado)
        mask_cerr = est.isin(cerrado_estados) & X_test["Mes"].isin((5, 6, 7))
        if mask_cerr.any():
            y_test_cerr = y_test[mask_cerr]
            y_pred_cerr = pipeline.predict(X_test[mask_cerr])
            acc_cerr = accuracy_score(y_test_cerr, y_pred_cerr)
            f1m_cerr = f1_score(y_test_cerr, y_pred_cerr, average="macro")
            metricas_ood["holdout_cerrado_transicao"] = {
                "n": int(mask_cerr.sum()),
                "accuracy": float(acc_cerr),
                "f1_macro": float(f1m_cerr),
                "estados": sorted(cerrado_estados),
                "meses": [5, 6, 7],
            }
            logger.info(
                "Hold-out Cerrado-transição: n=%d, acc=%.4f, f1_macro=%.4f",
                int(mask_cerr.sum()), acc_cerr, f1m_cerr,
            )

    baseline = _carregar_baseline_metrics(nome_modelo)
    comparativo: Dict = {"modelo_base": nome_modelo, "ood": metricas_ood}
    if baseline:
        b_acc = float(baseline.get("accuracy", 0.0))
        b_f1m = float(baseline.get("f1_macro", 0.0))
        b_rep = baseline.get("classification_report", {}) or {}
        b_f1_por_classe = _f1_per_class(b_rep)
        comparativo["baseline"] = {
            "accuracy": b_acc, "f1_macro": b_f1m,
            "f1_por_classe": b_f1_por_classe,
        }
        comparativo["delta"] = {
            "accuracy_pp": round(100 * (acc - b_acc), 3),
            "f1_macro_pp": round(100 * (f1m - b_f1m), 3),
            "f1_por_classe_pp": {
                c: round(100 * (f1_por_classe.get(c, 0) - b_f1_por_classe.get(c, 0)), 3)
                for c in ("Baixo", "Moderado", "Muito Alto")
            },
        }
        logger.info(
            "Δ acc=%.2f pp | Δ f1_macro=%.2f pp | Δ f1_MA=%.2f pp",
            comparativo["delta"]["accuracy_pp"],
            comparativo["delta"]["f1_macro_pp"],
            comparativo["delta"]["f1_por_classe_pp"].get("Muito Alto", 0),
        )

    return comparativo


def treinar(
    nome_modelo: str,
    dataset_path: Optional[str] = None,
) -> Dict:
    if nome_modelo not in MODEL_LIBRARY_IMPROVED:
        raise ValueError(
            f"Modelo {nome_modelo!r} não encontrado. "
            f"Disponíveis: {sorted(MODEL_LIBRARY_IMPROVED.keys())}"
        )

    alvos = _carregar_alvos()
    if not alvos:
        logger.warning("Sem pares-alvo OOD — retreinamento equivalente ao baseline.")

    logger.info("Carregando e preparando dados...")
    X, y, cat_features, num_features, df = carregar_dados(
        dataset_path=dataset_path,
        use_saved_thresholds=False,
        persist_thresholds=False,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )
    logger.info("Split estratificado: treino=%d, teste=%d", len(X_train), len(X_test))

    rng = np.random.default_rng(SEED)
    X_train_aug, y_train_aug, meta_os = aplicar_oversample_no_treino(
        X_train, y_train, alvos, rng
    )
    logger.info(
        "Oversampling: %d → %d (+%d, +%.1f%%)",
        meta_os["n_treino_antes"], meta_os["n_treino_depois"],
        meta_os["n_treino_depois"] - meta_os["n_treino_antes"],
        100 * (meta_os["n_treino_depois"] - meta_os["n_treino_antes"])
        / max(meta_os["n_treino_antes"], 1),
    )

    base_estimator = clone(MODEL_LIBRARY_IMPROVED[nome_modelo])
    preprocessor = PreProcessor(num_features=num_features, cat_features=cat_features)
    pipeline = Pipeline([
        ("preprocessor", clone(preprocessor.preprocessor)),
        ("modelo", base_estimator),
    ])

    logger.info("Treinando %s (oversample OOD)... isso pode demorar.", nome_modelo)
    t0 = time.time()
    pipeline.fit(X_train_aug, y_train_aug)
    dt = time.time() - t0
    logger.info("Treino concluído em %.1f min", dt / 60)

    model_path = MODEL_DIR / f"{nome_modelo}_ood.pkl"
    joblib.dump(pipeline, model_path)
    logger.info("Modelo salvo: %s", model_path)

    comparativo = avaliar_e_comparar(nome_modelo, pipeline, X_test, y_test, meta_os)
    comparativo["train_time_sec"] = round(dt, 1)
    comparativo["model_path"] = str(model_path)

    metricas_path = METRICS_DIR / f"{nome_modelo}_ood_metrics.json"
    with metricas_path.open("w", encoding="utf-8") as f:
        json.dump(comparativo["ood"], f, ensure_ascii=False, indent=2)
    with COMPARATIVO_PATH.open("w", encoding="utf-8") as f:
        json.dump(comparativo, f, ensure_ascii=False, indent=2)
    logger.info("Métricas: %s", metricas_path)
    logger.info("Comparativo: %s", COMPARATIVO_PATH)

    return comparativo


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--modelo",
        default="random_forest_balanced",
        choices=sorted(MODEL_LIBRARY_IMPROVED.keys()),
        help="Nome do modelo a retreinar (default: random_forest_balanced).",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Caminho do CSV de dataset (default: base_de_dados_enriquecido.csv via config).",
    )
    args = parser.parse_args()

    treinar(args.modelo, dataset_path=args.dataset)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
