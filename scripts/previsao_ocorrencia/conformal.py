"""Conformal prediction + métricas calibradas para o modelo de ocorrência (P6).

Substitui a heurística ad-hoc de incerteza (gap top-2 em operational_uncertainty.py)
por **conjuntos de predição com cobertura estatística garantida**.

Método: split-conformal classe-condicional (LAC — Least Ambiguous set-valued
Classifier; Sadinle, Lei & Wasserman 2019), variante Mondrian recomendada para
observação da Terra [mortier2024conformal]. Para cada classe k define-se um
limiar q_k no conjunto de calibração tal que P(p_k(x) ≥ q_k | Y=k) ≥ 1−α; o
conjunto de predição é {k : p_k(x) ≥ q_k}. Garante cobertura ≥ 1−α por classe.

Saídas (diagnóstico probabilístico, padrão [spatialuq2025]):
  • cobertura empírica por classe e marginal;
  • tamanho médio do conjunto e distribuição (singleton / ambíguo {0,1} / vazio);
  • ECE (Expected Calibration Error) e Brier da probabilidade P(fogo).

Split TEMPORAL (treino < ano_calib < ano_teste) para refletir uso operacional.
Caveat: conformal pressupõe permutabilidade; sob deriva temporal a cobertura é
aproximada — por isso calibramos no ano imediatamente anterior ao teste.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "dataset_ocorrencia_mensal.csv"
REL_DIR = ROOT / "modelos" / "relatorios"

FEATURES = [
    "LatBin", "LonBin", "mes_sin", "mes_cos", "estacao_seca",
    "DiaSemChuva", "Precipitacao", "RiscoFogo_inpe",
    "focos_lag1", "focos_lag2", "focos_lag3",
    "fogo_lag1", "fogo_lag2", "fogo_lag3",
    "focos_roll3", "focos_roll6", "focos_roll12",
    "fogo_roll3", "fogo_roll6", "fogo_roll12",
    "FRP_lag1", "meses_desde_fogo",
]


def ece(y, p, n_bins=10) -> float:
    """Expected Calibration Error (binário, P(classe=1))."""
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.digitize(p, bins) - 1
    idx = np.clip(idx, 0, n_bins - 1)
    e = 0.0
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            continue
        conf = p[m].mean()
        acc = y[m].mean()
        e += (m.mean()) * abs(acc - conf)
    return float(e)


def thresholds_lac(p_calib: np.ndarray, y_calib: np.ndarray, alpha: float) -> dict:
    """Limiar q_k por classe: quantil-alpha dos scores p_k entre exemplos
    verdadeiramente da classe k (garante cobertura ≥ 1−α por classe)."""
    q = {}
    for k in (0, 1):
        scores_k = p_calib[y_calib == k, k]
        # quantil inferior alpha: ponto abaixo do qual cai fração alpha
        q[k] = float(np.quantile(scores_k, alpha, method="lower"))
    return q


def conjuntos(p_test: np.ndarray, q: dict) -> np.ndarray:
    """Matriz booleana N×2: classe k incluída se p_k ≥ q_k."""
    return np.column_stack([p_test[:, 0] >= q[0], p_test[:, 1] >= q[1]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, default=str(DATASET))
    ap.add_argument("--alpha", type=float, default=0.1, help="1-alpha = cobertura alvo (default 0.1 → 90%).")
    ap.add_argument("--ano_calib", type=int, default=2022)
    ap.add_argument("--ano_teste", type=int, default=2023)
    args = ap.parse_args()

    df = pd.read_csv(args.dataset)
    tr = df[df["ano"] < args.ano_calib]
    cal = df[df["ano"] == args.ano_calib]
    te = df[df["ano"] == args.ano_teste]
    print(f"Treino<{args.ano_calib}: {len(tr):,} | calib {args.ano_calib}: {len(cal):,} | "
          f"teste {args.ano_teste}: {len(te):,}")

    m = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.08, max_depth=8, l2_regularization=1.0,
        max_leaf_nodes=63, class_weight="balanced", random_state=42,
    ).fit(tr[FEATURES], tr["alvo"])

    p_cal = m.predict_proba(cal[FEATURES])
    p_te = m.predict_proba(te[FEATURES])
    y_cal = cal["alvo"].to_numpy()
    y_te = te["alvo"].to_numpy()

    q = thresholds_lac(p_cal, y_cal, args.alpha)
    S = conjuntos(p_te, q)

    # Cobertura: o conjunto contém a classe verdadeira?
    contem_verd = S[np.arange(len(y_te)), y_te]
    cobertura_marginal = float(contem_verd.mean())
    cobertura_classe = {int(k): float(contem_verd[y_te == k].mean()) for k in (0, 1)}
    tam = S.sum(axis=1)
    dist = {
        "singleton": float((tam == 1).mean()),
        "ambiguo_{0,1}": float((tam == 2).mean()),
        "vazio": float((tam == 0).mean()),
    }

    # Diagnóstico probabilístico
    p1 = p_te[:, 1]
    brier = float(np.mean((p1 - y_te) ** 2))
    ece_v = ece(y_te, p1)

    resultado = {
        "metodo": "split-conformal classe-condicional (LAC, Sadinle 2019)",
        "alpha": args.alpha, "cobertura_alvo": 1 - args.alpha,
        "split": {"treino_ate": args.ano_calib - 1, "calib": args.ano_calib, "teste": args.ano_teste},
        "limiares_q": q,
        "cobertura_marginal": cobertura_marginal,
        "cobertura_por_classe": cobertura_classe,
        "tamanho_medio_conjunto": float(tam.mean()),
        "distribuicao_conjuntos": dist,
        "ece": ece_v, "brier": brier,
    }
    REL_DIR.mkdir(parents=True, exist_ok=True)
    out = REL_DIR / "previsao_ocorrencia_conformal.json"
    out.write_text(json.dumps(resultado, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n== Conformal (cobertura alvo {1-args.alpha:.0%}) ==")
    print(f"  cobertura marginal: {cobertura_marginal:.3f}")
    print(f"  cobertura por classe: nao-fogo={cobertura_classe[0]:.3f}  fogo={cobertura_classe[1]:.3f}")
    print(f"  tamanho médio do conjunto: {tam.mean():.3f}")
    print(f"  singletons={dist['singleton']:.3f}  ambíguos={dist['ambiguo_{0,1}']:.3f}  vazios={dist['vazio']:.3f}")
    print(f"  ECE={ece_v:.4f}  Brier={brier:.4f}")
    print(f"  salvo em: {out}")


if __name__ == "__main__":
    main()
