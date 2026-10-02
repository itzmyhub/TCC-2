"""Avalia a calibração isotônica das probabilidades do modelo de ocorrência.

O ``class_weight='balanced'`` desloca P(fogo) para cima: a ordenação das células
não muda, mas a probabilidade deixa de corresponder à frequência observada. Aqui
se ajusta uma regressão isotônica [zadrozny2002transforming] sobre as
probabilidades de um ano de calibração e se mede, num ano posterior, o efeito em
Brier, ECE e PR-AUC — mesmo protocolo de ``conformal.py`` (treino < 2022,
calibração 2022, teste 2023).

Saída: ``modelos/relatorios/calibracao_isotonica.json``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, roc_auc_score

from conformal import ece
from gerar_forecast_ocorrencia import FEATURES, FEATURES_BASE, montar_deter

ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "dataset_ocorrencia_mensal.csv"
DETER = ROOT / "scripts" / "previsao_ocorrencia" / "deter_celula_mes.csv"
REL = ROOT / "modelos" / "relatorios" / "calibracao_isotonica.json"


def novo_modelo():
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.08, max_depth=8, l2_regularization=1.0,
        max_leaf_nodes=63, class_weight="balanced", random_state=42)


def metricas(y, p) -> dict:
    return {"pr_auc": float(average_precision_score(y, p)),
            "roc_auc": float(roc_auc_score(y, p)),
            "brier": float(np.mean((p - y) ** 2)),
            "ece": ece(y, p),
            "p_media": float(p.mean()), "prevalencia": float(y.mean())}


def avaliar(df: pd.DataFrame, feats: list, ano_calib: int, ano_teste: int) -> dict:
    tr = df[df["ano"] < ano_calib]
    cal = df[df["ano"] == ano_calib]
    te = df[df["ano"] == ano_teste]
    m = novo_modelo().fit(tr[feats], tr["alvo"])
    p_cal = m.predict_proba(cal[feats])[:, 1]
    p_te = m.predict_proba(te[feats])[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(p_cal, cal["alvo"])
    y = te["alvo"].to_numpy()
    return {"bruta": metricas(y, p_te), "isotonica": metricas(y, iso.predict(p_te)),
            "n_teste": int(len(te))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=DATASET)
    ap.add_argument("--deter", type=Path, default=DETER)
    ap.add_argument("--ano_calib", type=int, default=2022)
    ap.add_argument("--ano_teste", type=int, default=2023)
    args = ap.parse_args()

    df = montar_deter(pd.read_csv(args.dataset), pd.read_csv(args.deter))
    out = {"protocolo": {"treino_ate": args.ano_calib - 1, "calibracao": args.ano_calib,
                         "teste": args.ano_teste}}
    for nome, feats in [("base", FEATURES_BASE), ("base_deter", FEATURES)]:
        r = avaliar(df, feats, args.ano_calib, args.ano_teste)
        out[nome] = r
        for k in ("bruta", "isotonica"):
            v = r[k]
            print(f"  {nome:10s} {k:9s} PR-AUC={v['pr_auc']:.3f} Brier={v['brier']:.3f} "
                  f"ECE={v['ece']:.3f} P médio={v['p_media']:.3f} (prevalência {v['prevalencia']:.3f})")
    REL.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"salvo em {REL}")


if __name__ == "__main__":
    main()
