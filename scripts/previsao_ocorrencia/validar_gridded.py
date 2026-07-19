"""P5 — Validação do ganho do clima gridado real + FWI.

Compara, no MESMO dataset (clima real, rows com cobertura) e validação temporal
em bloco (rolling-origin), quatro preditores de fogo observado em t+1:

  • ML BASE     — features causais sem clima real (histórico de fogo + sazonalidade);
  • ML GRIDDED  — BASE + clima real (temp/rh/vento/precip) + FWI (mean/max/dias>10);
  • FWI real    — fwi_mean do mês t como score (baseline físico, agora p/ TODO o dataset);
  • INPE RiscoFogo — coluna RiscoFogo do mês t como score.

Quantifica: (a) quanto o clima real + FWI agrega sobre o histórico de fogo;
(b) o benchmark FWI-vs-ML agora no dataset completo (não mais amostra de P3).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "dataset_ocorrencia_gridded.csv"
REL_DIR = ROOT / "modelos" / "relatorios"

FEATURES_BASE = [
    "LatBin", "LonBin", "mes_sin", "mes_cos", "estacao_seca",
    "DiaSemChuva", "Precipitacao", "RiscoFogo_inpe",
    "focos_lag1", "focos_lag2", "focos_lag3",
    "fogo_lag1", "fogo_lag2", "fogo_lag3",
    "focos_roll3", "focos_roll6", "focos_roll12",
    "fogo_roll3", "fogo_roll6", "fogo_roll12",
    "FRP_lag1", "meses_desde_fogo",
]
CLIMA_REAL = ["temp_mean", "rh_mean", "wind_mean", "precip_sum",
              "dias_secos", "fwi_mean", "fwi_max", "dias_fwi_gt10"]
FEATURES_GRID = FEATURES_BASE + CLIMA_REAL


def modelo():
    return HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.08, max_depth=8, l2_regularization=1.0,
        max_leaf_nodes=63, class_weight="balanced", random_state=42)


def auc_ap(y, s):
    y = np.asarray(y); s = np.asarray(s, dtype=float)
    if len(np.unique(y)) < 2:
        return {"roc_auc": float("nan"), "pr_auc": float("nan")}
    return {"roc_auc": float(roc_auc_score(y, s)), "pr_auc": float(average_precision_score(y, s))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, default=str(DATASET))
    ap.add_argument("--anos_teste", type=int, nargs="+", default=[2021, 2022, 2023])
    args = ap.parse_args()

    df = pd.read_csv(args.dataset)
    print(f"Dataset gridado: {len(df):,} linhas | prevalência {df['alvo'].mean():.3f}")

    folds = []
    for ano in args.anos_teste:
        tr = df[df["ano"] < ano]; te = df[df["ano"] == ano]
        if len(te) == 0 or tr["alvo"].nunique() < 2:
            continue
        mb = modelo().fit(tr[FEATURES_BASE], tr["alvo"])
        mg = modelo().fit(tr[FEATURES_GRID], tr["alvo"])
        fold = {
            "ano": int(ano),
            "ml_base": auc_ap(te["alvo"], mb.predict_proba(te[FEATURES_BASE])[:, 1]),
            "ml_gridded": auc_ap(te["alvo"], mg.predict_proba(te[FEATURES_GRID])[:, 1]),
            "fwi_real": auc_ap(te["alvo"], te["fwi_mean"]),
            "indice_inpe": auc_ap(te["alvo"], te["RiscoFogo_inpe"]),
        }
        folds.append(fold)
        print(f"  [{ano}] BASE PR-AUC={fold['ml_base']['pr_auc']:.3f} | "
              f"GRIDDED PR-AUC={fold['ml_gridded']['pr_auc']:.3f} | "
              f"FWI PR-AUC={fold['fwi_real']['pr_auc']:.3f} | "
              f"INPE PR-AUC={fold['indice_inpe']['pr_auc']:.3f}")

    def med(k):
        return {mk: float(np.nanmean([f[k][mk] for f in folds])) for mk in ("roc_auc", "pr_auc")}
    resumo = {k: med(k) for k in ("ml_base", "ml_gridded", "fwi_real", "indice_inpe")}
    ganho = resumo["ml_gridded"]["pr_auc"] - resumo["ml_base"]["pr_auc"]

    resultado = {"tarefa": "P5_validacao_clima_gridado", "n_linhas": int(len(df)),
                 "prevalencia": float(df["alvo"].mean()), "folds": folds,
                 "medias": resumo, "ganho_gridded_sobre_base_pr_auc": ganho}
    REL_DIR.mkdir(parents=True, exist_ok=True)
    out = REL_DIR / "previsao_ocorrencia_gridded.json"
    out.write_text(json.dumps(resultado, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n== MÉDIAS (rolling-origin) ==")
    print(f"  ML BASE     PR-AUC={resumo['ml_base']['pr_auc']:.3f} ROC-AUC={resumo['ml_base']['roc_auc']:.3f}")
    print(f"  ML GRIDDED  PR-AUC={resumo['ml_gridded']['pr_auc']:.3f} ROC-AUC={resumo['ml_gridded']['roc_auc']:.3f}")
    print(f"  FWI real    PR-AUC={resumo['fwi_real']['pr_auc']:.3f} ROC-AUC={resumo['fwi_real']['roc_auc']:.3f}")
    print(f"  INPE RF     PR-AUC={resumo['indice_inpe']['pr_auc']:.3f}")
    print(f"  >> ganho do clima real + FWI sobre o histórico-só: {ganho:+.3f} PR-AUC")
    print(f"  salvo em: {out}")


if __name__ == "__main__":
    main()
