"""Valida o ganho do driver antrópico (distância a cidades) sobre o histórico.

Mesmo protocolo do P5 (validar_gridded), para comparação justa: ML-base vs
ML+antrópico em validação TEMPORAL e ESPACIAL em bloco. O teste ESPACIAL
(leave-region-out) é o mais informativo: lat/lon não transferem para regiões
não vistas, mas a distância a cidades — proxy de acessibilidade — sim.
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
DATASET = ROOT / "dataset_ocorrencia_antropico.csv"
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
FEATURES_ANTRO = FEATURES_BASE + ["dist_cidade_km"]


def modelo():
    return HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.08, max_depth=8, l2_regularization=1.0,
        max_leaf_nodes=63, class_weight="balanced", random_state=42)


def auc_ap(y, s):
    y = np.asarray(y); s = np.asarray(s, dtype=float)
    if len(np.unique(y)) < 2:
        return {"roc_auc": float("nan"), "pr_auc": float("nan")}
    return {"roc_auc": float(roc_auc_score(y, s)), "pr_auc": float(average_precision_score(y, s))}


def avaliar(tr, te):
    mb = modelo().fit(tr[FEATURES_BASE], tr["alvo"])
    ma = modelo().fit(tr[FEATURES_ANTRO], tr["alvo"])
    return (auc_ap(te["alvo"], mb.predict_proba(te[FEATURES_BASE])[:, 1]),
            auc_ap(te["alvo"], ma.predict_proba(te[FEATURES_ANTRO])[:, 1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, default=str(DATASET))
    ap.add_argument("--anos_teste", type=int, nargs="+", default=[2021, 2022, 2023])
    args = ap.parse_args()

    df = pd.read_csv(args.dataset)
    print(f"Dataset: {len(df):,} linhas | prevalência {df['alvo'].mean():.3f}")

    # Temporal
    ft = []
    print("\n== TEMPORAL (rolling-origin) ==")
    for ano in args.anos_teste:
        tr, te = df[df["ano"] < ano], df[df["ano"] == ano]
        if len(te) == 0 or tr["alvo"].nunique() < 2:
            continue
        b, a = avaliar(tr, te)
        ft.append({"ano": int(ano), "base": b, "antro": a})
        print(f"  [{ano}] base PR-AUC={b['pr_auc']:.3f} | +antrópico PR-AUC={a['pr_auc']:.3f} "
              f"(Δ={a['pr_auc']-b['pr_auc']:+.3f})")

    # Espacial
    fe = []
    print("\n== ESPACIAL (leave-region-out, 5 blocos de longitude) ==")
    df["bloco"] = pd.qcut(df["LonBin"], 5, labels=False, duplicates="drop")
    for b_ in sorted(df["bloco"].dropna().unique()):
        tr, te = df[df["bloco"] != b_], df[df["bloco"] == b_]
        if te["alvo"].nunique() < 2:
            continue
        b, a = avaliar(tr, te)
        fe.append({"bloco": int(b_), "base": b, "antro": a})
        print(f"  [bloco {b_}] base PR-AUC={b['pr_auc']:.3f} | +antrópico PR-AUC={a['pr_auc']:.3f} "
              f"(Δ={a['pr_auc']-b['pr_auc']:+.3f})")

    def med(folds, k):
        return {mk: float(np.nanmean([f[k][mk] for f in folds])) for mk in ("roc_auc", "pr_auc")}

    res = {
        "tarefa": "driver_antropico_dist_cidade",
        "temporal": {"folds": ft, "media_base": med(ft, "base"), "media_antro": med(ft, "antro"),
                     "ganho_pr_auc": med(ft, "antro")["pr_auc"] - med(ft, "base")["pr_auc"]},
        "espacial": {"folds": fe, "media_base": med(fe, "base"), "media_antro": med(fe, "antro"),
                     "ganho_pr_auc": med(fe, "antro")["pr_auc"] - med(fe, "base")["pr_auc"]},
    }
    REL_DIR.mkdir(parents=True, exist_ok=True)
    out = REL_DIR / "driver_antropico.json"
    out.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n== RESUMO (Δ PR-AUC = ganho do driver antrópico sobre o histórico) ==")
    print(f"  TEMPORAL: base={res['temporal']['media_base']['pr_auc']:.3f} "
          f"+antrópico={res['temporal']['media_antro']['pr_auc']:.3f} "
          f"=> Δ={res['temporal']['ganho_pr_auc']:+.3f}")
    print(f"  ESPACIAL: base={res['espacial']['media_base']['pr_auc']:.3f} "
          f"+antrópico={res['espacial']['media_antro']['pr_auc']:.3f} "
          f"=> Δ={res['espacial']['ganho_pr_auc']:+.3f}")
    print(f"  salvo em: {out}")


if __name__ == "__main__":
    main()
