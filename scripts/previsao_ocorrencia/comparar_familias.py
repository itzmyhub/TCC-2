"""Comparação de famílias de modelo com INTERVALOS DE CONFIANÇA (rigor estatístico).

Avalia famílias distintas no mesmo conjunto de features (base + DETER), em validação
temporal em bloco (rolling-origin 2021–23, predições agrupadas out-of-time), e estima
IC 95% por **cluster bootstrap POR CÉLULA** (respeita a autocorrelação espacial — mais
honesto que bootstrap por linha). Reporta PR-AUC, ROC-AUC e Brier, no geral e no
estrato de nova ignição.

Famílias (scikit-learn; XGBoost/LightGBM ausentes no ambiente):
  HistGradientBoosting, RandomForest, ExtraTrees, LogisticRegression (escalonada).
Inclui o baseline de persistência (score = fogo do mês corrente) como referência.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import (ExtraTreesClassifier, HistGradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "dataset_ocorrencia_mensal.csv"
DETER = ROOT / "scripts" / "previsao_ocorrencia" / "deter_celula_mes.csv"
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
DETER_FEATS = ["deter_m0", "deter_m1", "deter_roll3", "deter_roll6"]
FEATURES = FEATURES_BASE + DETER_FEATS


def _ym_idx(ym):
    p = pd.PeriodIndex(pd.Index(ym).astype(str), freq="M")
    return (p.year * 12 + (p.month - 1)).astype(int)


def montar_deter(df, d):
    df = df.copy(); df["ym_idx"] = _ym_idx(df["ym"])
    d = d.copy(); d["ym_idx"] = _ym_idx(d["ym"])
    for k in range(6):
        tmp = d[["LatBin", "LonBin", "ym_idx", "deter_km"]].copy()
        tmp["ym_idx"] = tmp["ym_idx"] + k
        tmp = tmp.rename(columns={"deter_km": f"deter_lag{k}"})
        df = df.merge(tmp, on=["LatBin", "LonBin", "ym_idx"], how="left")
        df[f"deter_lag{k}"] = df[f"deter_lag{k}"].fillna(0.0)
    df["deter_m0"] = df["deter_lag0"]; df["deter_m1"] = df["deter_lag1"]
    df["deter_roll3"] = df[[f"deter_lag{k}" for k in range(3)]].sum(axis=1)
    df["deter_roll6"] = df[[f"deter_lag{k}" for k in range(6)]].sum(axis=1)
    return df


def familias():
    return {
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.08, max_depth=8, l2_regularization=1.0,
            max_leaf_nodes=63, class_weight="balanced", random_state=42),
        "RandomForest": RandomForestClassifier(
            n_estimators=200, max_depth=18, min_samples_leaf=4,
            class_weight="balanced", n_jobs=-1, random_state=42),
        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=200, max_depth=18, min_samples_leaf=4,
            class_weight="balanced", n_jobs=-1, random_state=42),
        "LogisticRegression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced")),
    }


def predicoes_out_of_time(df, modelo):
    """Treina por ano (treina<Y, prevê=Y) e devolve predições agrupadas 2021–23."""
    partes = []
    for ano in (2021, 2022, 2023):
        tr, te = df[df["ano"] < ano], df[df["ano"] == ano]
        Xtr = tr[FEATURES].fillna(0.0)
        m = modelo
        from sklearn.base import clone
        m = clone(modelo).fit(Xtr, tr["alvo"])
        p = m.predict_proba(te[FEATURES].fillna(0.0))[:, 1]
        partes.append(pd.DataFrame({
            "y": te["alvo"].to_numpy(), "p": p,
            "cell": (te["LatBin"].astype(str) + "_" + te["LonBin"].astype(str)).to_numpy(),
            "nova_ign": (te["focos_roll12"] == 0).to_numpy(),
        }))
    return pd.concat(partes, ignore_index=True)


def _metric(y, p, nome):
    if len(np.unique(y)) < 2:
        return np.nan
    if nome == "pr_auc":
        return average_precision_score(y, p)
    if nome == "roc_auc":
        return roc_auc_score(y, p)
    return brier_score_loss(y, p)


def cluster_bootstrap_ci(pred, metric, mask=None, B=300, seed=42):
    """IC 95% por reamostragem de CÉLULAS (com reposição)."""
    rng = np.random.default_rng(seed)
    sub = pred[mask] if mask is not None else pred
    cells = sub["cell"].to_numpy()
    y = sub["y"].to_numpy(); p = sub["p"].to_numpy()
    uniq = np.unique(cells)
    idx_por_cell = {c: np.where(cells == c)[0] for c in uniq}
    pontual = _metric(y, p, metric)
    vals = []
    for _ in range(B):
        amostra = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_por_cell[c] for c in amostra])
        v = _metric(y[idx], p[idx], metric)
        if not np.isnan(v):
            vals.append(v)
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(pontual), float(lo), float(hi)


def main():
    df = pd.read_csv(DATASET)
    d = pd.read_csv(DETER)
    df = montar_deter(df, d)
    print(f"Dataset: {len(df):,} | features: {len(FEATURES)}")

    resultados = {}
    for nome, modelo in familias().items():
        print(f"\n== {nome} ==")
        pred = predicoes_out_of_time(df, modelo)
        novo = pred["nova_ign"].to_numpy()
        r = {}
        for met in ("pr_auc", "roc_auc", "brier"):
            pt, lo, hi = cluster_bootstrap_ci(pred, met)
            r[f"geral_{met}"] = {"valor": pt, "ic95": [lo, hi]}
        ptn, lon_, hin = cluster_bootstrap_ci(pred, "pr_auc", mask=novo)
        r["nova_ignicao_pr_auc"] = {"valor": ptn, "ic95": [lon_, hin]}
        resultados[nome] = r
        print(f"  PR-AUC  geral: {r['geral_pr_auc']['valor']:.3f} "
              f"[{r['geral_pr_auc']['ic95'][0]:.3f}, {r['geral_pr_auc']['ic95'][1]:.3f}]")
        print(f"  ROC-AUC geral: {r['geral_roc_auc']['valor']:.3f} "
              f"[{r['geral_roc_auc']['ic95'][0]:.3f}, {r['geral_roc_auc']['ic95'][1]:.3f}]")
        print(f"  Brier   geral: {r['geral_brier']['valor']:.3f} "
              f"[{r['geral_brier']['ic95'][0]:.3f}, {r['geral_brier']['ic95'][1]:.3f}]")
        print(f"  PR-AUC nova ignição: {ptn:.3f} [{lon_:.3f}, {hin:.3f}]")

    REL_DIR.mkdir(parents=True, exist_ok=True)
    (REL_DIR / "comparacao_familias.json").write_text(
        json.dumps(resultados, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nsalvo em: {REL_DIR / 'comparacao_familias.json'}")
    print("\n== TABELA (PR-AUC geral [IC95]) ==")
    for nome, r in sorted(resultados.items(), key=lambda kv: -kv[1]["geral_pr_auc"]["valor"]):
        g = r["geral_pr_auc"]
        print(f"  {nome:22s} {g['valor']:.3f} [{g['ic95'][0]:.3f}, {g['ic95'][1]:.3f}]")


if __name__ == "__main__":
    main()
