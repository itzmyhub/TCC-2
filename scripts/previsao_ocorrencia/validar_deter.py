"""DETER mensal vs histórico — foco na NOVA IGNIÇÃO.

Features causais MENSAIS de desmatamento (≤ t): área DETER na célula no mês t
(`deter_m0`), no mês t-1 (`deter_m1`), e somas móveis dos últimos 3 e 6 meses
(`deter_roll3/6`). Tudo ≤ t → causal para prever fogo em t+1.

Compara ML-base vs ML+DETER em validação temporal em bloco, com foco no estrato de
NOVA IGNIÇÃO (sem foco nos últimos 12 m) — para verificar se a resolução mensal do
DETER amplia o ganho obtido com o PRODES anual (+0,025 PR-AUC na nova ignição).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

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
FEATURES_DETER = FEATURES_BASE + DETER_FEATS


def mk():
    return HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.08, max_depth=8, l2_regularization=1.0,
        max_leaf_nodes=63, class_weight="balanced", random_state=42)


def auc_ap(y, s):
    y = np.asarray(y); s = np.asarray(s, dtype=float)
    if len(np.unique(y)) < 2:
        return {"roc_auc": float("nan"), "pr_auc": float("nan"), "n": int(len(y)), "prev": float(np.mean(y))}
    return {"roc_auc": float(roc_auc_score(y, s)), "pr_auc": float(average_precision_score(y, s)),
            "n": int(len(y)), "prev": float(np.mean(y))}


def _ym_idx(ym: pd.Series) -> pd.Series:
    p = pd.PeriodIndex(ym.astype(str), freq="M")
    return p.year * 12 + (p.month - 1)


def montar_features_deter(df: pd.DataFrame, d: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ym_idx"] = _ym_idx(df["ym"])
    d = d.copy()
    d["ym_idx"] = _ym_idx(d["ym"])
    for k in range(6):
        tmp = d[["LatBin", "LonBin", "ym_idx", "deter_km"]].copy()
        tmp["ym_idx"] = tmp["ym_idx"] + k  # casa linha t com DETER de t-k
        tmp = tmp.rename(columns={"deter_km": f"deter_lag{k}"})
        df = df.merge(tmp, on=["LatBin", "LonBin", "ym_idx"], how="left")
        df[f"deter_lag{k}"] = df[f"deter_lag{k}"].fillna(0.0)
    df["deter_m0"] = df["deter_lag0"]
    df["deter_m1"] = df["deter_lag1"]
    df["deter_roll3"] = df[[f"deter_lag{k}" for k in range(3)]].sum(axis=1)
    df["deter_roll6"] = df[[f"deter_lag{k}" for k in range(6)]].sum(axis=1)
    return df


def main():
    df = pd.read_csv(DATASET)
    d = pd.read_csv(DETER)
    df = montar_features_deter(df, d)
    print(f"Dataset: {len(df):,} linhas | DETER recente (roll6>0): {(df['deter_roll6']>0).mean():.1%}")

    folds = []
    for ano in (2021, 2022, 2023):
        tr, te = df[df["ano"] < ano], df[df["ano"] == ano]
        if len(te) == 0 or tr["alvo"].nunique() < 2:
            continue
        mb = mk().fit(tr[FEATURES_BASE], tr["alvo"])
        md = mk().fit(tr[FEATURES_DETER], tr["alvo"])
        te = te.assign(pb=mb.predict_proba(te[FEATURES_BASE])[:, 1],
                       pd_=md.predict_proba(te[FEATURES_DETER])[:, 1])
        novo = te[te["focos_roll12"] == 0]
        rec = te[te["focos_roll12"] > 0]
        fold = {"ano": int(ano),
                "geral": {"base": auc_ap(te["alvo"], te["pb"]), "deter": auc_ap(te["alvo"], te["pd_"])},
                "nova_ignicao": {"base": auc_ap(novo["alvo"], novo["pb"]), "deter": auc_ap(novo["alvo"], novo["pd_"])},
                "recorrente": {"base": auc_ap(rec["alvo"], rec["pb"]), "deter": auc_ap(rec["alvo"], rec["pd_"])}}
        folds.append(fold)
        g, ni = fold["geral"], fold["nova_ignicao"]
        print(f"[{ano}] GERAL {g['base']['pr_auc']:.3f}->{g['deter']['pr_auc']:.3f} "
              f"(Δ={g['deter']['pr_auc']-g['base']['pr_auc']:+.3f}) | "
              f"NOVA IGNIÇÃO {ni['base']['pr_auc']:.3f}->{ni['deter']['pr_auc']:.3f} "
              f"(Δ={ni['deter']['pr_auc']-ni['base']['pr_auc']:+.3f})", flush=True)

    def media(chave):
        out = {m: float(np.nanmean([f[chave][m]["pr_auc"] for f in folds])) for m in ("base", "deter")}
        out["ganho"] = out["deter"] - out["base"]
        return out
    res = {"tarefa": "driver_deter_mensal",
           "media_geral": media("geral"), "media_nova_ignicao": media("nova_ignicao"),
           "media_recorrente": media("recorrente"), "folds": folds}
    REL_DIR.mkdir(parents=True, exist_ok=True)
    (REL_DIR / "driver_deter.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n== MÉDIAS (Δ PR-AUC do DETER mensal sobre o histórico) ==")
    for nome, k in [("GERAL", "media_geral"), ("NOVA IGNIÇÃO", "media_nova_ignicao"), ("RECORRENTE", "media_recorrente")]:
        print(f"  {nome:13s} base={res[k]['base']:.3f} +DETER={res[k]['deter']:.3f} => Δ={res[k]['ganho']:+.3f}")
    print(f"  (comparar: PRODES anual deu Δ=+0,025 na nova ignição)")


if __name__ == "__main__":
    main()
