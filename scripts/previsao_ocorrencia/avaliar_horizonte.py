"""Curva de SKILL por horizonte de previsão (t+1, t+2, t+3 meses).

Mostra que o sistema é um forecast com antecedência quantificada: como a PR-AUC
decai conforme se prevê 1, 2 ou 3 meses à frente (geral e no estrato de nova
ignição). Mesmo protocolo (features base+DETER, validação rolling-origin em bloco).

Saídas: modelos/relatorios/avaliacao_horizonte.json e figura PR-AUC vs horizonte.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
DETER = ROOT / "scripts" / "previsao_ocorrencia" / "deter_celula_mes.csv"
REL_DIR = ROOT / "modelos" / "relatorios"
DATASETS = {1: ROOT / "dataset_ocorrencia_mensal.csv",
            2: ROOT / "dataset_ocorrencia_h2.csv",
            3: ROOT / "dataset_ocorrencia_h3.csv"}

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


def avaliar(df):
    pg, pn = [], []
    for ano in (2021, 2022, 2023):
        tr, te = df[df["ano"] < ano], df[df["ano"] == ano]
        if len(te) == 0 or tr["alvo"].nunique() < 2:
            continue
        m = HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.08, max_depth=8, l2_regularization=1.0,
            max_leaf_nodes=63, class_weight="balanced", random_state=42).fit(tr[FEATURES], tr["alvo"])
        te = te.assign(p=m.predict_proba(te[FEATURES])[:, 1])
        pg.append(average_precision_score(te["alvo"], te["p"]))
        novo = te[te["focos_roll12"] == 0]
        if novo["alvo"].nunique() > 1:
            pn.append(average_precision_score(novo["alvo"], novo["p"]))
    return float(np.mean(pg)), float(np.mean(pn))


def main():
    d = pd.read_csv(DETER)
    res = {}
    print(f"{'horizonte':>9} {'PR-AUC geral':>13} {'PR-AUC nova ign.':>17} {'prevalência':>12}")
    for h, path in DATASETS.items():
        df = montar_deter(pd.read_csv(path), d)
        g, n = avaliar(df)
        res[f"t+{h}"] = {"pr_auc_geral": g, "pr_auc_nova_ignicao": n, "prevalencia": float(df["alvo"].mean())}
        print(f"{'t+'+str(h):>9} {g:>13.3f} {n:>17.3f} {df['alvo'].mean():>12.3f}")

    REL_DIR.mkdir(parents=True, exist_ok=True)
    (REL_DIR / "avaliacao_horizonte.json").write_text(
        json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")

    hs = list(DATASETS.keys())
    g = [res[f"t+{h}"]["pr_auc_geral"] for h in hs]
    n = [res[f"t+{h}"]["pr_auc_nova_ignicao"] for h in hs]
    plt.figure(figsize=(6, 4.2))
    plt.plot(hs, g, "o-", label="Geral", color="#d73027", linewidth=2)
    plt.plot(hs, n, "s--", label="Nova ignição", color="#4575b4", linewidth=2)
    plt.xticks(hs, [f"t+{h}" for h in hs])
    plt.xlabel("Horizonte de previsão (meses)")
    plt.ylabel("PR-AUC")
    plt.title("Decaimento do skill por horizonte")
    plt.grid(alpha=0.3); plt.legend(); plt.tight_layout()
    fig_path = REL_DIR / "avaliacao_horizonte.png"
    plt.savefig(fig_path, dpi=200)
    print(f"\nfigura: {fig_path}\njson: {REL_DIR / 'avaliacao_horizonte.json'}")


if __name__ == "__main__":
    main()
