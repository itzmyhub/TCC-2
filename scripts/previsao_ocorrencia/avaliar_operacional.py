"""Avaliação OPERACIONAL: priorização de alertas (curva recall@K / precision@K).

Traduz a PR-AUC em valor de decisão: dado um orçamento de vigilância (alertar as
top-K% células de maior risco previsto), quantos focos do próximo mês são
capturados (recall@K) e a que custo de falso-alarme (precision@K)? Compara o
modelo de ML com o índice operacional do INPE (Risco de Fogo) sob o MESMO orçamento
--- o contraste que demonstra ganho prático.

Inclui o ângulo de destaque: a mesma análise restrita às células RECÉM-DESMATADAS
(DETER nos últimos 6 meses), onde o modelo deve identificar a nova ignição que o
índice físico não capta.

Saídas: modelos/relatorios/avaliacao_operacional.json e figura recall@K.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
BUDGETS = [1, 2, 5, 10, 15, 20, 30]  # % de células alertadas


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


def predicoes_oot(df):
    partes = []
    for ano in (2021, 2022, 2023):
        tr, te = df[df["ano"] < ano], df[df["ano"] == ano]
        if len(te) == 0 or tr["alvo"].nunique() < 2:
            continue
        m = HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.08, max_depth=8, l2_regularization=1.0,
            max_leaf_nodes=63, class_weight="balanced", random_state=42).fit(tr[FEATURES], tr["alvo"])
        te = te.assign(p_ml=m.predict_proba(te[FEATURES])[:, 1])
        partes.append(te[["alvo", "p_ml", "RiscoFogo_inpe", "focos_roll12", "deter_roll6"]])
    return pd.concat(partes, ignore_index=True)


def recall_prec_at_k(y, score, k_pct):
    n = len(y)
    n_alert = max(1, int(np.ceil(k_pct / 100.0 * n)))
    ordem = np.argsort(-np.asarray(score, dtype=float))
    top = ordem[:n_alert]
    pos_top = int(np.asarray(y)[top].sum())
    total_pos = int(np.asarray(y).sum())
    recall = pos_top / total_pos if total_pos else float("nan")
    prec = pos_top / n_alert
    return recall, prec


def curva(sub, nome):
    out = {"n": int(len(sub)), "prevalencia": float(sub["alvo"].mean()), "budgets": {}}
    for k in BUDGETS:
        rm, pm = recall_prec_at_k(sub["alvo"], sub["p_ml"], k)
        ri, pi = recall_prec_at_k(sub["alvo"], sub["RiscoFogo_inpe"], k)
        out["budgets"][f"{k}%"] = {
            "ml": {"recall": rm, "precision": pm, "lift": rm / (k / 100.0)},
            "inpe": {"recall": ri, "precision": pi, "lift": ri / (k / 100.0)},
        }
    return out


def main():
    df = pd.read_csv(DATASET)
    d = pd.read_csv(DETER)
    df = montar_deter(df, d)
    pred = predicoes_oot(df)
    print(f"Predições out-of-time (2021–23): {len(pred):,} | prevalência {pred['alvo'].mean():.3f}")

    glob = curva(pred, "global")
    fresh = curva(pred[pred["deter_roll6"] > 0], "recem_desmatadas")
    novo = curva(pred[pred["focos_roll12"] == 0], "nova_ignicao")

    res = {"global": glob, "recem_desmatadas": fresh, "nova_ignicao": novo}
    REL_DIR.mkdir(parents=True, exist_ok=True)
    (REL_DIR / "avaliacao_operacional.json").write_text(
        json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")

    def linha(nome, c):
        print(f"\n== {nome} (n={c['n']:,}, prev={c['prevalencia']:.3f}) ==")
        print(f"  {'orçamento':>9} | {'recall ML':>9} {'prec ML':>8} {'lift ML':>7} | {'recall INPE':>11} {'lift INPE':>9}")
        for k in BUDGETS:
            b = c["budgets"][f"{k}%"]
            print(f"  {k:>7}% | {b['ml']['recall']:>9.3f} {b['ml']['precision']:>8.3f} {b['ml']['lift']:>7.2f} |"
                  f" {b['inpe']['recall']:>11.3f} {b['inpe']['lift']:>9.2f}")
    linha("GLOBAL", glob)
    linha("RECÉM-DESMATADAS (DETER 6m > 0)", fresh)
    linha("NOVA IGNIÇÃO (sem foco 12m)", novo)

    # Figura: recall@K (ML vs INPE), global e recém-desmatadas
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, (nome, c) in zip(axes, [("Global", glob), ("Células recém-desmatadas", fresh)]):
        ks = BUDGETS
        rml = [c["budgets"][f"{k}%"]["ml"]["recall"] for k in ks]
        rin = [c["budgets"][f"{k}%"]["inpe"]["recall"] for k in ks]
        ax.plot(ks, rml, "o-", label="Modelo ML", color="#d73027", linewidth=2)
        ax.plot(ks, rin, "s--", label="Índice INPE", color="#4575b4")
        ax.plot(ks, [k / 100 for k in ks], ":", color="gray", label="Aleatório")
        ax.set_xlabel("Orçamento de alertas (top-K% células)")
        ax.set_ylabel("Recall (focos do próx. mês capturados)")
        ax.set_title(nome)
        ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout()
    fig_path = REL_DIR / "avaliacao_operacional_recall_k.png"
    fig.savefig(fig_path, dpi=200)
    print(f"\nfigura: {fig_path}\njson: {REL_DIR / 'avaliacao_operacional.json'}")


if __name__ == "__main__":
    main()
