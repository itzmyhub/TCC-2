"""Treino e VALIDAÇÃO ESPAÇO-TEMPORAL EM BLOCO do modelo de ocorrência (P2).

Implementa a validação honesta que o estado da arte exige para dados
geográficos [roberts2017cv]: blocos temporais (rolling-origin por ano) e blocos
espaciais (leave-one-region-out), evitando o otimismo do split aleatório que
inflava a acurácia do pipeline original em ~12,8 pp.

Compara o modelo (HistGradientBoosting, sklearn) contra dois baselines
obrigatórios:
  • Persistência: "se queimou neste mês, queima no próximo" (alvo := fogo_t).
  • Climatologia: taxa-base histórica por (célula, mês-calendário), estimada
    SÓ no treino.

Métricas: ROC-AUC, Average Precision (PR-AUC, robusta a desbalanceamento),
Brier, e F1@0.5. Um modelo só "agrega valor" se superar os baselines.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             f1_score, roc_auc_score)

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


def novo_modelo() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.08, max_depth=8,
        l2_regularization=1.0, max_leaf_nodes=63,
        class_weight="balanced", random_state=42,
    )


def metricas(y, p) -> dict:
    yhat = (p >= 0.5).astype(int)
    out = {
        "roc_auc": float(roc_auc_score(y, p)) if y.nunique() > 1 else float("nan"),
        "pr_auc": float(average_precision_score(y, p)) if y.nunique() > 1 else float("nan"),
        "brier": float(brier_score_loss(y, p)),
        "f1@0.5": float(f1_score(y, yhat, zero_division=0)),
        "prevalencia": float(y.mean()),
        "n": int(len(y)),
    }
    return out


def baseline_persistencia(tr, te):
    # score = fogo do mês corrente (0/1)
    return metricas(te["alvo"], te["fogo"].astype(float).values)


def baseline_climatologia(tr, te):
    # taxa-base por (célula, mês) estimada no treino; fallback célula→global
    chave = ["LatBin", "LonBin", "mes"]
    tx = tr.groupby(chave)["alvo"].mean().rename("clim_cm")
    tx_cel = tr.groupby(["LatBin", "LonBin"])["alvo"].mean().rename("clim_c")
    glob = tr["alvo"].mean()
    te = te.merge(tx, on=chave, how="left").merge(tx_cel, on=["LatBin", "LonBin"], how="left")
    p = te["clim_cm"].fillna(te["clim_c"]).fillna(glob).values
    return metricas(te["alvo"], p)


def validacao_temporal(df: pd.DataFrame, anos_teste) -> list:
    folds = []
    for ano in anos_teste:
        tr = df[df["ano"] < ano]
        te = df[df["ano"] == ano]
        if len(te) == 0 or tr["alvo"].nunique() < 2:
            continue
        m = novo_modelo().fit(tr[FEATURES], tr["alvo"])
        p = m.predict_proba(te[FEATURES])[:, 1]
        folds.append({
            "ano_teste": int(ano),
            "modelo": metricas(te["alvo"], p),
            "baseline_persistencia": baseline_persistencia(tr, te),
            "baseline_climatologia": baseline_climatologia(tr, te.copy()),
        })
        f = folds[-1]
        print(f"  [temporal {ano}] modelo PR-AUC={f['modelo']['pr_auc']:.3f} "
              f"ROC-AUC={f['modelo']['roc_auc']:.3f} | "
              f"persist PR-AUC={f['baseline_persistencia']['pr_auc']:.3f} | "
              f"clim PR-AUC={f['baseline_climatologia']['pr_auc']:.3f}")
    return folds


def validacao_espacial(df: pd.DataFrame, n_blocos=5) -> list:
    # Blocos espaciais por faixas de longitude (regiões contíguas)
    df = df.copy()
    df["bloco"] = pd.qcut(df["LonBin"], n_blocos, labels=False, duplicates="drop")
    folds = []
    for b in sorted(df["bloco"].dropna().unique()):
        tr = df[df["bloco"] != b]
        te = df[df["bloco"] == b]
        if te["alvo"].nunique() < 2:
            continue
        m = novo_modelo().fit(tr[FEATURES], tr["alvo"])
        p = m.predict_proba(te[FEATURES])[:, 1]
        folds.append({"bloco": int(b), "modelo": metricas(te["alvo"], p),
                      "baseline_climatologia": baseline_climatologia(tr, te.copy())})
        f = folds[-1]
        print(f"  [espacial bloco {b}] modelo PR-AUC={f['modelo']['pr_auc']:.3f} "
              f"ROC-AUC={f['modelo']['roc_auc']:.3f} | "
              f"clim PR-AUC={f['baseline_climatologia']['pr_auc']:.3f}")
    return folds


def media(folds, chave="modelo"):
    if not folds:
        return {}
    ks = ["roc_auc", "pr_auc", "brier", "f1@0.5"]
    return {k: float(np.nanmean([f[chave][k] for f in folds])) for k in ks}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, default=str(DATASET))
    ap.add_argument("--anos_teste", type=int, nargs="+", default=[2021, 2022, 2023])
    args = ap.parse_args()

    print("Carregando dataset de ocorrência...")
    df = pd.read_csv(args.dataset)
    print(f"  {len(df):,} linhas | prevalência alvo = {df['alvo'].mean():.3f}")

    print("\n== Validação TEMPORAL em bloco (rolling-origin) ==")
    ft = validacao_temporal(df, args.anos_teste)
    print("\n== Validação ESPACIAL em bloco (leave-region-out) ==")
    fe = validacao_espacial(df, n_blocos=5)

    resultado = {
        "tarefa": "previsao_ocorrencia_fogo_t+1",
        "n_linhas": int(len(df)),
        "prevalencia": float(df["alvo"].mean()),
        "validacao_temporal": {
            "folds": ft,
            "media_modelo": media(ft, "modelo"),
            "media_persistencia": media(ft, "baseline_persistencia"),
            "media_climatologia": media(ft, "baseline_climatologia"),
        },
        "validacao_espacial": {
            "folds": fe,
            "media_modelo": media(fe, "modelo"),
            "media_climatologia": media(fe, "baseline_climatologia"),
        },
    }
    REL_DIR.mkdir(parents=True, exist_ok=True)
    out = REL_DIR / "previsao_ocorrencia_validacao.json"
    out.write_text(json.dumps(resultado, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n== RESUMO (médias) ==")
    mt = resultado["validacao_temporal"]
    print(f"  TEMPORAL  modelo: PR-AUC={mt['media_modelo']['pr_auc']:.3f} "
          f"ROC-AUC={mt['media_modelo']['roc_auc']:.3f} Brier={mt['media_modelo']['brier']:.3f}")
    print(f"            persistência: PR-AUC={mt['media_persistencia']['pr_auc']:.3f}")
    print(f"            climatologia: PR-AUC={mt['media_climatologia']['pr_auc']:.3f}")
    me = resultado["validacao_espacial"]
    print(f"  ESPACIAL  modelo: PR-AUC={me['media_modelo']['pr_auc']:.3f} "
          f"ROC-AUC={me['media_modelo']['roc_auc']:.3f}")
    print(f"  salvo em: {out}")


if __name__ == "__main__":
    main()
