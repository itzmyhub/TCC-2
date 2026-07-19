"""Teste decisivo: o DESMATAMENTO (PRODES) agrega sobre o histórico de fogo?

Features causais (estritamente ≤ t): desmatamento no(s) ANO(S) ANTERIOR(ES) da
célula — defor_lag1 (ano Y-1), defor_lag2 (Y-2), defor_cum3 (soma Y-3..Y-1).
Como o ano PRODES anterior é totalmente observado antes do ano Y, não há
vazamento ao prever fogo dentro de Y.

Avaliação (validação temporal em bloco) com a chave do achado anterior: além do
desempenho GERAL, mede o ganho ESPECIFICAMENTE no estrato de **nova ignição**
(células sem foco nos últimos 12 meses), onde a persistência é inútil e onde o
desmatamento — causa proximal antrópica — deveria provar seu valor.
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
DESMAT = ROOT / "scripts" / "previsao_ocorrencia" / "desmatamento_celula_ano.csv"
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
DEFOR = ["defor_lag1", "defor_lag2", "defor_cum3"]
FEATURES_DEF = FEATURES_BASE + DEFOR


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


def montar_features_defor(df: pd.DataFrame, d: pd.DataFrame) -> pd.DataFrame:
    def lag(df, k, nome):
        dd = d.copy()
        dd["ano"] = dd["ano"] + k  # desloca: linha do ano Y casa com defor de Y-k
        dd = dd.rename(columns={"defor_km": nome})
        return df.merge(dd[["LatBin", "LonBin", "ano", nome]], on=["LatBin", "LonBin", "ano"], how="left")
    df = lag(df, 1, "defor_lag1")
    df = lag(df, 2, "defor_lag2")
    df = lag(df, 3, "defor_lag3")
    for c in ("defor_lag1", "defor_lag2", "defor_lag3"):
        df[c] = df[c].fillna(0.0)
    df["defor_cum3"] = df[["defor_lag1", "defor_lag2", "defor_lag3"]].sum(axis=1)
    return df


def main():
    df = pd.read_csv(DATASET)
    d = pd.read_csv(DESMAT)
    df = montar_features_defor(df, d)
    cob = (df["defor_cum3"] > 0).mean()
    print(f"Dataset: {len(df):,} linhas | prevalência {df['alvo'].mean():.3f} | "
          f"linhas com desmatamento recente (cum3>0): {cob:.1%}")

    folds = []
    for ano in (2021, 2022, 2023):
        tr, te = df[df["ano"] < ano], df[df["ano"] == ano]
        if len(te) == 0 or tr["alvo"].nunique() < 2:
            continue
        mb = mk().fit(tr[FEATURES_BASE], tr["alvo"])
        md = mk().fit(tr[FEATURES_DEF], tr["alvo"])
        te = te.assign(pb=mb.predict_proba(te[FEATURES_BASE])[:, 1],
                       pd_=md.predict_proba(te[FEATURES_DEF])[:, 1])
        geral = {"base": auc_ap(te["alvo"], te["pb"]), "defor": auc_ap(te["alvo"], te["pd_"])}
        novo = te[te["focos_roll12"] == 0]
        recor = te[te["focos_roll12"] > 0]
        estrato = {
            "nova_ignicao": {"base": auc_ap(novo["alvo"], novo["pb"]), "defor": auc_ap(novo["alvo"], novo["pd_"])},
            "recorrente": {"base": auc_ap(recor["alvo"], recor["pb"]), "defor": auc_ap(recor["alvo"], recor["pd_"])},
        }
        folds.append({"ano": int(ano), "geral": geral, "estrato": estrato})
        print(f"\n[{ano}] GERAL base PR-AUC={geral['base']['pr_auc']:.3f} -> +defor {geral['defor']['pr_auc']:.3f} "
              f"(Δ={geral['defor']['pr_auc']-geral['base']['pr_auc']:+.3f})")
        ni = estrato["nova_ignicao"]
        print(f"     NOVA IGNIÇÃO (n={ni['base']['n']}, prev={ni['base']['prev']:.3f}): "
              f"base PR-AUC={ni['base']['pr_auc']:.3f} -> +defor {ni['defor']['pr_auc']:.3f} "
              f"(Δ={ni['defor']['pr_auc']-ni['base']['pr_auc']:+.3f})")

    def media(path):
        out = {}
        for m in ("base", "defor"):
            vals = []
            for f in folds:
                node = f
                for p in path:
                    node = node[p]
                vals.append(node[m]["pr_auc"])
            out[m] = float(np.nanmean(vals))
        out["ganho"] = out["defor"] - out["base"]
        return out

    res = {"tarefa": "driver_desmatamento_prodes",
           "media_geral": media(["geral"]),
           "media_nova_ignicao": media(["estrato", "nova_ignicao"]),
           "media_recorrente": media(["estrato", "recorrente"]),
           "folds": folds}
    REL_DIR.mkdir(parents=True, exist_ok=True)
    out = REL_DIR / "driver_desmatamento.json"
    out.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n== MÉDIAS (Δ PR-AUC = ganho do desmatamento sobre o histórico) ==")
    print(f"  GERAL:        base={res['media_geral']['base']:.3f} +defor={res['media_geral']['defor']:.3f} "
          f"=> Δ={res['media_geral']['ganho']:+.3f}")
    print(f"  NOVA IGNIÇÃO: base={res['media_nova_ignicao']['base']:.3f} +defor={res['media_nova_ignicao']['defor']:.3f} "
          f"=> Δ={res['media_nova_ignicao']['ganho']:+.3f}")
    print(f"  RECORRENTE:   base={res['media_recorrente']['base']:.3f} +defor={res['media_recorrente']['defor']:.3f} "
          f"=> Δ={res['media_recorrente']['ganho']:+.3f}")
    print(f"  salvo em: {out}")


if __name__ == "__main__":
    main()
