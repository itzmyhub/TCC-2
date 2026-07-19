"""Idade e FRONTEIRA de desmatamento — as dimensões que o DETER mensal apontou.

O DETER mostrou que a *resolução temporal* do desmatamento não amplia o ganho; a
hipótese é que outras DIMENSÕES o façam, computáveis dos dados já coletados:

  • IDADE: meses desde o último desmatamento na célula (recém-desmatado → mais
    propenso a queima de limpeza).
  • ACÚMULO: desmatamento acumulado na célula até t (estoque de área aberta).
  • FRONTEIRA (vizinhança): desmatamento nas 8 células vizinhas nos últimos 12
    meses — captura "célula na fronteira ativa" mesmo sem desmatamento próprio.

Compara incrementalmente, no estrato de NOVA IGNIÇÃO:
  base  →  base+DETER(lags simples)  →  base+DETER+IDADE/ACÚMULO/FRONTEIRA.
Tudo causal (≤ t). Fontes: deter_celula_mes.csv (DETER) — já coletado.
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
CELL = 0.25

FEATURES_BASE = [
    "LatBin", "LonBin", "mes_sin", "mes_cos", "estacao_seca",
    "DiaSemChuva", "Precipitacao", "RiscoFogo_inpe",
    "focos_lag1", "focos_lag2", "focos_lag3",
    "fogo_lag1", "fogo_lag2", "fogo_lag3",
    "focos_roll3", "focos_roll6", "focos_roll12",
    "fogo_roll3", "fogo_roll6", "fogo_roll12",
    "FRP_lag1", "meses_desde_fogo",
]
DETER_SIMPLES = ["deter_m0", "deter_m1", "deter_roll3", "deter_roll6"]
FRONTIER = ["meses_desde_desmat", "defor_acum", "defor_vizinhanca12"]


def mk():
    return HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.08, max_depth=8, l2_regularization=1.0,
        max_leaf_nodes=63, class_weight="balanced", random_state=42)


def auc_ap(y, s):
    y = np.asarray(y); s = np.asarray(s, dtype=float)
    if len(np.unique(y)) < 2:
        return {"pr_auc": float("nan"), "roc_auc": float("nan"), "n": int(len(y))}
    return {"pr_auc": float(average_precision_score(y, s)),
            "roc_auc": float(roc_auc_score(y, s)), "n": int(len(y))}


def _ym_idx(ym):
    p = pd.PeriodIndex(pd.Index(ym).astype(str), freq="M")
    return (p.year * 12 + (p.month - 1)).astype(int)


def construir(df, d):
    df = df.copy()
    df["ym_idx"] = _ym_idx(df["ym"])
    d = d.copy()
    d["ym_idx"] = _ym_idx(d["ym"])

    # --- DETER lags simples + rolling do próprio (deter_roll12 p/ vizinhança) ---
    for k in range(12):
        tmp = d[["LatBin", "LonBin", "ym_idx", "deter_km"]].copy()
        tmp["ym_idx"] = tmp["ym_idx"] + k
        tmp = tmp.rename(columns={"deter_km": f"_dl{k}"})
        df = df.merge(tmp, on=["LatBin", "LonBin", "ym_idx"], how="left")
        df[f"_dl{k}"] = df[f"_dl{k}"].fillna(0.0)
    df["deter_m0"] = df["_dl0"]
    df["deter_m1"] = df["_dl1"]
    df["deter_roll3"] = df[[f"_dl{k}" for k in range(3)]].sum(axis=1)
    df["deter_roll6"] = df[[f"_dl{k}" for k in range(6)]].sum(axis=1)
    df["deter_roll12_self"] = df[[f"_dl{k}" for k in range(12)]].sum(axis=1)

    # --- IDADE: meses desde o último mês com DETER>0 na célula (≤ t) ---
    # constrói, por célula, o último ym_idx com desmatamento <= cada t
    dd = d[d["deter_km"] > 0][["LatBin", "LonBin", "ym_idx"]].drop_duplicates()
    last_by_cell = {}
    for lat, lon, yi in dd.itertuples(index=False):
        last_by_cell.setdefault((lat, lon), []).append(yi)
    for k in last_by_cell:
        last_by_cell[k].sort()
    def meses_desde(row):
        arr = last_by_cell.get((row["LatBin"], row["LonBin"]))
        if not arr:
            return 99
        t = row["ym_idx"]
        # último desmatamento estritamente <= t
        import bisect
        i = bisect.bisect_right(arr, t) - 1
        return min(t - arr[i], 99) if i >= 0 else 99
    df["meses_desde_desmat"] = df.apply(meses_desde, axis=1)

    # --- ACÚMULO: DETER acumulado na célula até t (exclusivo) ---
    cum = {}
    acc_cell = {}
    # ordena por (cell, ym_idx) e acumula
    dsum = d.groupby(["LatBin", "LonBin", "ym_idx"])["deter_km"].sum().reset_index()
    dsum = dsum.sort_values(["LatBin", "LonBin", "ym_idx"])
    dsum["acum"] = dsum.groupby(["LatBin", "LonBin"])["deter_km"].cumsum() - dsum["deter_km"]
    df = df.merge(dsum[["LatBin", "LonBin", "ym_idx", "acum"]].rename(columns={"acum": "defor_acum"}),
                  on=["LatBin", "LonBin", "ym_idx"], how="left")
    # propaga o acumulado para meses sem registro (forward-fill por célula)
    df = df.sort_values(["LatBin", "LonBin", "ym_idx"])
    df["defor_acum"] = df.groupby(["LatBin", "LonBin"])["defor_acum"].ffill().fillna(0.0)

    # --- FRONTEIRA: soma do deter_roll12 das 8 células vizinhas (mesmo t) ---
    roll12 = {(r.LatBin, r.LonBin, r.ym_idx): r.deter_roll12_self
              for r in df[["LatBin", "LonBin", "ym_idx", "deter_roll12_self"]].itertuples(index=False)}
    viz = []
    offs = [(-CELL, -CELL), (-CELL, 0), (-CELL, CELL), (0, -CELL), (0, CELL),
            (CELL, -CELL), (CELL, 0), (CELL, CELL)]
    for lat, lon, yi in df[["LatBin", "LonBin", "ym_idx"]].itertuples(index=False):
        s = 0.0
        for dla, dlo in offs:
            s += roll12.get((round(lat + dla, 3), round(lon + dlo, 3), yi), 0.0)
        viz.append(s)
    df["defor_vizinhanca12"] = viz
    return df


def avaliar(df, feats):
    out = []
    for ano in (2021, 2022, 2023):
        tr, te = df[df["ano"] < ano], df[df["ano"] == ano]
        if len(te) == 0 or tr["alvo"].nunique() < 2:
            continue
        m = mk().fit(tr[feats], tr["alvo"])
        p = m.predict_proba(te[feats])[:, 1]
        te = te.assign(p=p)
        novo = te[te["focos_roll12"] == 0]
        out.append({"geral": auc_ap(te["alvo"], te["p"]),
                    "nova_ignicao": auc_ap(novo["alvo"], novo["p"])})
    g = float(np.nanmean([f["geral"]["pr_auc"] for f in out]))
    ni = float(np.nanmean([f["nova_ignicao"]["pr_auc"] for f in out]))
    return g, ni


def main():
    df = pd.read_csv(DATASET)
    d = pd.read_csv(DETER)
    print("Construindo features de idade/fronteira (pode levar ~1 min)...")
    df = construir(df, d)

    confs = {
        "base": FEATURES_BASE,
        "base+DETER_simples": FEATURES_BASE + DETER_SIMPLES,
        "base+DETER+frontier": FEATURES_BASE + DETER_SIMPLES + FRONTIER,
        "base+frontier_so": FEATURES_BASE + FRONTIER,
    }
    res = {}
    print(f"{'config':24s} {'PR-AUC geral':>13s} {'PR-AUC nova ign.':>17s}")
    for nome, feats in confs.items():
        g, ni = avaliar(df, feats)
        res[nome] = {"pr_auc_geral": g, "pr_auc_nova_ignicao": ni}
        print(f"{nome:24s} {g:13.3f} {ni:17.3f}")

    base_ni = res["base"]["pr_auc_nova_ignicao"]
    print(f"\nGanhos na NOVA IGNIÇÃO sobre base ({base_ni:.3f}):")
    for nome in confs:
        if nome != "base":
            print(f"  {nome:24s} Δ={res[nome]['pr_auc_nova_ignicao']-base_ni:+.3f}")
    REL_DIR.mkdir(parents=True, exist_ok=True)
    (REL_DIR / "driver_frontier.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nsalvo em: {REL_DIR / 'driver_frontier.json'}")


if __name__ == "__main__":
    main()
