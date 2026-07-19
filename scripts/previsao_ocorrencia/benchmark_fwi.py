"""Benchmark P3 — modelo de ML vs índices físicos de perigo de fogo.

Tese do TCC: um modelo de ML só "agrega valor" se PREVER FOGO OBSERVADO melhor
que o índice físico operacional. Aqui comparamos, na mesma tarefa (ocorrência de
foco em t+1) e validação temporal em bloco:

  • Modelo de ML (HistGradientBoosting, do pipeline P1/P2);
  • Índice de Risco de Fogo do INPE (operacional para o Brasil — a coluna
    RiscoFogo, agregada por célula-mês), usado como score;
  • Proxy de seca (Indice_Seca = DiaSemChuva/(Precip+ε));
  • [amostra] FWI canadense REAL [van Wagner & Pickett 1985; digiuseppe2024geff],
    computado dia-a-dia a partir de clima diário NASA POWER (T2M, RH2M, WS2M,
    PRECTOTCORR) e agregado a média mensal — para uma amostra de células
    (o FWI exige série diária contínua, indisponível nos CSVs de focos).

Parte A (offline, dataset completo) e Parte B (online, amostra com FWI real).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import fwi as fwi_mod  # mesmo diretório
import numpy as np
import pandas as pd
import requests
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "dataset_ocorrencia_mensal.csv"
REL_DIR = ROOT / "modelos" / "relatorios"
CACHE = ROOT / ".cache_umidade" / "nasa_daily"
NASA_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"

FEATURES = [
    "LatBin", "LonBin", "mes_sin", "mes_cos", "estacao_seca",
    "DiaSemChuva", "Precipitacao", "RiscoFogo_inpe",
    "focos_lag1", "focos_lag2", "focos_lag3",
    "fogo_lag1", "fogo_lag2", "fogo_lag3",
    "focos_roll3", "focos_roll6", "focos_roll12",
    "fogo_roll3", "fogo_roll6", "fogo_roll12",
    "FRP_lag1", "meses_desde_fogo",
]


def auc_ap(y, score):
    y = np.asarray(y); score = np.asarray(score, dtype=float)
    if len(np.unique(y)) < 2:
        return {"roc_auc": float("nan"), "pr_auc": float("nan"), "n": int(len(y)), "prev": float(y.mean())}
    return {"roc_auc": float(roc_auc_score(y, score)),
            "pr_auc": float(average_precision_score(y, score)),
            "n": int(len(y)), "prev": float(y.mean())}


# ---------- Parte A: índice INPE vs ML (dataset completo, blocos temporais) ----------
def parte_a(df: pd.DataFrame, anos_teste) -> dict:
    folds = []
    for ano in anos_teste:
        tr = df[df["ano"] < ano]
        te = df[df["ano"] == ano]
        if len(te) == 0 or tr["alvo"].nunique() < 2:
            continue
        m = HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.08, max_depth=8, l2_regularization=1.0,
            max_leaf_nodes=63, class_weight="balanced", random_state=42,
        ).fit(tr[FEATURES], tr["alvo"])
        p_ml = m.predict_proba(te[FEATURES])[:, 1]
        idx_seca = te["DiaSemChuva"] / (te["Precipitacao"] + 0.1)
        fold = {
            "ano_teste": int(ano),
            "ml": auc_ap(te["alvo"], p_ml),
            "indice_inpe_riscofogo": auc_ap(te["alvo"], te["RiscoFogo_inpe"]),
            "proxy_indice_seca": auc_ap(te["alvo"], idx_seca),
        }
        folds.append(fold)
        print(f"  [{ano}] ML PR-AUC={fold['ml']['pr_auc']:.3f} | "
              f"INPE-RiscoFogo PR-AUC={fold['indice_inpe_riscofogo']['pr_auc']:.3f} | "
              f"Indice_Seca PR-AUC={fold['proxy_indice_seca']['pr_auc']:.3f}")
    def med(k):
        return {mk: float(np.nanmean([f[k][mk] for f in folds])) for mk in ("roc_auc", "pr_auc")}
    return {"folds": folds, "media_ml": med("ml"),
            "media_indice_inpe": med("indice_inpe_riscofogo"),
            "media_proxy_seca": med("proxy_indice_seca")}


# ---------- Parte B: FWI real numa amostra de células ----------
def fetch_nasa_daily(lat, lon, start, end) -> pd.DataFrame:
    CACHE.mkdir(parents=True, exist_ok=True)
    fp = CACHE / f"nasa_{lat:.2f}_{lon:.2f}_{start}_{end}.json"
    if fp.exists():
        j = json.loads(fp.read_text())
    else:
        params = {"parameters": "T2M,RH2M,WS2M,PRECTOTCORR", "community": "AG",
                  "longitude": lon, "latitude": lat, "start": start, "end": end, "format": "JSON"}
        r = requests.get(NASA_URL, params=params, timeout=60)
        r.raise_for_status()
        j = r.json()
        fp.write_text(json.dumps(j))
    p = j["properties"]["parameter"]
    datas = sorted(p["T2M"].keys())
    rows = []
    for d in datas:
        t = p["T2M"][d]; rh = p["RH2M"][d]; ws = p["WS2M"][d]; pr = p["PRECTOTCORR"][d]
        if min(t, rh, ws, pr) <= -900:  # sentinela NASA
            continue
        rows.append({"data": pd.Timestamp(d), "temp": t,
                     "rh": float(np.clip(rh, 0, 100)),
                     "wind": max(0.0, ws) * 3.6,  # m/s -> km/h
                     "rain": max(0.0, pr)})
    return pd.DataFrame(rows)


def parte_b(df: pd.DataFrame, ano_teste: int, n_amostra: int, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    cells = df[["LatBin", "LonBin"]].drop_duplicates().reset_index(drop=True)
    sel = cells.iloc[rng.choice(len(cells), size=min(n_amostra, len(cells)), replace=False)]
    start = f"{ano_teste-1}0101"  # 1 ano de spin-up para FFMC/DMC/DC
    end = f"{ano_teste}1231"
    fwi_rows = []
    falhas = 0
    print(f"  buscando NASA POWER + FWI para {len(sel)} células ({start}–{end})...")
    for i, (_, c) in enumerate(sel.iterrows(), 1):
        try:
            daily = fetch_nasa_daily(c["LatBin"], c["LonBin"], start, end)
            if daily.empty:
                falhas += 1; continue
            serie = fwi_mod.fwi_serie(daily, lat=c["LatBin"])
            serie["ym"] = serie["data"].dt.to_period("M").astype(str)
            mensal = serie.groupby("ym")["fwi"].mean().rename("fwi_mensal").reset_index()
            mensal["LatBin"] = c["LatBin"]; mensal["LonBin"] = c["LonBin"]
            fwi_rows.append(mensal)
        except Exception as e:
            falhas += 1
            if falhas <= 3:
                print(f"    aviso: falha na célula {c['LatBin']},{c['LonBin']}: {e}")
        if i % 25 == 0:
            print(f"    {i}/{len(sel)} células processadas...")
    if not fwi_rows:
        return {"erro": "nenhum FWI computado (rede indisponível?)", "falhas": falhas}
    fwi_df = pd.concat(fwi_rows, ignore_index=True)

    # Treina ML em todos os anos < ano_teste, avalia no subconjunto amostrado do ano_teste
    tr = df[df["ano"] < ano_teste]
    m = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.08, max_depth=8, l2_regularization=1.0,
        max_leaf_nodes=63, class_weight="balanced", random_state=42,
    ).fit(tr[FEATURES], tr["alvo"])

    te = df[df["ano"] == ano_teste].merge(fwi_df, on=["LatBin", "LonBin", "ym"], how="inner")
    te = te.dropna(subset=["fwi_mensal"])
    if te["alvo"].nunique() < 2 or len(te) < 50:
        return {"erro": "subconjunto insuficiente após merge", "n": int(len(te)), "falhas": falhas}
    p_ml = m.predict_proba(te[FEATURES])[:, 1]
    res = {
        "ano_teste": ano_teste, "n_celulas_ok": len(fwi_df["LatBin"].astype(str).add(fwi_df["LonBin"].astype(str)).unique()),
        "n_linhas_avaliadas": int(len(te)), "falhas_fetch": falhas,
        "fwi_real": auc_ap(te["alvo"], te["fwi_mensal"]),
        "indice_inpe_riscofogo": auc_ap(te["alvo"], te["RiscoFogo_inpe"]),
        "ml": auc_ap(te["alvo"], p_ml),
        "fwi_medio_meses_com_fogo": float(te.loc[te["alvo"] == 1, "fwi_mensal"].mean()),
        "fwi_medio_meses_sem_fogo": float(te.loc[te["alvo"] == 0, "fwi_mensal"].mean()),
    }
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, default=str(DATASET))
    ap.add_argument("--anos_teste", type=int, nargs="+", default=[2021, 2022, 2023])
    ap.add_argument("--amostra_fwi", type=int, default=0,
                    help="N células para computar FWI real via NASA POWER (0 = pula a Parte B).")
    ap.add_argument("--ano_fwi", type=int, default=2023)
    args = ap.parse_args()

    df = pd.read_csv(args.dataset)
    print(f"Dataset: {len(df):,} linhas | prevalência {df['alvo'].mean():.3f}")

    print("\n== PARTE A: ML vs índice INPE (RiscoFogo) — dataset completo, blocos temporais ==")
    a = parte_a(df, args.anos_teste)
    print(f"  MÉDIA  ML PR-AUC={a['media_ml']['pr_auc']:.3f} ROC-AUC={a['media_ml']['roc_auc']:.3f} | "
          f"INPE-RiscoFogo PR-AUC={a['media_indice_inpe']['pr_auc']:.3f} ROC-AUC={a['media_indice_inpe']['roc_auc']:.3f}")

    resultado = {"tarefa": "benchmark_P3_ML_vs_indices_fisicos",
                 "parte_A_indice_inpe_vs_ml": a}

    if args.amostra_fwi > 0:
        print(f"\n== PARTE B: FWI canadense REAL (amostra de {args.amostra_fwi} células, ano {args.ano_fwi}) ==")
        b = parte_b(df, args.ano_fwi, args.amostra_fwi)
        resultado["parte_B_fwi_real_amostra"] = b
        if "erro" not in b:
            print(f"  n={b['n_linhas_avaliadas']} linhas | FWI-real PR-AUC={b['fwi_real']['pr_auc']:.3f} ROC-AUC={b['fwi_real']['roc_auc']:.3f}")
            print(f"  INPE-RiscoFogo PR-AUC={b['indice_inpe_riscofogo']['pr_auc']:.3f} | ML PR-AUC={b['ml']['pr_auc']:.3f}")
            print(f"  FWI médio: meses COM fogo={b['fwi_medio_meses_com_fogo']:.1f} vs SEM fogo={b['fwi_medio_meses_sem_fogo']:.1f}")
        else:
            print(f"  Parte B não concluída: {b['erro']}")

    REL_DIR.mkdir(parents=True, exist_ok=True)
    out = REL_DIR / "benchmark_p3_fwi.json"
    out.write_text(json.dumps(resultado, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nsalvo em: {out}")


if __name__ == "__main__":
    main()
