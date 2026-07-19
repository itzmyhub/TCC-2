"""Integração do pipeline de ocorrência ao produto (modelo serializado + mapa).

Treina o modelo final de previsão de ocorrência (features-base + desmatamento
DETER — o melhor conjunto), serializa-o, calibra conjuntos conformais e gera:
  • ``modelos/modelo_ocorrencia.pkl``        — modelo treinado (joblib);
  • ``modelos/ocorrencia_conformal.json``    — limiares LAC + cobertura;
  • ``dataset_forecast_celulas.csv``         — previsão por célula (último mês):
       P(fogo no próximo mês), rótulo conformal (fogo/não-fogo/incerto) e drivers;
  • ``mapa_forecast_ocorrencia.html``        — mapa interativo (folium) do forecast.

É a versão "produto" do pipeline reformulado: prevê **fogo observado no próximo mês**
por célula, com incerteza calibrada — diferente do app original (classe do índice).
"""
from __future__ import annotations

import json
from pathlib import Path

import folium
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "dataset_ocorrencia_mensal.csv"
DETER = ROOT / "scripts" / "previsao_ocorrencia" / "deter_celula_mes.csv"
MODELOS = ROOT / "modelos"

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


def thresholds_lac(p, y, alpha=0.1):
    q = {}
    for k in (0, 1):
        q[k] = float(np.quantile(p[y == k, k], alpha, method="lower"))
    return q


def rotulo_conformal(p0, p1, q):
    inc0, inc1 = p0 >= q[0], p1 >= q[1]
    if inc1 and not inc0:
        return "fogo"
    if inc0 and not inc1:
        return "nao-fogo"
    return "incerto"


def cor(p):
    # verde (baixo) -> amarelo -> vermelho (alto)
    if p < 0.1:
        return "#1a9850"
    if p < 0.25:
        return "#91cf60"
    if p < 0.5:
        return "#fee08b"
    if p < 0.75:
        return "#fc8d59"
    return "#d73027"


def main():
    print("Carregando dados e montando features (base + DETER)...")
    df = pd.read_csv(DATASET)
    d = pd.read_csv(DETER)
    df = montar_deter(df, d)

    # Modelo final: treina em tudo <= 2023; calibra conformal em 2023.
    tr = df[df["ano"] < 2023]
    cal = df[df["ano"] == 2023]
    print(f"Treino: {len(tr):,} | calibração (2023): {len(cal):,}")
    modelo = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.08, max_depth=8, l2_regularization=1.0,
        max_leaf_nodes=63, class_weight="balanced", random_state=42).fit(tr[FEATURES], tr["alvo"])
    pcal = modelo.predict_proba(cal[FEATURES])
    q = thresholds_lac(pcal, cal["alvo"].to_numpy())
    cobertura = float((pcal[np.arange(len(cal)), cal["alvo"].to_numpy()] >=
                       np.array([q[int(yy)] for yy in cal["alvo"]])).mean())

    # Modelo de produção: re-treina em TODOS os dados (≤2023) p/ uso operacional.
    modelo_prod = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.08, max_depth=8, l2_regularization=1.0,
        max_leaf_nodes=63, class_weight="balanced", random_state=42).fit(df[FEATURES], df["alvo"])

    MODELOS.mkdir(exist_ok=True)
    joblib.dump({"modelo": modelo_prod, "features": FEATURES}, MODELOS / "modelo_ocorrencia.pkl")
    (MODELOS / "ocorrencia_conformal.json").write_text(
        json.dumps({"metodo": "LAC (Sadinle 2019)", "alpha": 0.1, "limiares_q": q,
                    "cobertura_calib_2023": cobertura}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Modelo serializado. Cobertura conformal (calib 2023): {cobertura:.3f}")

    # Forecast por célula: último mês disponível de cada célula -> P(fogo próximo mês)
    ult = df.sort_values("ym_idx").groupby(["LatBin", "LonBin"]).tail(1).copy()
    proba = modelo_prod.predict_proba(ult[FEATURES])
    ult["p_fogo"] = proba[:, 1]
    ult["rotulo_conformal"] = [rotulo_conformal(proba[i, 0], proba[i, 1], q) for i in range(len(ult))]
    saida_cols = ["LatBin", "LonBin", "ym", "p_fogo", "rotulo_conformal",
                  "focos_roll12", "fogo_roll6", "meses_desde_fogo", "estacao_seca",
                  "deter_roll6", "deter_roll3"]
    fc = ult[saida_cols].sort_values("p_fogo", ascending=False).reset_index(drop=True)
    fc.to_csv(ROOT / "dataset_forecast_celulas.csv", index=False)
    print(f"Forecast por célula: {len(fc):,} células | P(fogo) médio={fc['p_fogo'].mean():.3f} "
          f"| conformal: {fc['rotulo_conformal'].value_counts().to_dict()}")

    # Mapa folium
    print("Gerando mapa interativo...")
    m = folium.Map(location=[-6.0, -55.0], zoom_start=5, tiles="CartoDB positron")
    for r in fc.itertuples(index=False):
        lat, lon, p = r.LatBin, r.LonBin, r.p_fogo
        popup = (f"<b>P(fogo próximo mês): {p:.0%}</b><br>conformal: {r.rotulo_conformal}<br>"
                 f"focos últimos 12m: {int(r.focos_roll12)}<br>"
                 f"meses desde fogo: {int(r.meses_desde_fogo)}<br>"
                 f"desmat. DETER 6m: {r.deter_roll6:.2f} km²<br>"
                 f"estação seca: {'sim' if r.estacao_seca else 'não'}")
        folium.Rectangle(
            bounds=[[lat - 0.125, lon - 0.125], [lat + 0.125, lon + 0.125]],
            color=None, weight=0, fill=True, fill_color=cor(p), fill_opacity=0.55,
            popup=folium.Popup(popup, max_width=260),
        ).add_to(m)
    legenda = ("<div style='position:fixed;bottom:30px;left:30px;z-index:9999;background:white;"
               "padding:10px;border:1px solid #999;font-size:12px'>"
               "<b>P(fogo no próximo mês)</b><br>"
               "<span style='color:#1a9850'>■</span> &lt;10%&nbsp;"
               "<span style='color:#91cf60'>■</span> 10–25%&nbsp;"
               "<span style='color:#fee08b'>■</span> 25–50%<br>"
               "<span style='color:#fc8d59'>■</span> 50–75%&nbsp;"
               "<span style='color:#d73027'>■</span> &gt;75%</div>")
    m.get_root().html.add_child(folium.Element(legenda))
    out_html = ROOT / "mapa_forecast_ocorrencia.html"
    m.save(str(out_html))
    print(f"Mapa salvo em: {out_html}")


if __name__ == "__main__":
    main()
