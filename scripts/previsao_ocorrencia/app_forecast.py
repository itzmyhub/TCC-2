"""App de PREVISÃO DE OCORRÊNCIA — serve o pipeline reformulado (P1/P2/P3/P6).

Independente do app original (`app_map_interativo.py`, que classifica o índice do
INPE), este serve o produto novo: **P(fogo observado no próximo mês)** por célula
0,25°, com rótulo conformal (fogo/não-fogo/incerto) e drivers (histórico de fogo +
desmatamento DETER). Consome os artefatos de `gerar_forecast_ocorrencia.py`.

Rotas:
  GET  /                 → mapa interativo do forecast (mapa_forecast_ocorrencia.html)
  POST /api/forecast     → {lat, lon} → previsão da célula correspondente
  GET  /api/forecast_top → top-N células de maior risco previsto
  GET  /api/status       → mês previsto e data do modelo

Para atualizar com os dados mais recentes: ``python atualizar_forecast.py``
(o app recarrega o forecast sozinho, sem reiniciar).
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, request

ROOT = Path(__file__).resolve().parents[2]
CELL = 0.25

app = Flask(__name__)
_FC_PATH = ROOT / "dataset_forecast_celulas.csv"
_MODELO = joblib.load(ROOT / "modelos" / "modelo_ocorrencia.pkl")
_CONF = json.loads((ROOT / "modelos" / "ocorrencia_conformal.json").read_text(encoding="utf-8"))
_HTML = ROOT / "mapa_forecast_ocorrencia.html"
_cache = {"mtime": None, "fc": None}


def forecast() -> pd.DataFrame:
    """Forecast atual; recarrega quando atualizar_forecast.py regrava o CSV."""
    mtime = _FC_PATH.stat().st_mtime
    if mtime != _cache["mtime"]:
        _cache["fc"], _cache["mtime"] = pd.read_csv(_FC_PATH), mtime
    return _cache["fc"]


def consultar(lat: float, lon: float) -> dict:
    """Snap para a célula 0,25° e retorna a previsão pré-computada."""
    latb = round(round(lat / CELL) * CELL, 3)
    lonb = round(round(lon / CELL) * CELL, 3)
    fc = forecast()
    m = fc[(fc.LatBin == latb) & (fc.LonBin == lonb)]
    if m.empty:
        return {"sucesso": False, "erro": "célula fora do domínio fire-prone monitorado",
                "celula": [latb, lonb]}
    r = m.iloc[0]
    return {
        "sucesso": True, "celula": [latb, lonb], "mes_referencia": r["ym"],
        "mes_previsto": r.get("mes_previsto"),
        "p_fogo_proximo_mes": round(float(r["p_fogo"]), 4),
        "rotulo_conformal": r["rotulo_conformal"],
        "drivers": {
            "focos_ultimos_12m": int(r["focos_roll12"]),
            "meses_desde_ultimo_fogo": int(r["meses_desde_fogo"]),
            "desmatamento_deter_6m_km2": round(float(r["deter_roll6"]), 3),
            "estacao_seca": bool(r["estacao_seca"]),
        },
        "cobertura_conformal": _CONF.get("cobertura_calib", _CONF.get("cobertura_calib_2023")),
    }


@app.route("/")
def index():
    if _HTML.exists():
        return _HTML.read_text(encoding="utf-8")
    return "Mapa não gerado. Rode gerar_forecast_ocorrencia.py.", 404


@app.route("/api/forecast", methods=["POST"])
def api_forecast():
    j = request.get_json(force=True, silent=True) or {}
    try:
        lat, lon = float(j["lat"]), float(j["lon"])
    except (KeyError, TypeError, ValueError):
        return jsonify({"sucesso": False, "erro": "envie {lat, lon} numéricos"}), 400
    return jsonify(consultar(lat, lon))


@app.route("/api/forecast_top")
def api_top():
    n = int(request.args.get("n", 20))
    top = forecast().nlargest(n, "p_fogo")[["LatBin", "LonBin", "p_fogo", "rotulo_conformal"]]
    return jsonify(top.to_dict(orient="records"))


@app.route("/api/status")
def api_status():
    fc = forecast()
    return jsonify({"mes_previsto": fc["mes_previsto"].max() if "mes_previsto" in fc else None,
                    "mes_features": fc["ym"].max(), "celulas": len(fc),
                    "modelo_treinado_ate": _MODELO.get("treinado_ate")})


if __name__ == "__main__":
    print(f"Forecast: {len(forecast()):,} células | modelo: {len(_MODELO['features'])} features")
    app.run(debug=True, host="0.0.0.0", port=5001)
