"""Auditoria operacional: cruza predições logadas com detecções FIRMS posteriores.

Lê `logs/auditoria_predicoes.jsonl`, para cada predição com idade ≥ `--min-days`
consulta a NASA FIRMS na vizinhança da coordenada e na janela de
[ref_data, ref_data + horizon_days], calcula o rótulo operacional
("ocorreu fogo / não ocorreu fogo") e mede F1 / precision / recall em
janelas móveis (default: predições dos últimos 7, 14, 30 dias).

Saídas:
- `modelos/relatorios/auditoria_operacional.json` (relatório agregado)
- `modelos/relatorios/auditoria_operacional.csv` (uma linha por predição
  auditada, para inspeção)

Uso:
    python scripts/auditar_predicoes.py --horizon-days 3 --radius-km 5

Cuidado de rate limit: a NASA FIRMS aceita ~5000 transações por 10 min na
chave atual. O script tem cache local e dorme entre chamadas se necessário.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("auditoria")

# Permite executar como script (`python scripts/auditar_predicoes.py`)
_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from audit_log import AUDIT_LOG_PATH, iter_audit_entries  # noqa: E402
from frp_api import FRPDataProvider  # noqa: E402

ROOT = _SCRIPTS_DIR.parent
RELATORIOS_DIR = ROOT / "modelos" / "relatorios"
RELATORIOS_DIR.mkdir(parents=True, exist_ok=True)
RELATORIO_JSON = RELATORIOS_DIR / "auditoria_operacional.json"
RELATORIO_CSV = RELATORIOS_DIR / "auditoria_operacional.csv"

# Cache do cruzamento FIRMS para não consultar a API duas vezes para a mesma
# tupla (lat_arred, lon_arred, data_iso, horizon).
CACHE_PATH = ROOT / ".cache_umidade" / "auditoria_firms_cache.json"


# ---- Mapeamento risco predito x rótulo operacional --------------------------
# A "verdade operacional" é binária: houve foco confirmado pelo FIRMS na
# janela seguinte (positivo) ou não (negativo). O risco predito é convertido
# para binário por: {Baixo} -> negativo; {Moderado, Muito Alto} -> positivo.
# Justificativa: a saída do sistema é interpretada operacionalmente como
# "alerta sim/não"; F1 binário é a métrica mais comparável com a literatura
# de detecção de risco.
POSITIVE_CLASSES = {"Moderado", "Muito Alto"}
DEFAULT_HORIZON_DAYS = 3   # janela após ref_data em que esperamos ver foco
DEFAULT_RADIUS_KM = 5.0    # raio para considerar foco "no mesmo local"
DEFAULT_MIN_AGE_DAYS = 3   # só auditar predições antigas o suficiente para que
                           # haja dados FIRMS NRT no horizonte
DEFAULT_FIRMS_DAYS = 5     # tamanho da janela FIRMS por chamada (limite NRT)


def _load_cache() -> Dict[str, Dict[str, Any]]:
    if not CACHE_PATH.exists():
        return {}
    try:
        with CACHE_PATH.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def _save_cache(cache: Dict[str, Dict[str, Any]]) -> None:
    try:
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with CACHE_PATH.open("w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False)
    except OSError as exc:
        logger.warning("Falha ao salvar cache de auditoria: %s", exc)


def _cache_key(lat: float, lon: float, end_date: datetime, horizon: int, radius: float) -> str:
    return (
        f"{lat:.3f}_{lon:.3f}_{end_date.strftime('%Y-%m-%d')}"
        f"_h{horizon}_r{radius:.1f}"
    )


def _query_firms_after_prediction(
    frp: FRPDataProvider,
    *,
    lat: float,
    lon: float,
    ref_data: datetime,
    horizon_days: int,
    radius_km: float,
    firms_days: int,
    cache: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Consulta FIRMS na janela [ref_data, ref_data + horizon_days]."""
    end_date = ref_data + timedelta(days=horizon_days)
    if end_date > datetime.now():
        end_date = datetime.now()
    days_back = min(horizon_days, firms_days)

    key = _cache_key(lat, lon, end_date, horizon_days, radius_km)
    if key in cache:
        return cache[key]

    try:
        result = frp.get_frp_from_nasa_firms(
            lat=lat,
            lon=lon,
            radius_km=radius_km,
            days_back=days_back,
            reference_date=end_date,
        )
    except Exception as exc:
        logger.warning("Erro FIRMS para (%.3f, %.3f) %s: %s", lat, lon, end_date.date(), exc)
        result = None

    detections = int((result or {}).get("detections", 0) or 0)
    frp_max = float((result or {}).get("frp_max", 0.0) or 0.0)
    frp_mean = float((result or {}).get("frp_mean", 0.0) or 0.0)
    entry = {
        "detections": detections,
        "frp_max": frp_max,
        "frp_mean": frp_mean,
        "end_date": end_date.strftime("%Y-%m-%d"),
        "days_back": days_back,
        "queried_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    cache[key] = entry
    return entry


def _classificar_binario(risco_predito: str) -> int:
    """1 = alerta (Moderado/Muito Alto); 0 = sem alerta (Baixo)."""
    return 1 if str(risco_predito) in POSITIVE_CLASSES else 0


def _truth_binario(firms_detections: int, frp_max: float) -> int:
    """1 = houve foco confirmado; 0 = não houve."""
    return 1 if (firms_detections > 0 or frp_max > 0) else 0


# A NASA FIRMS NRT guarda apenas ~60 dias de histórico. Para predições
# auditadas com ref_data anterior a esse limite, o "y_true=0" não é confiável
# (a ausência de detecção pode ser apenas ausência de dado no NRT).
FIRMS_NRT_HISTORY_DAYS = 60


def _metricas_binarias(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    """Calcula precision/recall/F1 + matriz de confusão."""
    tp = sum(1 for r in rows if r["y_true"] == 1 and r["y_pred"] == 1)
    fp = sum(1 for r in rows if r["y_true"] == 0 and r["y_pred"] == 1)
    fn = sum(1 for r in rows if r["y_true"] == 1 and r["y_pred"] == 0)
    tn = sum(1 for r in rows if r["y_true"] == 0 and r["y_pred"] == 0)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / len(rows) if rows else 0.0
    fnr = fn / (tp + fn) if (tp + fn) > 0 else 0.0  # custo de perder fogo real
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    return {
        "n": len(rows),
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "accuracy": round(accuracy, 4),
        "fnr": round(fnr, 4),
        "fpr": round(fpr, 4),
    }


def auditar(
    horizon_days: int,
    radius_km: float,
    min_age_days: int,
    firms_days: int,
    rolling_windows: Tuple[int, ...] = (7, 14, 30),
    pause_between_calls_s: float = 0.0,
) -> Dict[str, Any]:
    if not AUDIT_LOG_PATH.exists():
        logger.warning("Audit log não existe ainda: %s", AUDIT_LOG_PATH)
        return {"erro": "audit_log_vazio", "n_predicoes": 0}

    frp = FRPDataProvider()
    cache = _load_cache()

    rows: List[Dict[str, Any]] = []
    now = datetime.now()
    skipped_recentes = 0
    skipped_invalid = 0

    for entry in iter_audit_entries():
        try:
            ref_data = datetime.fromisoformat(entry["ref_data"])
        except (KeyError, ValueError):
            skipped_invalid += 1
            continue

        age_days = (now - ref_data).total_seconds() / 86400.0
        if age_days < min_age_days:
            skipped_recentes += 1
            continue

        firms = _query_firms_after_prediction(
            frp,
            lat=float(entry["lat"]),
            lon=float(entry["lon"]),
            ref_data=ref_data,
            horizon_days=horizon_days,
            radius_km=radius_km,
            firms_days=firms_days,
            cache=cache,
        )
        if pause_between_calls_s > 0:
            time.sleep(pause_between_calls_s)

        y_pred = _classificar_binario(entry.get("risco_predito", "Baixo"))
        y_true = _truth_binario(firms["detections"], firms["frp_max"])
        # Quando a predição é antiga demais para o NRT (>60d), o "y_true=0"
        # é não confiável — pode ser ausência de dado, não ausência de fogo.
        truth_confiavel = age_days <= FIRMS_NRT_HISTORY_DAYS

        rows.append({
            "prediction_id": entry.get("prediction_id"),
            "ref_data": entry["ref_data"],
            "lat": entry["lat"],
            "lon": entry["lon"],
            "estado": entry.get("estado"),
            "municipio": entry.get("municipio"),
            "modelo": entry.get("modelo"),
            "risco_predito": entry.get("risco_predito"),
            "confianca": entry.get("confianca"),
            "y_pred": y_pred,
            "y_true": y_true,
            "truth_confiavel": int(truth_confiavel),
            "firms_detections": firms["detections"],
            "firms_frp_max": firms["frp_max"],
            "horizon_days": horizon_days,
            "radius_km": radius_km,
            "age_days": round(age_days, 2),
        })

    _save_cache(cache)
    logger.info(
        "Auditadas %d predições (%d recentes ignoradas, %d inválidas)",
        len(rows), skipped_recentes, skipped_invalid,
    )

    if not rows:
        return {"n_predicoes": 0, "skipped_recentes": skipped_recentes}

    # Apenas linhas com truth confiável (≤ 60d) entram nas métricas oficiais.
    rows_confiaveis = [r for r in rows if r.get("truth_confiavel", 1)]

    # Métricas globais (apenas com truth confiável)
    metricas_global = _metricas_binarias(rows_confiaveis)

    # Métricas por janela rolling (todas dentro de 60d, então sempre confiáveis)
    metricas_rolling: Dict[str, Any] = {}
    for w in rolling_windows:
        cutoff = now - timedelta(days=w)
        subset = [r for r in rows_confiaveis if datetime.fromisoformat(r["ref_data"]) >= cutoff]
        metricas_rolling[f"ultimos_{w}_dias"] = _metricas_binarias(subset)

    # Métricas por classe predita (frequência de cada label)
    por_classe: Dict[str, Dict[str, int]] = defaultdict(lambda: {"n": 0, "y_true_pos": 0})
    for r in rows_confiaveis:
        cls = r["risco_predito"]
        por_classe[cls]["n"] += 1
        por_classe[cls]["y_true_pos"] += int(r["y_true"])

    # Métricas por estado (deriva regional)
    por_estado: Dict[str, Dict[str, float]] = {}
    estados_rows: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows_confiaveis:
        estados_rows[r.get("estado") or "DESCONHECIDO"].append(r)
    for est, srows in estados_rows.items():
        if len(srows) >= 5:
            por_estado[est] = _metricas_binarias(srows)

    relatorio = {
        "gerado_em_utc": datetime.now(timezone.utc).isoformat(),
        "parametros": {
            "horizon_days": horizon_days,
            "radius_km": radius_km,
            "min_age_days": min_age_days,
            "firms_days_per_call": firms_days,
            "rolling_windows_dias": list(rolling_windows),
            "firms_nrt_history_days": FIRMS_NRT_HISTORY_DAYS,
        },
        "n_predicoes_total_log": sum(1 for _ in iter_audit_entries()),
        "n_predicoes_auditadas": len(rows),
        "n_predicoes_confiaveis_para_metricas": len(rows_confiaveis),
        "n_predicoes_antigas_demais_para_NRT": len(rows) - len(rows_confiaveis),
        "n_predicoes_recentes_ignoradas": skipped_recentes,
        "metricas_global": metricas_global,
        "metricas_rolling": metricas_rolling,
        "metricas_por_estado": por_estado,
        "distribuicao_por_classe_predita": dict(por_classe),
        "nota": (
            "Métricas globais e rolling consideram apenas predições com "
            f"idade <= {FIRMS_NRT_HISTORY_DAYS} dias (limite NRT do VIIRS_SNPP). "
            "Predições mais antigas aparecem no CSV mas não nas métricas."
        ),
    }

    return relatorio, rows


def _write_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    import csv
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--horizon-days", type=int, default=DEFAULT_HORIZON_DAYS,
                        help="Janela após a predição em que se busca fogo confirmado (default: %(default)s).")
    parser.add_argument("--radius-km", type=float, default=DEFAULT_RADIUS_KM,
                        help="Raio em km para considerar foco no mesmo local (default: %(default)s).")
    parser.add_argument("--min-age-days", type=int, default=DEFAULT_MIN_AGE_DAYS,
                        help="Idade mínima da predição para auditar (default: %(default)s).")
    parser.add_argument("--firms-days", type=int, default=DEFAULT_FIRMS_DAYS,
                        help="Dias por chamada à NASA FIRMS NRT (1..5).")
    parser.add_argument("--pause", type=float, default=0.0,
                        help="Pausa entre chamadas FIRMS em segundos (rate limit).")
    parser.add_argument("--rolling", type=str, default="7,14,30",
                        help="Janelas rolling (dias) separadas por vírgula.")
    args = parser.parse_args()

    rolling = tuple(int(x) for x in args.rolling.split(",") if x.strip())

    resultado = auditar(
        horizon_days=args.horizon_days,
        radius_km=args.radius_km,
        min_age_days=args.min_age_days,
        firms_days=args.firms_days,
        rolling_windows=rolling,
        pause_between_calls_s=args.pause,
    )

    if isinstance(resultado, tuple):
        relatorio, rows = resultado
    else:
        relatorio, rows = resultado, []

    with RELATORIO_JSON.open("w", encoding="utf-8") as f:
        json.dump(relatorio, f, ensure_ascii=False, indent=2)
    logger.info("Relatório JSON: %s", RELATORIO_JSON)

    if rows:
        _write_csv(rows, RELATORIO_CSV)
        logger.info("Relatório CSV: %s", RELATORIO_CSV)

    g = relatorio.get("metricas_global", {})
    if g:
        logger.info(
            "RESUMO: n=%d  F1=%.3f  P=%.3f  R=%.3f  FNR=%.3f  FPR=%.3f  Acc=%.3f",
            g.get("n", 0), g.get("f1", 0), g.get("precision", 0),
            g.get("recall", 0), g.get("fnr", 0), g.get("fpr", 0), g.get("accuracy", 0),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
