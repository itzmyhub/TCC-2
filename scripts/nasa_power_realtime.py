"""
Cliente NASA POWER para previsão em tempo real, alinhado ao enriquecimento de treino
(`enriquecer_dados_umidade.py` usa RH2M na mesma API).

Uma requisição traz PRECTOTCORR + RH2M + WS2M + T2M_MAX em janela de N dias; calcula-se:
- precipitação do dia final;
- Precipitacao_ma7 = média dos últimos 7 dias;
- DiaSemChuva = dias consecutivos com chuva < 0,1 mm até o dia final;
- DiaSemChuva_ma7 = média aritmética dos «dias consecutivos secos» por dia (proxy da série usada
  no treino com rolling(7) sobre DiaSemChuva, ver carregar_dados._adicionar_features_temporais_janela);
- Umidade: RH2M do dia final;
- Vento: média móvel 7d de WS2M (m/s) e valor do dia final quando disponível;
- Temperatura: média móvel 7d de T2M_MAX (°C) quando disponível.
"""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import requests

logger = logging.getLogger(__name__)

NASA_POWER_DAILY_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"

# Mesmo padrão do cálculo no climate_api a partir de forecast
DRY_MM_THRESHOLD = 0.1


def _load_api_keys() -> List[str]:
    try:
        from config import NASA_POWER_API_KEY, NASA_POWER_API_KEYS
    except ImportError:
        return []
    keys: List[str] = []
    if NASA_POWER_API_KEY and isinstance(NASA_POWER_API_KEY, str):
        keys.append(NASA_POWER_API_KEY)
    if NASA_POWER_API_KEYS:
        for k in NASA_POWER_API_KEYS:
            if k and k not in keys:
                keys.append(k)
    return list(dict.fromkeys(keys))


def _get_next_key(keys: List[str], idx: int) -> Optional[str]:
    if not keys:
        return None
    return keys[idx % len(keys)]


def _parse_daily_parameter(data_json: Dict, param: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    try:
        param_block = data_json["properties"]["parameter"].get(param) or {}
    except (KeyError, TypeError):
        return out
    if not isinstance(param_block, dict):
        return out
    for date_str, val in param_block.items():
        if val is None or val == -999.0 or val == -999:
            continue
        try:
            out[date_str] = float(val)
        except (TypeError, ValueError):
            continue
    return out


def _consecutive_dry_streak_ordered(
    ordered_dates: List[str],
    prectot_by_date: Dict[str, float],
) -> int:
    """A partir do último dia em `ordered_dates`, conta dias consecutivos com P < limiar (só datas presentes)."""
    if not ordered_dates:
        return 0
    streak = 0
    for ds in reversed(ordered_dates):
        p = prectot_by_date.get(ds)
        if p is None:
            break
        if p < DRY_MM_THRESHOLD:
            streak += 1
        else:
            break
    return streak


def _diasem_reconstructed_series(
    date_keys_sorted: List[str],
    prectot_by_date: Dict[str, float],
) -> List[float]:
    """
    Por cada dia na janela, nº aproximado de dias consecutivos secos até aquele dia
    (o mesmo critério do streak, ancorado nesse dia).
    """
    dsem: List[float] = []
    for i, dks in enumerate(date_keys_sorted):
        sub = date_keys_sorted[: i + 1]
        dsem.append(float(_consecutive_dry_streak_ordered(sub, prectot_by_date)))
    return dsem


def fetch_nasa_power_window(
    lat: float,
    lon: float,
    end_date: datetime,
    lookback_days: int = 28,
    ma_window: int = 7,
) -> Optional[Dict[str, Any]]:
    """
    Uma requisição GET: PRECTOTCORR + RH2M, janela longa (padrão 28 dias) para
    estimar sequência de seca; `prec_ma7` e `diasem_ma7` usam os últimos `ma_window` dias.

    Returns:
        None se falha HTTP / payload inválido; senão dicionário com
        precipitacao, umidade, prec_ma7, dias_sem_chuva, diasem_ma7, fonte, meta.
    """
    if lookback_days < 7:
        lookback_days = 7
    if ma_window < 1:
        ma_window = 7
    end_d = end_date.replace(hour=0, minute=0, second=0, microsecond=0)
    start_d = end_d - timedelta(days=lookback_days - 1)
    s_str = start_d.strftime("%Y%m%d")
    e_str = end_d.strftime("%Y%m%d")

    keys = _load_api_keys()
    last_error: Optional[str] = None
    for attempt in range(len(keys) or 1):
        api_key = _get_next_key(keys, attempt)
        params: Dict[str, Any] = {
            "parameters": "PRECTOTCORR,RH2M,WS2M,T2M_MAX",
            "community": "AG",
            "longitude": float(lon),
            "latitude": float(lat),
            "start": s_str,
            "end": e_str,
            "format": "JSON",
        }
        if api_key:
            params["key"] = api_key
        try:
            r = requests.get(NASA_POWER_DAILY_URL, params=params, timeout=45)
            if r.status_code == 429:
                logger.warning("NASA POWER 429, tentando outra chave/novamente...")
                time.sleep(2.0 * (attempt + 1))
                last_error = "rate_limit"
                continue
            r.raise_for_status()
            j = r.json()
        except requests.RequestException as e:
            last_error = str(e)
            logger.warning("Erro requisição NASA POWER: %s", e)
            if attempt < max(0, len(keys) - 1):
                continue
            return None
        except json.JSONDecodeError as e:
            last_error = str(e)
            logger.warning("JSON inválido NASA POWER: %s", e)
            return None

        prect = _parse_daily_parameter(j, "PRECTOTCORR")
        rh2m = _parse_daily_parameter(j, "RH2M")
        ws2m = _parse_daily_parameter(j, "WS2M")
        t2max = _parse_daily_parameter(j, "T2M_MAX")

        if not prect:
            logger.info(
                "NASA POWER sem PRECTOTCORR na janela %s–%s (lat=%.4f, lon=%.4f) — %s",
                s_str, e_str, lat, lon, last_error or "dados vazios",
            )
            return None

        # Garante ordenação por data
        all_dates = sorted(set(prect.keys()) | set(rh2m.keys()))
        if e_str not in prect and all_dates:
            e_str = all_dates[-1]  # última disponível na resposta
        all_dates = [d for d in all_dates if d in prect]
        if not all_dates:
            all_dates = sorted(prect.keys())
        if not all_dates:
            return None

        prec_list = [prect.get(d, 0.0) for d in all_dates]
        ws_list = [max(0.0, ws2m.get(d, 0.0) or 0.0) for d in all_dates] if ws2m else []
        t2_list = [t2max.get(d) for d in all_dates] if t2max else []
        dsem_per_day = _diasem_reconstructed_series(all_dates, {d: prect.get(d, 0.0) for d in all_dates})

        e_str = all_dates[-1]
        precipitacao = float(prect.get(e_str, prec_list[-1]))
        prec_last7 = prec_list[-ma_window:] if len(prec_list) >= ma_window else prec_list
        dsem_last7 = dsem_per_day[-ma_window:] if len(dsem_per_day) >= ma_window else dsem_per_day
        prec_ma7 = float(sum(prec_last7) / len(prec_last7)) if prec_last7 else precipitacao
        dia_ma7 = float(sum(dsem_last7) / len(dsem_last7)) if dsem_last7 else 0.0
        dias_sem_chuva = float(_consecutive_dry_streak_ordered(all_dates, prect))
        umid = rh2m.get(e_str)
        if umid is not None and (umid == -999.0 or umid < 0):
            umid = None

        ws_last7 = ws_list[-ma_window:] if len(ws_list) >= ma_window else ws_list
        ws2m_ma7 = float(sum(ws_last7) / len(ws_last7)) if ws_last7 else None
        ws2m_hoje = float(ws2m.get(e_str)) if ws2m and e_str in ws2m else None
        if ws2m_hoje is not None and ws2m_hoje == -999.0:
            ws2m_hoje = None

        t2_vals = [float(x) for x in (t2_list[-ma_window:] if len(t2_list) >= ma_window else t2_list) if x is not None and x != -999.0]
        t2m_max_ma7 = float(sum(t2_vals) / len(t2_vals)) if t2_vals else None

        out_ret: Dict[str, Any] = {
            "precipitacao": max(0.0, precipitacao),
            "umidade": float(umid) if umid is not None else None,
            "prec_ma7": max(0.0, prec_ma7),
            "diasem_ma7": max(0.0, dia_ma7),
            "dias_sem_chuva": max(0.0, dias_sem_chuva),
            "fonte": "nasa_power",
            "sucesso": True,
            "janela_inicio": all_dates[0],
            "janela_fim": e_str,
        }
        if ws2m_ma7 is not None:
            out_ret["ws2m_ma7_ms"] = max(0.0, ws2m_ma7)
        if ws2m_hoje is not None:
            out_ret["ws2m_hoje_ms"] = max(0.0, ws2m_hoje)
        if t2m_max_ma7 is not None:
            out_ret["t2m_max_ma7_c"] = t2m_max_ma7

        return out_ret

    logger.error("NASA POWER esgotou tentativas. Último: %s", last_error)
    return None
