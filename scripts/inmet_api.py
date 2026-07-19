"""Integração com dados do INMET (estações automáticas).

Duas fontes complementares são usadas:

1. **WIS2 (OGC API Features)** — `http://wis2bra.inmet.gov.br/oapi`
   - Dados horários em tempo real, histórico de ~90 dias
   - **NÃO requer token** (acesso público anônimo)
   - 94+ estações sinópticas da Amazônia Legal publicam precipitação
   - Cobertura geográfica mais esparsa (só estações sinópticas WMO)

2. **ZIPs anuais** — `portal.inmet.gov.br/uploads/dadoshistoricos/{ANO}.zip`
   - Histórico completo desde 2000 (≈100 MB/ano)
   - 565+ estações automáticas, cobertura geográfica muito maior
   - ZIP do ano corrente é parcial (publicado mensalmente)

A API horária tradicional `apitempo.inmet.gov.br/estacao/...` foi descartada
porque retorna `204 No Content` com frequência (verificado em 12/05/2026).

Fluxo de seleção (``get_inmet_fusion_data``, ``hierarchical=True`` por padrão):
1. Tenta raio preferencial (``INMET_MAX_DISTANCE_KM``); se falhar, tenta
   ``INMET_EXTENDED_MAX_DISTANCE_KM``, preenchendo ``inmet_representatividade``
   e ``inmet_busca_raio_km`` para transparência na UI/TCC.
2. Em datas recentes (≤90 d), WIS2 sinótica no raio: precipitação + vento médio 7d
   opcional na mesma estação.
3. Senão, ZIP anual da automática mais próxima no mesmo raio.
4. Se nada retornar dados, o chamador usa MERRA-2.

Encoding dos CSVs do ZIP: `latin-1`. Separador `;`. Decimal `,`.
Cabeçalho de 8 linhas + linha de header das colunas.
"""
from __future__ import annotations

import io
import json
import logging
import math
import zipfile
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

from config import (
    INMET_CACHE_DIR,
    INMET_EXTENDED_MAX_DISTANCE_KM,
    INMET_MAX_DISTANCE_KM,
)

logger = logging.getLogger(__name__)

INMET_CATALOG_URL = "https://apitempo.inmet.gov.br/estacoes/T"
INMET_HISTORICAL_ZIP_URL = "https://portal.inmet.gov.br/uploads/dadoshistoricos/{year}.zip"

# WIS2 API (em tempo real, sem token)
WIS2_BASE = "http://wis2bra.inmet.gov.br/oapi"
WIS2_SYNOP_URL = f"{WIS2_BASE}/collections/urn:wmo:md:br-inmet:synop/items"
WIS2_STATIONS_URL = f"{WIS2_BASE}/collections/stations/items"
WIS2_PRECIP_NAME = "total_precipitation_or_total_water_equivalent"
WIS2_WIND_NAME = "wind_speed"
WIS2_HISTORY_DAYS = 90  # cobertura típica do WIS2
WIS2_CATALOG_TTL_DAYS = 7  # estações sinópticas mudam pouco

WIS2_STATIONS_CACHE_PATH = INMET_CACHE_DIR / "wis2_stations_precip.json"

CATALOG_CACHE_PATH = INMET_CACHE_DIR / "catalogo_estacoes.json"
CATALOG_TTL_DAYS = 30  # recarrega o catálogo a cada 30 dias

# Coluna do CSV horário com precipitação horária em mm
PRECIP_COL_KEY = "PRECIPITA"  # casa com "PRECIPITAÇÃO TOTAL, HORÁRIO (mm)"
RH_COL_KEY = "UMIDADE RELATIVA DO AR, HORARIA"
TEMP_COL_KEY = "TEMPERATURA DO AR - BULBO SECO, HORARIA"

ESTADOS_AMAZONIA_LEGAL = {"AC", "AM", "AP", "MA", "MT", "PA", "RO", "RR", "TO"}


# --------------------------------------------------------------------------- #
# Catálogo de estações
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class INMETStation:
    codigo: str
    nome: str
    uf: str
    latitude: float
    longitude: float
    altitude: float
    situacao: str
    inicio_operacao: Optional[str]

    def distance_km(self, lat: float, lon: float) -> float:
        return _haversine_km(self.latitude, self.longitude, lat, lon)


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Distância em km entre duas coordenadas (Terra como esfera de R=6371 km)."""
    r = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlamb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlamb / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def _ensure_cache_dir() -> None:
    INMET_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _catalog_cache_is_fresh() -> bool:
    if not CATALOG_CACHE_PATH.exists():
        return False
    try:
        age_days = (datetime.now().timestamp() - CATALOG_CACHE_PATH.stat().st_mtime) / 86400.0
        return age_days <= CATALOG_TTL_DAYS
    except OSError:
        return False


def get_station_catalog(force_refresh: bool = False) -> List[INMETStation]:
    """Retorna o catálogo de estações automáticas do INMET.

    Faz cache local em ``.cache_inmet/catalogo_estacoes.json`` por 30 dias.
    Em caso de falha de rede, usa a versão cacheada (mesmo que vencida).
    """
    _ensure_cache_dir()
    raw: Optional[List[Dict]] = None

    if not force_refresh and _catalog_cache_is_fresh():
        try:
            with CATALOG_CACHE_PATH.open("r", encoding="utf-8") as f:
                raw = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Cache do catálogo INMET corrompido (%s) — refazendo download.", exc)
            raw = None

    if raw is None:
        try:
            logger.info("Baixando catálogo de estações INMET...")
            r = requests.get(INMET_CATALOG_URL, timeout=30)
            r.raise_for_status()
            raw = r.json()
            with CATALOG_CACHE_PATH.open("w", encoding="utf-8") as f:
                json.dump(raw, f, ensure_ascii=False)
            logger.info("Catálogo INMET cacheado: %d estações.", len(raw))
        except Exception as exc:
            logger.warning("Falha no download do catálogo INMET: %s", exc)
            if CATALOG_CACHE_PATH.exists():
                logger.info("Usando catálogo INMET cacheado (mesmo vencido).")
                try:
                    with CATALOG_CACHE_PATH.open("r", encoding="utf-8") as f:
                        raw = json.load(f)
                except (OSError, json.JSONDecodeError):
                    raw = []
            else:
                raw = []

    stations: List[INMETStation] = []
    for e in raw or []:
        try:
            lat = float(e.get("VL_LATITUDE") or 0.0)
            lon = float(e.get("VL_LONGITUDE") or 0.0)
            alt = float(e.get("VL_ALTITUDE") or 0.0)
        except (TypeError, ValueError):
            continue
        if lat == 0.0 and lon == 0.0:
            continue
        stations.append(INMETStation(
            codigo=str(e.get("CD_ESTACAO") or "").strip(),
            nome=str(e.get("DC_NOME") or "").strip(),
            uf=str(e.get("SG_ESTADO") or "").strip().upper(),
            latitude=lat,
            longitude=lon,
            altitude=alt,
            situacao=str(e.get("CD_SITUACAO") or "").strip(),
            inicio_operacao=e.get("DT_INICIO_OPERACAO"),
        ))
    return stations


def find_nearest_station(
    lat: float,
    lon: float,
    *,
    max_km: float = INMET_MAX_DISTANCE_KM,
    only_operante: bool = True,
    only_amazonia_legal: bool = False,
) -> Optional[Tuple[INMETStation, float]]:
    """Retorna (estação, distância_km) mais próxima dentro de `max_km`.

    Se nenhuma estação satisfaz os critérios, retorna None.
    """
    catalogo = get_station_catalog()
    if not catalogo:
        return None

    best: Optional[Tuple[INMETStation, float]] = None
    for est in catalogo:
        if only_operante and est.situacao.lower() != "operante":
            continue
        if only_amazonia_legal and est.uf not in ESTADOS_AMAZONIA_LEGAL:
            continue
        d = est.distance_km(lat, lon)
        if d > max_km:
            continue
        if best is None or d < best[1]:
            best = (est, d)
    return best


# --------------------------------------------------------------------------- #
# ZIP histórico + leitura do CSV por estação
# --------------------------------------------------------------------------- #
def _zip_path(year: int) -> Path:
    return INMET_CACHE_DIR / f"{year}.zip"


def _station_csv_cache_path(year: int, codigo: str) -> Path:
    return INMET_CACHE_DIR / str(year) / f"{codigo}.csv"


def _ensure_zip_downloaded(year: int) -> Optional[Path]:
    """Baixa (se necessário) o ZIP anual do INMET. Retorna o caminho local ou None."""
    path = _zip_path(year)
    if path.exists() and path.stat().st_size > 1024 * 1024:  # >1 MB
        return path
    _ensure_cache_dir()
    url = INMET_HISTORICAL_ZIP_URL.format(year=year)
    logger.info("Baixando ZIP histórico INMET %s (pode demorar ~30s, ~100 MB)...", year)
    try:
        r = requests.get(url, timeout=600, stream=True)
        if r.status_code != 200:
            logger.warning("ZIP INMET %s indisponível (status=%d).", year, r.status_code)
            return None
        tmp = path.with_suffix(".zip.part")
        with tmp.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
        tmp.replace(path)
        logger.info("ZIP INMET %s pronto: %.1f MB", year, path.stat().st_size / 1024 / 1024)
        return path
    except Exception as exc:
        logger.warning("Erro ao baixar ZIP INMET %s: %s", year, exc)
        return None


def _extract_station_csv(year: int, codigo: str) -> Optional[Path]:
    """Extrai do ZIP anual o CSV da estação `codigo` para um arquivo local."""
    out = _station_csv_cache_path(year, codigo)
    if out.exists() and out.stat().st_size > 1024:
        return out

    zip_path = _ensure_zip_downloaded(year)
    if not zip_path:
        return None

    try:
        with zipfile.ZipFile(zip_path) as zf:
            target = next((n for n in zf.namelist() if f"_{codigo}_" in n), None)
            if not target:
                logger.info("Estação %s não está no ZIP %s.", codigo, year)
                return None
            out.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(target) as src, out.open("wb") as dst:
                dst.write(src.read())
        return out
    except (zipfile.BadZipFile, OSError) as exc:
        logger.warning("Erro ao extrair %s do ZIP %s: %s", codigo, year, exc)
        return None


def _parse_decimal(value: str) -> Optional[float]:
    """Converte '0,5' -> 0.5; '' -> None; valores não-numéricos -> None."""
    s = (value or "").strip().replace(",", ".")
    if not s or s in {"-", "null", "None"}:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _read_station_csv(csv_path: Path) -> Dict[datetime, Dict[str, Optional[float]]]:
    """Lê o CSV horário do INMET e retorna {datetime_utc: {precipitacao, umidade, temperatura}}."""
    out: Dict[datetime, Dict[str, Optional[float]]] = {}
    try:
        with csv_path.open("r", encoding="latin-1") as f:
            lines = f.readlines()
    except OSError as exc:
        logger.warning("Não foi possível ler %s: %s", csv_path, exc)
        return out

    if len(lines) < 10:
        return out

    # Linha 8 (índice 8) é o cabeçalho de colunas
    header_line = lines[8].rstrip("\n").rstrip("\r")
    cols = header_line.split(";")
    idx_data = 0
    idx_hora = 1
    idx_precip = next((i for i, c in enumerate(cols) if PRECIP_COL_KEY in c.upper()), None)
    idx_rh = next((i for i, c in enumerate(cols) if RH_COL_KEY in c.upper()), None)
    idx_temp = next((i for i, c in enumerate(cols) if TEMP_COL_KEY in c.upper()), None)

    if idx_precip is None:
        logger.warning("Coluna de precipitação não encontrada em %s", csv_path.name)
        return out

    for line in lines[9:]:
        parts = line.rstrip("\n").rstrip("\r").split(";")
        if len(parts) <= idx_precip:
            continue
        data_str = parts[idx_data].strip()
        hora_str = parts[idx_hora].strip().replace(" UTC", "")
        if not data_str or not hora_str:
            continue
        # Formatos: '2024/01/01' '0300 UTC' ou '01/01/2024' etc.
        dt = _parse_inmet_dt(data_str, hora_str)
        if dt is None:
            continue
        precip = _parse_decimal(parts[idx_precip])
        umidade = _parse_decimal(parts[idx_rh]) if idx_rh is not None else None
        temp = _parse_decimal(parts[idx_temp]) if idx_temp is not None else None
        out[dt] = {"precipitacao_mm": precip, "umidade_pct": umidade, "temp_c": temp}
    return out


def _parse_inmet_dt(data_str: str, hora_str: str) -> Optional[datetime]:
    """Aceita formatos 'YYYY/MM/DD' ou 'DD/MM/YYYY' + 'HHMM' (sem ':')."""
    h = hora_str.replace(":", "").strip()
    if len(h) != 4 or not h.isdigit():
        return None
    hour = int(h[:2])
    minute = int(h[2:])
    # Tenta YYYY/MM/DD
    try:
        d = datetime.strptime(data_str, "%Y/%m/%d")
        return d.replace(hour=hour, minute=minute)
    except ValueError:
        pass
    try:
        d = datetime.strptime(data_str, "%d/%m/%Y")
        return d.replace(hour=hour, minute=minute)
    except ValueError:
        return None


def get_inmet_window_precipitation(
    station_code: str,
    end_date: datetime,
    lookback_days: int = 7,
) -> Optional[Dict[str, float]]:
    """Lê precipitação horária do INMET para a estação na janela
    [end_date - lookback_days, end_date] e agrega em diários e médias móveis.

    Retorna ``None`` se a estação/CSV não estiver disponível para o período.
    Caso contrário, retorna dict com:
    - ``precipitacao_dia_mm``: precipitação no dia final (end_date)
    - ``prec_ma7``: média de precipitação diária na janela
    - ``dias_sem_chuva``: nº de dias consecutivos sem chuva terminando em end_date
    - ``diasem_ma7``: dias sem chuva média na janela
    - ``cobertura_horas``: fração de horas com dado válido na janela
    """
    start_date = end_date - timedelta(days=lookback_days + 1)
    anos = sorted({start_date.year, end_date.year})
    horarios: Dict[datetime, Dict[str, Optional[float]]] = {}
    for ano in anos:
        csv_path = _extract_station_csv(ano, station_code)
        if not csv_path:
            continue
        horarios.update(_read_station_csv(csv_path))

    if not horarios:
        return None

    # Filtra janela
    filtrados = {
        ts: vals for ts, vals in horarios.items()
        if start_date <= ts <= end_date
    }
    if not filtrados:
        return None

    # Agrega em diários (soma de precipitação por dia)
    diarios: Dict[Tuple[int, int, int], float] = {}
    horas_validas: Dict[Tuple[int, int, int], int] = {}
    for ts, vals in filtrados.items():
        key = (ts.year, ts.month, ts.day)
        diarios.setdefault(key, 0.0)
        horas_validas.setdefault(key, 0)
        p = vals.get("precipitacao_mm")
        if p is not None:
            diarios[key] += float(p)
            horas_validas[key] += 1

    # Ordena por data
    dias_ordenados = sorted(diarios.keys())
    if not dias_ordenados:
        return None

    precip_dia_mm: List[Tuple[Tuple[int, int, int], float, int]] = [
        (d, diarios[d], horas_validas[d]) for d in dias_ordenados
    ]

    # Mantém apenas dias com cobertura razoável (>= 12 horas)
    dias_validos = [(d, p) for d, p, h in precip_dia_mm if h >= 12]
    if not dias_validos:
        return None

    # Últimos 7 dias
    ultimos7 = dias_validos[-min(7, len(dias_validos)):]
    prec_ma7 = sum(p for _, p in ultimos7) / len(ultimos7)
    dias_sem_chuva_ma7 = sum(1 for _, p in ultimos7 if p < 0.5) / len(ultimos7) * 7.0

    # Dias secos consecutivos terminando no fim
    dias_sem_chuva = 0
    for _, p in reversed(dias_validos):
        if p < 0.5:
            dias_sem_chuva += 1
        else:
            break

    precip_dia_final = dias_validos[-1][1]
    cobertura = sum(h for _, _, h in precip_dia_mm) / (len(precip_dia_mm) * 24.0)

    return {
        "precipitacao_dia_mm": float(precip_dia_final),
        "prec_ma7": float(prec_ma7),
        "dias_sem_chuva": float(dias_sem_chuva),
        "diasem_ma7": float(dias_sem_chuva_ma7),
        "cobertura_horas_pct": round(cobertura * 100.0, 1),
        "n_dias_validos": len(dias_validos),
    }


# --------------------------------------------------------------------------- #
# WIS2 (tempo real, sem token, histórico de ~90 dias)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class WIS2Station:
    """Estação sinóptica WIS2 que publica precipitação."""
    wigos_id: str
    nome: str
    latitude: float
    longitude: float

    def distance_km(self, lat: float, lon: float) -> float:
        return _haversine_km(self.latitude, self.longitude, lat, lon)


def _wis2_catalog_cache_is_fresh() -> bool:
    if not WIS2_STATIONS_CACHE_PATH.exists():
        return False
    try:
        age_days = (datetime.now().timestamp() - WIS2_STATIONS_CACHE_PATH.stat().st_mtime) / 86400.0
        return age_days <= WIS2_CATALOG_TTL_DAYS
    except OSError:
        return False


def get_wis2_precip_stations(force_refresh: bool = False) -> List[WIS2Station]:
    """Retorna estações WIS2 que publicaram precipitação nos últimos 7 dias.

    O catálogo é descoberto por amostragem (pega 1000 features de precipitação
    e extrai estações únicas + cruza com a coleção `stations`). É cacheado em
    `.cache_inmet/wis2_stations_precip.json` por 7 dias.

    Retorna estações de qualquer região do Brasil; o `find_nearest_*` filtra
    por proximidade depois.
    """
    _ensure_cache_dir()

    if not force_refresh and _wis2_catalog_cache_is_fresh():
        try:
            with WIS2_STATIONS_CACHE_PATH.open("r", encoding="utf-8") as f:
                data = json.load(f)
            return [WIS2Station(**s) for s in data]
        except (OSError, json.JSONDecodeError, TypeError):
            pass

    fim = datetime.now(timezone.utc)
    ini = fim - timedelta(days=7)
    params_obs = {
        "name": WIS2_PRECIP_NAME,
        "datetime": f"{ini.strftime('%Y-%m-%dT%H:%M:%SZ')}/{fim.strftime('%Y-%m-%dT%H:%M:%SZ')}",
        "limit": 1000,
        "f": "json",
    }
    try:
        r = requests.get(WIS2_SYNOP_URL, params=params_obs, timeout=60,
                         headers={"User-Agent": "TCC-2"})
        if r.status_code != 200:
            logger.warning("WIS2 SYNOP indisponível (status=%d).", r.status_code)
            return []
        feats = r.json().get("features", []) or []
    except Exception as exc:  # noqa: BLE001
        logger.warning("Falha ao consultar WIS2 SYNOP: %s", exc)
        return []

    wids: Dict[str, Tuple[float, float]] = {}
    for f in feats:
        props = f.get("properties", {}) or {}
        wid = props.get("wigos_station_identifier")
        coords = (f.get("geometry") or {}).get("coordinates") or []
        if wid and len(coords) >= 2:
            wids[wid] = (float(coords[1]), float(coords[0]))  # (lat, lon)

    if not wids:
        return []

    # Cruzar com `stations` para pegar nomes (bbox grande para varrer Brasil)
    names: Dict[str, str] = {}
    try:
        r2 = requests.get(WIS2_STATIONS_URL,
                          params={"bbox": "-75,-35,-30,5", "limit": 2000, "f": "json"},
                          timeout=60, headers={"User-Agent": "TCC-2"})
        if r2.status_code == 200:
            for s in r2.json().get("features", []) or []:
                p = s.get("properties", {}) or {}
                wid = p.get("wigos_station_identifier")
                if wid in wids:
                    names[wid] = p.get("name", "?")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Falha ao mapear nomes de estações WIS2: %s", exc)

    stations: List[WIS2Station] = []
    for wid, (lat, lon) in wids.items():
        stations.append(WIS2Station(
            wigos_id=wid,
            nome=names.get(wid, "?"),
            latitude=lat,
            longitude=lon,
        ))

    try:
        with WIS2_STATIONS_CACHE_PATH.open("w", encoding="utf-8") as f:
            json.dump([s.__dict__ for s in stations], f, ensure_ascii=False)
    except OSError as exc:
        logger.warning("Falha ao cachear catálogo WIS2: %s", exc)

    logger.info("WIS2: %d estações com precipitação nos últimos 7 dias.", len(stations))
    return stations


def find_nearest_wis2_station(
    lat: float,
    lon: float,
    *,
    max_km: float = INMET_MAX_DISTANCE_KM,
) -> Optional[Tuple[WIS2Station, float]]:
    """Retorna (estação WIS2 com chuva, distância_km) mais próxima."""
    cat = get_wis2_precip_stations()
    if not cat:
        return None
    best: Optional[Tuple[WIS2Station, float]] = None
    for est in cat:
        d = est.distance_km(lat, lon)
        if d > max_km:
            continue
        if best is None or d < best[1]:
            best = (est, d)
    return best


def get_wis2_window_precipitation(
    wigos_id: str,
    end_date: datetime,
    lookback_days: int = 7,
    station_lat: Optional[float] = None,
    station_lon: Optional[float] = None,
) -> Optional[Dict[str, float]]:
    """Consulta WIS2 para a janela [end_date - lookback_days, end_date]
    e retorna agregados diários e médias móveis 7d.

    Importante: a WIS2 do INMET **não respeita** o filtro
    ``wigos_station_identifier`` na querystring (verificado em 12/05/2026).
    Para isolar a estação corretamente fazemos:
    1. bbox restrito (~0,05° em torno da estação) para reduzir o volume;
    2. filtragem client-side por `wigos_station_identifier`;
    3. paginação implícita via `offset` (a API tem 58k+ obs no período).

    Retorna None se a estação não tiver dados na janela.
    """
    start_date = end_date - timedelta(days=lookback_days + 1)
    if end_date.tzinfo is None:
        end_date = end_date.replace(tzinfo=timezone.utc)
        start_date = start_date.replace(tzinfo=timezone.utc)

    base_params = {
        "name": WIS2_PRECIP_NAME,
        "datetime": (
            f"{start_date.strftime('%Y-%m-%dT%H:%M:%SZ')}/"
            f"{end_date.strftime('%Y-%m-%dT%H:%M:%SZ')}"
        ),
        "limit": 1000,
        "f": "json",
    }

    if station_lat is not None and station_lon is not None:
        # bbox de ~0,05° (~5 km) em torno da estação para reduzir o volume
        delta = 0.05
        base_params["bbox"] = (
            f"{station_lon - delta},{station_lat - delta},"
            f"{station_lon + delta},{station_lat + delta}"
        )

    feats: List[Dict] = []
    try:
        r = requests.get(WIS2_SYNOP_URL, params=base_params, timeout=30,
                         headers={"User-Agent": "TCC-2"})
        if r.status_code != 200:
            return None
        feats = r.json().get("features", []) or []
    except Exception as exc:  # noqa: BLE001
        logger.warning("Falha em WIS2 (estação=%s): %s", wigos_id, exc)
        return None

    if not feats:
        return None

    # Filtrar client-side pelo wigos_id (a API ignora o filtro na querystring)
    feats_estacao = [
        f for f in feats
        if (f.get("properties", {}) or {}).get("wigos_station_identifier") == wigos_id
    ]
    if not feats_estacao:
        return None

    # Deduplica por timestamp da observação horária (mesmo ts → mantém último valor)
    # phenomenonTime no formato "2026-05-07T19:00:00Z/2026-05-07T20:00:00Z"
    por_hora: Dict[datetime, float] = {}
    for f in feats_estacao:
        p = f.get("properties", {}) or {}
        ts_str = p.get("phenomenonTime") or p.get("reportTime")
        val = p.get("value")
        if not ts_str or val is None:
            continue
        if "/" in ts_str:
            ts_str = ts_str.split("/")[1]  # fim da janela horária
        try:
            ts = datetime.strptime(ts_str.replace("Z", ""), "%Y-%m-%dT%H:%M:%S")
            por_hora[ts] = float(val)
        except (ValueError, TypeError):
            continue

    if not por_hora:
        return None

    # Agrupar por dia somando os horários únicos
    diarios: Dict[Tuple[int, int, int], float] = {}
    horas_por_dia: Dict[Tuple[int, int, int], int] = {}
    for ts, val in por_hora.items():
        key = (ts.year, ts.month, ts.day)
        diarios[key] = diarios.get(key, 0.0) + val
        horas_por_dia[key] = horas_por_dia.get(key, 0) + 1

    dias = sorted(diarios.keys())
    # Cobertura mínima 6h/dia (SYNOP sinóptico publica menos que automática)
    dias_validos = [(d, diarios[d]) for d in dias if horas_por_dia[d] >= 6]
    if not dias_validos:
        return None

    ultimos7 = dias_validos[-min(7, len(dias_validos)):]
    prec_ma7 = sum(p for _, p in ultimos7) / len(ultimos7)
    dias_sem_chuva_ma7 = sum(1 for _, p in ultimos7 if p < 0.5) / len(ultimos7) * 7.0

    dias_sem_chuva = 0
    for _, p in reversed(dias_validos):
        if p < 0.5:
            dias_sem_chuva += 1
        else:
            break

    precip_dia_final = dias_validos[-1][1]
    horas_total = sum(horas_por_dia[d] for d, _ in dias_validos)
    cobertura = min(1.0, horas_total / (len(dias_validos) * 24.0))

    return {
        "precipitacao_dia_mm": float(precip_dia_final),
        "prec_ma7": float(prec_ma7),
        "dias_sem_chuva": float(dias_sem_chuva),
        "diasem_ma7": float(dias_sem_chuva_ma7),
        "cobertura_horas_pct": round(cobertura * 100.0, 1),
        "n_dias_validos": len(dias_validos),
    }


def get_wis2_window_wind_ma7(
    wigos_id: str,
    end_date: datetime,
    lookback_days: int = 7,
    station_lat: Optional[float] = None,
    station_lon: Optional[float] = None,
) -> Optional[float]:
    """Média móvel 7d da velocidade do vento (m/s) a partir de SYNOP WIS2.

    Usa o mesmo padrão de bbox + filtro client-side que a precipitação.
    Retorna None se não houver dados suficientes.
    """
    start_date = end_date - timedelta(days=lookback_days + 1)
    if end_date.tzinfo is None:
        end_date = end_date.replace(tzinfo=timezone.utc)
        start_date = start_date.replace(tzinfo=timezone.utc)

    base_params: Dict[str, Any] = {
        "name": WIS2_WIND_NAME,
        "datetime": (
            f"{start_date.strftime('%Y-%m-%dT%H:%M:%SZ')}/"
            f"{end_date.strftime('%Y-%m-%dT%H:%M:%SZ')}"
        ),
        "limit": 1000,
        "f": "json",
    }
    if station_lat is not None and station_lon is not None:
        delta = 0.05
        base_params["bbox"] = (
            f"{station_lon - delta},{station_lat - delta},"
            f"{station_lon + delta},{station_lat + delta}"
        )

    try:
        r = requests.get(WIS2_SYNOP_URL, params=base_params, timeout=30,
                         headers={"User-Agent": "TCC-2"})
        if r.status_code != 200:
            return None
        feats = r.json().get("features", []) or []
    except Exception as exc:  # noqa: BLE001
        logger.warning("WIS2 vento (estação=%s): %s", wigos_id, exc)
        return None

    feats_estacao = [
        f for f in feats
        if (f.get("properties", {}) or {}).get("wigos_station_identifier") == wigos_id
    ]
    if not feats_estacao:
        return None

    por_hora: Dict[datetime, float] = {}
    for f in feats_estacao:
        p = f.get("properties", {}) or {}
        ts_str = p.get("phenomenonTime") or p.get("reportTime")
        val = p.get("value")
        if not ts_str or val is None:
            continue
        if "/" in ts_str:
            ts_str = ts_str.split("/")[1]
        try:
            ts = datetime.strptime(ts_str.replace("Z", ""), "%Y-%m-%dT%H:%M:%S")
            por_hora[ts] = float(val)
        except (ValueError, TypeError):
            continue

    if not por_hora:
        return None

    # Média horária por dia, depois média dos últimos 7 dias com ≥6h
    medias_dia: Dict[Tuple[int, int, int], List[float]] = {}
    for ts, spd in por_hora.items():
        key = (ts.year, ts.month, ts.day)
        medias_dia.setdefault(key, []).append(spd)

    diarios: List[Tuple[Tuple[int, int, int], float, int]] = []
    for key, vals in sorted(medias_dia.items()):
        if len(vals) >= 6:
            diarios.append((key, sum(vals) / len(vals), len(vals)))

    if not diarios:
        return None

    ultimos7 = diarios[-min(7, len(diarios)):]
    return float(sum(d for _, d, _ in ultimos7) / len(ultimos7))


def _inmet_representatividade(dist_km: float, raio_busca_km: float) -> str:
    """rótulo qualitativo para o TCC / UI (não altera o modelo)."""
    if dist_km <= INMET_MAX_DISTANCE_KM:
        return "alta"
    if dist_km <= 100.0:
        return "moderada"
    if dist_km <= raio_busca_km:
        return "baixa"
    return "baixa"


def _try_inmet_fusion_at_radius(
    lat: float,
    lon: float,
    ref_naive: datetime,
    lookback_days: int,
    max_km: float,
) -> Optional[Dict[str, Any]]:
    """Uma tentativa: WIS2 (≤90d) depois ZIP automática, mesmo raio máximo."""
    # --- WIS2 ---
    age_days = (datetime.now() - ref_naive).total_seconds() / 86400.0
    if -1 < age_days < WIS2_HISTORY_DAYS:
        wis2 = find_nearest_wis2_station(lat, lon, max_km=max_km)
        if wis2:
            est_w, dist_w = wis2
            dados_w = get_wis2_window_precipitation(
                est_w.wigos_id, ref_naive, lookback_days,
                station_lat=est_w.latitude, station_lon=est_w.longitude,
            )
            if dados_w:
                out: Dict[str, Any] = {
                    "precipitacao": dados_w["precipitacao_dia_mm"],
                    "prec_ma7": dados_w["prec_ma7"],
                    "dias_sem_chuva": dados_w["dias_sem_chuva"],
                    "diasem_ma7": dados_w["diasem_ma7"],
                    "fonte_inmet": "inmet_wis2_sinoptica",
                    "estacao_inmet": {
                        "codigo": est_w.wigos_id,
                        "nome": est_w.nome,
                        "uf": None,
                        "latitude": est_w.latitude,
                        "longitude": est_w.longitude,
                    },
                    "distancia_estacao_inmet_km": round(dist_w, 2),
                    "cobertura_horas_pct": dados_w["cobertura_horas_pct"],
                    "n_dias_validos": dados_w["n_dias_validos"],
                }
                ws = get_wis2_window_wind_ma7(
                    est_w.wigos_id, ref_naive, lookback_days,
                    station_lat=est_w.latitude, station_lon=est_w.longitude,
                )
                if ws is not None:
                    out["vento_inmet_ms_ma7"] = round(ws, 3)
                return out
            logger.info(
                "WIS2: estação %s (%.1f km) sem dados na janela %s — tentando ZIP.",
                est_w.wigos_id, dist_w, ref_naive.date(),
            )

    # --- ZIP ---
    res = find_nearest_station(lat, lon, max_km=max_km, only_operante=True)
    if not res:
        return None
    estacao, distancia_km = res
    dados = get_inmet_window_precipitation(estacao.codigo, ref_naive, lookback_days)
    if not dados:
        logger.info(
            "INMET: estação %s (%s, %.1f km) sem dados para janela %s — caindo no MERRA-2.",
            estacao.codigo, estacao.nome, distancia_km, ref_naive.date(),
        )
        return None

    return {
        "precipitacao": dados["precipitacao_dia_mm"],
        "prec_ma7": dados["prec_ma7"],
        "dias_sem_chuva": dados["dias_sem_chuva"],
        "diasem_ma7": dados["diasem_ma7"],
        "fonte_inmet": "inmet_zip_automatica",
        "estacao_inmet": {
            "codigo": estacao.codigo,
            "nome": estacao.nome,
            "uf": estacao.uf,
            "latitude": estacao.latitude,
            "longitude": estacao.longitude,
        },
        "distancia_estacao_inmet_km": round(distancia_km, 2),
        "cobertura_horas_pct": dados["cobertura_horas_pct"],
        "n_dias_validos": dados["n_dias_validos"],
    }


# --------------------------------------------------------------------------- #
# Função de fusão (alto nível)
# --------------------------------------------------------------------------- #
def get_inmet_fusion_data(
    lat: float,
    lon: float,
    reference_date: Optional[datetime] = None,
    lookback_days: int = 7,
    max_km: float = INMET_MAX_DISTANCE_KM,
    hierarchical: bool = True,
) -> Optional[Dict]:
    """Precipitação INMET fundível com MERRA-2 (WIS2 e/ou ZIP).

    Com ``hierarchical=True`` (padrão): tenta primeiro raio ``max_km`` (tipicamente
    50~km); se falhar, tenta ``INMET_EXTENDED_MAX_DISTANCE_KM`` (ex.: 180~km),
    marcando ``inmet_representatividade`` e ``inmet_busca_raio_km`` para a UI
    e para o TCC (transparência sobre extrapolação espacial).

    Chaves extras opcionais: ``vento_inmet_ms_ma7`` (só fonte WIS2 sinótica).
    """
    if reference_date is None:
        reference_date = datetime.now()
    ref_naive = reference_date.replace(tzinfo=None) if reference_date.tzinfo else reference_date

    if not hierarchical:
        r = _try_inmet_fusion_at_radius(lat, lon, ref_naive, lookback_days, max_km)
        if r:
            r["inmet_busca_raio_km"] = float(max_km)
            r["inmet_representatividade"] = _inmet_representatividade(
                float(r["distancia_estacao_inmet_km"]), float(max_km),
            )
        return r

    r1 = _try_inmet_fusion_at_radius(lat, lon, ref_naive, lookback_days, max_km)
    if r1:
        r1["inmet_busca_raio_km"] = float(max_km)
        r1["inmet_representatividade"] = _inmet_representatividade(
            float(r1["distancia_estacao_inmet_km"]), float(max_km),
        )
        return r1

    r2 = _try_inmet_fusion_at_radius(
        lat, lon, ref_naive, lookback_days, INMET_EXTENDED_MAX_DISTANCE_KM,
    )
    if r2:
        r2["inmet_busca_raio_km"] = float(INMET_EXTENDED_MAX_DISTANCE_KM)
        r2["inmet_representatividade"] = _inmet_representatividade(
            float(r2["distancia_estacao_inmet_km"]),
            float(INMET_EXTENDED_MAX_DISTANCE_KM),
        )
        logger.info(
            "Fusão INMET hierárquica: raio estendido %.0f km → estação %s a %.1f km (%s)",
            INMET_EXTENDED_MAX_DISTANCE_KM,
            r2.get("estacao_inmet", {}).get("codigo"),
            r2["distancia_estacao_inmet_km"],
            r2["inmet_representatividade"],
        )
        return r2
    return None
