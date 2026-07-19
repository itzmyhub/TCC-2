"""
feature_lookup.py — Pré-cálculo de features Tier 1 para predição pontual.

Motivação
---------
As features avançadas (`features_avancadas.py`) dependem de séries temporais
por célula espacial (rolling em DateTimeIndex, soma acumulada, SPI). No app
do mapa o usuário consulta **um único ponto**, então não temos histórico
local para calcular essas features em tempo real.

Solução adotada
---------------
1. Lê uma vez o dataset enriquecido (`base_de_dados_enriquecido.csv`).
2. Pré-computa medianas (representam o "comportamento típico") em três
   níveis de granularidade decrescente:
       a) `(LatBin, LonBin, Mes)`   — mais granular (~0.25°, ≈27 km).
       b) `(Estado, Mes)`           — granularidade média.
       c) `(Mes)`                   — fallback global.
3. Para cada ponto consultado, faz lookup em cascata a→b→c.
4. Recalcula features determinísticas (Temp_Climatologica, KBDI proxy,
   VPD proxy, Aridez De Martonne, Anomalia_Precipitacao_rel) com os
   valores **observados em tempo real** (NASA POWER), preservando a
   sensibilidade do modelo à precipitação/secura local.

Limitação (documentada no TCC)
------------------------------
Medianas espaciais/sazonais são **proxies estáticos**. SPI, anomalia
absoluta e histórico de incêndios não capturam a anomalia exata do
ponto na data consultada — capturam o "valor típico daquela célula
naquele mês". Como o objetivo do app é classificação de risco em tempo
real (não previsão precisa de área queimada), isso é aceitável.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Resolução espacial igual à de features_avancadas (≈27 km).
# ---------------------------------------------------------------------------
GRID_RESOLUTION_DEG = 0.25

# Features Tier 1 que vamos popular via lookup ou recálculo.
# Mantém ordem igual à do preprocessor (ver `modelos/preprocessor_metadata.json`).
ADVANCED_FEATURES: List[str] = [
    "Precipitacao_ma14", "Precipitacao_ma30", "Precipitacao_ma90",
    "DiaSemChuva_ma14", "DiaSemChuva_ma30", "DiaSemChuva_ma90",
    "Precipitacao_acum_30d", "Precipitacao_acum_90d",
    "Precipitacao_acum_180d", "Precipitacao_acum_365d",
    "SPI_1m", "SPI_3m", "SPI_6m",
    "Anomalia_Precipitacao", "Anomalia_Precipitacao_rel",
    "Temp_Climatologica", "KBDI_proxy", "Aridez_DeMartonne", "VPD_proxy",
    "Incendios_Ultimos_90_Dias", "Incendios_Ultimos_180_Dias",
    "Incendios_Ultimos_365_Dias", "Media_FRP_Celula_30d",
    "Dias_Secos_90d",
]

# Features SEMPRE recalculadas no momento da consulta (com clima real).
# Não vêm do lookup — o lookup fornece apenas os "blocos de construção".
RECALCULATED_FEATURES = {
    "Temp_Climatologica",
    "KBDI_proxy",
    "Aridez_DeMartonne",
    "VPD_proxy",
    "Anomalia_Precipitacao",
    "Anomalia_Precipitacao_rel",
}

# Climatologia mensal de precipitação por estado (mm/dia, valor médio
# usado como fallback se o lookup de município falhar).
PRECIP_CLIMATOLOGIA_ESTADO_MES_DEFAULT: Dict[Tuple[str, int], float] = {}


class _FeatureLookup:
    """Cache singleton com as medianas das features Tier 1."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._loaded = False
        self._by_cell: Dict[Tuple[int, int, int], Dict[str, float]] = {}
        self._by_estado: Dict[Tuple[str, int], Dict[str, float]] = {}
        self._by_mes: Dict[int, Dict[str, float]] = {}
        # Climatologia de precipitação (mensal, por município/estado) para
        # cálculo correto de Anomalia_Precipitacao em tempo real.
        self._precip_clim_mun: Dict[Tuple[str, int], float] = {}
        self._precip_clim_estado: Dict[Tuple[str, int], float] = {}
        self._precip_clim_mes: Dict[int, float] = {}
        self._dataset_path: Optional[Path] = None
        self._dataset_mtime: Optional[float] = None
        self._n_rows: int = 0

    # -------------------------------------------------------------------
    def _ensure_loaded(self, dataset_path: Path) -> None:
        with self._lock:
            mtime = dataset_path.stat().st_mtime if dataset_path.exists() else None
            if (
                self._loaded
                and self._dataset_path == dataset_path
                and self._dataset_mtime == mtime
            ):
                return
            self._load(dataset_path)
            self._loaded = True
            self._dataset_path = dataset_path
            self._dataset_mtime = mtime

    # -------------------------------------------------------------------
    def _load(self, dataset_path: Path) -> None:
        inicio = time.time()
        logger.info("Pré-calculando lookups Tier 1 a partir de %s ...", dataset_path)

        cols_necessarias = [
            "Estado", "Municipio", "Latitude", "Longitude", "Mes", "Precipitacao",
        ] + [c for c in ADVANCED_FEATURES if c not in RECALCULATED_FEATURES]
        # Inclui também Temp_Climatologica/Aridez/KBDI no lookup como
        # "fallback histórico", mesmo que o caminho normal recalcule.
        cols_extra = ["Temp_Climatologica", "Aridez_DeMartonne", "KBDI_proxy", "VPD_proxy"]
        cols_necessarias = list(dict.fromkeys(cols_necessarias + cols_extra))

        try:
            header = pd.read_csv(dataset_path, nrows=0).columns.tolist()
        except Exception as exc:  # pragma: no cover
            logger.error("Não foi possível ler header de %s: %s", dataset_path, exc)
            return

        cols_presentes = [c for c in cols_necessarias if c in header]
        ausentes = set(cols_necessarias) - set(cols_presentes)
        if ausentes:
            logger.warning(
                "Colunas ausentes no dataset enriquecido (lookup parcial): %s",
                sorted(ausentes),
            )
        if "Estado" not in cols_presentes or "Mes" not in cols_presentes:
            logger.warning("Dataset enriquecido sem Estado/Mes — lookups vazios.")
            return

        df = pd.read_csv(dataset_path, usecols=cols_presentes, low_memory=False)
        df["Estado"] = df["Estado"].astype(str).str.strip().str.upper()
        if "Municipio" in df.columns:
            df["Municipio"] = df["Municipio"].astype(str).str.strip().str.upper()
        df["Mes"] = pd.to_numeric(df["Mes"], errors="coerce").fillna(0).astype(int)

        cols_numericas = [c for c in cols_presentes if c not in ("Estado", "Municipio")]
        for c in cols_numericas:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        df["_LatBin"] = (df["Latitude"] / GRID_RESOLUTION_DEG).round().astype("int32")
        df["_LonBin"] = (df["Longitude"] / GRID_RESOLUTION_DEG).round().astype("int32")

        feature_cols = [c for c in ADVANCED_FEATURES if c in df.columns]
        self._n_rows = len(df)

        # (LatBin, LonBin, Mes) — médias por célula sazonal
        g_cell = df.groupby(["_LatBin", "_LonBin", "Mes"])[feature_cols].median()
        for (lb, lob, mes_i), row in g_cell.iterrows():
            self._by_cell[(int(lb), int(lob), int(mes_i))] = {
                k: float(v) for k, v in row.items() if pd.notna(v)
            }

        # (Estado, Mes)
        g_est = df.groupby(["Estado", "Mes"])[feature_cols].median()
        for (est, mes_i), row in g_est.iterrows():
            self._by_estado[(str(est), int(mes_i))] = {
                k: float(v) for k, v in row.items() if pd.notna(v)
            }

        # (Mes) — fallback global
        g_mes = df.groupby("Mes")[feature_cols].median()
        for mes_i, row in g_mes.iterrows():
            self._by_mes[int(mes_i)] = {k: float(v) for k, v in row.items() if pd.notna(v)}

        # Climatologia de precipitação (mensal) — usada para Anomalia em tempo real
        if "Precipitacao" in df.columns and "Municipio" in df.columns:
            p_mun = df.groupby(["Municipio", "Mes"])["Precipitacao"].mean()
            for (mun, mes_i), v in p_mun.items():
                if pd.notna(v):
                    self._precip_clim_mun[(str(mun), int(mes_i))] = float(v)
        if "Precipitacao" in df.columns:
            p_est = df.groupby(["Estado", "Mes"])["Precipitacao"].mean()
            for (est, mes_i), v in p_est.items():
                if pd.notna(v):
                    self._precip_clim_estado[(str(est), int(mes_i))] = float(v)
            p_mes = df.groupby("Mes")["Precipitacao"].mean()
            for mes_i, v in p_mes.items():
                if pd.notna(v):
                    self._precip_clim_mes[int(mes_i)] = float(v)

        elapsed = time.time() - inicio
        logger.info(
            "Lookups Tier 1 prontos em %.1fs: %d células, %d (Estado,Mes), %d meses.",
            elapsed, len(self._by_cell), len(self._by_estado), len(self._by_mes),
        )

    # -------------------------------------------------------------------
    def get_lookup_dict(
        self,
        lat: float,
        lon: float,
        estado: str,
        mes: int,
    ) -> Tuple[Dict[str, float], str]:
        """Retorna dict de features Tier 1 médias para o ponto.

        Cascata (mais granular → mais grosseira):
          1. célula central + 8 vizinhas imediatas (≈25 km)         → 'celula'
          2. anel 5×5 centrado no ponto (≈55 km)                     → 'celula_ext'
          3. anel 5×5 no mesmo Estado, mês ±1 (sazonalmente vizinho) → 'celula_mes_vizinho'
          4. mediana (Estado, Mês)                                   → 'estado_mes'
          5. mediana (Mês) global                                    → 'mes_global'
        """
        if not self._loaded:
            return {}, "vazio"

        lat_b = int(round(lat / GRID_RESOLUTION_DEG))
        lon_b = int(round(lon / GRID_RESOLUTION_DEG))
        mes_i = int(mes) if mes else 0
        est = (estado or "DESCONHECIDO").strip().upper()

        # (1) 3×3 — vizinhança imediata, próxima do treino original
        for offset_lat in (0, -1, 1):
            for offset_lon in (0, -1, 1):
                v = self._by_cell.get((lat_b + offset_lat, lon_b + offset_lon, mes_i))
                if v:
                    return dict(v), "celula"

        # (2) 5×5 — anel externo (≈55 km). Útil em regiões com baixa
        # densidade de amostras no dataset enriquecido (ex.: Cerrado-TO
        # em meses de transição).
        for offset_lat in (-2, 2, -1, 1, 0):
            for offset_lon in (-2, 2, -1, 1, 0):
                if abs(offset_lat) < 2 and abs(offset_lon) < 2:
                    continue  # já foi coberto pelo passo (1)
                v = self._by_cell.get((lat_b + offset_lat, lon_b + offset_lon, mes_i))
                if v:
                    return dict(v), "celula_ext"

        # (3) Mês adjacente (sazonalmente próximo) — preserva o sinal
        # espacial local mesmo que o mês específico não tenha amostras.
        meses_adj = [mes_i - 1 if mes_i > 1 else 12,
                     mes_i + 1 if mes_i < 12 else 1]
        for mes_alt in meses_adj:
            for offset_lat in range(-2, 3):
                for offset_lon in range(-2, 3):
                    v = self._by_cell.get((lat_b + offset_lat, lon_b + offset_lon, mes_alt))
                    if v:
                        return dict(v), "celula_mes_vizinho"

        v = self._by_estado.get((est, mes_i))
        if v:
            return dict(v), "estado_mes"

        v = self._by_mes.get(mes_i)
        if v:
            return dict(v), "mes_global"

        return {}, "vazio"

    # -------------------------------------------------------------------
    def get_precip_climatologia(
        self,
        municipio: Optional[str],
        estado: Optional[str],
        mes: int,
    ) -> Tuple[Optional[float], str]:
        if not self._loaded:
            return None, "vazio"
        mes_i = int(mes) if mes else 0
        mun = (municipio or "").strip().upper()
        est = (estado or "").strip().upper()
        if mun:
            v = self._precip_clim_mun.get((mun, mes_i))
            if v is not None:
                return float(v), "municipio_mes"
        if est:
            v = self._precip_clim_estado.get((est, mes_i))
            if v is not None:
                return float(v), "estado_mes"
        v = self._precip_clim_mes.get(mes_i)
        if v is not None:
            return float(v), "mes_global"
        return None, "vazio"


# Singleton compartilhado
_LOOKUP = _FeatureLookup()


def _carregar_lookup_se_necessario(dataset_path: Path) -> None:
    if not dataset_path.exists():
        logger.warning(
            "Dataset enriquecido %s não encontrado — features Tier 1 serão 0.",
            dataset_path,
        )
        return
    _LOOKUP._ensure_loaded(dataset_path)


def _temp_climatologica_estado_mes(estado: str, mes: int) -> float:
    """Lookup de temperatura média mensal climatológica do estado.

    Importa do features_avancadas (única fonte da verdade)."""
    try:
        from features_avancadas import _temp_climatologica
        return float(_temp_climatologica(estado, mes))
    except Exception:
        # fallback: ~26°C (média Amazônia)
        return 26.5


def get_advanced_features_for_point(
    lat: float,
    lon: float,
    estado: Optional[str],
    municipio: Optional[str],
    mes: int,
    precipitacao_atual: float,
    dias_sem_chuva_atual: float,
    prec_ma7: float,
    dsem_ma7: float,
    indice_seca: float,
    dataset_path: Path,
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """Retorna dict com as 24 features Tier 1 para 1 ponto, e metadata.

    Estratégia:
        - Features históricas (Precipitacao_acum_*, SPI_*, Incendios_*, ...)
          vêm do lookup espaço-sazonal (celula → estado_mes → mes_global).
        - Features "rolling curtas" (Precipitacao_ma14/30/90, DiaSemChuva_ma14/30/90)
          são preenchidas com o valor atual (prec_ma7 / dsem_ma7) se disponível,
          assumindo que MAs longas convergem à média estável local.
        - Temp_Climatologica, KBDI_proxy, Aridez_DeMartonne, VPD_proxy,
          Anomalia_Precipitacao* são RECALCULADAS com clima atual.

    Returns
    -------
    (features_dict, metadata) onde metadata['granularidade'] indica a
    resolução do lookup que foi usado (informativo para a UI).
    """
    _carregar_lookup_se_necessario(dataset_path)

    base_features, granularidade = _LOOKUP.get_lookup_dict(lat, lon, estado or "", mes)

    out: Dict[str, float] = {}

    # ----------------- Camada 1 — Médias móveis estendidas ---------------
    # Preferimos preencher com prec_ma7 / dsem_ma7 (vindo da NASA POWER,
    # janela 7d real do clima_provider) — assim a MA respeita o clima
    # atual do ponto, em vez da mediana histórica.
    for col_alvo, default in [
        ("Precipitacao_ma14", prec_ma7),
        ("Precipitacao_ma30", prec_ma7),
        ("Precipitacao_ma90", prec_ma7),
        ("DiaSemChuva_ma14", dsem_ma7),
        ("DiaSemChuva_ma30", dsem_ma7),
        ("DiaSemChuva_ma90", dsem_ma7),
    ]:
        out[col_alvo] = float(default if default is not None else base_features.get(col_alvo, 0.0))

    # ----------------- Camada 2 — Acumulados ----------------------------
    # Para acumulados longos, usar lookup (mediana espaço-sazonal).
    for col in (
        "Precipitacao_acum_30d", "Precipitacao_acum_90d",
        "Precipitacao_acum_180d", "Precipitacao_acum_365d",
    ):
        out[col] = float(base_features.get(col, 0.0))

    # ----------------- Camada 3 — SPI -----------------------------------
    for col in ("SPI_1m", "SPI_3m", "SPI_6m"):
        out[col] = float(base_features.get(col, 0.0))

    # ----------------- Camada 4 — Anomalia ------------------------------
    p_clim, _ = _LOOKUP.get_precip_climatologia(municipio, estado, mes)
    if p_clim is not None:
        out["Anomalia_Precipitacao"] = float(precipitacao_atual - p_clim)
        out["Anomalia_Precipitacao_rel"] = float(
            np.clip(out["Anomalia_Precipitacao"] / (p_clim + 1e-6), -5, 5)
        )
    else:
        out["Anomalia_Precipitacao"] = float(base_features.get("Anomalia_Precipitacao", 0.0))
        out["Anomalia_Precipitacao_rel"] = float(base_features.get("Anomalia_Precipitacao_rel", 0.0))

    # ----------------- Camada 5 — KBDI proxy / De Martonne / VPD --------
    temp_clim = _temp_climatologica_estado_mes(estado or "", mes)
    out["Temp_Climatologica"] = float(temp_clim)
    # KBDI ≈ dsem_ma * Temp / (prec_ma + 1)
    out["KBDI_proxy"] = float(
        (dsem_ma7 or 0.0) * temp_clim / ((prec_ma7 or 0.0) + 1.0)
    )
    # De Martonne com P acumulada de 365d (do lookup) / (T + 10)
    p_acum_365 = out["Precipitacao_acum_365d"]
    out["Aridez_DeMartonne"] = float(p_acum_365 / (temp_clim + 10.0))
    # VPD proxy: T² × Indice_Seca/1000
    out["VPD_proxy"] = float((temp_clim ** 2) * float(np.clip(indice_seca or 0.0, 0, 1000)) / 1000.0)

    # ----------------- Camada 6 — Histórico estendido -------------------
    for col in (
        "Incendios_Ultimos_90_Dias",
        "Incendios_Ultimos_180_Dias",
        "Incendios_Ultimos_365_Dias",
        "Media_FRP_Celula_30d",
    ):
        out[col] = float(base_features.get(col, 0.0))

    # ----------------- Camada 7 — Estação seca acumulada ----------------
    out["Dias_Secos_90d"] = float(base_features.get("Dias_Secos_90d", 0.0))

    metadata = {
        "granularidade": granularidade,
        "precip_climatologia_mm_dia": p_clim,
        "temp_climatologica_c": temp_clim,
        "dataset_path": str(dataset_path),
    }
    return out, metadata


def get_precip_climatologia(
    municipio: Optional[str],
    estado: Optional[str],
    mes: int,
    dataset_path: Path,
) -> Tuple[Optional[float], str]:
    """Climatologia de precipitação para comparação com o clima atual."""
    _carregar_lookup_se_necessario(dataset_path)
    return _LOOKUP.get_precip_climatologia(municipio, estado, mes)


__all__ = [
    "ADVANCED_FEATURES",
    "get_advanced_features_for_point",
    "get_precip_climatologia",
    "GRID_RESOLUTION_DEG",
]
