"""
features_avancadas.py — Features físico-climáticas derivadas para risco de fogo.

Gera, a partir das colunas (Latitude, Longitude, Ano, Mes, Dia, Precipitacao,
DiaSemChuva, FRP, Estado), um conjunto de features que a literatura recente
identifica como preditoras-chave de risco/severidade de fogo, sem depender de
APIs externas em tempo real.

Camadas de features (Tier 1 do plano de melhorias):
    1. Médias móveis estendidas (14, 30, 90 dias) por (Latitude, Longitude)
    2. Precipitação acumulada com lags longos (30, 90, 180, 365 dias)
    3. Standardized Precipitation Index (SPI-1, SPI-3, SPI-6) por célula espacial
    4. Anomalia de precipitação vs. climatologia local (município x mês)
    5. KBDI proxy / Drought Code simplificado (precipitação + temperatura
       climatológica mensal por estado)
    6. Histórico estendido de incêndios (90, 180, 365 dias) por célula
    7. Estação seca acumulada (dias com P<5mm acumulados em janela de 90 dias)

Fundamentação na literatura:
    - Seager et al. (AMS 2015): VPD e prior-year cold-season precipitation como
      preditores de área queimada.
    - Forests 2024 (Cerrado-Amazônia): KBDI, P-EVAP, FMA+ entre os melhores
      índices de risco de fogo na transição Cerrado-Amazônia; KBDI mostrou
      melhor desempenho em Canaã dos Carajás (Amazônia oriental).
    - npj Natural Hazards 2025: SPI derivado de precipitação observada prevê
      anomalia de área queimada com até 1 mês de antecedência em ~68% das
      áreas queimáveis.
    - IndJST 2025 (Sumathi & Rajesh): RFE identificou temperatura, vento e
      umidade como mais relevantes; aqui usamos temperatura climatológica
      como proxy estável (sem custo de API).
    - Sci. Reports 2025 (Gangwon, KR; Germany): NDVI e sazonalidade dominam
      a importância em modelos year-round; capturamos sazonalidade via
      MAs longos + KBDI proxy.

Limitações conhecidas (documentar no TCC):
    - SPI usa climatologia in-sample (sem split temporal). Aceitável porque a
      média de precipitação não tem feedback direto com o label RiscoFogo; o
      ganho de informação é de natureza espacial-sazonal.
    - Temperatura é climatologia mensal por estado (estática), não real-time.
      É um proxy razoável dado que a variabilidade interanual de T na
      Amazônia é menor que a variabilidade de precipitação.
    - KBDI aqui é simplificado (sem recursão diária completa do KBDI clássico
      de Keetch-Byram 1968), mas captura a essência: déficit hídrico
      acumulado modulado por temperatura.

Uso programático:
    from features_avancadas import adicionar_features_avancadas
    df_enriquecido = adicionar_features_avancadas(df_original)

Uso CLI:
    python scripts/features_avancadas.py \
        --input base_de_dados_com_historico.csv \
        --output base_de_dados_enriquecido.csv
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Climatologia de temperatura mensal por estado (°C)
# Aproximação das normais climatológicas INMET 1981-2010 (médias de capitais).
# Fonte: portal.inmet.gov.br/normais. Valores arredondados para 0.5°C.
# Documento como limitação no TCC: usamos média estadual estática, não
# variabilidade interanual nem espacial intra-estadual.
# ---------------------------------------------------------------------------
TEMP_CLIMATOLOGIA_ESTADO_MES: dict = {
    'ACRE':         [25.5, 25.5, 25.5, 25.5, 24.5, 23.5, 23.5, 25.0, 26.0, 26.0, 26.0, 25.5],
    'AMAPA':        [26.0, 26.0, 26.0, 26.0, 26.5, 26.5, 26.5, 27.0, 27.5, 27.5, 27.0, 26.5],
    'AMAZONAS':     [26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 27.0, 27.5, 28.0, 28.0, 27.5, 27.0],
    'MARANHAO':     [26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 27.5, 28.5, 28.5, 28.0, 27.0],
    'MARANHÃO':     [26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 27.5, 28.5, 28.5, 28.0, 27.0],
    'MATO GROSSO':  [26.0, 26.0, 26.0, 25.5, 24.0, 23.0, 23.0, 25.0, 27.0, 27.0, 26.5, 26.0],
    'PARA':         [26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 27.0, 27.5, 27.5, 27.0, 26.5],
    'PARÁ':         [26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 27.0, 27.5, 27.5, 27.0, 26.5],
    'RONDONIA':     [25.5, 25.5, 25.5, 25.5, 24.5, 23.5, 23.5, 25.0, 26.5, 26.5, 26.0, 25.5],
    'RONDÔNIA':     [25.5, 25.5, 25.5, 25.5, 24.5, 23.5, 23.5, 25.0, 26.5, 26.5, 26.0, 25.5],
    'RORAIMA':      [27.0, 27.0, 27.5, 27.5, 27.0, 26.5, 26.5, 27.0, 27.5, 28.0, 27.5, 27.0],
    'TOCANTINS':    [26.5, 26.5, 27.0, 27.0, 26.0, 25.0, 25.0, 27.0, 28.5, 28.0, 27.0, 26.5],
}

# Valor default caso o estado não esteja mapeado (média ponderada da Amazônia)
TEMP_CLIMATOLOGIA_DEFAULT = [26.3, 26.3, 26.4, 26.4, 25.8, 25.4, 25.5, 26.5, 27.4, 27.4, 27.0, 26.5]

# Resolução das células espaciais para SPI e histórico estendido (em graus).
# 0.25° ≈ 27 km — granularidade suficiente para clima e histórico de fogo.
GRID_RESOLUTION_DEG = 0.25

# Janelas de tempo (dias) usadas em features acumuladas / históricas
JANELAS_DIAS = (14, 30, 90)
JANELAS_LAG_DIAS = (30, 90, 180, 365)
JANELAS_HISTORICO_DIAS = (90, 180, 365)
SPI_LAGS_MESES = (1, 3, 6)

EPS = 1e-6


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _ensure_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """Garante presença de coluna `_Data` (datetime) baseada em (Ano, Mes, Dia)."""
    if '_Data' in df.columns and pd.api.types.is_datetime64_any_dtype(df['_Data']):
        return df

    required = {'Ano', 'Mes', 'Dia'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Colunas obrigatórias para construir data: {sorted(missing)}")

    df = df.copy()
    df['_Data'] = pd.to_datetime(
        df['Ano'].astype(str)
        + '-'
        + df['Mes'].astype(int).astype(str).str.zfill(2)
        + '-'
        + df['Dia'].astype(int).astype(str).str.zfill(2),
        format='%Y-%m-%d',
        errors='coerce',
    )
    return df


def _normalizar_estado(estado: str) -> str:
    if not isinstance(estado, str):
        return ''
    return estado.strip().upper()


def _adicionar_bins_espaciais(df: pd.DataFrame, grid: float = GRID_RESOLUTION_DEG) -> pd.DataFrame:
    """Cria colunas auxiliares `_LatBin`, `_LonBin` para agrupamento espacial.

    Crucial porque cada foco no dataset é geralmente um ponto único
    (Lat, Lon) — agrupar por coordenada exata colapsa as janelas
    temporais. Discretizar em células de ~0.25° (≈27 km) recupera o
    sinal de série temporal local.
    """
    if '_LatBin' in df.columns and '_LonBin' in df.columns:
        return df
    df = df.copy()
    df['_LatBin'] = (df['Latitude'] / grid).round().astype('int32')
    df['_LonBin'] = (df['Longitude'] / grid).round().astype('int32')
    return df


def _sanitizar_numericos(df: pd.DataFrame) -> pd.DataFrame:
    """Converte sentinelas (-999) para NaN nas colunas numéricas-base.
    Necessário porque o pipeline atual (`carregar_dados.limpar_dados`) faz
    isso antes de chamar features avançadas, mas se este módulo for chamado
    isolado (CLI ou testes), os -999 ainda estão presentes.
    """
    df = df.copy()
    for col in ('Precipitacao', 'DiaSemChuva', 'FRP'):
        if col in df.columns:
            df[col] = df[col].replace(-999, np.nan)
    if 'FRP' in df.columns:
        df['FRP'] = df['FRP'].fillna(0.0)
    return df


def _garantir_indice_seca(df: pd.DataFrame) -> pd.DataFrame:
    """Garante que `Indice_Seca` exista (caso este módulo seja chamado fora
    do `carregar_dados.py`)."""
    if 'Indice_Seca' in df.columns:
        return df
    if {'DiaSemChuva', 'Precipitacao'}.issubset(df.columns):
        df = df.copy()
        df['Indice_Seca'] = df['DiaSemChuva'].fillna(0) / (df['Precipitacao'].fillna(0) + 0.1)
    return df


def _temp_climatologica(estado: str, mes: int) -> float:
    """Lookup de temperatura média mensal climatológica do estado."""
    chave = _normalizar_estado(estado)
    serie = TEMP_CLIMATOLOGIA_ESTADO_MES.get(chave)
    if serie is None:
        serie = TEMP_CLIMATOLOGIA_DEFAULT
    if pd.isna(mes):
        return float(np.mean(serie))
    idx = int(mes) - 1
    if not 0 <= idx <= 11:
        return float(np.mean(serie))
    return float(serie[idx])


# ---------------------------------------------------------------------------
# Camada 1 — Médias móveis estendidas
# ---------------------------------------------------------------------------
def _adicionar_medias_moveis(df: pd.DataFrame, janelas: Iterable[int] = JANELAS_DIAS) -> pd.DataFrame:
    """Calcula MAs por **célula espacial** (LatBin, LonBin) ordenado pelo
    tempo, com janela em dias-corridos via DateTimeIndex.

    Agrupar por célula espacial (≈0.25°) — em vez de (Lat, Lon) literal — é
    crítico, pois cada foco é um ponto único e a granularidade exata
    colapsa qualquer janela temporal."""
    if not all(c in df.columns for c in ('_LatBin', '_LonBin', '_Data')):
        return df

    inicio = time.time()
    df = df.sort_values(['_LatBin', '_LonBin', '_Data']).reset_index(drop=True)

    for col, prefixo in [('Precipitacao', 'Precipitacao'), ('DiaSemChuva', 'DiaSemChuva')]:
        if col not in df.columns:
            continue
        for n in janelas:
            nome = f'{prefixo}_ma{n}'
            if nome in df.columns:
                continue
            df[nome] = (
                df.set_index('_Data')
                  .groupby(['_LatBin', '_LonBin'])[col]
                  .transform(lambda s: s.rolling(f'{n}D', min_periods=1).mean())
                  .reset_index(drop=True)
            )

    logger.info(
        "Camada 1 (médias móveis %s, grid %.2f°) concluída em %.1fs",
        list(janelas), GRID_RESOLUTION_DEG, time.time() - inicio,
    )
    return df


# ---------------------------------------------------------------------------
# Camada 2 — Precipitação acumulada (lags longos)
# ---------------------------------------------------------------------------
def _adicionar_acumulados(df: pd.DataFrame, janelas: Iterable[int] = JANELAS_LAG_DIAS) -> pd.DataFrame:
    """Soma rolling de Precipitacao por célula espacial."""
    if not all(c in df.columns for c in ('_LatBin', '_LonBin', '_Data', 'Precipitacao')):
        return df

    inicio = time.time()
    df = df.sort_values(['_LatBin', '_LonBin', '_Data']).reset_index(drop=True)

    for n in janelas:
        nome = f'Precipitacao_acum_{n}d'
        if nome in df.columns:
            continue
        df[nome] = (
            df.set_index('_Data')
              .groupby(['_LatBin', '_LonBin'])['Precipitacao']
              .transform(lambda s: s.rolling(f'{n}D', min_periods=1).sum())
              .reset_index(drop=True)
        )

    logger.info(
        "Camada 2 (precipitação acumulada %s, grid %.2f°) concluída em %.1fs",
        list(janelas), GRID_RESOLUTION_DEG, time.time() - inicio,
    )
    return df


# ---------------------------------------------------------------------------
# Camada 3 — SPI (Standardized Precipitation Index)
# ---------------------------------------------------------------------------
def _adicionar_spi(df: pd.DataFrame, lags_meses: Iterable[int] = SPI_LAGS_MESES,
                   grid: float = GRID_RESOLUTION_DEG) -> pd.DataFrame:
    """Calcula SPI-1, SPI-3, SPI-6 por célula espacial e mês.

    Implementação:
        1. Discretiza (Lat, Lon) em células de `grid`° (≈27 km a 0.25°).
        2. Agrega Precipitacao média por (cell, ano, mês).
        3. Para cada lag k, calcula soma rolling de k meses por célula.
        4. z-score por (cell, mês_calendário) usando todos os anos como
           climatologia (μ, σ).
        5. Faz join de volta no df original por (cell, ano, mês).
    """
    if not all(c in df.columns for c in ('Latitude', 'Longitude', 'Ano', 'Mes', 'Precipitacao')):
        return df

    inicio = time.time()
    df = df.copy()
    if '_LatBin' not in df.columns:
        df['_LatBin'] = (df['Latitude'] / grid).round().astype('int32')
        df['_LonBin'] = (df['Longitude'] / grid).round().astype('int32')

    # Precipitação média mensal por célula
    pm = (
        df.groupby(['_LatBin', '_LonBin', 'Ano', 'Mes'])['Precipitacao']
          .mean()
          .reset_index()
          .rename(columns={'Precipitacao': '_P_mensal'})
    )
    # ordenar por tempo dentro de cada célula
    pm = pm.sort_values(['_LatBin', '_LonBin', 'Ano', 'Mes']).reset_index(drop=True)

    for k in lags_meses:
        col_acum = f'_P_acum_{k}m'
        if k == 1:
            pm[col_acum] = pm['_P_mensal']
        else:
            pm[col_acum] = (
                pm.groupby(['_LatBin', '_LonBin'])['_P_mensal']
                  .transform(lambda s: s.rolling(k, min_periods=1).sum())
            )

        # climatologia (μ, σ) por (célula, mês_calendário)
        clim = (
            pm.groupby(['_LatBin', '_LonBin', 'Mes'])[col_acum]
              .agg(['mean', 'std'])
              .reset_index()
              .rename(columns={'mean': f'_mu_{k}m', 'std': f'_sd_{k}m'})
        )
        pm = pm.merge(clim, on=['_LatBin', '_LonBin', 'Mes'], how='left')

        spi_col = f'SPI_{k}m'
        sd = pm[f'_sd_{k}m'].replace(0, np.nan)
        pm[spi_col] = (pm[col_acum] - pm[f'_mu_{k}m']) / (sd + EPS)
        pm[spi_col] = pm[spi_col].fillna(0.0).clip(-3.5, 3.5)

    # join SPI de volta no df original (sem dropar bins, usados em outras camadas)
    spi_cols = [f'SPI_{k}m' for k in lags_meses]
    df = df.merge(
        pm[['_LatBin', '_LonBin', 'Ano', 'Mes'] + spi_cols],
        on=['_LatBin', '_LonBin', 'Ano', 'Mes'],
        how='left',
    )
    for c in spi_cols:
        df[c] = df[c].fillna(0.0)

    logger.info("Camada 3 (SPI %s meses) concluída em %.1fs", list(lags_meses), time.time() - inicio)
    return df


# ---------------------------------------------------------------------------
# Camada 4 — Anomalia de precipitação por município x mês
# ---------------------------------------------------------------------------
def _adicionar_anomalia_precipitacao(df: pd.DataFrame) -> pd.DataFrame:
    if not all(c in df.columns for c in ('Municipio', 'Mes', 'Precipitacao')):
        return df

    inicio = time.time()
    df = df.copy()
    clim = (
        df.groupby(['Municipio', 'Mes'])['Precipitacao']
          .mean()
          .reset_index()
          .rename(columns={'Precipitacao': '_P_clim_mun_mes'})
    )
    df = df.merge(clim, on=['Municipio', 'Mes'], how='left')
    df['Anomalia_Precipitacao'] = df['Precipitacao'] - df['_P_clim_mun_mes']
    # ratio relativo: quanto está acima/abaixo da média histórica do município
    df['Anomalia_Precipitacao_rel'] = (
        df['Anomalia_Precipitacao'] / (df['_P_clim_mun_mes'] + EPS)
    ).clip(-5, 5)
    df.drop(columns=['_P_clim_mun_mes'], inplace=True)

    logger.info("Camada 4 (anomalia de precipitação) concluída em %.1fs", time.time() - inicio)
    return df


# ---------------------------------------------------------------------------
# Camada 5 — KBDI proxy / Drought Code simplificado / Aridez
# ---------------------------------------------------------------------------
def _adicionar_kbdi_proxy(df: pd.DataFrame) -> pd.DataFrame:
    """Adiciona Temp_Climatologica + KBDI proxy + Aridez de De Martonne.

    Não é o KBDI canônico de Keetch-Byram 1968 (esse exige recursão diária
    sobre temperatura máxima e precipitação observada). Aqui usamos uma
    versão simplificada que captura a essência do índice: déficit hídrico
    acumulado modulado por demanda evaporativa (T climatológica)."""

    if 'Estado' not in df.columns or 'Mes' not in df.columns:
        return df

    inicio = time.time()
    df = df.copy()
    # Lookup vetorizado por (Estado, Mes)
    df['Temp_Climatologica'] = [
        _temp_climatologica(e, m) for e, m in zip(df['Estado'].values, df['Mes'].values)
    ]

    base_p_ma = 'Precipitacao_ma30' if 'Precipitacao_ma30' in df.columns else 'Precipitacao_ma7'
    base_dsc_ma = 'DiaSemChuva_ma30' if 'DiaSemChuva_ma30' in df.columns else 'DiaSemChuva_ma7'

    if base_p_ma in df.columns and base_dsc_ma in df.columns:
        df['KBDI_proxy'] = (
            df[base_dsc_ma].fillna(0)
            * df['Temp_Climatologica']
            / (df[base_p_ma].fillna(0) + 1.0)
        )
    elif {'DiaSemChuva', 'Precipitacao'}.issubset(df.columns):
        df['KBDI_proxy'] = (
            df['DiaSemChuva'].fillna(0)
            * df['Temp_Climatologica']
            / (df['Precipitacao'].fillna(0) + 1.0)
        )

    # Aridez de De Martonne (anualizada via P acumulada de 365 dias)
    if 'Precipitacao_acum_365d' in df.columns:
        df['Aridez_DeMartonne'] = df['Precipitacao_acum_365d'] / (df['Temp_Climatologica'] + 10.0)
    elif 'Precipitacao_acum_180d' in df.columns:
        df['Aridez_DeMartonne'] = (df['Precipitacao_acum_180d'] * 2) / (df['Temp_Climatologica'] + 10.0)

    # VPD proxy (vapor pressure deficit grosseiro): T² × Indice_Seca/100
    if 'Indice_Seca' in df.columns:
        df['VPD_proxy'] = (df['Temp_Climatologica'] ** 2) * df['Indice_Seca'].clip(0, 1000) / 1000.0

    logger.info("Camada 5 (KBDI proxy / De Martonne / VPD proxy) concluída em %.1fs",
                time.time() - inicio)
    return df


# ---------------------------------------------------------------------------
# Camada 6 — Histórico estendido de incêndios
# ---------------------------------------------------------------------------
def _adicionar_historico_estendido(df: pd.DataFrame,
                                    janelas: Iterable[int] = JANELAS_HISTORICO_DIAS,
                                    grid: float = GRID_RESOLUTION_DEG) -> pd.DataFrame:
    """Conta focos em janelas longas (90/180/365 dias) por célula espacial.

    Usa o próprio dataset (cada linha é um foco detectado) como evidência
    histórica. Para eficiência, agrega contagem por (cell, dia) e aplica
    rolling sum em DateTimeIndex por célula; depois faz join de volta.
    """
    if not all(c in df.columns for c in ('_LatBin', '_LonBin', '_Data')):
        return df

    inicio = time.time()
    df = df.copy()

    # contagem de focos por (célula, dia)
    daily = (
        df.groupby(['_LatBin', '_LonBin', '_Data'])
          .size()
          .reset_index(name='_focos_dia')
    )
    daily = daily.sort_values(['_LatBin', '_LonBin', '_Data']).reset_index(drop=True)

    for n in janelas:
        col = f'Incendios_Ultimos_{n}_Dias'
        daily[col] = (
            daily.set_index('_Data')
                 .groupby(['_LatBin', '_LonBin'])['_focos_dia']
                 .transform(lambda s: s.rolling(f'{n}D', min_periods=1).sum())
                 .reset_index(drop=True)
        )

    # Calculo de FRP médio na célula nos últimos 30 dias (proxy de severidade)
    if 'FRP' in df.columns:
        frp_daily = (
            df.groupby(['_LatBin', '_LonBin', '_Data'])['FRP']
              .mean()
              .reset_index()
        )
        frp_daily = frp_daily.sort_values(['_LatBin', '_LonBin', '_Data']).reset_index(drop=True)
        frp_daily['Media_FRP_Celula_30d'] = (
            frp_daily.set_index('_Data')
                     .groupby(['_LatBin', '_LonBin'])['FRP']
                     .transform(lambda s: s.rolling('30D', min_periods=1).mean())
                     .reset_index(drop=True)
        )
        daily = daily.merge(
            frp_daily[['_LatBin', '_LonBin', '_Data', 'Media_FRP_Celula_30d']],
            on=['_LatBin', '_LonBin', '_Data'],
            how='left',
        )

    cols_join = [c for c in daily.columns if c.startswith('Incendios_Ultimos_') or c == 'Media_FRP_Celula_30d']
    df = df.merge(
        daily[['_LatBin', '_LonBin', '_Data'] + cols_join],
        on=['_LatBin', '_LonBin', '_Data'],
        how='left',
    )
    df[cols_join] = df[cols_join].fillna(0.0)

    logger.info(
        "Camada 6 (histórico estendido %s dias) concluída em %.1fs",
        list(janelas), time.time() - inicio,
    )
    return df


# ---------------------------------------------------------------------------
# Camada 7 — Estação seca acumulada (proxy do Drought Code)
# ---------------------------------------------------------------------------
def _adicionar_estacao_seca_acumulada(df: pd.DataFrame, limiar_p_mm: float = 5.0) -> pd.DataFrame:
    """Conta dias com Precipitacao < limiar_p_mm em janela de 90d por
    célula espacial. Proxy simples do Canadian Drought Code."""
    if not all(c in df.columns for c in ('_LatBin', '_LonBin', '_Data', 'Precipitacao')):
        return df

    inicio = time.time()
    df = df.sort_values(['_LatBin', '_LonBin', '_Data']).reset_index(drop=True)
    df['_dia_seco'] = (df['Precipitacao'].fillna(0) < limiar_p_mm).astype('int8')
    df['Dias_Secos_90d'] = (
        df.set_index('_Data')
          .groupby(['_LatBin', '_LonBin'])['_dia_seco']
          .transform(lambda s: s.rolling('90D', min_periods=1).sum())
          .reset_index(drop=True)
    )
    df.drop(columns=['_dia_seco'], inplace=True)
    logger.info("Camada 7 (estação seca acumulada 90d) concluída em %.1fs", time.time() - inicio)
    return df


# ---------------------------------------------------------------------------
# API pública — pipeline completo
# ---------------------------------------------------------------------------
ADVANCED_FEATURES_NUM: List[str] = [
    # Camada 1
    'Precipitacao_ma14', 'Precipitacao_ma30', 'Precipitacao_ma90',
    'DiaSemChuva_ma14', 'DiaSemChuva_ma30', 'DiaSemChuva_ma90',
    # Camada 2
    'Precipitacao_acum_30d', 'Precipitacao_acum_90d',
    'Precipitacao_acum_180d', 'Precipitacao_acum_365d',
    # Camada 3
    'SPI_1m', 'SPI_3m', 'SPI_6m',
    # Camada 4
    'Anomalia_Precipitacao', 'Anomalia_Precipitacao_rel',
    # Camada 5
    'Temp_Climatologica', 'KBDI_proxy', 'Aridez_DeMartonne', 'VPD_proxy',
    # Camada 6
    'Incendios_Ultimos_90_Dias', 'Incendios_Ultimos_180_Dias',
    'Incendios_Ultimos_365_Dias', 'Media_FRP_Celula_30d',
    # Camada 7
    'Dias_Secos_90d',
]


def adicionar_features_avancadas(df: pd.DataFrame, drop_helpers: bool = True) -> pd.DataFrame:
    """Aplica todas as camadas de features avançadas.

    Parameters
    ----------
    df : pd.DataFrame
        Deve conter Latitude, Longitude, Ano, Mes, Dia, Precipitacao,
        DiaSemChuva, FRP (opcional), Estado, Municipio.
    drop_helpers : bool
        Se True, remove colunas auxiliares (`_Data`) antes de retornar.

    Returns
    -------
    pd.DataFrame com as features avançadas adicionadas. Linhas e índice
    preservados em ordem (Latitude, Longitude, _Data) — não preserva ordem
    original; reordene se necessário.
    """
    inicio_total = time.time()
    n0 = len(df)
    logger.info("=== Features avançadas: iniciando enriquecimento de %d linhas ===", n0)

    df = _sanitizar_numericos(df)
    df = _garantir_indice_seca(df)
    df = _ensure_datetime(df)
    df = _adicionar_bins_espaciais(df)

    df = _adicionar_medias_moveis(df)
    df = _adicionar_acumulados(df)
    df = _adicionar_spi(df)
    df = _adicionar_anomalia_precipitacao(df)
    df = _adicionar_kbdi_proxy(df)
    df = _adicionar_historico_estendido(df)
    df = _adicionar_estacao_seca_acumulada(df)

    if drop_helpers:
        for col in ('_Data', '_LatBin', '_LonBin'):
            if col in df.columns:
                df = df.drop(columns=[col])

    novas = [c for c in ADVANCED_FEATURES_NUM if c in df.columns]
    logger.info(
        "=== Features avançadas: %d features adicionadas em %.1fs (linhas: %d → %d) ===",
        len(novas), time.time() - inicio_total, n0, len(df),
    )
    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _cli(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Enriquece o dataset com features físicas/climáticas avançadas (Tier 1)."
    )
    parser.add_argument(
        '--input', type=Path, default=Path('base_de_dados_com_historico.csv'),
        help='CSV de entrada (default: base_de_dados_com_historico.csv).',
    )
    parser.add_argument(
        '--output', type=Path, default=Path('base_de_dados_enriquecido.csv'),
        help='CSV de saída (default: base_de_dados_enriquecido.csv).',
    )
    parser.add_argument(
        '--parquet', action='store_true',
        help='Salva também em formato Parquet (mais rápido para retreinos).',
    )
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='[%(levelname)s] %(asctime)s %(name)s: %(message)s',
        datefmt='%H:%M:%S',
    )

    if not args.input.exists():
        logger.error("Arquivo de entrada não encontrado: %s", args.input)
        return 1

    logger.info("Carregando %s ...", args.input)
    df = pd.read_csv(args.input)
    logger.info("Dataset carregado: %d linhas, %d colunas", len(df), df.shape[1])

    df_out = adicionar_features_avancadas(df)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Salvando %s ...", args.output)
    df_out.to_csv(args.output, index=False)
    logger.info("CSV salvo (%d linhas, %d colunas).", len(df_out), df_out.shape[1])

    if args.parquet:
        parquet_path = args.output.with_suffix('.parquet')
        try:
            df_out.to_parquet(parquet_path, index=False)
            logger.info("Parquet salvo: %s", parquet_path)
        except Exception as e:  # pragma: no cover
            logger.warning("Falha ao salvar parquet (%s). Apenas CSV foi gerado.", e)

    return 0


if __name__ == '__main__':
    sys.exit(_cli())
