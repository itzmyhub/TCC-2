"""Módulo para integração com APIs climáticas."""
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests

from config import (
    DATASET_PATH,
    HG_WEATHER_API_KEY,
    HISTORICAL_TRAINING_DATASET_PATH,
    INMET_EXTENDED_MAX_DISTANCE_KM,
    INMET_MAX_DISTANCE_KM,
)

from nasa_power_realtime import fetch_nasa_power_window

try:
    from inmet_api import get_inmet_fusion_data
except Exception as _exc:  # noqa: BLE001
    logger_init = logging.getLogger(__name__)
    logger_init.warning("Módulo INMET indisponível (%s) — pipeline segue só com NASA POWER.", _exc)
    get_inmet_fusion_data = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


class ClimateDataProvider:
    """Classe para obter dados climáticos de diferentes fontes."""
    
    def __init__(self, api_key: str = HG_WEATHER_API_KEY):
        self.api_key = api_key
        self.historical_data = None
        self._load_historical_data()
        # Medianas (Estado, Mes) do dataset de treino para aproximar histórico de incêndio no mapa
        self._fire_by_estado_mes: Optional[Dict[Tuple[str, int], Dict[str, float]]] = None
        self._fire_by_mes: Optional[Dict[int, Dict[str, float]]] = None
    
    def _load_historical_data(self):
        """Carrega dados históricos para fallback."""
        try:
            if DATASET_PATH.exists():
                df = pd.read_csv(DATASET_PATH, nrows=10000)  # Carregar amostra para performance
                # Converter colunas se necessário
                if 'Precipitacao' in df.columns:
                    df['Precipitacao'] = pd.to_numeric(df['Precipitacao'], errors='coerce')
                self.historical_data = df
                logger.info("Dados históricos carregados para fallback")
        except Exception as e:
            logger.warning("Não foi possível carregar dados históricos: %s", e)
            self.historical_data = None

    def _load_training_fire_lookups(self) -> None:
        if self._fire_by_estado_mes is not None:
            return
        fire_cols: List[str] = [
            "Incendios_Ultimos_7_Dias",
            "Incendios_Ultimos_30_Dias",
            "Dias_Desde_Ultimo_Incendio",
            "Media_FRP_Ultimos_7_Dias",
            "Max_FRP_Ultimos_7_Dias",
        ]
        path = HISTORICAL_TRAINING_DATASET_PATH
        if not path.exists():
            path = DATASET_PATH
        if not path.exists():
            logger.info("Sem CSV de treino para medianas de incêndio: %s", path)
            self._fire_by_estado_mes = {}
            self._fire_by_mes = {}
            return
        try:
            df0 = pd.read_csv(path, nrows=0)
            cols = [c for c in (["Estado", "Mes"] + fire_cols) if c in df0.columns]
            if "Estado" not in cols or "Mes" not in cols:
                self._fire_by_estado_mes = {}
                self._fire_by_mes = {}
                return
            df = pd.read_csv(path, usecols=cols, low_memory=False)
            for c in fire_cols:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce")
            df["Estado"] = df["Estado"].astype(str).str.strip().str.upper()
            df["Mes"] = pd.to_numeric(df["Mes"], errors="coerce").fillna(0).astype(int)
            present = [c for c in fire_cols if c in df.columns]
            if not present:
                self._fire_by_estado_mes = {}
                self._fire_by_mes = {}
                return
            g = (
                df.groupby(["Estado", "Mes"])[present]
                .median()
            )
            self._fire_by_estado_mes = {}
            for (est, mes_i), row in g.iterrows():
                self._fire_by_estado_mes[(str(est), int(mes_i))] = {k: float(row[k]) for k in present if pd.notna(row[k])}
            g2 = df.groupby("Mes")[present].median()
            self._fire_by_mes = {}
            for mes_i, row in g2.iterrows():
                self._fire_by_mes[int(mes_i)] = {k: float(row[k]) for k in present if pd.notna(row[k])}
            logger.info(
                "Carregadas medianas de incêndio (treino) para %d pares (Estado,Mes) de %s",
                len(self._fire_by_estado_mes),
                path.name,
            )
        except Exception as e:
            logger.warning("Falha ao carregar medianas de incêndio do treino: %s", e)
            self._fire_by_estado_mes = {}
            self._fire_by_mes = {}

    def get_training_fire_proxies(self, estado: str, mes: int) -> Dict[str, float]:
        """Valores medianos do dataset de treino alinhados a Estado/mês (aproxima distribuição usada no modelo)."""
        self._load_training_fire_lookups()
        est = (estado or "DESCONHECIDO").strip().upper()
        defaults: Dict[str, float] = {
            "Incendios_Ultimos_7_Dias": 0.0,
            "Incendios_Ultimos_30_Dias": 0.0,
            "Dias_Desde_Ultimo_Incendio": 365.0,
            "Media_FRP_Ultimos_7_Dias": 0.0,
            "Max_FRP_Ultimos_7_Dias": 0.0,
        }
        if not self._fire_by_estado_mes:
            return defaults
        d = self._fire_by_estado_mes.get((est, int(mes)))
        if not d:
            d = self._fire_by_mes.get(int(mes)) if self._fire_by_mes else None
        if not d:
            return defaults
        out = defaults.copy()
        out.update(d)
        return out

    def get_climate_data_from_api(self, lat: float, lon: float) -> Optional[Dict]:
        """Obtém dados climáticos da API HG Weather."""
        try:
            # URL correta da API HG Weather
            url = "https://api.hgbrasil.com/weather"
            # A API HG Weather pode usar 'woeid' para cidade ou 'lat'/'lon' para coordenadas
            # Vamos tentar ambos os formatos
            params = {
                'key': self.api_key,
                'lat': lat,
                'lon': lon,
                'user_ip': 'remote',  # Para identificar origem
            }
            
            logger.info("Consultando API HG Weather: lat=%.4f, lon=%.4f", lat, lon)
            response = requests.get(url, params=params, timeout=15)
            logger.info("Resposta API: status=%d", response.status_code)
            
            response.raise_for_status()
            data = response.json()
            
            logger.info("Dados recebidos da API: %s", str(data)[:200])
            
            # Verificar estrutura da resposta
            if 'results' in data:
                results = data['results']
                logger.info("Estrutura da resposta: %s", list(results.keys())[:10])
                
                # Tentar obter precipitação de diferentes campos possíveis
                precipitacao = 0.0
                precipitacao_encontrada = False
                
                # Campo 'rain' pode estar em results diretamente ou no forecast
                if 'rain' in results and results['rain'] is not None:
                    precipitacao = float(results['rain'])
                    precipitacao_encontrada = True
                    logger.info("Precipitação encontrada em 'rain': %.2f", precipitacao)
                elif 'precipitation' in results and results['precipitation'] is not None:
                    precipitacao = float(results['precipitation'])
                    precipitacao_encontrada = True
                    logger.info("Precipitação encontrada em 'precipitation': %.2f", precipitacao)
                elif 'forecast' in results and isinstance(results['forecast'], list) and len(results['forecast']) > 0:
                    # Tentar pegar do forecast (previsão do dia atual)
                    today = results['forecast'][0]
                    logger.info("Campos disponíveis no forecast[0]: %s", list(today.keys()) if isinstance(today, dict) else 'não é dict')
                    
                    if isinstance(today, dict):
                        if 'rain' in today and today['rain'] is not None:
                            precipitacao = float(today['rain'])
                            precipitacao_encontrada = True
                            logger.info("Precipitação encontrada em forecast[0]['rain']: %.2f", precipitacao)
                        elif 'precipitation' in today and today['precipitation'] is not None:
                            precipitacao = float(today['precipitation'])
                            precipitacao_encontrada = True
                            logger.info("Precipitação encontrada em forecast[0]['precipitation']: %.2f", precipitacao)
                        elif 'rain_probability' in today:
                            # Se não tem chuva prevista mas tem probabilidade, usar 0
                            precipitacao = 0.0
                            logger.info("Chuva não prevista (probabilidade: %s%%)", today.get('rain_probability', 'N/A'))
                
                if not precipitacao_encontrada:
                    logger.warning("Precipitação não encontrada na resposta da API. Campos disponíveis: %s", list(results.keys()))
                
                # Tentar calcular dias sem chuva baseado no forecast
                # Se temos forecast, podemos contar quantos dias consecutivos sem chuva
                dias_sem_chuva_estimado = None
                if 'forecast' in results and isinstance(results['forecast'], list) and len(results['forecast']) > 0:
                    dias_consecutivos_sem_chuva = 0
                    for dia_forecast in results['forecast']:
                        if isinstance(dia_forecast, dict):
                            chuva_dia = dia_forecast.get('rain', 0) or dia_forecast.get('precipitation', 0) or 0
                            try:
                                chuva_valor = float(chuva_dia) if chuva_dia else 0.0
                                if chuva_valor < 0.1:  # Menos de 0.1mm = sem chuva significativa
                                    dias_consecutivos_sem_chuva += 1
                                else:
                                    break  # Parar no primeiro dia com chuva
                            except (ValueError, TypeError):
                                # Se não conseguir converter, considerar sem chuva
                                dias_consecutivos_sem_chuva += 1
                    if dias_consecutivos_sem_chuva > 0:
                        dias_sem_chuva_estimado = float(dias_consecutivos_sem_chuva)
                        logger.info("Dias sem chuva estimados do forecast: %.1f dias", dias_sem_chuva_estimado)
                
                resultado = {
                    'precipitacao': precipitacao,
                    'temperatura': results.get('temp', None),
                    'umidade': results.get('humidity', None),
                    'descricao': results.get('description', ''),
                    'fonte': 'hg_weather',
                    'timestamp': datetime.now().isoformat(),
                    'sucesso': True,
                    'resposta_completa': results,  # Para debug
                    'dias_sem_chuva_estimado': dias_sem_chuva_estimado  # Estimado do forecast se disponível
                }
                
                logger.info("Dados extraídos: precipitação=%.2f mm, temperatura=%s°C, umidade=%s%%",
                           precipitacao, 
                           results.get('temp', 'N/A'),
                           results.get('humidity', 'N/A'))
                return resultado
            else:
                logger.warning("Resposta da API HG Weather não contém 'results': %s", str(data)[:200])
                return None
                
        except requests.exceptions.RequestException as e:
            logger.error("Erro ao consultar API HG Weather: %s", e, exc_info=True)
            if hasattr(e, 'response') and e.response is not None:
                logger.error("Resposta do erro: %s", e.response.text[:500])
            return None
        except Exception as e:
            logger.error("Erro inesperado ao obter dados da API: %s", e, exc_info=True)
            return None
    
    def estimate_days_without_rain(self, lat: float, lon: float, mes: int) -> float:
        """Estima dias sem chuva baseado em dados históricos."""
        if self.historical_data is None:
            # Estimativa simples baseada no mês
            # Meses mais secos na Amazônia: julho-setembro (7-9)
            dry_season_months = [7, 8, 9]
            if mes in dry_season_months:
                return 15.0  # Estimativa para estação seca
            else:
                return 5.0  # Estimativa para estação chuvosa
        
        try:
            # Filtrar dados por coordenadas próximas e mês
            lat_range = 2.0  # Buscar dentro de 2 graus
            lon_range = 2.0
            
            filtered = self.historical_data[
                (abs(self.historical_data['Latitude'] - lat) <= lat_range) &
                (abs(self.historical_data['Longitude'] - lon) <= lon_range) &
                (self.historical_data['Mes'] == mes)
            ]
            
            if len(filtered) > 0:
                # Calcular média de dias sem chuva quando precipitação é muito baixa
                low_precip = filtered[filtered['Precipitacao'] < 0.5]
                if len(low_precip) > 0 and 'DiaSemChuva' in low_precip.columns:
                    return float(low_precip['DiaSemChuva'].median())
                else:
                    # Se não tiver coluna DiaSemChuva, estimar baseado em precipitação
                    return float(filtered['Precipitacao'].median()) if len(filtered) > 0 else 5.0
            else:
                # Estimativa baseada no mês
                dry_season_months = [7, 8, 9]
                return 15.0 if mes in dry_season_months else 5.0
                
        except Exception as e:
            logger.warning("Erro ao estimar dias sem chuva: %s", e)
            # Fallback: estimativa simples
            dry_season_months = [7, 8, 9]
            return 15.0 if mes in dry_season_months else 5.0
    
    def get_historical_climate_data(self, lat: float, lon: float, mes: int) -> Dict:
        """Obtém dados climáticos baseados em histórico do dataset."""
        precipitacao = 0.0
        dias_sem_chuva = self.estimate_days_without_rain(lat, lon, mes)
        
        if self.historical_data is not None:
            try:
                # Buscar dados próximos à coordenada no mesmo mês
                lat_range = 1.0
                lon_range = 1.0
                
                filtered = self.historical_data[
                    (abs(self.historical_data['Latitude'] - lat) <= lat_range) &
                    (abs(self.historical_data['Longitude'] - lon) <= lon_range) &
                    (self.historical_data['Mes'] == mes)
                ]
                
                if len(filtered) > 0:
                    precipitacao = float(filtered['Precipitacao'].median())
                    if 'DiaSemChuva' in filtered.columns:
                        dias_sem_chuva = float(filtered['DiaSemChuva'].median())
            except Exception as e:
                logger.warning("Erro ao buscar dados históricos: %s", e)
        
        return {
            'precipitacao': precipitacao,
            'dias_sem_chuva': dias_sem_chuva,
            'fonte': 'historico',
            'timestamp': datetime.now().isoformat(),
            'sucesso': True
        }
    
    def get_climate_data(
        self,
        lat: float,
        lon: float,
        mes: Optional[int] = None,
        use_api: bool = True,
        reference_date: Optional[datetime] = None,
        prefer_nasa: bool = True,
        use_hg_fallback: bool = True,
    ) -> Dict:
        """
        Obtém dados alinhados ao treino: prioriza NASA POWER (PRECTOTCORR + RH2M, janela 28d, ma7),
        depois HG Weather, depois amostra histórica do CSV.
        """
        if mes is None:
            mes = datetime.now().month
        if reference_date is None:
            reference_date = datetime.now()
        if not use_api:
            return self.get_historical_climate_data(lat, lon, mes)

        if prefer_nasa:
            nasa = fetch_nasa_power_window(lat, lon, reference_date, lookback_days=28, ma_window=7)
            if nasa and nasa.get("sucesso"):
                precip = float(nasa.get("precipitacao", 0.0) or 0.0)
                dsem = float(nasa.get("dias_sem_chuva", 0.0) or 0.0)
                pma7 = float(nasa.get("prec_ma7", precip) or 0.0)
                dma7 = float(nasa.get("diasem_ma7", dsem) or 0.0)
                out: Dict[str, Any] = {
                    "precipitacao": precip,
                    "dias_sem_chuva": dsem,
                    "dias_sem_chuva_estimado": None,
                    "prec_ma7": pma7,
                    "diasem_ma7": dma7,
                    "umidade": nasa.get("umidade"),
                    "temperatura": None,
                    "descricao": "NASA POWER MERRA-2 (reanálise, mesma família do enriquecimento de Umidade)",
                    "fonte": "nasa_power",
                    "timestamp": datetime.now().isoformat(),
                    "sucesso": True,
                    "janela_nasa": f"{nasa.get('janela_inicio')}-{nasa.get('janela_fim')}",
                }
                if nasa.get("ws2m_ma7_ms") is not None:
                    out["ws2m_ma7_ms"] = float(nasa["ws2m_ma7_ms"])
                if nasa.get("ws2m_hoje_ms") is not None:
                    out["ws2m_hoje_ms"] = float(nasa["ws2m_hoje_ms"])
                if nasa.get("t2m_max_ma7_c") is not None:
                    out["t2m_max_ma7_c"] = float(nasa["t2m_max_ma7_c"])
                logger.info(
                    "Clima NASA POWER: P=%.2f mm, P_ma7=%.2f, Dsem=%.1f, Dsem_ma7=%.1f, RH=%s",
                    precip, pma7, dsem, dma7, out.get("umidade"),
                )

                # Fusão com INMET (precipitação local) quando há estação ≤ 50 km.
                # INMET reflete medições in situ — mais fiel ao microclima do que
                # a célula MERRA-2 (~50 km). Mantemos NASA POWER para `umidade`
                # (RH) e como fallback, mas sobrescrevemos as colunas de chuva.
                if get_inmet_fusion_data is not None:
                    try:
                        inmet = get_inmet_fusion_data(
                            lat=lat,
                            lon=lon,
                            reference_date=reference_date,
                            lookback_days=7,
                        )
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("Falha na fusão INMET (segue só NASA): %s", exc)
                        inmet = None
                    if inmet:
                        merra_precip = pma7
                        merra_dsem = dma7
                        out["precipitacao_merra"] = precip
                        out["prec_ma7_merra"] = pma7
                        out["diasem_ma7_merra"] = dma7
                        out["precipitacao"] = float(inmet["precipitacao"])
                        out["prec_ma7"] = float(inmet["prec_ma7"])
                        out["dias_sem_chuva"] = float(inmet["dias_sem_chuva"])
                        out["diasem_ma7"] = float(inmet["diasem_ma7"])
                        out["fonte"] = "inmet_fundido_nasa_power"
                        out["fonte_inmet"] = inmet["fonte_inmet"]
                        out["estacao_inmet"] = inmet["estacao_inmet"]
                        out["distancia_estacao_inmet_km"] = inmet["distancia_estacao_inmet_km"]
                        out["cobertura_inmet_pct"] = inmet["cobertura_horas_pct"]
                        if inmet.get("vento_inmet_ms_ma7") is not None:
                            out["vento_inmet_ms_ma7"] = float(inmet["vento_inmet_ms_ma7"])
                        if inmet.get("inmet_representatividade"):
                            out["inmet_representatividade"] = inmet["inmet_representatividade"]
                        if inmet.get("inmet_busca_raio_km") is not None:
                            out["inmet_busca_raio_km"] = float(inmet["inmet_busca_raio_km"])
                        raio_txt = (
                            f"{inmet.get('inmet_busca_raio_km', INMET_MAX_DISTANCE_KM):.0f} km"
                        )
                        rep = inmet.get("inmet_representatividade", "—")
                        out["descricao"] = (
                            "Fusão: precipitação INMET (raio de busca "
                            f"{raio_txt}, representatividade {rep}) + RH/vento/temperatura NASA POWER MERRA-2"
                        )
                        logger.info(
                            "Fusão INMET (%s @ %.1f km, cob=%.0f%%, rep=%s): "
                            "P_ma7 %.2f → %.2f, Dsem_ma7 %.1f → %.1f mm/d",
                            inmet["estacao_inmet"]["codigo"],
                            inmet["distancia_estacao_inmet_km"],
                            inmet["cobertura_horas_pct"],
                            inmet.get("inmet_representatividade", "?"),
                            merra_precip, out["prec_ma7"],
                            merra_dsem, out["diasem_ma7"],
                        )
                return out

        if not use_hg_fallback:
            logger.info("Sem NASA e HG desligado — histórico CSV para lat=%.4f, lon=%.4f", lat, lon)
            return self.get_historical_climate_data(lat, lon, mes)

        api_data = self.get_climate_data_from_api(lat, lon)
        if api_data and api_data.get("sucesso"):
            precipitacao_api = api_data.get("precipitacao", 0.0)
            dias_sem_chuva_estimado = api_data.get("dias_sem_chuva_estimado")
            if dias_sem_chuva_estimado is not None:
                dias_sem_chuva = float(dias_sem_chuva_estimado)
            else:
                dias_sem_chuva = float(
                    self.estimate_days_without_rain(lat, lon, mes)
                )
            api_data["dias_sem_chuva"] = dias_sem_chuva
            p = float(precipitacao_api or 0.0)
            d = float(dias_sem_chuva)
            api_data["prec_ma7"] = p
            api_data["diasem_ma7"] = d
            api_data["fonte_hg"] = "hg_weather"
            if precipitacao_api is not None:
                logger.info(
                    "Dados obtidos da API HG (fallback): precip=%.2f, dsem=%.1f; ma7=proxy(valor instantâneo).",
                    p, d,
                )
                return api_data
            logger.warning("HG retornou sucesso sem precipitação. Usando fallback histórico.")

        logger.info("Usando dados históricos (amostra CSV) para lat=%.4f, lon=%.4f", lat, lon)
        h = self.get_historical_climate_data(lat, lon, mes)
        p0 = float(h.get("precipitacao", 0.0) or 0.0)
        d0 = float(h.get("dias_sem_chuva", 0.0) or 0.0)
        h["prec_ma7"] = p0
        h["diasem_ma7"] = d0
        h["fonte"] = h.get("fonte", "historico") or "historico"
        return h


# Instância global
climate_provider = ClimateDataProvider()

