"""Módulo para buscar dados de FRP (Fire Radiative Power) de APIs de satélite."""
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple

import numpy as np
import requests

logger = logging.getLogger(__name__)


class FRPDataProvider:
    """Classe para obter dados de FRP de APIs de satélite térmico."""

    def __init__(self, map_key: Optional[str] = None):
        # NASA FIRMS API endpoint
        # A API da NASA FIRMS requer MAP_KEY para autenticação
        # Formato: https://firms.modaps.eosdis.nasa.gov/api/country/csv/{MAP_KEY}/{source}/{country}/{date}
        # Ou: https://firms.modaps.eosdis.nasa.gov/api/area/csv/{MAP_KEY}/{source}/{north}/{south}/{east}/{west}/{date}
        # Sem MAP_KEY, a API retorna erro 500
        # Obter MAP_KEY de: https://firms.modaps.eosdis.nasa.gov/mapserver/mapkey_status/
        # A MAP_KEY é gratuita e pode ser obtida registrando-se no site da NASA FIRMS
        try:
            from config import NASA_FIRMS_MAP_KEY
            config_key = NASA_FIRMS_MAP_KEY
        except ImportError:
            config_key = None
        
        self.map_key = map_key or config_key or os.getenv('NASA_FIRMS_MAP_KEY', None)
        
        # Endpoints da API
        if self.map_key:
            self.firms_api_country = f"https://firms.modaps.eosdis.nasa.gov/api/country/csv/{self.map_key}"
            self.firms_api_area = f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{self.map_key}"
            logger.info("NASA FIRMS MAP_KEY configurada")
        else:
            # Sem MAP_KEY, não podemos usar a API
            self.firms_api_country = None
            self.firms_api_area = None
            logger.warning("NASA FIRMS MAP_KEY não configurada. API não estará disponível.")
            logger.info("Para obter uma MAP_KEY: https://firms.modaps.eosdis.nasa.gov/mapserver/mapkey_status/")
        
        self.timeout = 15

    # Limite atual da NASA FIRMS (NRT): 1..5 dias por chamada.
    # Referência empírica (HTTP 400 "Invalid day range. Expects [1..5].").
    FIRMS_MAX_DAYS = 5

    def get_frp_from_nasa_firms(
        self,
        lat: float,
        lon: float,
        radius_km: float = 10.0,
        days_back: int = 5,
        reference_date: Optional[datetime] = None,
    ) -> Optional[Dict]:
        """
        Busca dados de FRP da NASA FIRMS para uma coordenada.

        Args:
            lat: Latitude
            lon: Longitude
            radius_km: Raio de busca em km (padrão: 10km)
            days_back: Número de dias para buscar dados. NRT aceita 1..5; valores
                maiores são truncados em 5 com aviso.
            reference_date: Data final da janela (opcional). Se omitido, a API
                retorna os últimos `days_back` dias até "hoje".

        Returns:
            Dicionário com dados de FRP ou None se não encontrar
        """
        try:
            if days_back > self.FIRMS_MAX_DAYS:
                logger.info(
                    "days_back=%d excede limite FIRMS NRT (%d). Truncando.",
                    days_back, self.FIRMS_MAX_DAYS,
                )
                days_back = self.FIRMS_MAX_DAYS
            if days_back < 1:
                days_back = 1
            # Calcular janela explícita (apenas para logs/diagnóstico)
            end_date = reference_date or datetime.now()
            start_date = end_date - timedelta(days=days_back)
            start_date_str = start_date.strftime("%Y-%m-%d")
            end_date_str = end_date.strftime("%Y-%m-%d")

            # API da NASA FIRMS - usar endpoint de dados MODIS ou VIIRS
            # Formato: /country/country_name/modis/{start_date}/{end_date}
            # Ou usar endpoint de área: /area/{source}/{north}/{south}/{east}/{west}/{start_date}/{end_date}
            
            # Calcular bounding box simples (aproximado)
            # 1 grau ≈ 111 km
            lat_offset = radius_km / 111.0
            lon_offset = radius_km / (111.0 * abs(np.cos(np.radians(lat))))

            north = lat + lat_offset
            south = lat - lat_offset
            east = lon + lon_offset
            west = lon - lon_offset

            # Verificar se temos MAP_KEY configurada
            if not self.map_key:
                logger.warning("NASA FIRMS MAP_KEY não configurada. Não é possível buscar dados de FRP.")
                logger.info("Configure a variável de ambiente NASA_FIRMS_MAP_KEY ou passe map_key no construtor.")
                return {
                    'frp': 0.0,
                    'frp_max': 0.0,
                    'frp_mean': 0.0,
                    'detections': 0,
                    'fonte': 'nasa_firms_fallback',
                    'sucesso': True,
                    'erro': 'MAP_KEY não configurada',
                }
            
            # NASA FIRMS API: formato correto com MAP_KEY
            # Formato correto: /area/csv/{MAP_KEY}/{dataset}/{days}/{west},{south},{east},{north}
            # ONDE:
            #   - {dataset} pode ser: 'VIIRS_SNPP_NRT', 'MODIS_NRT', 'VIIRS_NOAA20_NRT', etc.
            #   - {days} é o número de dias (não a data!)
            #   - {west},{south},{east},{north} são coordenadas com vírgulas
            # NOTA: A API não usa data específica, mas sim número de dias de histórico
            
            # Tentar múltiplas fontes (datasets)
            # VIIRS_SNPP_NRT: mais recente, melhor resolução (NRT = Near Real Time)
            # MODIS_NRT: dados mais antigos mas estáveis
            sources_to_try = ['VIIRS_SNPP_NRT', 'MODIS_NRT']
            
            # Número de dias para buscar (a API retorna dados dos últimos N dias)
            days_to_request = min(days_back, self.FIRMS_MAX_DAYS)
            
            logger.info(
                "Consultando NASA FIRMS API: lat=%.4f, lon=%.4f, raio=%.1fkm, período=%s a %s",
                lat, lon, radius_km, start_date_str, end_date_str
            )
            
            response = None
            data = None
            url_used = None
            
            # Sufixo opcional /YYYY-MM-DD para janela ancorada em data passada.
            # Sem data → últimos N dias até hoje.
            date_suffix = f"/{end_date_str}" if reference_date is not None else ""

            # Tentar diferentes fontes (datasets)
            for source in sources_to_try:
                try:
                    # Formato correto da API NASA FIRMS:
                    # /api/area/csv/{MAP_KEY}/{dataset}/{west},{south},{east},{north}/{days}[/{date}]
                    area_params = f"{west:.4f},{south:.4f},{east:.4f},{north:.4f}"
                    url = f"{self.firms_api_area}/{source}/{area_params}/{days_to_request}{date_suffix}"
                    url_used = url
                    
                    logger.info("Tentando NASA FIRMS: %s para últimos %d dias", source, days_to_request)
                    logger.info("URL completa: %s", url)
                    response = requests.get(url, timeout=self.timeout)
                    
                    logger.info("Resposta NASA FIRMS: status=%d", response.status_code)
                    
                    if response.status_code == 200:
                        data = response.text
                        # Verificar se a resposta contém dados válidos ou apenas mensagem de erro
                        if data and len(data.strip()) > 0:
                            # Verificar se não é uma mensagem de erro
                            if 'Invalid' not in data and 'Error' not in data and 'invalid' not in data.lower():
                                logger.info("Sucesso! Dados obtidos de %s para últimos %d dias", source, days_to_request)
                                break
                            else:
                                # Resposta contém erro, tentar próxima fonte
                                logger.warning("Resposta contém erro: %s", data[:200])
                                continue
                        else:
                            # Resposta vazia, tentar próxima fonte
                            logger.debug("Resposta vazia, tentando próxima fonte...")
                            continue
                    elif response.status_code == 404:
                        # Dataset não disponível, tentar próxima
                        logger.debug("Dataset %s não disponível (404), tentando próxima...", source)
                        continue
                    else:
                        response.raise_for_status()
                        
                except requests.exceptions.RequestException as e:
                    logger.warning("Erro ao buscar %s: %s", source, e)
                    continue
            
            # Se ainda não conseguiu por área, tentar país (Brasil) como último recurso
            if not data or not response or (response and response.status_code != 200) or (data and ('Invalid' in data or 'Error' in data or 'invalid' in data.lower())):
                logger.info("Endpoint de área não funcionou, tentando endpoint de país (Brasil) como alternativa...")
                for source in sources_to_try:
                    try:
                        # Formato para país: /api/country/csv/{MAP_KEY}/{dataset}/{country}/{days}[/{date}]
                        url = f"{self.firms_api_country}/{source}/BR/{days_to_request}{date_suffix}"
                        url_used = url
                        
                        logger.info("Tentando NASA FIRMS (Brasil): %s para últimos %d dias", source, days_to_request)
                        logger.info("URL completa: %s", url)
                        response = requests.get(url, timeout=self.timeout)
                        
                        logger.info("Resposta NASA FIRMS: status=%d", response.status_code)
                        
                        if response.status_code == 200:
                            data = response.text
                            if data and len(data.strip()) > 0:
                                if 'Invalid' not in data and 'Error' not in data and 'invalid' not in data.lower():
                                    logger.info("Sucesso! Dados obtidos de %s (Brasil) para últimos %d dias", source, days_to_request)
                                    # Filtrar dados pela área de interesse
                                    data = self._filter_data_by_area(data, lat, lon, radius_km)
                                    break
                                else:
                                    logger.warning("Resposta contém erro: %s", data[:200])
                                    continue
                            else:
                                continue
                        elif response.status_code == 404:
                            continue
                        else:
                            response.raise_for_status()
                            
                    except requests.exceptions.RequestException as e:
                        logger.warning("Erro ao buscar %s (Brasil): %s", source, e)
                        continue
            
            # Se ainda não conseguiu, retornar sem dados
            if not data or not response or response.status_code != 200:
                logger.warning("Não foi possível obter dados da NASA FIRMS após tentar múltiplas fontes e datas")
                if response:
                    logger.warning("Último status: %d, URL: %s", response.status_code, url_used)
                else:
                    logger.warning("Nenhuma resposta recebida. Verifique a MAP_KEY e conexão.")
                return {
                    'frp': 0.0,
                    'frp_max': 0.0,
                    'frp_mean': 0.0,
                    'detections': 0,
                    'fonte': 'nasa_firms_fallback',
                    'sucesso': True,
                    'erro': f'API retornou status {response.status_code if response else "N/A"}',
                }

            # data já foi atribuído acima quando response.status_code == 200
            logger.info("Resposta NASA FIRMS (primeiros 500 chars): %s", data[:500] if data else "vazio")

            if not data or len(data.strip()) == 0:
                logger.info("Nenhum dado de FRP encontrado na área (resposta vazia)")
                return {
                    'frp': 0.0,
                    'frp_max': 0.0,
                    'frp_mean': 0.0,
                    'detections': 0,
                    'fonte': 'nasa_firms',
                    'sucesso': True,
                }

            # Parse do CSV retornado
            lines = data.strip().split('\n')
            logger.info("Total de linhas na resposta: %d", len(lines))
            
            if len(lines) < 2:  # Header + dados
                logger.info("Nenhum dado de FRP encontrado na área (menos de 2 linhas)")
                return {
                    'frp': 0.0,
                    'frp_max': 0.0,
                    'frp_mean': 0.0,
                    'detections': 0,
                    'fonte': 'nasa_firms',
                    'sucesso': True,
                }

            # Parse do header para encontrar índice da coluna FRP
            header = lines[0]
            header_parts = [h.strip() for h in lines[0].split(',')]
            frp_col_idx = None
            
            logger.info("Header completo: %s", header)
            logger.info("Colunas encontradas (%d): %s", len(header_parts), header_parts)
            
            for i, col in enumerate(header_parts):
                if 'frp' in col.lower():
                    frp_col_idx = i
                    logger.info("Coluna FRP encontrada no índice %d: '%s'", i, col)
                    break
            
            if frp_col_idx is None and len(header_parts) >= 13:
                # Baseado no header: latitude,longitude,bright_ti4,scan,track,acq_date,acq_time,satellite,instrument,confidence,version,bright_ti5,frp,daynight
                # FRP está no índice 12 (0-indexed)
                frp_col_idx = 12
                logger.info("Coluna FRP não encontrada no header, usando índice padrão 12 (baseado no formato padrão)")
            elif frp_col_idx is None:
                logger.warning("Não foi possível identificar coluna FRP. Header: %s", header[:200])
                frp_col_idx = -1  # Usar última coluna

            # Processar linhas (pular header)
            frp_values = []
            linhas_processadas = 0
            
            for line_idx, line in enumerate(lines[1:], start=2):
                if not line.strip():
                    continue
                
                parts = [p.strip() for p in line.split(',')]
                linhas_processadas += 1
                
                if len(parts) >= 6:
                    try:
                        # Tentar obter FRP da coluna identificada
                        if frp_col_idx >= 0 and frp_col_idx < len(parts):
                            frp_str = parts[frp_col_idx]
                        elif frp_col_idx == -1:
                            # Usar última coluna
                            frp_str = parts[-1]
                        else:
                            continue
                        
                        frp_val = float(frp_str)
                        
                        if frp_val > 0:
                            frp_values.append(frp_val)
                            logger.debug("FRP encontrado na linha %d: %.2f MW", line_idx, frp_val)
                    except (ValueError, IndexError) as e:
                        logger.debug("Erro ao processar linha %d de FRP: %s (parts: %s)", 
                                    line_idx, e, parts[:5] if len(parts) > 5 else parts)
                        continue
                
            logger.info("Linhas processadas: %d | Valores FRP encontrados: %d", linhas_processadas, len(frp_values))

            if not frp_values:
                logger.info("Nenhum valor de FRP > 0 encontrado na área")
                return {
                    'frp': 0.0,
                    'frp_max': 0.0,
                    'frp_mean': 0.0,
                    'detections': 0,
                    'fonte': 'nasa_firms',
                    'sucesso': True,
                }

            # Calcular estatísticas
            frp_max = max(frp_values)
            frp_mean = sum(frp_values) / len(frp_values)
            # Usar média ou máximo? Para previsão de risco, máximo pode ser mais relevante
            # Mas média pode ser mais estável
            frp_representativo = frp_max  # Usar máximo para indicar pior caso

            logger.info(
                "FRP encontrado: max=%.2f MW, média=%.2f MW, detecções=%d",
                frp_max, frp_mean, len(frp_values)
            )

            return {
                'frp': float(frp_representativo),
                'frp_max': float(frp_max),
                'frp_mean': float(frp_mean),
                'detections': len(frp_values),
                'fonte': 'nasa_firms',
                'sucesso': True,
                'periodo_dias': days_back,
            }

        except requests.exceptions.RequestException as e:
            logger.warning("Erro ao consultar NASA FIRMS API: %s", e)
            return None
        except Exception as e:
            logger.error("Erro inesperado ao obter FRP: %s", e, exc_info=True)
            return None

    def get_frp(
        self,
        lat: float,
        lon: float,
        radius_km: float = 10.0,
        days_back: int = 7,
    ) -> float:
        """
        Obtém valor de FRP para uma coordenada.

        Args:
            lat: Latitude
            lon: Longitude
            radius_km: Raio de busca em km
            days_back: Dias para buscar dados

        Returns:
            Valor de FRP (MW) ou 0.0 se não encontrar
        """
        frp_data = self.get_frp_from_nasa_firms(lat, lon, radius_km, days_back)
        
        if frp_data and frp_data.get('sucesso'):
            return frp_data.get('frp', 0.0)
        
        return 0.0
    
    def _filter_data_by_area(self, csv_data: str, lat: float, lon: float, radius_km: float) -> str:
        """
        Filtra dados CSV da NASA FIRMS para uma área específica.
        Usado quando obtemos dados do Brasil inteiro.
        
        Args:
            csv_data: Dados CSV retornados pela API
            lat: Latitude do centro
            lon: Longitude do centro
            radius_km: Raio em km
        
        Returns:
            CSV filtrado ou vazio se nenhum dado estiver na área
        """
        try:
            lines = csv_data.strip().split('\n')
            if len(lines) < 2:
                return ''
            
            # Calcular bounding box
            lat_offset = radius_km / 111.0
            lon_offset = radius_km / (111.0 * abs(np.cos(np.radians(lat))))
            
            north = lat + lat_offset
            south = lat - lat_offset
            east = lon + lon_offset
            west = lon - lon_offset
            
            # Header
            filtered_lines = [lines[0]]
            
            # Filtrar linhas por coordenadas (latitude e longitude geralmente estão nas primeiras colunas)
            for line in lines[1:]:
                if not line.strip():
                    continue
                parts = line.split(',')
                if len(parts) >= 2:
                    try:
                        line_lat = float(parts[0])  # Primeira coluna geralmente é latitude
                        line_lon = float(parts[1])  # Segunda coluna geralmente é longitude
                        
                        if south <= line_lat <= north and west <= line_lon <= east:
                            filtered_lines.append(line)
                    except (ValueError, IndexError):
                        continue
            
            return '\n'.join(filtered_lines) if len(filtered_lines) > 1 else ''
            
        except Exception as e:
            logger.warning("Erro ao filtrar dados por área: %s", e)
            return csv_data  # Retornar dados originais em caso de erro


# Instância global (tentará obter MAP_KEY de variável de ambiente)
frp_provider = FRPDataProvider()

