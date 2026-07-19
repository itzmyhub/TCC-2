"""
Script para enriquecer a base de dados com dados históricos reais de umidade relativa do ar.

Este script busca dados de umidade de fontes confiáveis (INMET, NASA, APIs meteorológicas)
e adiciona à base de dados existente.

Fontes possíveis:
- INMET (Instituto Nacional de Meteorologia) - dados históricos brasileiros
- NASA POWER API - dados de reanálise global
- OpenWeatherMap Historical API
- Outras APIs meteorológicas

IMPORTANTE: Este script deve ser executado ANTES do treinamento para enriquecer
a base de dados com dados reais de umidade.
"""

import logging
import os
import signal
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Tuple, Set
from datetime import datetime, timedelta
import requests
import time
import json
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from threading import Lock, Semaphore

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR.parent / 'base_de_dados.csv'
OUTPUT_PATH = BASE_DIR.parent / 'base_de_dados_com_umidade.csv'
CACHE_DIR = BASE_DIR.parent / '.cache_umidade'
CACHE_DIR.mkdir(exist_ok=True)
CACHE_FILE = CACHE_DIR / 'umidade_cache.json'

# Cache global em memória
_umidade_cache: Dict[str, Optional[float]] = {}

# Rate limiting
# NASA POWER API limits (sem chave registrada):
# - 30 requisições/hora por IP
# - 50 requisições/dia por IP
# Com chave registrada: ~1000 requisições/hora
# Usaremos limite conservador baseado na presença de chave API

# Verificar se tem chave(s) API
try:
    from config import NASA_POWER_API_KEY, NASA_POWER_API_KEYS
    # Suportar tanto uma chave única quanto múltiplas chaves
    if hasattr(NASA_POWER_API_KEY, '__iter__') and not isinstance(NASA_POWER_API_KEY, str):
        # Se for lista/tupla, usar como múltiplas chaves
        API_KEYS = list(NASA_POWER_API_KEY) if NASA_POWER_API_KEY else []
    elif NASA_POWER_API_KEY:
        API_KEYS = [NASA_POWER_API_KEY]
    else:
        API_KEYS = []
    
    # Verificar se há múltiplas chaves em NASA_POWER_API_KEYS
    if hasattr(NASA_POWER_API_KEYS, '__iter__') and not isinstance(NASA_POWER_API_KEYS, str):
        API_KEYS.extend(list(NASA_POWER_API_KEYS))
    elif NASA_POWER_API_KEYS:
        API_KEYS.append(NASA_POWER_API_KEYS)
        
except (ImportError, AttributeError):
    # Fallback: usar chaves hardcoded se não estiverem em config
    API_KEYS = [
        'K5GKN9bHblP1I73AWT8ZjbLW8ZeuGrV00WUUn9og',
        'BuERPEkGGVJdWiPqsEhQzxRclsJ0xSE3bpDbCl0b'
    ]

# Remover duplicatas mantendo ordem
API_KEYS = list(dict.fromkeys(API_KEYS))
HAS_API_KEY = len(API_KEYS) > 0
NUM_API_KEYS = len(API_KEYS)

if HAS_API_KEY:
    # Com múltiplas chaves, podemos aumentar o rate limit proporcionalmente
    # Cada chave = 1000 req/hora, mas vamos ser conservadores
    REQUESTS_PER_HOUR_PER_KEY = 1000
    REQUESTS_PER_MINUTE = min(16 * NUM_API_KEYS, 50)  # Máximo 50 req/min para evitar sobrecarga
    REQUESTS_PER_HOUR = REQUESTS_PER_HOUR_PER_KEY * NUM_API_KEYS
    logger.info(
        "Chave(s) API NASA POWER detectada(s): %d chave(s). "
        "Limite: %d req/min (~%d req/hora total)",
        NUM_API_KEYS, REQUESTS_PER_MINUTE, REQUESTS_PER_HOUR
    )
else:
    REQUESTS_PER_MINUTE = 1  # 30 req/hora / 60 min = ~0.5 req/min, usando 1 para ser conservador
    REQUESTS_PER_HOUR = 30   # Limite sem chave API
    logger.warning(
        "Chave API NASA POWER não encontrada. "
        "Limite sem chave: %d req/min (~%d req/hora). "
        "Para aumentar limite, configure NASA_POWER_API_KEY ou NASA_POWER_API_KEYS em config.py. "
        "Obter chave grátis em: https://api.nasa.gov/",
        REQUESTS_PER_MINUTE, REQUESTS_PER_HOUR
    )

# Índice para rotação de chaves
_api_key_index = 0
_api_key_lock = Lock()

_rate_limit_lock = Lock()
_rate_limit_times = []  # Timestamps das últimas requisições
_rate_limit_semaphore = Semaphore(REQUESTS_PER_MINUTE)


class APIRequestFailure(Exception):
    """Exceção levantada quando há falha crítica na requisição à API.
    
    Diferencia falhas reais (que devem parar o processamento) de dados não encontrados
    (que podem continuar processando).
    """
    pass


def _cache_key(lat: float, lon: float, data: datetime) -> str:
    """Gera chave única para cache baseada em lat, lon e data."""
    data_str = data.strftime('%Y%m%d')
    # Arredondar coordenadas para reduzir cache (mesma área ~0.01 grau = ~1km)
    lat_rounded = round(lat, 2)
    lon_rounded = round(lon, 2)
    return f"{lat_rounded:.2f}_{lon_rounded:.2f}_{data_str}"


def _load_cache() -> Dict[str, Optional[float]]:
    """Carrega cache de disco."""
    if CACHE_FILE.exists():
        try:
            with open(CACHE_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.warning("Erro ao carregar cache: %s", e)
            backup = CACHE_FILE.with_suffix(
                f".json.broken.{int(time.time())}"
            )
            try:
                CACHE_FILE.rename(backup)
                logger.warning(
                    "Cache inválido movido para %s — recomeçando cache vazio.",
                    backup,
                )
            except OSError:
                logger.warning("Não foi possível renomear cache corrompido.")
    return {}


def _save_cache(cache: Dict[str, Optional[float]]) -> None:
    """Salva cache em disco."""
    try:
        with open(CACHE_FILE, 'w') as f:
            json.dump(cache, f)
    except Exception as e:
        logger.warning("Erro ao salvar cache: %s", e)


def _get_next_api_key() -> Optional[str]:
    """Obtém próxima chave API usando rotação round-robin."""
    global _api_key_index
    
    if not API_KEYS:
        return None
    
    with _api_key_lock:
        key = API_KEYS[_api_key_index]
        _api_key_index = (_api_key_index + 1) % len(API_KEYS)
        return key


def _wait_for_rate_limit():
    """Aguarda respeitando rate limit da API."""
    global _rate_limit_times
    
    with _rate_limit_lock:
        now = time.time()
        # Remover timestamps antigos (> 1 minuto)
        _rate_limit_times = [t for t in _rate_limit_times if now - t < 60]
        
        # Se atingiu o limite, aguardar
        if len(_rate_limit_times) >= REQUESTS_PER_MINUTE:
            sleep_time = 60 - (now - _rate_limit_times[0]) + 0.1
            if sleep_time > 0:
                logger.debug("Rate limit atingido, aguardando %.1f segundos...", sleep_time)
                time.sleep(sleep_time)
                # Remover timestamps antigos novamente
                _rate_limit_times = [t for t in _rate_limit_times if time.time() - t < 60]
        
        # Registrar nova requisição
        _rate_limit_times.append(time.time())


def buscar_umidade_nasa_power(
    lat: float, 
    lon: float, 
    data: datetime, 
    use_cache: bool = True,
    max_retries: int = 3,
    base_delay: float = 1.0
) -> Optional[float]:
    """
    Busca umidade relativa da NASA POWER API.
    
    NASA POWER (Prediction Of Worldwide Energy Resources) fornece dados meteorológicos
    históricos baseados em reanálise e observações.
    
    API: https://power.larc.nasa.gov/api/pages/
    
    Args:
        lat: Latitude
        lon: Longitude
        data: Data
        use_cache: Se True, usa cache para evitar requisições repetidas
        max_retries: Número máximo de tentativas em caso de falha
        base_delay: Delay base para backoff exponencial
    
    Returns:
        float: Umidade relativa (%%) se encontrada, None se não existir dados para a data
        
    Raises:
        APIRequestFailure: Se houver falha crítica na requisição após todas as tentativas
    """
    # Verificar cache primeiro
    if use_cache:
        cache_key = _cache_key(lat, lon, data)
        if cache_key in _umidade_cache:
            return _umidade_cache[cache_key]
    
    data_str = data.strftime('%Y%m%d')
    
    # Retry com backoff exponencial
    for attempt in range(max_retries + 1):
        try:
            # Respeitar rate limit
            _wait_for_rate_limit()
            
            # Formato da API NASA POWER
            # https://power.larc.nasa.gov/api/temporal/daily/point?parameters=RH2M&community=AG&longitude={lon}&latitude={lat}&start={YYYYMMDD}&end={YYYYMMDD}&format=JSON
            
            url = "https://power.larc.nasa.gov/api/temporal/daily/point"
            params = {
                'parameters': 'RH2M',  # Relative Humidity at 2 Meters
                'community': 'AG',  # Agriculture
                'longitude': lon,
                'latitude': lat,
                'start': data_str,
                'end': data_str,
                'format': 'JSON'
            }
            
            # Adicionar chave API se disponível (usando rotação)
            # Nota: NASA POWER pode não usar chave API da mesma forma que outras APIs NASA
            # Mas vamos tentar adicionar se estiver disponível
            api_key = _get_next_api_key()
            if api_key:
                params['key'] = api_key
            
            response = requests.get(url, params=params, timeout=30)  # Aumentado para 30s
            status_code = response.status_code
            
            logger.debug("NASA POWER request: lat=%.4f, lon=%.4f, data=%s, status=%d (tentativa %d/%d)", 
                        lat, lon, data_str, status_code, attempt + 1, max_retries + 1)
            
            # Verificar rate limit (429 Too Many Requests)
            if status_code == 429:
                retry_after = response.headers.get('Retry-After')
                if retry_after:
                    wait_time = int(retry_after)
                else:
                    # Backoff exponencial: 2^attempt * base_delay
                    wait_time = (2 ** attempt) * base_delay * 60  # Converter para segundos
                
                if attempt < max_retries:
                    logger.warning(
                        "Rate limit atingido (429) para lat=%.4f, lon=%.4f. "
                        "Aguardando %d segundos antes de retry... (tentativa %d/%d)",
                        lat, lon, wait_time, attempt + 1, max_retries + 1
                    )
                    time.sleep(wait_time)
                    continue
                else:
                    # Rate limit persistente após todas as tentativas = FALHA CRÍTICA
                    logger.error(
                        "Rate limit atingido (429) após %d tentativas para lat=%.4f, lon=%.4f, data=%s. "
                        "FALHA CRÍTICA: Parando processamento.",
                        max_retries + 1, lat, lon, data_str
                    )
                    raise APIRequestFailure(
                        f"Rate limit persistente (429) após {max_retries + 1} tentativas. "
                        f"Recomendado: reduzir workers ou aguardar antes de continuar."
                    )
            
            # Outros erros HTTP (503 Service Unavailable, etc.)
            elif status_code >= 500:
                if attempt < max_retries:
                    wait_time = (2 ** attempt) * base_delay
                    logger.warning(
                        "Erro do servidor (%d) para lat=%.4f, lon=%.4f. "
                        "Aguardando %.1f segundos antes de retry... (tentativa %d/%d)",
                        status_code, lat, lon, wait_time, attempt + 1, max_retries + 1
                    )
                    time.sleep(wait_time)
                    continue
                else:
                    # Erro do servidor persistente após todas as tentativas = FALHA CRÍTICA
                    logger.error(
                        "Erro do servidor (%d) após %d tentativas para lat=%.4f, lon=%.4f, data=%s. "
                        "FALHA CRÍTICA: Parando processamento.",
                        status_code, max_retries + 1, lat, lon, data_str
                    )
                    raise APIRequestFailure(
                        f"Erro do servidor ({status_code}) persistente após {max_retries + 1} tentativas."
                    )
            
            # Verificar cabeçalhos de rate limit se disponíveis
            if 'X-RateLimit-Remaining' in response.headers:
                remaining = int(response.headers['X-RateLimit-Remaining'])
                if remaining < 10:
                    logger.warning(
                        "Poucas requisições restantes no rate limit: %d. "
                        "Considere reduzir workers ou aguardar.",
                        remaining
                    )
            
            # Sucesso!
            if status_code == 200:
                data_json = response.json()
                
                # Formato da resposta: {"properties": {"parameter": {"RH2M": {"20240101": 85.09}}}}
                if 'properties' in data_json and 'parameter' in data_json['properties']:
                    rh2m = data_json['properties']['parameter'].get('RH2M', {})
                    
                    # RH2M é um dicionário com datas como chaves: {"20240101": 85.09}
                    if isinstance(rh2m, dict):
                        if data_str in rh2m:
                            valor = rh2m[data_str]
                            if valor is not None and valor != -999.0:  # -999 é fill_value da API
                                resultado = float(valor)
                                # Salvar no cache
                                if use_cache:
                                    cache_key = _cache_key(lat, lon, data)
                                    _umidade_cache[cache_key] = resultado
                                logger.debug("Umidade encontrada: %.2f%% para %s", resultado, data_str)
                                return resultado
                            else:
                                logger.debug("Valor inválido (None ou -999) para %s", data_str)
                        elif len(rh2m) > 0:
                            # Se não encontrar a data exata, pegar o primeiro valor disponível
                            valores = [v for v in rh2m.values() if v is not None and v != -999.0]
                            if valores:
                                resultado = float(valores[0])
                                # Salvar no cache
                                if use_cache:
                                    cache_key = _cache_key(lat, lon, data)
                                    _umidade_cache[cache_key] = resultado
                                logger.debug("Usando primeiro valor disponível: %.2f%%", resultado)
                                return resultado
                        else:
                            # RH2M vazio = dado não existe para essa data (NÃO é falha)
                            logger.debug("RH2M vazio para lat=%.4f, lon=%.4f, data=%s (dado não existe)", lat, lon, data_str)
                            # Retornar None para indicar que não há dados, mas não é falha
                            # Salvar None no cache para não tentar novamente
                            if use_cache:
                                cache_key = _cache_key(lat, lon, data)
                                _umidade_cache[cache_key] = None
                            return None
                    else:
                        logger.debug("RH2M não é um dicionário: %s", type(rh2m))
                else:
                    logger.debug("Estrutura de resposta inesperada. Keys: %s", list(data_json.keys()) if isinstance(data_json, dict) else 'não é dict')
                
                # Se chegou aqui, API retornou 200 mas não conseguiu extrair dados válidos
                # Isso pode indicar que o dado não existe ou estrutura inesperada
                # Como a API retornou 200, não é falha crítica - continuar processamento
                logger.debug("Dado não encontrado para lat=%.4f, lon=%.4f, data=%s (estrutura inesperada ou sem dados)", lat, lon, data_str)
                if use_cache:
                    cache_key = _cache_key(lat, lon, data)
                    _umidade_cache[cache_key] = None
                return None
            
            # Outros status codes (401, 403, 404, etc.)
            elif status_code not in (429, 500, 502, 503, 504):
                # 404 = recurso não encontrado (pode não ter dados) - NÃO é falha crítica
                if status_code == 404:
                    logger.debug(
                        "Recurso não encontrado (404) para lat=%.4f, lon=%.4f, data=%s (dado pode não existir). Continuando...",
                        lat, lon, data_str
                    )
                    if use_cache:
                        cache_key = _cache_key(lat, lon, data)
                        _umidade_cache[cache_key] = None
                    return None
                # Outros 4xx (401, 403, etc.) podem indicar problema de autenticação/acesso = FALHA CRÍTICA
                else:
                    logger.error(
                        "Erro HTTP %d para lat=%.4f, lon=%.4f, data=%s. FALHA CRÍTICA: Parando processamento.",
                        status_code, lat, lon, data_str
                    )
                    raise APIRequestFailure(
                        f"Erro HTTP {status_code} na requisição. Verifique autenticação/acesso à API."
                    )
            
        except requests.exceptions.Timeout:
            if attempt < max_retries:
                wait_time = (2 ** attempt) * base_delay
                logger.warning(
                    "Timeout para lat=%.4f, lon=%.4f, data=%s. Aguardando %.1f segundos antes de retry... (tentativa %d/%d)",
                    lat, lon, data_str, wait_time, attempt + 1, max_retries + 1
                )
                time.sleep(wait_time)
                continue
            else:
                # Timeout persistente após todas as tentativas = FALHA CRÍTICA
                logger.error(
                    "Timeout persistente após %d tentativas para lat=%.4f, lon=%.4f, data=%s. "
                    "FALHA CRÍTICA: Parando processamento.",
                    max_retries + 1, lat, lon, data_str
                )
                raise APIRequestFailure(
                    f"Timeout persistente após {max_retries + 1} tentativas. "
                    "Verifique conexão com a API."
                )
                
        except requests.exceptions.RequestException as e:
            if attempt < max_retries:
                wait_time = (2 ** attempt) * base_delay
                logger.warning(
                    "Erro de conexão para lat=%.4f, lon=%.4f, data=%s: %s. "
                    "Aguardando %.1f segundos antes de retry... (tentativa %d/%d)",
                    lat, lon, data_str, str(e), wait_time, attempt + 1, max_retries + 1
                )
                time.sleep(wait_time)
                continue
            else:
                # Erro de conexão persistente após todas as tentativas = FALHA CRÍTICA
                logger.error(
                    "Erro de conexão persistente após %d tentativas para lat=%.4f, lon=%.4f, data=%s: %s. "
                    "FALHA CRÍTICA: Parando processamento.",
                    max_retries + 1, lat, lon, data_str, str(e)
                )
                raise APIRequestFailure(
                    f"Erro de conexão persistente após {max_retries + 1} tentativas: {str(e)}"
                ) from e
                 
        except APIRequestFailure:
            # Re-levantar exceções de falha crítica
            raise
                 
        except Exception as e:
            # Erro inesperado = FALHA CRÍTICA
            logger.error(
                "Erro inesperado ao buscar umidade NASA POWER para lat=%.4f, lon=%.4f, data=%s: %s. "
                "FALHA CRÍTICA: Parando processamento.",
                lat, lon, data_str, e
            )
            raise APIRequestFailure(
                f"Erro inesperado na requisição: {str(e)}"
            ) from e
    
    # Este ponto nunca deveria ser alcançado se o código acima estiver correto
    # Mas por segurança, se chegar aqui sem dados e sem exceção, assumir que dado não existe
    logger.warning(
        "Fim inesperado do loop de retry para lat=%.4f, lon=%.4f, data=%s. "
        "Assumindo que dado não existe.",
        lat, lon, data_str
    )
    if use_cache:
        cache_key = _cache_key(lat, lon, data)
        _umidade_cache[cache_key] = None
    return None


def buscar_umidade_inmet(lat: float, lon: float, data: datetime) -> Optional[float]:
    """
    Busca umidade relativa do INMET (Instituto Nacional de Meteorologia).
    
    NOTA: A API pública do INMET pode ter limitações. Verificar documentação atual.
    """
    try:
        # INMET API - formato pode variar
        # Exemplo: https://apitempo.inmet.gov.br/estacao/{codigo}/{data_inicio}/{data_fim}
        # Requer código de estação meteorológica próxima
        
        # Por enquanto, retornar None - implementar quando tiver acesso à API
        logger.debug("Busca INMET não implementada ainda - requer código de estação")
        return None
    except Exception as e:
        logger.debug("Erro ao buscar umidade INMET: %s", e)
        return None


def enriquecer_dataset_com_umidade(
    input_path: Path,
    output_path: Path,
    fonte: str = 'nasa_power',
    amostra: Optional[int] = None,
    delay_entre_requisicoes: float = 0.1,
    max_workers: int = 5,
    salvar_cache_periodicamente: int = 100,
    resume: bool = True,
) -> pd.DataFrame:
    """
    Enriquece dataset com dados reais de umidade relativa.

    Resume automaticamente a partir do ponto de interrupção:
    - Se ``output_path`` já existir com registros parcialmente enriquecidos, esses
      valores são preservados e apenas as linhas ainda sem umidade são processadas.
    - O cache em disco é salvo a cada ``salvar_cache_periodicamente`` requisições E
      ao receber SIGINT/KeyboardInterrupt, garantindo que nenhum trabalho seja perdido.

    Args:
        input_path: Caminho para o CSV original (sem coluna Umidade).
        output_path: Caminho para salvar/retomar CSV enriquecido.
        fonte: Fonte de dados ('nasa_power', 'inmet', etc.)
        amostra: Se fornecido, processa apenas N primeiras linhas (útil para testes).
        delay_entre_requisicoes: Delay em segundos entre requisições (padrão: 0.1).
        max_workers: Número de workers para processamento paralelo.
        salvar_cache_periodicamente: Salvar cache e CSV a cada N requisições (padrão: 100).
        resume: Se True (padrão), retoma a partir do output parcial sem perguntar.

    Returns:
        DataFrame enriquecido com coluna 'Umidade'.
    """
    # Carregar cache
    global _umidade_cache
    _umidade_cache = _load_cache()
    cache_size = len(_umidade_cache)
    logger.info("Cache carregado: %d entradas", cache_size)

    # --- Estratégia de resume ---
    # Sempre carregamos o input_path (dataset completo) para garantir que temos
    # todas as linhas. Se output_path existir COM O MESMO número de linhas, copiamos
    # a coluna Umidade já preenchida. Caso contrário (output é de run antigo com
    # --amostra ou menor), ignoramos o output e usamos somente o cache.
    logger.info("Carregando dataset original: %s", input_path)
    df = pd.read_csv(input_path)
    logger.info("Dataset original: %d linhas, %d colunas", df.shape[0], df.shape[1])

    if resume and output_path.exists():
        df_partial = pd.read_csv(output_path)
        if len(df_partial) == len(df) and 'Umidade' in df_partial.columns:
            # Output compatível — reutilizar coluna Umidade já preenchida
            df['Umidade'] = df_partial['Umidade'].values
            ja_enriquecidos = int(df['Umidade'].notna().sum())
            logger.info(
                "Output compatível encontrado — retomando. Já enriquecidos: %d / %d",
                ja_enriquecidos, len(df),
            )
        else:
            # Output incompatível (ex.: run antigo com --amostra). Usar somente cache.
            logger.warning(
                "Output parcial tem %d linhas (input tem %d) — ignorando output, "
                "usando apenas cache para não repetir chamadas à API.",
                len(df_partial), len(df),
            )
            # A coluna Umidade será preenchida via cache na etapa de processamento
    
    # Limitar amostra se solicitado
    if amostra and amostra < len(df):
        df = df.head(amostra)
        logger.info("Processando amostra de %d linhas", len(df))
    
    # Criar coluna Umidade se não existir
    if 'Umidade' not in df.columns:
        df['Umidade'] = np.nan

    # Verificar se tem colunas necessárias
    colunas_necessarias = ['Latitude', 'Longitude', 'Ano', 'Mes', 'Dia']
    faltando = [c for c in colunas_necessarias if c not in df.columns]
    if faltando:
        raise ValueError(f"Colunas necessárias não encontradas: {faltando}")

    # Contador mutável compartilhado com o handler de sinal
    _state = {
        'df': df,
        'resultados_grupos': {},
        'output_path': output_path,
        'interrompido': False,
    }

    def _salvar_progresso_parcial(signum=None, frame=None):
        """Salva cache e CSV parcial ao receber SIGINT ou ao ser chamado explicitamente."""
        logger.warning("Sinal de interrupção recebido — salvando progresso...")
        try:
            _save_cache(_umidade_cache)
            logger.info("Cache salvo (%d entradas).", len(_umidade_cache))
        except Exception as e:
            logger.warning("Erro ao salvar cache: %s", e)
        try:
            _df = _state['df']
            _rg = _state['resultados_grupos']
            if _rg:
                _df['Umidade'] = _df['_cache_key'].map(_rg).combine_first(_df['Umidade'])
            # Remover coluna auxiliar temporária, se existir
            _df_out = _df.drop(columns=['_cache_key'], errors='ignore')
            _df_out.to_csv(_state['output_path'], index=False)
            salvos = int(_df_out['Umidade'].notna().sum())
            logger.info(
                "CSV parcial salvo: %d registros com umidade em %s",
                salvos, _state['output_path'],
            )
        except Exception as e:
            logger.error("Erro ao salvar CSV parcial: %s", e)
        _state['interrompido'] = True
        if signum is not None:
            sys.exit(0)

    # Registrar handler para SIGINT (Ctrl+C) e SIGTERM
    signal.signal(signal.SIGINT, _salvar_progresso_parcial)
    signal.signal(signal.SIGTERM, _salvar_progresso_parcial)

    # Processar cada linha
    total = len(df)
    encontrados = 0
    nao_encontrados = 0

    logger.info("Buscando dados de umidade de: %s", fonte)
    logger.info("Isso pode levar algum tempo devido ao delay entre requisições...")
    
    # Verificar range de anos no dataset
    if 'Ano' in df.columns:
        anos_min = int(df['Ano'].min())
        anos_max = int(df['Ano'].max())
        logger.info("Range de anos no dataset: %d a %d", anos_min, anos_max)
        if anos_min < 1981:
            logger.warning("Dataset contém anos anteriores a 1981. NASA POWER pode não ter dados para esses anos.")
    
    # OTIMIZAÇÃO: Agrupar por coordenadas/data para reduzir requisições
    logger.info("Agrupando registros por coordenadas e data para otimizar requisições...")
    df['_cache_key'] = df.apply(
        lambda row: _cache_key(
            float(row['Latitude']), 
            float(row['Longitude']), 
            datetime(int(row['Ano']), int(row['Mes']), int(row['Dia']))
        ), axis=1
    )
    
    # Processar apenas grupos únicos
    grupos_unicos = df[df['Umidade'].isna()].groupby('_cache_key').first().reset_index()
    total_requisicoes = len(grupos_unicos)
    logger.info("Total de requisições únicas necessárias: %d (de %d registros)", 
               total_requisicoes, len(df))
    
    if total_requisicoes == 0:
        logger.info("Todos os registros já têm umidade. Nenhuma requisição necessária.")
        df.drop(columns=['_cache_key'], inplace=True)
        return df
    
    # Função para processar um grupo único
    def processar_grupo(row_tuple):
        idx, row = row_tuple
        cache_key = row['_cache_key']
        
        # Verificar cache primeiro
        if cache_key in _umidade_cache:
            valor = _umidade_cache[cache_key]
            if valor is not None:
                return cache_key, valor, 'cache'
            return cache_key, None, 'cache_none'
        
        lat = float(row['Latitude'])
        lon = float(row['Longitude'])
        ano = int(row['Ano'])
        mes = int(row['Mes'])
        dia = int(row['Dia'])
        
        try:
            data = datetime(ano, mes, dia)
            ano_atual = datetime.now().year
            if ano < 1981:
                return cache_key, None, 'data_antiga'
            if ano > ano_atual:
                data = datetime(ano_atual, mes, dia)
        except ValueError:
            return cache_key, None, 'data_invalida'
        
        umidade = None
        try:
            if fonte == 'nasa_power':
                umidade = buscar_umidade_nasa_power(lat, lon, data, use_cache=True)
            elif fonte == 'inmet':
                umidade = buscar_umidade_inmet(lat, lon, data)
            
            # Se chegou aqui sem exceção, a requisição foi bem-sucedida
            # umidade pode ser None (dado não existe) ou float (dado encontrado)
            return cache_key, umidade, 'api'
            
        except APIRequestFailure as e:
            # Falha crítica na requisição - propagar exceção para parar processamento
            logger.error(
                "FALHA CRÍTICA ao buscar umidade para cache_key=%s (lat=%.4f, lon=%.4f, data=%s-%s-%s): %s. "
                "Parando processamento.",
                cache_key, lat, lon, ano, mes, dia, e
            )
            # Re-levantar para parar processamento
            raise
        except Exception as e:
            # Outros erros inesperados = FALHA CRÍTICA
            logger.error(
                "Erro inesperado ao buscar umidade para cache_key=%s (lat=%.4f, lon=%.4f, data=%s-%s-%s): %s. "
                "FALHA CRÍTICA: Parando processamento.",
                cache_key, lat, lon, ano, mes, dia, e
            )
            raise APIRequestFailure(
                f"Erro inesperado ao processar grupo: {str(e)}"
            ) from e
    
    # Ajustar workers baseado no rate limit
    # Com rate limit de 20 req/min, 5 workers = ~100 req/min (sem considerar tempo de resposta)
    # Mas o rate limit vai controlar isso automaticamente
    max_workers_effective = min(max_workers, REQUESTS_PER_MINUTE // 4)  # Máximo de 1/4 do limite por worker
    if max_workers > max_workers_effective:
        logger.warning(
            "Número de workers reduzido de %d para %d para respeitar rate limit da API (%d req/min)",
            max_workers, max_workers_effective, REQUESTS_PER_MINUTE
        )
        max_workers = max_workers_effective
    
    # Processar grupos únicos em paralelo
    logger.info(
        "Processando %d requisições únicas com %d workers em paralelo... "
        "(Rate limit: %d req/min = ~%.0f req/hora)",
        total_requisicoes, max_workers, REQUESTS_PER_MINUTE, REQUESTS_PER_MINUTE * 60
    )
    resultados_grupos: Dict[str, Optional[float]] = {}
    _state['resultados_grupos'] = resultados_grupos
    encontrados = 0
    nao_encontrados = 0
    cache_hits = 0
    processados = 0

    inicio = time.time()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(processar_grupo, (idx, row)): (idx, row)
                  for idx, row in grupos_unicos.iterrows()}

        for future in as_completed(futures):
            try:
                cache_key, umidade, origem = future.result()
                resultados_grupos[cache_key] = umidade
                processados += 1

                if umidade is not None:
                    encontrados += 1
                elif origem in ('cache', 'cache_none'):
                    cache_hits += 1
                else:
                    nao_encontrados += 1

            except APIRequestFailure as e:
                logger.error(
                    "FALHA CRÍTICA detectada: %s — salvando progresso e interrompendo.",
                    e,
                )
                _salvar_progresso_parcial()
                raise

            except Exception as e:
                logger.error(
                    "Erro inesperado no future: %s — salvando progresso e interrompendo.", e
                )
                try:
                    idx, row = futures[future]
                    logger.error("cache_key do future que falhou: %s", row.get('_cache_key'))
                except Exception:
                    pass
                _salvar_progresso_parcial()
                raise APIRequestFailure(
                    f"Erro inesperado no processamento paralelo: {str(e)}"
                ) from e

            # Log de progresso e salvamento periódico
            if processados % 100 == 0:
                tempo_decorrido = time.time() - inicio
                tempo_medio = tempo_decorrido / processados if processados > 0 else 0
                tempo_restante = tempo_medio * (total_requisicoes - processados)
                taxa_sucesso = (encontrados / processados) * 100 if processados > 0 else 0

                logger.info(
                    "Progresso: %d/%d (%.1f%%) | Encontrados: %d (%.1f%%) | "
                    "Cache hits: %d | Tempo: %.1fs | Restante: ~%.0fmin",
                    processados, total_requisicoes,
                    (processados / total_requisicoes) * 100,
                    encontrados, taxa_sucesso, cache_hits,
                    tempo_decorrido, tempo_restante / 60,
                )

            # Salvar cache E CSV periodicamente
            if processados % salvar_cache_periodicamente == 0:
                _save_cache(_umidade_cache)
                # Aplicar resultados parciais e salvar CSV
                df['Umidade'] = df['_cache_key'].map(resultados_grupos).combine_first(df['Umidade'])
                df.drop(columns=['_cache_key'], errors='ignore').to_csv(output_path, index=False)
                # Recriar coluna auxiliar para continuar iterando
                df['_cache_key'] = df.apply(
                    lambda row: _cache_key(
                        float(row['Latitude']),
                        float(row['Longitude']),
                        datetime(int(row['Ano']), int(row['Mes']), int(row['Dia']))
                    ), axis=1
                )
                logger.info(
                    "Checkpoint salvo: cache=%d entradas, CSV com %d registros enriquecidos",
                    len(_umidade_cache),
                    int(df['Umidade'].notna().sum()),
                )
    
    # Restaurar handlers padrão de sinal
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    signal.signal(signal.SIGTERM, signal.SIG_DFL)

    # Aplicar resultados finais preservando valores já existentes
    logger.info("Aplicando resultados finais aos %d registros...", len(df))
    df['Umidade'] = df['_cache_key'].map(resultados_grupos).combine_first(df['Umidade'])
    df.drop(columns=['_cache_key'], inplace=True)
    
    encontrados_total = df['Umidade'].notna().sum()
    nao_encontrados_total = df['Umidade'].isna().sum()
    
    tempo_total = time.time() - inicio
    logger.info("Processamento concluído em %.1f segundos!", tempo_total)
    logger.info("Total de registros: %d", len(df))
    logger.info("Umidade encontrada: %d (%.1f%%)", encontrados_total, 
               encontrados_total / len(df) * 100 if len(df) > 0 else 0)
    logger.info("Umidade não encontrada: %d (%.1f%%)", nao_encontrados_total,
               nao_encontrados_total / len(df) * 100 if len(df) > 0 else 0)
    logger.info("Cache hits: %d", cache_hits)
    logger.info("Requisições únicas processadas: %d", total_requisicoes)
    
    # Salvar cache final
    _save_cache(_umidade_cache)
    logger.info("Cache final salvo: %d entradas", len(_umidade_cache))
    
    # Salvar
    df.to_csv(output_path, index=False)
    logger.info("Dataset enriquecido salvo em: %s", output_path)
    
    return df


def main():
    """Função principal."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Enriquece base de dados com dados históricos reais de umidade relativa'
    )
    parser.add_argument(
        '--input',
        type=str,
        default=str(DATA_PATH),
        help='Caminho para CSV de entrada (padrão: base_de_dados.csv)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=str(OUTPUT_PATH),
        help='Caminho para CSV de saída (padrão: base_de_dados_com_umidade.csv)'
    )
    parser.add_argument(
        '--fonte',
        type=str,
        default='nasa_power',
        choices=['nasa_power', 'inmet'],
        help='Fonte de dados (padrão: nasa_power)'
    )
    parser.add_argument(
        '--amostra',
        type=int,
        default=None,
        help='Processar apenas N primeiras linhas (útil para testes)'
    )
    parser.add_argument(
        '--delay',
        type=float,
        default=0.1,
        help='Delay em segundos entre requisições (padrão: 0.1)'
    )
    parser.add_argument(
        '--max_workers',
        type=int,
        default=5,
        help='Número de workers para processamento paralelo (padrão: 5)'
    )
    parser.add_argument(
        '--salvar_cache_periodicamente',
        type=int,
        default=100,
        help='Salvar cache e CSV a cada N requisições (padrão: 100)',
    )
    parser.add_argument(
        '--no-resume',
        action='store_true',
        default=False,
        help='Ignorar output parcial existente e começar do zero (padrão: retomar)',
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        logger.error("Arquivo de entrada não encontrado: %s", input_path)
        return

    enriquecer_dataset_com_umidade(
        input_path=input_path,
        output_path=output_path,
        fonte=args.fonte,
        amostra=args.amostra,
        delay_entre_requisicoes=args.delay,
        max_workers=args.max_workers,
        salvar_cache_periodicamente=args.salvar_cache_periodicamente,
        resume=not args.no_resume,
    )


if __name__ == '__main__':
    main()

