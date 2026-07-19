"""Configurações da aplicação."""
import os
from pathlib import Path

# Chave API HG Weather
HG_WEATHER_API_KEY = '14068910'

# Chave API NASA FIRMS (MAP_KEY)
# Obter de: https://firms.modaps.eosdis.nasa.gov/mapserver/mapkey_status/
NASA_FIRMS_MAP_KEY = '5bcd249ca2d9e80a86ff67e0320c7873'

# Chave(s) API NASA POWER
# Suporta uma chave única ou múltiplas chaves para rotação (aumenta rate limit)
# Obter gratuitamente de: https://api.nasa.gov/
NASA_POWER_API_KEY = 'K5GKN9bHblP1I73AWT8ZjbLW8ZeuGrV00WUUn9og'
# Múltiplas chaves (opcional): descomente e adicione mais chaves para aumentar rate limit
NASA_POWER_API_KEYS = [
    'K5GKN9bHblP1I73AWT8ZjbLW8ZeuGrV00WUUn9og',
    'BuERPEkGGVJdWiPqsEhQzxRclsJ0xSE3bpDbCl0b'
]

# Diretórios base
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
MODEL_DIR = PROJECT_ROOT / 'modelos'
#codigo da mary
Incendio_cerrado = 'palmas_calor_40_graus_queimadas'

DATA_DIR = PROJECT_ROOT

# Caminhos dos shapefiles da Amazônia Legal
AMAZONIA_LEGAL_SHP_2024 = DATA_DIR / 'Limites_Amazonia_Legal_2024_shp' / 'Limites_Amazonia_Legal_2024.shp'
BRAZILIAN_LEGAL_AMAZON_SHP = DATA_DIR / 'brazilian_legal_amazon' / 'brazilian_legal_amazon.shp'

# Metadados do preprocessor
PREPROCESSOR_METADATA_PATH = MODEL_DIR / 'preprocessor_metadata.json'

# Cache de geocodificação
GEOCODE_CACHE_PATH = MODEL_DIR / 'geocode_cache.json'

# Dataset histórico para fallback
DATASET_PATH = DATA_DIR / 'base_de_dados.csv'

# Base com histórico de incêndios (alinhada ao treino) — medianas para features no mapa
HISTORICAL_TRAINING_DATASET_PATH = DATA_DIR / 'base_de_dados_com_historico.csv'

# Cache INMET (catálogo de estações + dados anuais)
INMET_CACHE_DIR = PROJECT_ROOT / '.cache_inmet'
INMET_MAX_DISTANCE_KM = 50.0  # raio preferencial (alta representatividade espacial)
# Segunda tentativa de fusão INMET (WIS2 sinótica ou ZIP automática) quando não há
# estação dentro do raio preferencial — típico do Cerrado/TO com rede esparsa.
INMET_EXTENDED_MAX_DISTANCE_KM = 180.0

# Configurações do mapa
DEFAULT_MAP_VIEW = (-5, -60)
DEFAULT_MAP_ZOOM = 5

