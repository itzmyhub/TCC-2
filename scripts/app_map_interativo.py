"""Aplicação Flask para mapa interativo de risco de incêndio na Amazônia Legal."""
import __main__ as _main
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import folium
import joblib
import pandas as pd

# Stacking (ensemble_stacking) picklou _LabelEncodingWrapper; joblib precisa achar a classe
import treinamento_modelo

if hasattr(treinamento_modelo, "_LabelEncodingWrapper"):
    _main._LabelEncodingWrapper = treinamento_modelo._LabelEncodingWrapper
from flask import Flask, jsonify, render_template_string, request
from geopy.exc import GeocoderTimedOut
from geopy.geocoders import Nominatim

from audit_log import log_prediction
from climate_api import climate_provider
from config import (
    AMAZONIA_LEGAL_SHP_2024,
    BRAZILIAN_LEGAL_AMAZON_SHP,
    DATA_DIR,
    DEFAULT_MAP_VIEW,
    DEFAULT_MAP_ZOOM,
    GEOCODE_CACHE_PATH,
    MODEL_DIR,
    PREPROCESSOR_METADATA_PATH,
)
from explainer import explicar_predicao
from operational_uncertainty import (
    fire_weather_proxy_heuristic,
    multiclass_operational_uncertainty,
)
from feature_lookup import (
    get_advanced_features_for_point,
    get_precip_climatologia,
)
from frp_api import frp_provider
from pre_processor import PreProcessor

# Caminhos auxiliares para Tier 1 lookups e explicabilidade
ENRICHED_DATASET_PATH = DATA_DIR / "base_de_dados_enriquecido.csv"
SHAP_IMPORTANCE_PATH = MODEL_DIR / "relatorios" / "shap_feature_importance.json"
PREDICTION_THRESHOLDS_PATH = MODEL_DIR / "prediction_thresholds.json"


def _carregar_thresholds_otimizados() -> Optional[Dict[str, Any]]:
    """Carrega thresholds multi-classe otimizados pelo `ajustar_threshold.py`.

    Retorna ``None`` se o arquivo não existir; o app cai no argmax padrão.
    """
    try:
        if not PREDICTION_THRESHOLDS_PATH.exists():
            return None
        with PREDICTION_THRESHOLDS_PATH.open('r', encoding='utf-8') as fp:
            return json.load(fp)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Falha ao carregar thresholds otimizados: %s", exc)
        return None


def _aplicar_thresholds_multiclasse(
    probabilidades: Dict[str, float],
    thr_moderado: float,
    thr_muito_alto: float,
) -> str:
    """Decide a classe a partir das probabilidades usando os thresholds otimizados.

    Lógica documentada em `prediction_thresholds.json`:
      - se P(Moderado) >= thr_moderado → "Moderado"
      - senão se P(Muito Alto) >= thr_muito_alto → "Muito Alto"
      - senão → "Baixo"
    """
    p_mod = float(probabilidades.get('Moderado', 0.0) or 0.0)
    p_alt = float(probabilidades.get('Muito Alto', 0.0) or 0.0)
    if p_mod >= thr_moderado:
        return 'Moderado'
    if p_alt >= thr_muito_alto:
        return 'Muito Alto'
    return 'Baixo'

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

app = Flask(__name__)
geolocator = Nominatim(user_agent="tcc-mapeamento-interativo")


# Cache de geocodificação
def carregar_cache() -> Dict[str, Tuple[float, float]]:
    """Carrega cache de geocodificação."""
    if GEOCODE_CACHE_PATH.exists():
        with GEOCODE_CACHE_PATH.open('r', encoding='utf-8') as fp:
            cache = json.load(fp)
        return {k: tuple(v) for k, v in cache.items()}
    return {}


def salvar_cache(cache: Dict[str, Tuple[float, float]]) -> None:
    """Salva cache de geocodificação."""
    GEOCODE_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    serializavel = {k: list(v) for k, v in cache.items()}
    with GEOCODE_CACHE_PATH.open('w', encoding='utf-8') as fp:
        json.dump(serializavel, fp, ensure_ascii=False, indent=2)


def reverse_geocode(lat: float, lon: float, cache: Dict) -> Tuple[Optional[str], Optional[str]]:
    """Geocodificação reversa para obter Estado e Município."""
    chave = f"{lat:.4f}|{lon:.4f}"
    if chave in cache:
        estado, municipio = cache[chave]
        logger.debug("Estado/Município obtidos do cache: %s / %s", estado, municipio)
        return estado, municipio
    
    try:
        logger.info("Fazendo geocodificação reversa para lat=%.4f, lon=%.4f...", lat, lon)
        location = geolocator.reverse(f"{lat}, {lon}", timeout=15, language='pt')
        
        if location:
            address = location.raw.get('address', {})
            logger.info("Endereço retornado: %s", address)
            
            # Tentar diferentes campos para estado e município
            estado = address.get('state') or address.get('region') or address.get('state_district')
            
            # Tentar diferentes campos para município
            municipio = (address.get('city') or 
                        address.get('town') or 
                        address.get('village') or
                        address.get('municipality') or
                        address.get('city_district'))
            
            logger.info("Estado extraído: %s | Município extraído: %s", estado, municipio)
            
            if estado and municipio:
                estado_str = str(estado).upper()
                municipio_str = str(municipio).upper()
                cache[chave] = (estado_str, municipio_str)
                logger.info("Geocodificação bem-sucedida: %s / %s", estado_str, municipio_str)
                return estado_str, municipio_str
            else:
                logger.warning("Estado ou município não encontrado. Address completo: %s", address)
        else:
            logger.warning("Geocodificação não retornou location")
    except GeocoderTimedOut:
        logger.warning("Timeout na geocodificação reversa")
    except Exception as e:
        logger.warning("Erro ao fazer geocodificação reversa: %s", e, exc_info=True)
    
    return None, None


def carregar_modelo(nome_modelo: Optional[str] = None):
    """Carrega modelo treinado.

    Quando nenhum nome é informado, prefere modelos ensemble mais recentes
    (ordem: `ensemble_stacking_gbm` → `ensemble_stacking` → `random_forest_balanced`
    → primeiro `.pkl` em ordem alfabética).
    """
    modelos = sorted(MODEL_DIR.glob('*.pkl'))
    if not modelos:
        raise FileNotFoundError(f"Nenhum modelo encontrado em {MODEL_DIR}")

    if nome_modelo:
        caminho = MODEL_DIR / f'{nome_modelo}.pkl'
        if not caminho.exists():
            raise FileNotFoundError(f"Modelo '{nome_modelo}' não encontrado")
        return joblib.load(caminho), caminho.stem

    preferidos = (
        'ensemble_stacking_gbm',
        'ensemble_stacking',
        'random_forest_balanced',
    )
    for nome in preferidos:
        caminho = MODEL_DIR / f'{nome}.pkl'
        if caminho.exists():
            return joblib.load(caminho), caminho.stem
    return joblib.load(modelos[0]), modelos[0].stem


def listar_modelos():
    """Lista modelos disponíveis."""
    modelos = sorted(MODEL_DIR.glob('*.pkl'))
    return [m.stem for m in modelos]


_NUM_FEATURES_TREINO: Optional[set] = None


def _num_features_treino() -> set:
    global _NUM_FEATURES_TREINO
    if _NUM_FEATURES_TREINO is None:
        if PREPROCESSOR_METADATA_PATH.exists():
            with PREPROCESSOR_METADATA_PATH.open("r", encoding="utf-8") as fp:
                meta = json.load(fp)
            _NUM_FEATURES_TREINO = set(meta.get("num_features", []))
        else:
            _NUM_FEATURES_TREINO = set()
    return _NUM_FEATURES_TREINO


def preparar_dados_previsao(
    lat: float,
    lon: float,
    estado: Optional[str],
    municipio: Optional[str],
    clima_data: Dict,
    ano: int,
    mes: int,
    dia: int,
    hora: int,
    frp: float = 0.0,
) -> pd.DataFrame:
    """Prepara dados para previsão no formato esperado pelo modelo."""
    import numpy as np
    
    # Garantir Estado e Município
    if not estado or not municipio:
        cache = carregar_cache()
        estado_rev, municipio_rev = reverse_geocode(lat, lon, cache)
        estado = estado or estado_rev or "DESCONHECIDO"
        municipio = municipio or municipio_rev or "DESCONHECIDO"
        salvar_cache(cache)
    
    # Calcular features derivadas (mesmas do treinamento)
    # Estação do ano
    def estacao(mes: int) -> str:
        if mes in [7, 8, 9]:
            return 'Seca_Alta'
        elif mes in [6, 10]:
            return 'Seca_Transicao'
        elif mes in [11, 12, 1, 2, 3]:
            return 'Chuvosa'
        else:  # 4, 5
            return 'Transicao_Chuvosa'
    
    # Período do dia
    def periodo_dia(hora: int) -> str:
        if 5 <= hora < 12:
            return 'Manha'
        elif 12 <= hora < 18:
            return 'Tarde'
        elif 18 <= hora < 22:
            return 'Noite'
        else:
            return 'Madrugada'
    
    precipitacao = float(clima_data.get('precipitacao', 0.0) or 0.0)
    dias_sem_chuva = float(clima_data.get('dias_sem_chuva', 0) or 0.0)
    prec_ma7 = float(
        clima_data.get('prec_ma7', clima_data.get('precipitacao', 0.0)) or 0.0
    )
    dia_ma7 = float(
        clima_data.get('diasem_ma7', clima_data.get('dias_sem_chuva', 0.0)) or 0.0
    )
    umidade = clima_data.get('umidade')  # Umidade relativa do ar (%)

    inc7 = float(clima_data.get('Incendios_Ultimos_7_Dias', 0.0) or 0.0)
    inc30 = float(clima_data.get('Incendios_Ultimos_30_Dias', 0.0) or 0.0)
    ddes = float(clima_data.get('Dias_Desde_Ultimo_Incendio', 365.0) or 365.0)
    mfrp = float(clima_data.get('Media_FRP_Ultimos_7_Dias', 0.0) or 0.0)
    xfrp = float(clima_data.get('Max_FRP_Ultimos_7_Dias', 0.0) or 0.0)

    indice_seca = dias_sem_chuva / (precipitacao + 0.1)

    # Preparar dados base
    dados = {
        'DiaSemChuva': [dias_sem_chuva],
        'Precipitacao': [precipitacao],
        'Latitude': [lat],
        'Longitude': [lon],
        'FRP': [frp],
        'Ano': [ano],
        'Mes': [mes],
        'Dia': [dia],
        'Hora': [hora],
        'Estado': [estado.upper()],
        'Municipio': [municipio.upper()],
        # Features derivadas
        'Mes_sin': [np.sin(2 * np.pi * mes / 12)],
        'Mes_cos': [np.cos(2 * np.pi * mes / 12)],
        'Periodo_Critico': [1 if mes in [7, 8, 9] else 0],
        'Indice_Seca': [indice_seca],
        'Estacao': [estacao(mes)],
        'Periodo_Dia': [periodo_dia(hora)],
        'Precipitacao_ma7': [prec_ma7],
        'DiaSemChuva_ma7': [dia_ma7],
        'Incendios_Ultimos_7_Dias': [inc7],
        'Incendios_Ultimos_30_Dias': [inc30],
        'Dias_Desde_Ultimo_Incendio': [ddes],
        'Media_FRP_Ultimos_7_Dias': [mfrp],
        'Max_FRP_Ultimos_7_Dias': [xfrp],
    }

    # ----------------------------------------------------------------------
    # Features Tier 1 (físico-climáticas) — ESSENCIAIS para modelos enriquecidos.
    # Sem isso, o modelo Stacking/RF Tier 1 receberia NaNs ou colunas faltando.
    # ----------------------------------------------------------------------
    adv_features, adv_meta = get_advanced_features_for_point(
        lat=lat,
        lon=lon,
        estado=estado,
        municipio=municipio,
        mes=int(mes),
        precipitacao_atual=precipitacao,
        dias_sem_chuva_atual=dias_sem_chuva,
        prec_ma7=prec_ma7,
        dsem_ma7=dia_ma7,
        indice_seca=indice_seca,
        dataset_path=ENRICHED_DATASET_PATH,
    )
    num_features_treino = _num_features_treino()
    adicionadas: list = []
    for k, v in adv_features.items():
        if not num_features_treino or k in num_features_treino:
            dados[k] = [v]
            adicionadas.append(k)
    if adicionadas:
        logger.info(
            "Features Tier 1 adicionadas (%d, granularidade=%s): %s",
            len(adicionadas), adv_meta.get('granularidade'),
            ', '.join(adicionadas[:6]) + ('…' if len(adicionadas) > 6 else ''),
        )

    # Só envia Umidade se o treino do preprocessor listar a coluna (evita quebra do Pipeline)
    if umidade is not None and "Umidade" in num_features_treino:
        dados["Umidade"] = [float(umidade)]
        logger.info("Umidade enviada ao modelo: %.1f%%", umidade)
    elif umidade is not None:
        logger.info("Umidade (%.1f%%) obtida do clima mas omitida: modelo não treinou com coluna Umidade", umidade)

    dados = pd.DataFrame(dados)
    dados.attrs["adv_meta"] = adv_meta

    return dados


@app.route('/')
def index():
    """Rota principal que serve o mapa interativo."""
    # Carregar shapefile se disponível e calcular bounds
    shapefile_path = None
    bounds = None
    gdf = None
    
    if AMAZONIA_LEGAL_SHP_2024.exists():
        shapefile_path = AMAZONIA_LEGAL_SHP_2024
    else:
        # Tentar alternativa
        if BRAZILIAN_LEGAL_AMAZON_SHP.exists():
            shapefile_path = BRAZILIAN_LEGAL_AMAZON_SHP
    
    # Calcular bounds do shapefile se disponível
    if shapefile_path:
        try:
            import geopandas as gpd
            gdf = gpd.read_file(shapefile_path)
            # Obter bounds (minx, miny, maxx, maxy)
            bounds = gdf.total_bounds  # [minx, miny, maxx, maxy]
            # Converter para formato do Folium: [[miny, minx], [maxy, maxx]]
            bounds_folium = [[bounds[1], bounds[0]], [bounds[3], bounds[2]]]
            logger.info("Shapefile carregado. Bounds: %s", bounds_folium)
        except ImportError:
            logger.warning("geopandas não instalado. Usando bounding box aproximada.")
            bounds_folium = [[-18, -74], [6, -44]]
        except Exception as e:
            logger.warning("Erro ao carregar shapefile: %s", e)
            bounds_folium = [[-18, -74], [6, -44]]
    else:
        # Bounding box aproximada da Amazônia Legal
        bounds_folium = [[-18, -74], [6, -44]]
    
    # Calcular centro e zoom baseado nos bounds
    center_lat = (bounds_folium[0][0] + bounds_folium[1][0]) / 2
    center_lon = (bounds_folium[0][1] + bounds_folium[1][1]) / 2
    
    # Criar mapa
    mapa = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=DEFAULT_MAP_ZOOM,
        tiles='OpenStreetMap'
    )
    
    # Adicionar limites da Amazônia Legal se shapefile disponível
    if gdf is not None:
        folium.GeoJson(
            gdf.to_json(),
            style_function=lambda feature: {
                'fillColor': 'lightgreen',
                'color': 'darkgreen',
                'weight': 2,
                'fillOpacity': 0.2,
            },
            tooltip='Amazônia Legal'
        ).add_to(mapa)
        logger.info("Shapefile da Amazônia Legal carregado")
        # Ajustar bounds do mapa para o shapefile
        mapa.fit_bounds(bounds_folium)
    else:
        # Bounding box aproximada
        bbox_coords = [
            [-18, -74], [6, -74], [6, -44], [-18, -44], [-18, -74]
        ]
        folium.Polygon(
            locations=bbox_coords,
            color='darkgreen',
            fillColor='lightgreen',
            fillOpacity=0.2,
            weight=2,
            tooltip='Amazônia Legal (aproximado)'
        ).add_to(mapa)
        # Ajustar bounds do mapa
        mapa.fit_bounds(bounds_folium)
    
    # Script de clique no iframe do Folium: apenas detecta o clique e
    # despacha um evento para o documento principal lidar (toda a UI vive lá).
    click_script = """
    <script>
    (function() {
        function findLeafletMap() {
            for (var key in window) {
                try {
                    if (key.startsWith('map_') && window[key] && window[key]._container) {
                        return window[key];
                    }
                } catch(e) {}
            }
            return null;
        }

        function postClickToParent(lat, lon) {
            try {
                var tgt = (window.parent && window.parent !== window) ? window.parent : window.top;
                tgt.postMessage({ tipo: 'mapa_clique', lat: lat, lon: lon }, '*');
            } catch(e) {
                console.error('Falha ao notificar parent do clique:', e);
            }
        }

        function init() {
            var m = findLeafletMap();
            if (!m) { setTimeout(init, 200); return; }
            m.on('click', function(e) {
                postClickToParent(e.latlng.lat, e.latlng.lng);
            });
            // Exponha o mapa para receber comandos do parent (marcadores)
            window.__mapaPrincipal = m;
            window.addEventListener('message', function(ev) {
                if (!ev.data || ev.data.tipo !== 'mapa_marcador') return;
                try {
                    if (window.__currentMarker) m.removeLayer(window.__currentMarker);
                    var cor = ev.data.cor || 'blue';
                    window.__currentMarker = L.marker([ev.data.lat, ev.data.lon], {
                        icon: L.icon({
                            iconUrl: 'https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-' + cor + '.png',
                            iconSize: [25, 41], iconAnchor: [12, 41], popupAnchor: [1, -34],
                            shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-shadow.png',
                            shadowSize: [41, 41]
                        })
                    }).addTo(m);
                    if (ev.data.centralizar) {
                        var zoomAlvo = Math.max(m.getZoom(), 7);
                        m.setView([ev.data.lat, ev.data.lon], zoomAlvo, { animate: true });
                    }
                } catch(err) { console.error(err); }
            });
        }
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', function() { setTimeout(init, 300); });
        } else {
            setTimeout(init, 300);
        }
    })();
    </script>
    """
    mapa.get_root().html.add_child(folium.Element(click_script))
    
    # Obter HTML do mapa
    mapa_html = mapa._repr_html_()
    
    html_content = _renderizar_pagina(mapa_html)
    return html_content


# ---------------------------------------------------------------------------
# Página HTML do app (shell em torno do iframe Folium)
# ---------------------------------------------------------------------------
def _renderizar_pagina(mapa_html: str) -> str:
    """Retorna o HTML completo da página principal.

    Separado em função para manter `index()` legível e permitir reutilização
    em testes/embed futuro. Usa Jinja-like substitution via {map_html}.
    """
    return _PAGINA_HTML.replace("{{MAPA_HTML}}", mapa_html)


_PAGINA_HTML = r"""<!DOCTYPE html>
<html lang="pt-br">
<head>
    <title>Risco de Incêndio · Amazônia Legal</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.7.1/dist/leaflet.css" />
    <style>
        :root {
            --c-baixo: #16a34a;
            --c-moderado: #d97706;
            --c-alto: #dc2626;
            --c-bg: #f5f6f8;
            --c-card: #ffffff;
            --c-border: #e3e4e8;
            --c-text: #1f2328;
            --c-muted: #57606a;
            --c-accent: #2b7de9;
        }
        * { box-sizing: border-box; }
        body {
            margin: 0; padding: 0;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background: var(--c-bg);
            color: var(--c-text);
            font-size: 13px;
            line-height: 1.45;
        }
        #app { display: flex; height: 100vh; width: 100vw; overflow: hidden; }
        #map-area { flex: 1; position: relative; min-width: 300px; }
        #map-area iframe, #map-area .folium-map { height: 100% !important; width: 100% !important; }
        #side {
            width: 420px;
            max-width: 50vw;
            background: var(--c-bg);
            border-left: 1px solid var(--c-border);
            overflow-y: auto;
            padding: 16px;
        }
        .header { display: flex; align-items: center; justify-content: space-between; }
        .header h1 { margin: 0; font-size: 16px; }
        .header .subtitle { color: var(--c-muted); font-size: 12px; margin-top: 2px; }
        .toolbar {
            background: var(--c-card);
            border: 1px solid var(--c-border);
            border-radius: 8px;
            padding: 10px 12px;
            margin-top: 12px;
        }
        .toolbar label { font-size: 11px; color: var(--c-muted); text-transform: uppercase; letter-spacing: .03em; }
        .toolbar select {
            width: 100%; margin-top: 4px;
            padding: 7px 8px; border: 1px solid var(--c-border); border-radius: 6px;
            background: white; font-size: 13px;
        }
        .toolbar .hint { color: var(--c-muted); font-size: 12px; margin-top: 8px; }
        .form-row { display: flex; gap: 8px; margin-top: 8px; }
        .form-field { flex: 1; display: flex; flex-direction: column; }
        .form-field .field-label {
            font-size: 11px; color: var(--c-muted);
            text-transform: uppercase; letter-spacing: .03em; margin-bottom: 2px;
        }
        .form-field input {
            width: 100%; padding: 6px 8px;
            border: 1px solid var(--c-border); border-radius: 6px;
            background: white; font-size: 13px; font-variant-numeric: tabular-nums;
        }
        .form-field input:focus {
            outline: none; border-color: var(--c-accent);
            box-shadow: 0 0 0 2px rgba(43, 125, 233, 0.15);
        }
        .form-actions { display: flex; gap: 8px; margin-top: 10px; }
        .form-actions button {
            flex: 1; padding: 8px 12px; border-radius: 6px;
            border: 1px solid var(--c-accent); background: var(--c-accent);
            color: white; font-size: 13px; font-weight: 600; cursor: pointer;
            transition: background .15s ease;
        }
        .form-actions button:hover { background: #1e6ad9; }
        .form-actions button.ghost {
            background: white; color: var(--c-accent); flex: 0 0 auto; min-width: 72px;
        }
        .form-actions button.ghost:hover { background: #eaf3ff; }
        .form-erro {
            margin-top: 8px; padding: 6px 10px;
            background: #fff1f2; border-left: 3px solid var(--c-alto);
            color: #991b1b; font-size: 12px; border-radius: 4px;
        }

        .card {
            background: var(--c-card);
            border: 1px solid var(--c-border);
            border-radius: 8px;
            padding: 12px 14px;
            margin-top: 12px;
        }
        .card h3 {
            font-size: 12px; text-transform: uppercase; letter-spacing: .05em;
            color: var(--c-muted); margin: 0 0 10px 0; font-weight: 600;
        }
        .card .row { display: flex; justify-content: space-between; gap: 8px; padding: 3px 0; }
        .card .row .k { color: var(--c-muted); }
        .card .row .v { font-weight: 600; text-align: right; }
        .badge-risco {
            display: inline-block; padding: 6px 12px; border-radius: 999px;
            font-weight: 700; font-size: 16px; color: white;
        }
        .badge-risco.baixo { background: var(--c-baixo); }
        .badge-risco.moderado { background: var(--c-moderado); }
        .badge-risco.alto { background: var(--c-alto); }
        .badge-risco.unknown { background: #6b7280; }

        .prob-bar { display: flex; flex-direction: column; gap: 6px; margin-top: 8px; }
        .prob-row { display: grid; grid-template-columns: 70px 1fr 50px; align-items: center; gap: 8px; }
        .prob-row .label { font-size: 12px; color: var(--c-muted); }
        .prob-row .track { background: #eef0f3; height: 8px; border-radius: 4px; overflow: hidden; position: relative; }
        .prob-row .fill { height: 100%; border-radius: 4px; transition: width .3s ease; }
        .prob-row .pct { font-size: 12px; text-align: right; font-variant-numeric: tabular-nums; font-weight: 600; }
        .fill.baixo { background: var(--c-baixo); }
        .fill.moderado { background: var(--c-moderado); }
        .fill.alto { background: var(--c-alto); }

        .explain-item {
            border-left: 3px solid var(--c-border);
            padding: 6px 10px;
            margin-bottom: 6px;
            background: #fbfbfc;
            border-radius: 0 4px 4px 0;
        }
        .explain-item.aumenta { border-left-color: var(--c-alto); }
        .explain-item.reduz { border-left-color: var(--c-baixo); }
        .explain-item .feat { font-weight: 600; font-size: 12px; }
        .explain-item .desc { color: var(--c-muted); font-size: 11.5px; }
        .explain-item .num {
            font-size: 12px; margin-top: 4px;
            display: flex; gap: 10px; flex-wrap: wrap; font-variant-numeric: tabular-nums;
        }
        .explain-item .pill {
            background: #eef0f3; border-radius: 4px; padding: 1px 6px;
        }
        .explain-item .pill.aumenta { background: #fee2e2; color: #991b1b; }
        .explain-item .pill.reduz { background: #dcfce7; color: #166534; }
        .explain-item .pill.neutra { background: #eef0f3; color: var(--c-muted); }

        .legend { display: flex; gap: 12px; margin-top: 8px; font-size: 12px; }
        .legend .item { display: inline-flex; align-items: center; gap: 6px; }
        .legend .dot { width: 10px; height: 10px; border-radius: 50%; }

        #loading {
            display: none; margin-top: 12px; padding: 10px 12px;
            background: #eaf3ff; border-left: 4px solid var(--c-accent); border-radius: 6px;
        }
        #empty-state {
            background: #fff; border: 1px dashed var(--c-border); border-radius: 8px;
            padding: 18px; color: var(--c-muted); text-align: center; margin-top: 12px;
        }
        #empty-state h4 { color: var(--c-text); margin: 0 0 8px 0; font-size: 14px; }
        .small { font-size: 11.5px; color: var(--c-muted); }
        details { margin-top: 8px; }
        details summary {
            cursor: pointer; color: var(--c-accent); font-size: 12px;
        }
        .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 4px 12px; }
        .grid-2 .row { padding: 2px 0; }
        .anomalia.positiva { color: var(--c-baixo); }
        .anomalia.negativa { color: var(--c-alto); }
        .conf-trace {
            margin-top: 6px; padding: 6px 8px; background: #f8f9fa;
            border-radius: 4px; font-size: 11.5px; color: var(--c-muted);
        }
        @media (max-width: 900px) {
            #app { flex-direction: column; }
            #side { width: 100%; max-width: 100%; max-height: 55vh; border-left: 0; border-top: 1px solid var(--c-border); }
            #map-area { min-height: 45vh; }
        }
    </style>
</head>
<body>
<div id="app">
    <div id="map-area">{{MAPA_HTML}}</div>
    <aside id="side">
        <div class="header">
            <div>
                <h1>Risco de incêndio · Amazônia Legal</h1>
                <div class="subtitle">Modelo ML + clima NASA POWER + FIRMS</div>
            </div>
        </div>

        <div class="toolbar" id="busca-form">
            <label>Consulta por coordenada e data</label>
            <div class="form-row">
                <div class="form-field">
                    <span class="field-label">Latitude</span>
                    <input id="in-lat" type="text" inputmode="decimal"
                           autocomplete="off" spellcheck="false"
                           placeholder="-10,18 (ex.: Palmas-TO)">
                </div>
                <div class="form-field">
                    <span class="field-label">Longitude</span>
                    <input id="in-lon" type="text" inputmode="decimal"
                           autocomplete="off" spellcheck="false"
                           placeholder="-48,33 (ex.: Palmas-TO)">
                </div>
            </div>
            <div class="form-row">
                <div class="form-field">
                    <span class="field-label">Data</span>
                    <input id="in-data" type="date">
                </div>
                <div class="form-field">
                    <span class="field-label">Hora</span>
                    <input id="in-hora" type="time" step="3600" value="12:00">
                </div>
            </div>
            <div class="form-actions">
                <button id="btn-consultar" type="button">Consultar</button>
                <button id="btn-agora" type="button" class="ghost" title="Preencher data/hora com o momento atual">Agora</button>
            </div>
            <div id="busca-erro" class="form-erro" style="display:none;"></div>
            <div class="hint">Limite NASA POWER: dados até cerca de 2–3 dias atrás. Datas mais recentes podem cair em fallback (HG / histórico).</div>
        </div>

        <div class="toolbar">
            <label for="modelo">Modelo</label>
            <select id="modelo">
                <option value="">Padrão (primeiro disponível)</option>
            </select>
            <div class="hint">Ou clique direto no mapa para usar a data/hora do formulário (default: agora).</div>
            <div class="legend">
                <span class="item"><span class="dot" style="background: var(--c-baixo);"></span>Baixo</span>
                <span class="item"><span class="dot" style="background: var(--c-moderado);"></span>Moderado</span>
                <span class="item"><span class="dot" style="background: var(--c-alto);"></span>Muito Alto</span>
            </div>
        </div>

        <div id="loading"></div>

        <div id="empty-state">
            <h4>Nenhum ponto consultado</h4>
            <div>Clique em qualquer local do mapa dentro da Amazônia Legal. A previsão usa precipitação NASA POWER (últimos 28 d), histórico de focos NASA FIRMS e features climáticas avançadas (SPI, KBDI proxy, anomalia, etc.).</div>
        </div>

        <div id="erro-comunicacao" style="display:none;"></div>

        <div id="resultado" style="display:none;">
            <!-- Card: Risco -->
            <div class="card" id="card-risco">
                <h3>Previsão</h3>
                <div style="display:flex; justify-content: space-between; align-items: center;">
                    <div>
                        <div id="risco-label" class="badge-risco unknown">—</div>
                        <div class="small" style="margin-top:6px;" id="local-label">—</div>
                    </div>
                    <div style="text-align:right;">
                        <div class="small">Modelo</div>
                        <div style="font-weight:600; font-size:12px;" id="modelo-label">—</div>
                        <div class="small" id="modelo-metricas">—</div>
                    </div>
                </div>
                <div class="prob-bar" id="prob-bar"></div>
                <div class="conf-trace" id="trace-modelo"></div>
                <div class="conf-trace" id="trace-thresholds"></div>
            </div>

            <!-- Card: Por que esse risco -->
            <div class="card" id="card-explain">
                <h3>Por que esse risco?</h3>
                <div id="explicacao-content">—</div>
                <details>
                    <summary>Como esta explicação é calculada?</summary>
                    <div class="small" style="margin-top:6px;">
                        Contribuição aproximada por feature, calculada como
                        <code>importância global × anormalidade local (z-score) × sinal físico</code>.
                        A importância global vem do Random Forest treinado e a anormalidade compara o valor da feature no ponto com a distribuição do dataset enriquecido.
                    </div>
                </details>
            </div>

            <!-- Card: Dados climáticos -->
            <div class="card" id="card-clima">
                <h3>Dados climáticos atuais</h3>
                <div class="grid-2" id="clima-grid"></div>
                <div class="conf-trace" id="trace-clima"></div>
            </div>

            <!-- Card: Índices de seca -->
            <div class="card" id="card-seca">
                <h3>Índices de seca e aridez</h3>
                <div class="grid-2" id="seca-grid"></div>
                <div class="small" style="margin-top:8px;">
                    SPI: 0 = normal, &lt; -1 = seca, &gt; +1 = chuvoso. KBDI proxy e VPD proxy crescem com déficit hídrico × temperatura.
                </div>
            </div>

            <!-- Card: Acumulados e anomalia -->
            <div class="card" id="card-acum">
                <h3>Precipitação acumulada e anomalia</h3>
                <div class="grid-2" id="acum-grid"></div>
                <div id="anomalia-narrativa" class="small" style="margin-top:8px;"></div>
            </div>

            <!-- Card: Histórico de fogo -->
            <div class="card" id="card-fogo">
                <h3>Histórico de focos (NASA FIRMS + treino)</h3>
                <div class="grid-2" id="fogo-grid"></div>
            </div>
        </div>
    </aside>
</div>

<script src="https://unpkg.com/leaflet@1.7.1/dist/leaflet.js"></script>
<script>
(function() {
    var modelos = [];

    function $(id) { return document.getElementById(id); }
    function fmtNum(v, dec) {
        if (v === null || v === undefined || isNaN(v)) return '—';
        var d = (dec === undefined) ? 2 : dec;
        return Number(v).toLocaleString('pt-BR', { minimumFractionDigits: d, maximumFractionDigits: d });
    }
    function ucfirst(s) { if (!s) return ''; return s.charAt(0).toUpperCase() + s.slice(1); }
    function riscoClass(r) {
        if (!r) return 'unknown';
        var s = String(r).toLowerCase();
        if (s.indexOf('alto') >= 0) return 'alto';
        if (s.indexOf('moder') >= 0) return 'moderado';
        if (s.indexOf('baix') >= 0) return 'baixo';
        return 'unknown';
    }
    function corLeafletDoRisco(r) {
        var c = riscoClass(r);
        if (c === 'alto') return 'red';
        if (c === 'moderado') return 'orange';
        if (c === 'baixo') return 'green';
        return 'blue';
    }

    // ------------ Modelos disponíveis ------------
    fetch('/api/models')
        .then(r => r.json())
        .then(function(data) {
            if (!data.sucesso) return;
            modelos = data.modelos || [];
            var sel = $('modelo');
            modelos.forEach(function(m) {
                var opt = document.createElement('option');
                opt.value = m; opt.textContent = m;
                sel.appendChild(opt);
            });
            var preferidos = ['ensemble_stacking_gbm', 'ensemble_stacking', 'random_forest_balanced', 'random_forest'];
            for (var i = 0; i < preferidos.length; i++) {
                var p = preferidos[i];
                var idx = modelos.findIndex(function(m){ return m === p || m.indexOf(p) >= 0; });
                if (idx >= 0) { sel.value = modelos[idx]; break; }
            }
        });

    // ------------ Inicializa form com data/hora "agora" ------------
    function pad2(n) { return (n < 10 ? '0' : '') + n; }
    function setAgora() {
        var d = new Date();
        $('in-data').value = d.getFullYear() + '-' + pad2(d.getMonth() + 1) + '-' + pad2(d.getDate());
        $('in-hora').value = pad2(d.getHours()) + ':00';
    }
    setAgora();
    $('btn-agora').addEventListener('click', function() { setAgora(); limparErroForm(); });

    // Limita data máxima ao dia de hoje (NASA POWER é reanálise, sem futuro)
    (function() {
        var d = new Date();
        var hoje = d.getFullYear() + '-' + pad2(d.getMonth() + 1) + '-' + pad2(d.getDate());
        $('in-data').max = hoje;
    })();

    function mostrarErroForm(msg) {
        var box = $('busca-erro');
        box.textContent = msg;
        box.style.display = 'block';
    }
    function limparErroForm() {
        var box = $('busca-erro');
        box.style.display = 'none'; box.textContent = '';
    }

    // Aceita "-10,18", "-10.18", " -10,1800 ", etc. Retorna NaN se inválido.
    function parseCoord(raw) {
        if (raw === null || raw === undefined) return NaN;
        var s = String(raw).trim();
        if (!s) return NaN;
        s = s.replace(/\s+/g, '').replace(',', '.');
        if (!/^-?\d+(\.\d+)?$/.test(s)) return NaN;
        return parseFloat(s);
    }

    function parseFormulario() {
        var rawLat = $('in-lat').value;
        var rawLon = $('in-lon').value;
        var lat = parseCoord(rawLat);
        var lon = parseCoord(rawLon);
        var dataStr = $('in-data').value;
        var horaStr = $('in-hora').value || '12:00';

        if (isNaN(lat) || isNaN(lon)) {
            mostrarErroForm('Informe latitude e longitude válidas (use ponto ou vírgula como separador decimal, ex.: -10,18).');
            return null;
        }
        // Detecta inversão clássica: lat caiu no range de lon e vice-versa.
        var latParecLon = (lat <= -44 && lat >= -74);
        var lonParecLat = (lon >= -18 && lon <= 6);
        if (latParecLon && lonParecLat) {
            mostrarErroForm(
                'Parece que latitude e longitude estão trocadas: '
                + 'lat=' + lat + ' caberia em longitude e lon=' + lon + ' caberia em latitude. '
                + 'Clique novamente em "Consultar" se desejar trocar automaticamente.'
            );
            // Troca os valores nos inputs para o usuário só apertar Consultar de novo.
            $('in-lat').value = String(lon).replace('.', ',');
            $('in-lon').value = String(lat).replace('.', ',');
            return null;
        }
        if (lat < -18 || lat > 6 || lon < -74 || lon > -44) {
            mostrarErroForm(
                'Coordenada fora do bounding box da Amazônia Legal '
                + '(lat ∈ [-18, 6], lon ∈ [-74, -44]). '
                + 'Você digitou lat=' + lat + ', lon=' + lon + '. '
                + 'Exemplo válido (Palmas-TO): lat=-10,18 lon=-48,33.'
            );
            return null;
        }
        if (!dataStr) {
            mostrarErroForm('Selecione uma data.');
            return null;
        }
        var partesData = dataStr.split('-');
        var ano = parseInt(partesData[0], 10);
        var mes = parseInt(partesData[1], 10);
        var dia = parseInt(partesData[2], 10);
        var hora = parseInt((horaStr.split(':')[0] || '12'), 10);

        var alvo = new Date(ano, mes - 1, dia, hora);
        if (alvo > new Date()) {
            mostrarErroForm('Data/hora no futuro: o modelo usa reanálise climática (NASA POWER), sem previsão futura.');
            return null;
        }

        limparErroForm();
        return { lat: lat, lon: lon, ano: ano, mes: mes, dia: dia, hora: hora };
    }

    function preencherForm(lat, lon) {
        $('in-lat').value = Number(lat).toFixed(4);
        $('in-lon').value = Number(lon).toFixed(4);
    }

    // ------------ Recebe clique do iframe do Folium ------------
    window.addEventListener('message', function(ev) {
        if (!ev.data || ev.data.tipo !== 'mapa_clique') return;
        // Preenche o form e usa a data/hora atual nos inputs (ou now caso vazio)
        preencherForm(ev.data.lat, ev.data.lon);
        var parsed = parseFormulario();
        if (!parsed) return;
        consultar(parsed, /*centralizar=*/false);
    });

    $('btn-consultar').addEventListener('click', function() {
        var parsed = parseFormulario();
        if (!parsed) return;
        consultar(parsed, /*centralizar=*/true);
    });

    function consultar(params, centralizar) {
        var modelo = $('modelo').value || null;
        $('empty-state').style.display = 'none';
        $('resultado').style.display = 'none';
        $('loading').style.display = 'block';
        var dataHumana = String(params.ano) + '-' + pad2(params.mes) + '-' + pad2(params.dia)
            + ' ' + pad2(params.hora) + 'h';
        $('loading').innerHTML = '<strong>Consultando ' + dataHumana + '…</strong>'
            + '<br><span class="small">Clima NASA POWER · Focos FIRMS · Features Tier 1 · Modelo '
            + (modelo || 'padrão') + '</span>';

        var payload = {
            lat: params.lat, lon: params.lon,
            ano: params.ano, mes: params.mes, dia: params.dia, hora: params.hora,
            modelo: modelo
        };

        fetch('/api/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        })
        .then(r => r.json())
        .then(function(data) {
            $('loading').style.display = 'none';
            if (!data.sucesso) {
                mostrarErro('Falha na previsão', data.erro || 'erro desconhecido');
                return;
            }
            esconderErro();
            try {
                renderResultado(data, params.lat, params.lon);
            } catch (renderErr) {
                console.error('Erro ao renderizar resultado:', renderErr);
                mostrarErro('Erro de renderização', renderErr.message + ' (verifique o console do navegador)');
                return;
            }
            // Pinta marcador no iframe via postMessage (e centraliza se a consulta veio do form)
            var iframe = document.querySelector('#map-area iframe');
            if (iframe && iframe.contentWindow) {
                iframe.contentWindow.postMessage({
                    tipo: 'mapa_marcador',
                    lat: params.lat, lon: params.lon,
                    cor: corLeafletDoRisco(data.risco),
                    centralizar: !!centralizar
                }, '*');
            }
        })
        .catch(function(err) {
            $('loading').style.display = 'none';
            mostrarErro('Erro de comunicação', err.message);
        });
    }

    function mostrarErro(titulo, mensagem) {
        var box = $('erro-comunicacao');
        box.style.display = 'block';
        box.innerHTML = '<div class="card" style="border-left:4px solid var(--c-alto);">'
            + '<h3 style="color: var(--c-alto);">' + titulo + '</h3>'
            + '<div>' + (mensagem || 'sem detalhes') + '</div>'
            + '</div>';
    }

    function esconderErro() {
        var box = $('erro-comunicacao');
        if (box) { box.style.display = 'none'; box.innerHTML = ''; }
    }

    function renderResultado(data, lat, lon) {
        $('resultado').style.display = 'block';
        var du = data.dados_usados || {};
        var ctx = data.contexto_historico || {};
        var explic = data.explicacao || [];

        // Risco
        var rc = riscoClass(data.risco);
        var elRisco = $('risco-label');
        elRisco.className = 'badge-risco ' + rc;
        elRisco.textContent = data.risco || '—';

        $('local-label').innerHTML =
            '<strong>' + (data.municipio || 'Município ?') + '</strong>'
            + ' / ' + (data.estado || '—')
            + ' &middot; ' + fmtNum(lat, 4) + ', ' + fmtNum(lon, 4);

        $('modelo-label').textContent = data.modelo_usado || '—';
        if (data.metricas_modelo && data.metricas_modelo.accuracy != null) {
            var acc = (data.metricas_modelo.accuracy * 100).toFixed(1);
            var f1 = data.metricas_modelo.f1_macro != null ? (data.metricas_modelo.f1_macro * 100).toFixed(1) : null;
            $('modelo-metricas').textContent = 'acc ' + acc + '% · F1m ' + (f1 || '—') + '%';
        } else {
            $('modelo-metricas').textContent = '';
        }

        // Probabilidades
        var pb = $('prob-bar');
        pb.innerHTML = '';
        if (data.probabilidades) {
            var ordemRiscos = ['Baixo', 'Moderado', 'Muito Alto'];
            var chaves = Object.keys(data.probabilidades).sort(function(a, b) {
                return ordemRiscos.indexOf(a) - ordemRiscos.indexOf(b);
            });
            chaves.forEach(function(k) {
                var v = data.probabilidades[k] || 0;
                var cls = riscoClass(k);
                var div = document.createElement('div');
                div.className = 'prob-row';
                div.innerHTML = '<span class="label">' + k + '</span>'
                    + '<div class="track"><div class="fill ' + cls + '" style="width:' + (v * 100).toFixed(1) + '%;"></div></div>'
                    + '<span class="pct">' + (v * 100).toFixed(1) + '%</span>';
                pb.appendChild(div);
            });
        }

        $('trace-modelo').innerHTML = 'Fonte clima: <strong>' + (data.fonte_dados || '—') + '</strong>'
            + (data.janela_clima_nasa ? ' (' + data.janela_clima_nasa + ')' : '')
            + ' &middot; Tier 1 lookup: <strong>' + (ctx.tier1_lookup_granularidade || '—') + '</strong>';

        var trThr = $('trace-thresholds');
        if (data.thresholds_aplicados) {
            var th = data.thresholds_aplicados;
            var f1m = th.f1_macro_esperado != null ? (th.f1_macro_esperado * 100).toFixed(1) + '%' : '—';
            var f1mod = th.f1_moderado_esperado != null ? (th.f1_moderado_esperado * 100).toFixed(1) + '%' : '—';
            trThr.innerHTML = 'Decisão por thresholds calibrados (<strong>' + th.estrategia + '</strong>): '
                + 'P(Moderado)&ge;<strong>' + th.threshold_moderado.toFixed(2) + '</strong>, '
                + 'P(Muito Alto)&ge;<strong>' + th.threshold_muito_alto.toFixed(2) + '</strong>'
                + ' &middot; F1-macro esperado <strong>' + f1m + '</strong>, '
                + 'F1-Moderado <strong>' + f1mod + '</strong>.';
        } else {
            trThr.innerHTML = 'Decisão por <strong>argmax</strong> (sem thresholds calibrados).';
        }

        // Explicação
        var ex = $('explicacao-content');
        if (explic.length === 0) {
            ex.innerHTML = '<div class="small">Sem explicação disponível (modelo ou features fora do escopo).</div>';
        } else {
            ex.innerHTML = explic.map(function(it) {
                var pillCls = it.direcao_risco;
                var p10 = it.p10 != null ? fmtNum(it.p10, 2) : '—';
                var p90 = it.p90 != null ? fmtNum(it.p90, 2) : '—';
                return '<div class="explain-item ' + pillCls + '">'
                    + '<div class="feat">' + it.descricao
                        + ' <span class="pill ' + pillCls + '">' + (pillCls === 'aumenta' ? 'AUMENTA RISCO' : pillCls === 'reduz' ? 'REDUZ RISCO' : 'neutro') + '</span>'
                    + '</div>'
                    + '<div class="num">'
                        + '<span class="pill">valor: ' + fmtNum(it.valor, 2) + '</span>'
                        + '<span class="pill">típico (p10–p90): ' + p10 + ' – ' + p90 + '</span>'
                        + '<span class="pill">z = ' + fmtNum(it.z_score, 2) + '</span>'
                    + '</div>'
                    + '</div>';
            }).join('');
        }

        // Dados climáticos
        var umidade = du.Umidade_rel;
        $('clima-grid').innerHTML = [
            ['Precipitação atual (mm)', fmtNum(du.Precipitacao, 2)],
            ['Precip. média 7d (mm)', fmtNum(du.Precipitacao_ma7, 2)],
            ['Dias sem chuva', fmtNum(du.DiaSemChuva, 0)],
            ['Dias sem chuva (média 7d)', fmtNum(du.DiaSemChuva_ma7, 1)],
            ['Umidade relativa (%)', umidade != null ? fmtNum(umidade, 1) : '—'],
            ['Temp. climatológica (°C)', fmtNum(du.Temp_Climatologica, 1)],
            ['Período crítico (Jul-Set)?', du.Periodo_Critico ? 'Sim' : 'Não'],
            ['Estação / Período', (du.Estacao || '—') + ' · ' + (du.Periodo_Dia || '—')],
        ].map(function(r){ return '<div class="row"><span class="k">'+r[0]+'</span></div><div class="row"><span class="v">'+r[1]+'</span></div>'; }).join('');

        var dataConsultadaStr =
            String(du.Ano) + '-' + String(du.Mes).padStart(2,'0') + '-' + String(du.Dia).padStart(2,'0')
            + ' ' + String(du.Hora).padStart(2,'0') + 'h';
        var hoje = new Date();
        var hojeStr = hoje.getFullYear() + '-' + pad2(hoje.getMonth() + 1) + '-' + pad2(hoje.getDate());
        var dataConsultadaYMD =
            String(du.Ano) + '-' + String(du.Mes).padStart(2,'0') + '-' + String(du.Dia).padStart(2,'0');
        var sufixoHist = (dataConsultadaYMD !== hojeStr) ? ' &middot; <em>consulta histórica</em>' : '';
        $('trace-clima').innerHTML =
            'Data consultada: <strong>' + dataConsultadaStr + '</strong>'
            + (data.janela_clima_nasa ? ' &middot; Janela NASA POWER: ' + data.janela_clima_nasa : '')
            + sufixoHist;

        // Índices de seca
        $('seca-grid').innerHTML = [
            ['Índice de Seca', fmtNum(du.Indice_Seca, 2)],
            ['SPI-1m', fmtNum(du.SPI_1m, 2)],
            ['SPI-3m', fmtNum(du.SPI_3m, 2)],
            ['SPI-6m', fmtNum(du.SPI_6m, 2)],
            ['KBDI proxy', fmtNum(du.KBDI_proxy, 1)],
            ['Aridez De Martonne', fmtNum(du.Aridez_DeMartonne, 1)],
            ['VPD proxy', fmtNum(du.VPD_proxy, 2)],
            ['Dias secos 90d (P<5mm)', fmtNum(du.Dias_Secos_90d, 0)],
        ].map(function(r){ return '<div class="row"><span class="k">'+r[0]+'</span></div><div class="row"><span class="v">'+r[1]+'</span></div>'; }).join('');

        // Acumulados e anomalia
        $('acum-grid').innerHTML = [
            ['Acum. 30 dias (mm)', fmtNum(du.Precipitacao_acum_30d, 1)],
            ['Acum. 90 dias (mm)', fmtNum(du.Precipitacao_acum_90d, 1)],
            ['Acum. 180 dias (mm)', fmtNum(du.Precipitacao_acum_180d, 1)],
            ['Acum. 365 dias (mm)', fmtNum(du.Precipitacao_acum_365d, 1)],
            ['Anomalia P (mm)', fmtNum(du.Anomalia_Precipitacao, 2)],
            ['Anomalia P relativa', fmtNum(du.Anomalia_Precipitacao_rel, 2)],
        ].map(function(r){ return '<div class="row"><span class="k">'+r[0]+'</span></div><div class="row"><span class="v">'+r[1]+'</span></div>'; }).join('');

        var pclim = ctx.precipitacao_climatologia_mm_dia;
        var an = du.Anomalia_Precipitacao;
        if (pclim != null && an != null) {
            var direcao = an < 0 ? 'abaixo' : 'acima';
            var cls = an < 0 ? 'negativa' : 'positiva';
            $('anomalia-narrativa').innerHTML =
                'Precipitação atual está <span class="anomalia ' + cls + '">' + fmtNum(Math.abs(an), 2) + ' mm '
                + direcao + '</span> da média histórica do ' + (ctx.precipitacao_origem === 'municipio_mes' ? 'município' : 'estado')
                + ' nesse mês (' + fmtNum(pclim, 2) + ' mm/dia).';
        } else {
            $('anomalia-narrativa').innerHTML = '';
        }

        // Histórico de fogo
        $('fogo-grid').innerHTML = [
            ['Focos últimos 7d', fmtNum(du.Incendios_Ultimos_7_Dias, 1)],
            ['Focos últimos 30d', fmtNum(du.Incendios_Ultimos_30_Dias, 1)],
            ['Focos últimos 90d (célula)', fmtNum(du.Incendios_Ultimos_90_Dias, 1)],
            ['Focos últimos 180d (célula)', fmtNum(du.Incendios_Ultimos_180_Dias, 1)],
            ['Focos últimos 365d (célula)', fmtNum(du.Incendios_Ultimos_365_Dias, 1)],
            ['Dias desde último foco', fmtNum(du.Dias_Desde_Ultimo_Incendio, 0)],
            ['FRP médio 7d (MW)', fmtNum(du.Media_FRP_Ultimos_7_Dias, 2)],
            ['FRP máx. 7d (MW)', fmtNum(du.Max_FRP_Ultimos_7_Dias, 2)],
            ['FRP no ponto agora (MW)', fmtNum(du.FRP, 2)],
            ['FRP médio célula 30d (MW)', fmtNum(du.Media_FRP_Celula_30d, 2)],
        ].map(function(r){ return '<div class="row"><span class="k">'+r[0]+'</span></div><div class="row"><span class="v">'+r[1]+'</span></div>'; }).join('');

        $('resultado').scrollTop = 0;
    }
})();
</script>
</body>
</html>"""


@app.route('/api/models', methods=['GET'])
def api_models():
    """Endpoint para listar modelos disponíveis (com métricas, se houver)."""
    try:
        modelos = listar_modelos()
        info = []
        for nome in modelos:
            metricas = None
            metricas_path = MODEL_DIR / 'relatorios' / f'{nome}_metrics.json'
            if metricas_path.exists():
                try:
                    with metricas_path.open('r', encoding='utf-8') as fp:
                        m = json.load(fp)
                    metricas = {
                        'accuracy': m.get('accuracy'),
                        'f1_macro': m.get('f1_macro'),
                    }
                except Exception:
                    pass
            info.append({'nome': nome, 'metricas': metricas})
        return jsonify({
            'sucesso': True,
            'modelos': modelos,
            'modelos_info': info,
        })
    except Exception as e:
        logger.error("Erro ao listar modelos: %s", e)
        return jsonify({
            'sucesso': False,
            'erro': str(e)
        }), 500


@app.route('/api/explain', methods=['POST'])
def api_explain():
    """Endpoint dedicado para gerar/refazer a explicação local.

    Útil para front-ends que queiram apenas a explicação sem refazer
    todo o pipeline (útil também para testes manuais)."""
    try:
        data = request.get_json() or {}
        valores = data.get('valores') or {}
        risco = data.get('risco', '')
        top_n = int(data.get('top_n', 6))
        explicacao = explicar_predicao(
            valores_features={k: float(v) for k, v in valores.items() if isinstance(v, (int, float))},
            risco_predito=str(risco),
            importance_path=SHAP_IMPORTANCE_PATH,
            dataset_path=ENRICHED_DATASET_PATH,
            top_n=top_n,
        )
        return jsonify({'sucesso': True, 'explicacao': explicacao})
    except Exception as e:
        logger.error("Erro em /api/explain: %s", e, exc_info=True)
        return jsonify({'sucesso': False, 'erro': str(e)}), 500


@app.route('/api/climate', methods=['POST'])
def api_climate():
    """Endpoint para obter dados climáticos de uma coordenada."""
    try:
        data = request.get_json()
        lat = float(data.get('lat'))
        lon = float(data.get('lon'))
        mes = data.get('mes', datetime.now().month)
        use_api = data.get('use_api', True)
        ref = data.get('reference_date')  # opcional: ISO ou {ano,mes,dia}
        if isinstance(ref, str) and ref:
            ref_d = datetime.fromisoformat(ref.replace('Z', '+00:00'))
        elif isinstance(ref, dict) and 'ano' in ref:
            ref_d = datetime(int(ref['ano']), int(ref.get('mes', mes)), int(ref.get('dia', 1)))
        else:
            ref_d = datetime.now()
        clima_data = climate_provider.get_climate_data(
            lat=lat,
            lon=lon,
            mes=mes,
            use_api=use_api,
            reference_date=ref_d,
            prefer_nasa=data.get('preferir_nasa', True),
            use_hg_fallback=data.get('usar_hg_como_fallback', True),
        )
        
        return jsonify({
            'sucesso': True,
            **clima_data
        })
    except Exception as e:
        logger.error("Erro ao obter dados climáticos: %s", e)
        return jsonify({
            'sucesso': False,
            'erro': str(e)
        }), 500


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """Endpoint para fazer previsão de risco de incêndio."""
    try:
        data = request.get_json()
        lat = float(data.get('lat'))
        lon = float(data.get('lon'))
        # Se modelo vier como string vazia, tratar como None
        nome_modelo = data.get('modelo')
        logger.info("Modelo recebido da requisição (raw): %s (tipo: %s)", nome_modelo, type(nome_modelo))
        
        # Tratar diferentes formatos
        if nome_modelo is None or nome_modelo == '' or nome_modelo == 'null':
            nome_modelo = None
        else:
            # Garantir que é string e remover espaços
            nome_modelo = str(nome_modelo).strip()
            if nome_modelo == '':
                nome_modelo = None
        
        logger.info("Recebida requisição: lat=%.4f, lon=%.4f, modelo=%s", lat, lon, nome_modelo)
        
        # Data/hora — sanitiza (JSON pode chegar como string/float) e valida.
        agora = datetime.now()

        def _to_int(value: Any, default: int) -> int:
            if value is None or value == '':
                return int(default)
            try:
                return int(value)
            except (TypeError, ValueError):
                return int(default)

        ano = _to_int(data.get('ano'), agora.year)
        mes = _to_int(data.get('mes'), agora.month)
        dia = _to_int(data.get('dia'), agora.day)
        hora = _to_int(data.get('hora'), agora.hour)
        if not (1 <= mes <= 12):
            return jsonify({'sucesso': False, 'erro': f'Mês inválido: {mes}'}), 400
        if not (1 <= dia <= 31):
            return jsonify({'sucesso': False, 'erro': f'Dia inválido: {dia}'}), 400
        if not (0 <= hora <= 23):
            return jsonify({'sucesso': False, 'erro': f'Hora inválida: {hora}'}), 400
        try:
            ref_data = datetime(ano, mes, dia, hora)
        except ValueError as exc:
            return jsonify({'sucesso': False, 'erro': f'Data inválida: {exc}'}), 400
        if ref_data > agora:
            return jsonify({
                'sucesso': False,
                'erro': (
                    'Data/hora no futuro não suportada: o pipeline usa reanálise climática '
                    '(NASA POWER), que não fornece previsão. Use uma data passada ou hoje.'
                ),
            }), 400
        logger.info("Data consultada: %s", ref_data.isoformat())
        
        # Geocodificação cedo: medianas de incêndio (treino) dependem de Estado
        cache = carregar_cache()
        estado, municipio = reverse_geocode(lat, lon, cache)
        salvar_cache(cache)
        
        # Dados climáticos (NASA POWER primeiro, alinhado ao enriquecimento Umidade / PRECTOT)
        use_api = data.get('usar_api_clima', True)
        prefer_nasa = data.get("preferir_nasa", data.get("preferir_clima_treino", True))
        use_hg = data.get("usar_hg_como_fallback", True)
        logger.info("Obtendo clima: use_api=%s, prefer_nasa=%s, ref_data=%s", use_api, prefer_nasa, ref_data.date())
        clima_data = climate_provider.get_climate_data(
            lat=lat,
            lon=lon,
            mes=mes,
            use_api=use_api,
            reference_date=ref_data,
            prefer_nasa=prefer_nasa,
            use_hg_fallback=use_hg,
        )
        # Medianas do dataset de treino (distribuição de histórico de incêndio)
        fprox = climate_provider.get_training_fire_proxies(estado or "DESCONHECIDO", int(mes))
        clima_data = {**fprox, **clima_data}
        logger.info(
            "Clima: fonte=%s, P=%.2f, P_ma7=%.2f, Dsem=%.1f, Dsem_ma7=%.1f | treino-Inc7=%.1f",
            clima_data.get("fonte"),
            clima_data.get("precipitacao", 0) or 0,
            clima_data.get("prec_ma7", clima_data.get("precipitacao", 0)) or 0,
            clima_data.get("dias_sem_chuva", 0) or 0,
            clima_data.get("diasem_ma7", clima_data.get("dias_sem_chuva", 0)) or 0,
            fprox.get("Incendios_Ultimos_7_Dias", 0) or 0,
        )
        
        # Permitir override manual (se fornecido explicitamente no request)
        # Mas usar dados da API/clima se disponíveis
        if 'precipitacao' in data and data['precipitacao'] is not None:
            precipitacao = float(data['precipitacao'])
            logger.info("Usando precipitação fornecida manualmente: %.2f mm", precipitacao)
        else:
            precipitacao = clima_data.get('precipitacao', 0.0)
            logger.info("Usando precipitação da fonte de dados: %.2f mm", precipitacao)
        
        if 'dias_sem_chuva' in data and data['dias_sem_chuva'] is not None:
            dias_sem_chuva = float(data['dias_sem_chuva'])
            logger.info("Usando dias_sem_chuva fornecidos manualmente: %s", dias_sem_chuva)
        else:
            dias_sem_chuva = float(clima_data.get('dias_sem_chuva', 0) or 0)
            logger.info("Usando dias_sem_chuva da fonte de dados: %s", dias_sem_chuva)
        
        # FRP (FIRMS) + mesma chamada alimenta Media/Max de FRP (colunas de treino)
        frp = data.get('frp')
        frp_data = None
        if frp is None and use_api:
            logger.info(
                "Buscando FRP (NASA FIRMS) para lat=%.4f, lon=%.4f, ref=%s...",
                lat, lon, ref_data.date(),
            )
            try:
                # days_back=5 (limite atual da NASA FIRMS NRT) e reference_date
                # ancorando a janela na data consultada (não em datetime.now()).
                frp_data = frp_provider.get_frp_from_nasa_firms(
                    lat=lat, lon=lon, radius_km=10.0,
                    days_back=5, reference_date=ref_data,
                )
                if frp_data and frp_data.get('sucesso'):
                    frp = frp_data.get('frp', 0.0)
                    logger.info(
                        "FRP FIRMS: %.2f MW (detecções: %d)",
                        frp, frp_data.get('detections', 0)
                    )
                else:
                    frp = 0.0
            except Exception as e:
                logger.warning("Erro ao buscar FRP: %s. FRP=0.0", e)
                frp = 0.0
        else:
            frp = float(frp) if frp is not None else 0.0
            if data.get('frp') is not None:
                logger.info("FRP fornecido manualmente: %.2f MW", frp)
        logger.info("FRP ponto: %.2f (0 = sem fogo ativo no raio)", frp)

        # Sobrescreve apenas as features que no TREINO já eram observação real
        # individual (FRP médio/máx no entorno + dias desde último foco).
        #
        # IMPORTANTE: NÃO sobrescrevemos `Incendios_Ultimos_7/30_Dias` aqui.
        # No pipeline de inferência, essas colunas vêm da mediana (Estado, Mês)
        # do dataset de treino. Trocar pelo valor real medido pelo FIRMS leva a
        # OOD (out-of-distribution) — testes empíricos mostraram que injetar
        # contagens reais altas em meses não-críticos AUMENTA a confiança em
        # "Baixo" (o modelo aprendeu o sinal como sazonal, não causal).
        # Esse comportamento está documentado como limitação no TCC.
        if frp_data and frp_data.get("sucesso"):
            detec = int(frp_data.get("detections", 0) or 0)
            frp_max = float(frp_data.get("frp_max", 0.0) or 0.0)
            frp_mean = float(frp_data.get("frp_mean", 0.0) or 0.0)
            if detec > 0 or frp_max > 0:
                clima_data["Media_FRP_Ultimos_7_Dias"] = frp_mean
                clima_data["Max_FRP_Ultimos_7_Dias"] = frp_max
                # Se há detecção FIRMS na janela, o último foco real foi ≤ 5 d atrás.
                clima_data["Dias_Desde_Ultimo_Incendio"] = min(
                    float(clima_data.get("Dias_Desde_Ultimo_Incendio", 365.0) or 365.0),
                    5.0,
                )
                logger.info(
                    "FIRMS NRT atualiza FRP/DiasDesdeUlt (Inc7/30 mantidos como "
                    "mediana de treino p/ não sair da distribuição): detec=%d, "
                    "FRP_max=%.2f, DiasDesdeUlt=%s",
                    detec, frp_max, clima_data["Dias_Desde_Ultimo_Incendio"],
                )
        
        clima_data['precipitacao'] = precipitacao
        clima_data['dias_sem_chuva'] = float(dias_sem_chuva)
        if data.get('precipitacao') is not None:
            clima_data['prec_ma7'] = float(precipitacao)
        if data.get('dias_sem_chuva') is not None:
            clima_data['diasem_ma7'] = float(dias_sem_chuva)
        
        # Preparar dados
        dados = preparar_dados_previsao(
            lat=lat,
            lon=lon,
            estado=estado,
            municipio=municipio,
            clima_data=clima_data,
            ano=ano,
            mes=mes,
            dia=dia,
            hora=hora,
            frp=frp,
        )
        
        # Carregar modelo
        logger.info("Carregando modelo: %s", nome_modelo or "padrão")
        modelo, modelo_nome = carregar_modelo(nome_modelo)
        logger.info("Modelo carregado: %s", modelo_nome)
        
        # Fazer previsão (o modelo é um Pipeline que inclui preprocessor)
        logger.info("Fazendo previsão com dados: %s", dados.to_dict('records')[0])
        previsao = modelo.predict(dados)
        risco = previsao[0]
        logger.info("Previsão (argmax): risco=%s", risco)

        # Obter confiança e probabilidades de todas as classes se disponível
        confianca = None
        probabilidades = None
        if hasattr(modelo, 'predict_proba'):
            probas = modelo.predict_proba(dados)
            confianca = float(probas.max())
            classes = modelo.classes_ if hasattr(modelo, 'classes_') else ['Baixo', 'Moderado', 'Muito Alto']
            probabilidades = {
                str(classe): float(prob)
                for classe, prob in zip(classes, probas[0])
            }
            logger.info("Confiança: %.2f%% | Probabilidades: %s",
                       confianca * 100,
                       {k: f"{v:.2%}" for k, v in probabilidades.items()})

        # Aplicar thresholds otimizados (default: f1_macro). Mantém compatibilidade
        # se o arquivo `prediction_thresholds.json` não existir (cai no argmax).
        estrategia_thresh = str(
            data.get('estrategia_thresholds', data.get('usar_thresholds', 'f1_macro'))
        ).lower().strip()
        thresholds_aplicados: Optional[Dict[str, Any]] = None
        if probabilidades is not None and estrategia_thresh != 'argmax':
            thr_info = _carregar_thresholds_otimizados()
            if thr_info:
                bloco_key = (
                    'thresholds_otimizados_f1_moderado'
                    if estrategia_thresh in {'f1_moderado', 'moderado'}
                    else 'thresholds_otimizados_f1_macro'
                )
                bloco = thr_info.get(bloco_key) or thr_info.get('thresholds_otimizados_f1_macro')
                if bloco:
                    thr_mod = float(bloco.get('threshold_moderado', 0.5))
                    thr_alt = float(bloco.get('threshold_muito_alto', 0.5))
                    risco_thr = _aplicar_thresholds_multiclasse(probabilidades, thr_mod, thr_alt)
                    if risco_thr != risco:
                        logger.info(
                            "Threshold override (%s): %s → %s | thr_mod=%.2f thr_alto=%.2f",
                            estrategia_thresh, risco, risco_thr, thr_mod, thr_alt,
                        )
                    risco = risco_thr
                    thresholds_aplicados = {
                        'estrategia': estrategia_thresh,
                        'threshold_moderado': thr_mod,
                        'threshold_muito_alto': thr_alt,
                        'f1_macro_esperado': bloco.get('f1_macro_esperado'),
                        'f1_moderado_esperado': bloco.get('f1_moderado_esperado'),
                        'accuracy_esperada': bloco.get('accuracy_esperada'),
                    }
        
        # Obter métricas do modelo se disponíveis
        metricas_modelo = None
        try:
            metricas_path = MODEL_DIR / 'relatorios' / f'{modelo_nome}_metrics.json'
            if metricas_path.exists():
                with metricas_path.open('r', encoding='utf-8') as f:
                    metricas_modelo = json.load(f)
                    # Extrair apenas informações relevantes
                    metricas_modelo = {
                        'accuracy': metricas_modelo.get('accuracy'),
                        'f1_macro': metricas_modelo.get('f1_macro'),
                    }
        except Exception as e:
            logger.warning("Não foi possível carregar métricas do modelo: %s", e)
        
        # Construir o dicionário rico de features usadas (para UI + explainer)
        row = dados.to_dict('records')[0]
        adv_meta = dados.attrs.get('adv_meta', {})

        vento_ma7: Optional[float] = None
        vento_fonte: Optional[str] = None
        if clima_data.get("vento_inmet_ms_ma7") is not None:
            vento_ma7 = float(clima_data["vento_inmet_ms_ma7"])
            vento_fonte = "inmet_wis2"
        elif clima_data.get("ws2m_ma7_ms") is not None:
            vento_ma7 = float(clima_data["ws2m_ma7_ms"])
            vento_fonte = "nasa_merra_ws2m_ma7"
        incerteza_operacional = multiclass_operational_uncertainty(probabilidades)
        fwi_proxy_val = fire_weather_proxy_heuristic(
            dia_sem_chuva_ma7=float(row.get("DiaSemChuva_ma7") or 0.0),
            umidade_pct=clima_data.get("umidade"),
            vento_ms_ma7=vento_ma7,
        )

        # Climatologia local (P média do município no mês) para contexto histórico
        p_clim_mes, p_clim_origem = get_precip_climatologia(
            municipio, estado, int(mes), ENRICHED_DATASET_PATH,
        )

        # Explicação local (Top-N features que mais influenciam esta previsão)
        try:
            valores_para_explainer = {k: v for k, v in row.items() if isinstance(v, (int, float))}
            explicacao = explicar_predicao(
                valores_features=valores_para_explainer,
                risco_predito=str(risco),
                importance_path=SHAP_IMPORTANCE_PATH,
                dataset_path=ENRICHED_DATASET_PATH,
                top_n=6,
            )
        except Exception as exc:
            logger.warning("Falha ao gerar explicação local: %s", exc)
            explicacao = []

        # ------------------- Dados usados (estendido p/ UI) -------------------
        def _f(k: str, default: float = 0.0) -> float:
            v = row.get(k, default)
            try:
                return float(v if v is not None else default)
            except (TypeError, ValueError):
                return float(default)

        dados_usados = {
            # Bloco "Dados climáticos atuais" (NASA POWER / HG)
            'DiaSemChuva': _f('DiaSemChuva'),
            'Precipitacao': _f('Precipitacao'),
            'Precipitacao_ma7': _f('Precipitacao_ma7'),
            'DiaSemChuva_ma7': _f('DiaSemChuva_ma7'),
            'Umidade_rel': clima_data.get('umidade'),
            'Temp_Climatologica': _f('Temp_Climatologica'),
            # Bloco "Índices de seca/aridez"
            'Indice_Seca': _f('Indice_Seca'),
            'KBDI_proxy': _f('KBDI_proxy'),
            'Aridez_DeMartonne': _f('Aridez_DeMartonne'),
            'VPD_proxy': _f('VPD_proxy'),
            'SPI_1m': _f('SPI_1m'),
            'SPI_3m': _f('SPI_3m'),
            'SPI_6m': _f('SPI_6m'),
            # Bloco "Precipitação acumulada"
            'Precipitacao_acum_30d': _f('Precipitacao_acum_30d'),
            'Precipitacao_acum_90d': _f('Precipitacao_acum_90d'),
            'Precipitacao_acum_180d': _f('Precipitacao_acum_180d'),
            'Precipitacao_acum_365d': _f('Precipitacao_acum_365d'),
            'Dias_Secos_90d': _f('Dias_Secos_90d'),
            # Bloco "Anomalia vs. histórico"
            'Anomalia_Precipitacao': _f('Anomalia_Precipitacao'),
            'Anomalia_Precipitacao_rel': _f('Anomalia_Precipitacao_rel'),
            # Bloco "Histórico de fogo"
            'Incendios_Ultimos_7_Dias': _f('Incendios_Ultimos_7_Dias'),
            'Incendios_Ultimos_30_Dias': _f('Incendios_Ultimos_30_Dias'),
            'Incendios_Ultimos_90_Dias': _f('Incendios_Ultimos_90_Dias'),
            'Incendios_Ultimos_180_Dias': _f('Incendios_Ultimos_180_Dias'),
            'Incendios_Ultimos_365_Dias': _f('Incendios_Ultimos_365_Dias'),
            'Dias_Desde_Ultimo_Incendio': _f('Dias_Desde_Ultimo_Incendio', 365.0),
            'Media_FRP_Ultimos_7_Dias': _f('Media_FRP_Ultimos_7_Dias'),
            'Max_FRP_Ultimos_7_Dias': _f('Max_FRP_Ultimos_7_Dias'),
            'Media_FRP_Celula_30d': _f('Media_FRP_Celula_30d'),
            # Bloco "Localização e tempo"
            'Latitude': float(lat),
            'Longitude': float(lon),
            'FRP': float(frp),
            'Ano': int(ano),
            'Mes': int(mes),
            'Dia': int(dia),
            'Hora': int(hora),
            'Periodo_Critico': int(row.get('Periodo_Critico', 0)),
            'Estacao': row.get('Estacao'),
            'Periodo_Dia': row.get('Periodo_Dia'),
            # Estado da arte leve: vento + proxy de *fire weather* (não são colunas do treino)
            'Vento_ma7_ms': vento_ma7,
            'Vento_fonte': vento_fonte,
            'Tmax_ma7_merra_c': (
                float(clima_data['t2m_max_ma7_c'])
                if clima_data.get('t2m_max_ma7_c') is not None else None
            ),
            'FWI_fire_weather_proxy': fwi_proxy_val,
            'Incerteza_gap_top2': incerteza_operacional.get('gap_top2'),
            'Incerteza_alta': incerteza_operacional.get('incerteza_alta'),
        }

        contexto_historico = {
            'precipitacao_climatologia_mm_dia': p_clim_mes,
            'precipitacao_origem': p_clim_origem,
            'tier1_lookup_granularidade': adv_meta.get('granularidade'),
            'estacao_inmet': clima_data.get('estacao_inmet'),
            'distancia_estacao_inmet_km': clima_data.get('distancia_estacao_inmet_km'),
            'cobertura_inmet_pct': clima_data.get('cobertura_inmet_pct'),
            'inmet_representatividade': clima_data.get('inmet_representatividade'),
            'inmet_busca_raio_km': clima_data.get('inmet_busca_raio_km'),
            'precipitacao_merra_ma7': clima_data.get('prec_ma7_merra'),
            'diasem_merra_ma7': clima_data.get('diasem_ma7_merra'),
            'vento_merra_ma7_ms': clima_data.get('ws2m_ma7_ms'),
            'vento_inmet_ma7_ms': clima_data.get('vento_inmet_ms_ma7'),
            't2m_max_ma7_merra_c': clima_data.get('t2m_max_ma7_c'),
        }

        # Auditoria operacional: registra a predição em JSONL append-only
        # para posterior cruzamento com FIRMS (ver scripts/auditar_predicoes.py).
        features_chave_audit = {
            'KBDI_proxy': dados_usados.get('KBDI_proxy'),
            'VPD_proxy': dados_usados.get('VPD_proxy'),
            'Indice_Seca': dados_usados.get('Indice_Seca'),
            'SPI_3m': dados_usados.get('SPI_3m'),
            'Precipitacao_ma7': dados_usados.get('Precipitacao_ma7'),
            'DiaSemChuva_ma7': dados_usados.get('DiaSemChuva_ma7'),
            'Precipitacao_acum_180d': dados_usados.get('Precipitacao_acum_180d'),
            'Dias_Secos_90d': dados_usados.get('Dias_Secos_90d'),
            'Anomalia_Precipitacao_rel': dados_usados.get('Anomalia_Precipitacao_rel'),
            'Incendios_Ultimos_7_Dias': dados_usados.get('Incendios_Ultimos_7_Dias'),
            'Incendios_Ultimos_30_Dias': dados_usados.get('Incendios_Ultimos_30_Dias'),
            'Dias_Desde_Ultimo_Incendio': dados_usados.get('Dias_Desde_Ultimo_Incendio'),
            'Media_FRP_Ultimos_7_Dias': dados_usados.get('Media_FRP_Ultimos_7_Dias'),
            'Max_FRP_Ultimos_7_Dias': dados_usados.get('Max_FRP_Ultimos_7_Dias'),
            'FRP': dados_usados.get('FRP'),
            'FWI_fire_weather_proxy': fwi_proxy_val,
            'Incerteza_gap_top2': incerteza_operacional.get('gap_top2'),
        }
        prediction_id = log_prediction(
            lat=lat,
            lon=lon,
            estado=estado,
            municipio=municipio,
            ref_data_iso=ref_data.isoformat(),
            risco=str(risco),
            confianca=confianca,
            probabilidades=probabilidades,
            modelo=modelo_nome,
            fonte_clima=clima_data.get('fonte', 'desconhecido'),
            fonte_clima_detalhe={
                'janela_nasa': clima_data.get('janela_nasa'),
                'fonte_inmet': clima_data.get('fonte_inmet'),
                'distancia_estacao_inmet_km': clima_data.get('distancia_estacao_inmet_km'),
                'inmet_representatividade': clima_data.get('inmet_representatividade'),
                'inmet_busca_raio_km': clima_data.get('inmet_busca_raio_km'),
            },
            granularidade_tier1=adv_meta.get('granularidade'),
            features_chave=features_chave_audit,
            thresholds_aplicados=thresholds_aplicados,
            extra={
                'vento_fonte': vento_fonte,
                'Vento_ma7_ms': vento_ma7,
            },
        )

        # Preparar resposta
        resposta = {
            'sucesso': True,
            'prediction_id': prediction_id,
            'risco': risco,
            'confianca': confianca,  # Probabilidade da classe predita (máxima)
            'probabilidades': probabilidades,  # Probabilidades de todas as classes
            'municipio': municipio,
            'estado': estado,
            'modelo_usado': modelo_nome,
            'metricas_modelo': metricas_modelo,  # Acurácia e F1 do modelo no teste
            'thresholds_aplicados': thresholds_aplicados,  # None = argmax padrão
            'fonte_dados': clima_data.get('fonte', 'desconhecido'),
            'janela_clima_nasa': clima_data.get('janela_nasa'),
            'dados_usados': dados_usados,
            'contexto_historico': contexto_historico,
            'incerteza_operacional': incerteza_operacional,
            'explicacao': explicacao,
            'aviso': (
                'Clima: NASA POWER (reanálise) quando disponível, alinhado ao enriquecimento de Umidade; '
                'HG é fallback. INMET: busca hierárquica (raio preferencial depois estendido) com rótulo de '
                'representatividade espacial. Vento WS2M (MERRA-2) e, se WIS2, vento na estação sinótica. '
                'FWI_fire_weather_proxy combina seca média na semana, aridez (1−RH) e vento (indicador interpretável, '
                'não substitui FWI canadense). Incerteza operacional: gap entre as duas classes mais prováveis. '
                'Histórico de incêndio: medianas do dataset de treino (Estado+mês); '
                'FIRMS atualiza FRP e média/máx na janela. '
                'Features Tier 1 (SPI, KBDI proxy, anomalias, etc.) preenchidas via lookup '
                'espaço-sazonal do dataset enriquecido. '
                'Confiança = probabilidade da classe predita pelo modelo calibrado. '
                'Explicação = importância global × anormalidade local × sinal físico esperado. '
                'Quando thresholds_aplicados ≠ null, a classe foi decidida por thresholds calibrados '
                '(F1-macro/F1-Moderado) em vez do argmax.'
            ),
        }

        return jsonify(resposta)
        
    except Exception as e:
        logger.error("Erro ao fazer previsão: %s", e, exc_info=True)
        return jsonify({
            'sucesso': False,
            'erro': str(e)
        }), 500


if __name__ == '__main__':
    logger.info("Iniciando aplicação Flask...")
    logger.info("Acesse http://localhost:5000 para ver o mapa interativo")
    app.run(debug=True, host='0.0.0.0', port=5000)

