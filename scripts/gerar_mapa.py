import argparse
import json
import logging
import time
import webbrowser
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import folium
import joblib
import numpy as np
import pandas as pd
from folium import Element
from folium.plugins import MarkerCluster
from geopy.exc import GeocoderTimedOut
from geopy.geocoders import Nominatim

from carregar_dados import _adicionar_features_derivadas, _adicionar_features_temporais_janela
from pre_processor import PreProcessor


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR.parent / 'modelos'
METRICS_DIR = MODEL_DIR / 'relatorios'
PREPROCESSOR_METADATA_PATH = MODEL_DIR / 'preprocessor_metadata.json'
GEOCODE_CACHE_PATH = MODEL_DIR / 'geocode_cache.json'

DEFAULT_MAP_PATH = BASE_DIR / 'mapa_risco_amazonia_com_previsoes.html'
DEFAULT_VIEW = (-5, -60)
DEFAULT_ZOOM = 4

# Ordem de preferência quando --modelo não é informado (alinhado ao treino recente)
MODELOS_PREFERIDOS = [
    'ensemble_stacking',
    'ensemble_voting_soft',
    'random_forest_balanced',
    'random_forest_smote',
    'logistic_regression_balanced',
    'random_forest',
    'logistic_regression',
]

geolocator = Nominatim(user_agent="tcc-mapeamento")


def carregar_preprocessor_metadata() -> PreProcessor:
    if not PREPROCESSOR_METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Metadados do pré-processador não encontrados em {PREPROCESSOR_METADATA_PATH}. "
            "Execute o treinamento antes de gerar o mapa."
        )
    return PreProcessor.load_metadata(str(PREPROCESSOR_METADATA_PATH))


def carregar_cache() -> Dict[str, Tuple[float, float]]:
    if GEOCODE_CACHE_PATH.exists():
        with GEOCODE_CACHE_PATH.open('r', encoding='utf-8') as fp:
            cache = json.load(fp)
        return {k: tuple(v) for k, v in cache.items()}
    return {}


def salvar_cache(cache: Dict[str, Tuple[float, float]]) -> None:
    serializavel = {k: list(v) for k, v in cache.items()}
    with GEOCODE_CACHE_PATH.open('w', encoding='utf-8') as fp:
        json.dump(serializavel, fp, ensure_ascii=False, indent=2)
    logger.info("Cache de geocodificação salvo em %s", GEOCODE_CACHE_PATH)


def geocodificar_municipio(municipio: str, estado: str, cache: Dict[str, Tuple[float, float]]):
    chave = f"{municipio.upper()}|{estado.upper()}"
    if chave in cache:
        return cache[chave]
    try:
        localizacao = geolocator.geocode(f"{municipio}, {estado}, Brasil", timeout=15)
        if localizacao:
            coordenadas = (localizacao.latitude, localizacao.longitude)
            cache[chave] = coordenadas
            return coordenadas
        logger.warning("Não foi possível geocodificar %s - %s", municipio, estado)
        return None, None
    except GeocoderTimedOut:
        logger.warning("Timeout ao geocodificar %s - %s. Tentando novamente...", municipio, estado)
        time.sleep(1)
        return geocodificar_municipio(municipio, estado, cache)


def carregar_modelo(selecao: Optional[str]) -> Path:
    disponiveis = {p.stem: p for p in MODEL_DIR.glob('*.pkl')}
    if not disponiveis:
        raise FileNotFoundError(f"Nenhum modelo encontrado em {MODEL_DIR}. Execute o treinamento primeiro.")

    if selecao:
        caminho = disponiveis.get(selecao)
        if caminho is None:
            raise FileNotFoundError(
                f"Modelo '{selecao}' não encontrado. Opções: {', '.join(sorted(disponiveis))}"
            )
        return caminho

    for nome in MODELOS_PREFERIDOS:
        if nome in disponiveis:
            logger.info("Modelo padrão (preferência): %s", nome)
            return disponiveis[nome]

    primeiro = sorted(disponiveis.values(), key=lambda p: p.stem)[0]
    logger.warning("Nenhum modelo preferido encontrado; usando %s", primeiro.stem)
    return primeiro


def carregar_metricas_modelo(stem: str) -> Optional[dict]:
    path = METRICS_DIR / f'{stem}_metrics.json'
    if not path.exists():
        return None
    try:
        with path.open('r', encoding='utf-8') as fp:
            return json.load(fp)
    except Exception as exc:
        logger.warning("Não foi possível ler métricas de %s: %s", path, exc)
        return None


def carregar_exemplos(
    csv_path: Optional[Path],
    amostra_dataset: Optional[Path],
    n_amostra: Optional[int],
) -> pd.DataFrame:
    if amostra_dataset is not None:
        n = n_amostra if n_amostra and n_amostra > 0 else 500
        logger.info("Amostrando %d linhas de %s", n, amostra_dataset)
        return pd.read_csv(amostra_dataset, nrows=n)

    if csv_path:
        logger.info("Carregando exemplos a partir de %s", csv_path)
        return pd.read_csv(csv_path)

    logger.info("Usando amostra padrão de municípios (cenários ilustrativos).")
    return pd.DataFrame(
        {
            'DiaSemChuva': [3, 10, 8, 12, 0, 5],
            'Precipitacao': [0.2, 2.1, 0.0, 4.5, 0.1, 3.8],
            'Latitude': [None] * 6,
            'Longitude': [None] * 6,
            'FRP': [5.0, 12.0, 0.0, 50.0, 1.5, 2.0],
            'Ano': [2023, 2024, 2025, 2023, 2022, 2021],
            'Mes': [8, 7, 9, 5, 6, 12],
            'Dia': [17, 22, 14, 3, 18, 25],
            'Hora': [16, 18, 17, 15, 19, 16],
            'Estado': ['AMAZONAS', 'PARÁ', 'RONDÔNIA', 'ACRE', 'MATO GROSSO', 'TOCANTINS'],
            'Municipio': ['MANAUS', 'BELÉM', 'PORTO VELHO', 'RIO BRANCO', 'CUIABÁ', 'PALMAS'],
        }
    )


def preparar_features_para_modelo(df: pd.DataFrame, metadata: PreProcessor) -> pd.DataFrame:
    """
    Replica o enriquecimento usado em treino/app (derivadas + janela 7d + defaults de histórico).
    """
    df = df.copy()
    df.replace(-999, np.nan, inplace=True)
    if 'FRP' in df.columns:
        df['FRP'] = pd.to_numeric(df['FRP'], errors='coerce').fillna(0.0)

    df = _adicionar_features_derivadas(df)
    df = _adicionar_features_temporais_janela(df)

    alvo_num = list(metadata.num_features)
    alvo_cat = list(metadata.cat_features)
    alvo: List[str] = alvo_num + alvo_cat

    defaults = {
        'Incendios_Ultimos_7_Dias': 0,
        'Incendios_Ultimos_30_Dias': 0,
        'Dias_Desde_Ultimo_Incendio': 365.0,
        'Media_FRP_Ultimos_7_Dias': 0.0,
        'Max_FRP_Ultimos_7_Dias': 0.0,
    }

    for col in alvo:
        if col in df.columns:
            continue
        if col in defaults:
            df[col] = defaults[col]
        elif col == 'Precipitacao_ma7' and 'Precipitacao' in df.columns:
            df[col] = df['Precipitacao']
        elif col == 'DiaSemChuva_ma7' and 'DiaSemChuva' in df.columns:
            df[col] = df['DiaSemChuva']
        elif col in alvo_num:
            df[col] = 0.0
        else:
            df[col] = 'DESCONHECIDO'

    for c in alvo_cat:
        if c in df.columns:
            df[c] = df[c].fillna('DESCONHECIDO').astype(str).str.upper()

    faltando = [c for c in alvo if c not in df.columns]
    if faltando:
        raise ValueError(f"Colunas ainda ausentes após enriquecimento: {faltando}")

    return df[alvo].copy()


def preencher_coordenadas(exemplos: pd.DataFrame, cache: Dict[str, Tuple[float, float]]) -> pd.DataFrame:
    if 'Latitude' not in exemplos.columns or 'Longitude' not in exemplos.columns:
        raise ValueError("As colunas 'Latitude' e 'Longitude' são obrigatórias no conjunto de dados.")

    exemplos = exemplos.copy()
    for idx, row in exemplos.iterrows():
        if pd.notna(row['Latitude']) and pd.notna(row['Longitude']):
            continue
        lat, lon = geocodificar_municipio(row['Municipio'], row['Estado'], cache)
        exemplos.at[idx, 'Latitude'] = lat
        exemplos.at[idx, 'Longitude'] = lon

    exemplos = exemplos.dropna(subset=['Latitude', 'Longitude'])
    return exemplos


def obter_confiancas(modelo, dados: pd.DataFrame):
    if hasattr(modelo, 'predict_proba'):
        probas = modelo.predict_proba(dados)
        return probas.max(axis=1), probas
    return None, None


def _html_legenda_titulo(modelo_stem: str, metricas: Optional[dict]) -> str:
    acc_txt = ""
    if metricas and metricas.get('accuracy') is not None:
        acc_txt = f"<br><span style='font-size:11px'>Acurácia (hold-out): {metricas['accuracy']*100:.2f}%</span>"
    return f"""
    <div style="position:fixed;top:10px;left:50px;width:420px;z-index:9999;
                background-color:white;padding:10px 14px;border:2px solid #333;
                font-size:14px;box-shadow:0 2px 6px rgba(0,0,0,0.25);">
        <strong>Previsão de risco de incêndio</strong><br>
        <span style="font-size:12px">Modelo: <code>{modelo_stem}</code></span>
        {acc_txt}
    </div>
    <div style="position:fixed;bottom:24px;left:24px;width:220px;z-index:9999;
                background-color:rgba(255,255,255,0.92);padding:10px 12px;border:1px solid #888;
                font-size:12px;line-height:1.5;">
        <strong>Legenda</strong><br>
        <span style="color:green">●</span> Baixo<br>
        <span style="color:orange">●</span> Moderado<br>
        <span style="color:red">●</span> Muito Alto
    </div>
    """


def gerar_mapa(
    modelo_path: Path,
    csv_path: Optional[Path],
    amostra_dataset: Optional[Path],
    n_amostra: Optional[int],
    abrir_no_navegador: bool,
    saida: Path,
):
    metadata_preprocessor = carregar_preprocessor_metadata()
    cache = carregar_cache()

    exemplos_raw = carregar_exemplos(csv_path, amostra_dataset, n_amostra)

    colunas_geo = {'Municipio', 'Estado', 'Latitude', 'Longitude'}
    if not colunas_geo.issubset(set(exemplos_raw.columns)):
        raise ValueError(
            f"O CSV precisa conter colunas: {sorted(colunas_geo)}. "
            "Use --csv com essas colunas ou --amostra-dataset com base histórica."
        )

    exemplos_geo = preencher_coordenadas(exemplos_raw, cache)
    if exemplos_geo.empty:
        raise ValueError("Nenhum registro com coordenadas válidas para plotar no mapa.")

    salvar_cache(cache)

    exemplos = preparar_features_para_modelo(exemplos_geo, metadata_preprocessor)

    modelo = joblib.load(modelo_path)
    previsoes = modelo.predict(exemplos)
    exemplos_geo = exemplos_geo.loc[exemplos.index].copy()
    exemplos_geo['RiscoPrevisto'] = previsoes

    conf, probas = obter_confiancas(modelo, exemplos)
    if conf is not None:
        exemplos_geo['Confianca'] = conf

    classes = list(modelo.classes_) if hasattr(modelo, 'classes_') else []
    logger.info("Previsões geradas:\n%s", exemplos_geo[['Municipio', 'Estado', 'RiscoPrevisto']])

    mapa = folium.Map(location=DEFAULT_VIEW, zoom_start=DEFAULT_ZOOM, tiles='OpenStreetMap')
    folium.TileLayer('CartoDB positron', name='CartoDB claro').add_to(mapa)

    modelo_stem = modelo_path.stem
    metricas = carregar_metricas_modelo(modelo_stem)
    mapa.get_root().html.add_child(Element(_html_legenda_titulo(modelo_stem, metricas)))

    cluster = MarkerCluster(name='Focos').add_to(mapa)
    cor_risco = {'Baixo': 'green', 'Moderado': 'orange', 'Muito Alto': 'red'}

    for pos, (_, row) in enumerate(exemplos_geo.iterrows()):
        popup_parts = [
            f"<b>{row['Municipio']}</b> ({row['Estado']})",
            f"Risco: <b style='color:{cor_risco.get(row['RiscoPrevisto'], 'gray')}'>{row['RiscoPrevisto']}</b>",
        ]
        if 'Confianca' in row and pd.notna(row['Confianca']):
            popup_parts.append(f"Confiança (classe predita): {row['Confianca']*100:.1f}%")
        if probas is not None and classes and pos < probas.shape[0]:
            linha_prob = '<br>'.join(
                f"{c}: {probas[pos, j]*100:.1f}%"
                for j, c in enumerate(classes)
            )
            popup_parts.append(f"<small>{linha_prob}</small>")

        folium.Marker(
            location=[row['Latitude'], row['Longitude']],
            popup=folium.Popup('<br>'.join(popup_parts), max_width=280),
            icon=folium.Icon(color=cor_risco.get(row['RiscoPrevisto'], 'blue')),
        ).add_to(cluster)

    folium.LayerControl(collapsed=False).add_to(mapa)

    saida.parent.mkdir(parents=True, exist_ok=True)
    mapa.save(saida)
    logger.info("Mapa salvo em %s", saida)
    if abrir_no_navegador:
        webbrowser.open("file://" + str(saida.resolve()))


def main():
    parser = argparse.ArgumentParser(
        description="Gera mapa HTML com previsões de risco (compatível com modelos recentes / ensembles)."
    )
    parser.add_argument('--modelo', type=str, help="Nome do modelo salvo (sem extensão .pkl), ex.: ensemble_stacking")
    parser.add_argument('--csv', type=str, help="CSV com colunas mínimas + clima (Municipio, Estado, Lat/Lon, …)")
    parser.add_argument(
        '--amostra-dataset',
        type=str,
        help="CSV grande (ex.: base_de_dados_com_historico.csv); usa as primeiras N linhas com --amostra",
    )
    parser.add_argument(
        '--amostra',
        type=int,
        default=400,
        help="Número de linhas ao ler de --amostra-dataset (padrão: 400)",
    )
    parser.add_argument('--saida', type=str, default=str(DEFAULT_MAP_PATH), help="Caminho do HTML de saída")
    parser.add_argument('--no-browser', action='store_true', help="Não abrir o mapa automaticamente no navegador")
    args = parser.parse_args()

    modelo_path = carregar_modelo(args.modelo)
    csv_path = Path(args.csv) if args.csv else None
    amostra_ds = Path(args.amostra_dataset) if args.amostra_dataset else None
    if amostra_ds and not amostra_ds.exists():
        raise FileNotFoundError(amostra_ds)
    saida = Path(args.saida)

    gerar_mapa(
        modelo_path,
        csv_path,
        amostra_ds,
        args.amostra,
        abrir_no_navegador=not args.no_browser,
        saida=saida,
    )


if __name__ == '__main__':
    main()
