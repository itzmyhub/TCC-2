import json
import logging
import os
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from features_avancadas import (
        ADVANCED_FEATURES_NUM,
        adicionar_features_avancadas,
    )
    HAS_FEATURES_AVANCADAS = True
except ImportError:  # pragma: no cover
    HAS_FEATURES_AVANCADAS = False
    ADVANCED_FEATURES_NUM = []


logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, '..', 'base_de_dados.csv')
MODELOS_DIR = os.path.join(BASE_DIR, '..', 'modelos')
RISK_THRESHOLD_FILE = os.path.join(MODELOS_DIR, 'risk_thresholds.json')

NUM_FEATURES = [
    'DiaSemChuva',
    'Precipitacao',
    'Latitude',
    'Longitude',
    'FRP',
    'Ano',
    'Mes',
    'Dia',
    'Hora',
    # 'Umidade',  # Adicionar apenas se dados reais estiverem disponíveis
    # Para enriquecer com dados reais, usar: scripts/enriquecer_dados_umidade.py
    # Features de histórico de incêndios (adicionadas por adicionar_historico_incendios.py)
    # 'Incendios_Ultimos_7_Dias',
    # 'Incendios_Ultimos_30_Dias',
    # 'Dias_Desde_Ultimo_Incendio',
    # 'Media_FRP_Ultimos_7_Dias',
    # 'Max_FRP_Ultimos_7_Dias',
]

CAT_FEATURES = ['Estado', 'Municipio']

REQUIRED_COLUMNS = set(NUM_FEATURES + CAT_FEATURES + ['RiscoFogo'])

DEFAULT_RISK_QUANTILES = (0.33, 0.66)


def _ensure_model_dir() -> None:
    os.makedirs(MODELOS_DIR, exist_ok=True)


def carregar_csv(caminho: str) -> pd.DataFrame:
    if not os.path.exists(caminho):
        raise FileNotFoundError(f"Arquivo não encontrado: {caminho}")
    logger.info("Carregando CSV de %s", caminho)
    return pd.read_csv(caminho)


def validar_colunas(df: pd.DataFrame, required_columns: Iterable[str]) -> None:
    missing = set(required_columns) - set(df.columns)
    if missing:
        raise ValueError(f"Colunas obrigatórias ausentes no dataset: {sorted(missing)}")


def _clip_outliers(df: pd.DataFrame, columns: Iterable[str], lower_q: float = 0.01, upper_q: float = 0.99) -> pd.DataFrame:
    df = df.copy()
    for col in columns:
        if col not in df.columns:
            continue
        lower, upper = df[col].quantile([lower_q, upper_q])
        df[col] = df[col].clip(lower=lower, upper=upper)
    return df


def limpar_dados(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.drop(columns=['Satelite', 'Pais', 'Bioma'], errors='ignore', inplace=True)
    df.replace(-999, np.nan, inplace=True)
    
    # Tratar FRP: substituir NaN por 0.0 (sem fogo ativo)
    # FRP > 0 indica fogo ativo/recente, então é importante para previsões
    if 'FRP' in df.columns:
        nulos_antes = df['FRP'].isna().sum()
        df['FRP'] = df['FRP'].fillna(0.0)
        logger.info("FRP: %d valores NaN substituídos por 0.0 (sem fogo ativo)", nulos_antes)
    
    # Tratamento de Umidade:
    # IMPORTANTE: Para um projeto científico, não podemos inventar dados.
    # Opções:
    # 1. Se a coluna 'Umidade' existir nos dados históricos, usar esses valores
    # 2. Se não existir, deixar como NaN e o preprocessor tratará (imputação ou remoção)
    # 3. Na aplicação em tempo real, usar valores reais da API HG Weather
    # 
    # NOTA: Se precisar enriquecer a base de dados com umidade histórica real,
    # usar o script enriquecer_dados_umidade.py para buscar de fontes confiáveis
    # (INMET, NASA, etc.)
    
    if 'Umidade' not in df.columns:
        # Não inventar dados - deixar como NaN para ser tratado pelo preprocessor
        # ou criar script separado para enriquecer com dados reais
        logger.warning(
            "Coluna 'Umidade' não encontrada no dataset. "
            "Para um projeto científico, é necessário usar dados reais. "
            "Considere usar o script 'enriquecer_dados_umidade.py' para buscar "
            "dados históricos de umidade de fontes confiáveis (INMET, NASA, etc.). "
            "Por enquanto, a coluna será adicionada como NaN."
        )
        df['Umidade'] = np.nan
    else:
        # Se existir, tratar valores ausentes
        nulos_umidade = df['Umidade'].isna().sum()
        if nulos_umidade > 0:
            logger.warning(
                "Umidade: %d valores NaN encontrados. "
                "Para um projeto científico, considere buscar dados reais "
                "ao invés de imputar. Por enquanto, mantendo como NaN.",
                nulos_umidade
            )
            # Não imputar - deixar NaN para ser tratado pelo preprocessor
            # ou buscar dados reais
    
    df.dropna(subset=['RiscoFogo'], inplace=True)
    df = df[df['RiscoFogo'] >= 0]
    df.drop_duplicates(inplace=True)
    
    # IMPORTANTE: NÃO clipar RiscoFogo (o target não deve ser modificado)
    # Clipar apenas features numéricas
    features_para_clip = [f for f in NUM_FEATURES if f in df.columns]
    df = _clip_outliers(df, features_para_clip)
    
    logger.info("Dataset após limpeza: %d linhas, %d colunas", df.shape[0], df.shape[1])
    return df


def _compute_risk_thresholds(
    series: pd.Series, 
    quantiles: Tuple[float, float] = None, 
    use_semantic: bool = True
) -> Dict[str, float]:
    """
    Computa thresholds de risco.
    
    Args:
        series: Série com valores de RiscoFogo
        quantiles: Quantis a usar (se None, usa DEFAULT_RISK_QUANTILES)
        use_semantic: Se True, usa thresholds semânticos ao invés de apenas quantis
    """
    if quantiles is None:
        quantiles = DEFAULT_RISK_QUANTILES
    
    if use_semantic:
        # Thresholds semânticos baseados em conhecimento do domínio
        # Valores mais "redondos" e com melhor separação das classes
        moderate_min = 0.50  # Risco moderado começa em 0.5
        very_high_min = 0.80  # Risco muito alto começa em 0.8
        
        # Validar que os thresholds fazem sentido com os dados
        q33, q66 = series.quantile(list(DEFAULT_RISK_QUANTILES)).values
        
        # Se thresholds semânticos estão muito longe dos quantis, usar quantis
        if abs(moderate_min - q33) > 0.15 or abs(very_high_min - q66) > 0.15:
            logger.warning(
                "Thresholds semânticos muito diferentes dos quantis. "
                "Quantis: moderado=%.2f, muito_alto=%.2f. "
                "Usando quantis.",
                q33, q66
            )
            moderate_min = float(q33)
            very_high_min = float(q66)
        else:
            logger.info(
                "Usando thresholds semânticos: moderado=%.2f, muito_alto=%.2f "
                "(quantis seriam: %.2f, %.2f)",
                moderate_min, very_high_min, q33, q66
            )
    else:
        # Usar quantis como antes
        q_values = series.quantile(list(quantiles))
        moderate_min = float(q_values.iloc[0])
        very_high_min = float(q_values.iloc[1])
    
    thresholds = {
        'moderate_min': moderate_min,
        'very_high_min': very_high_min,
        'quantiles': list(quantiles),
        'method': 'semantic' if use_semantic else 'quantiles',
    }
    return thresholds


def _save_risk_thresholds(thresholds: Dict[str, float]) -> None:
    _ensure_model_dir()
    with open(RISK_THRESHOLD_FILE, 'w', encoding='utf-8') as fp:
        json.dump(thresholds, fp, indent=2)
    logger.info("Limiar de risco salvo em %s", RISK_THRESHOLD_FILE)


def carregar_risk_thresholds() -> Optional[Dict[str, float]]:
    if not os.path.exists(RISK_THRESHOLD_FILE):
        return None
    with open(RISK_THRESHOLD_FILE, 'r', encoding='utf-8') as fp:
        thresholds = json.load(fp)
    logger.info("Limiar de risco carregado de %s", RISK_THRESHOLD_FILE)
    return thresholds


def _adicionar_features_derivadas(df: pd.DataFrame) -> pd.DataFrame:
    """Adiciona features derivadas baseadas em conhecimento do domínio."""
    df = df.copy()
    
    # Estação do ano (baseada na Amazônia Legal)
    # Temporada de seca: julho-setembro (maior risco)
    # Temporada chuvosa: dezembro-março (menor risco)
    def estacao(mes: int) -> str:
        if mes in [7, 8, 9]:
            return 'Seca_Alta'  # Pico da seca
        elif mes in [6, 10]:
            return 'Seca_Transicao'
        elif mes in [11, 12, 1, 2, 3]:
            return 'Chuvosa'
        else:  # 4, 5
            return 'Transicao_Chuvosa'
    
    df['Estacao'] = df['Mes'].apply(estacao)
    
    # Sazonalidade cíclica (sin/cos para capturar padrões anuais)
    df['Mes_sin'] = np.sin(2 * np.pi * df['Mes'] / 12)
    df['Mes_cos'] = np.cos(2 * np.pi * df['Mes'] / 12)
    
    # Indicador de período crítico (julho-setembro = período de maior risco)
    df['Periodo_Critico'] = df['Mes'].isin([7, 8, 9]).astype(int)
    
    # Razão dias sem chuva / precipitação (normalizada)
    # Alta razão = condições mais secas
    df['Indice_Seca'] = df['DiaSemChuva'] / (df['Precipitacao'] + 0.1)  # +0.1 para evitar divisão por zero
    
    # Hora do dia categorizada (manhã/tarde/noite/madrugada)
    def periodo_dia(hora: int) -> str:
        if 5 <= hora < 12:
            return 'Manha'
        elif 12 <= hora < 18:
            return 'Tarde'
        elif 18 <= hora < 22:
            return 'Noite'
        else:
            return 'Madrugada'
    
    df['Periodo_Dia'] = df['Hora'].apply(periodo_dia)
    
    logger.info("Features derivadas adicionadas: Estacao, Mes_sin, Mes_cos, Periodo_Critico, Indice_Seca, Periodo_Dia")
    return df


def _adicionar_features_temporais_janela(df: pd.DataFrame) -> pd.DataFrame:
    """
    Médias móveis de 7 dias por (Latitude, Longitude), ordenado no tempo.
    Requer Ano, Mes, Dia e colunas numéricas alvo.
    """
    df = df.copy()
    req = ['Ano', 'Mes', 'Dia', 'Latitude', 'Longitude']
    if not all(c in df.columns for c in req):
        return df

    df['_Data'] = pd.to_datetime(
        df['Ano'].astype(str)
        + '-'
        + df['Mes'].astype(str).str.zfill(2)
        + '-'
        + df['Dia'].astype(str).str.zfill(2),
        format='%Y-%m-%d',
        errors='coerce',
    )
    if df['_Data'].isna().all():
        df.drop(columns=['_Data'], inplace=True, errors='ignore')
        return df

    df = df.sort_values(['Latitude', 'Longitude', '_Data'])
    grp = df.groupby(['Latitude', 'Longitude'], group_keys=False)
    for col, nome_ma in [
        ('Precipitacao', 'Precipitacao_ma7'),
        ('DiaSemChuva', 'DiaSemChuva_ma7'),
    ]:
        if col in df.columns:
            df[nome_ma] = grp[col].transform(lambda s: s.rolling(7, min_periods=1).mean())

    df.drop(columns=['_Data'], inplace=True)
    logger.info(
        "Features temporais (janela 7d por local): Precipitacao_ma7, DiaSemChuva_ma7 (quando colunas existem)"
    )
    return df


def classificar_risco(
    df: pd.DataFrame,
    thresholds: Optional[Dict[str, float]] = None,
    persist_thresholds: bool = False,
    use_semantic_thresholds: bool = True,
) -> pd.DataFrame:
    df = df.copy()
    if thresholds is None:
        thresholds = _compute_risk_thresholds(df['RiscoFogo'], use_semantic=use_semantic_thresholds)
        if persist_thresholds:
            _save_risk_thresholds(thresholds)

    very_high_min = thresholds['very_high_min']
    moderate_min = thresholds['moderate_min']

    def categorize(risco: float) -> str:
        if pd.isna(risco):
            return np.nan
        if risco >= very_high_min:
            return 'Muito Alto'
        if risco >= moderate_min:
            return 'Moderado'
        return 'Baixo'

    df['RiscoClassificado'] = df['RiscoFogo'].apply(categorize)
    df.attrs['risk_thresholds'] = thresholds
    logger.info(
        "Classificação de risco aplicada (moderado >= %.2f, muito alto >= %.2f)",
        moderate_min,
        very_high_min,
    )
    return df


def _aplicar_features_avancadas(df: pd.DataFrame, enable: bool = True) -> pd.DataFrame:
    """Aplica camada de features avançadas (Tier 1) se disponível e habilitado.

    Idempotente: se o CSV já vem pré-enriquecido (todas as colunas avançadas
    presentes), apenas registra e retorna sem recomputar.
    """
    if not enable or not HAS_FEATURES_AVANCADAS:
        if not HAS_FEATURES_AVANCADAS:
            logger.warning(
                "Módulo features_avancadas indisponível — Tier 1 ignorado. "
                "Modelos treinarão apenas com features básicas."
            )
        return df

    if ADVANCED_FEATURES_NUM and all(c in df.columns for c in ADVANCED_FEATURES_NUM):
        logger.info(
            "Features avançadas (Tier 1) já presentes no CSV — pulando recomputo."
        )
        return df

    logger.info("Aplicando features avançadas (Tier 1) a %d linhas...", len(df))
    return adicionar_features_avancadas(df)


def carregar_dados(
    dataset_path: Optional[str] = None,
    use_saved_thresholds: bool = True,
    persist_thresholds: bool = True,
    use_semantic_thresholds: bool = True,
    use_advanced_features: bool = True,
) -> Tuple[pd.DataFrame, pd.Series, list, list, pd.DataFrame]:
    csv_path = dataset_path or DATA_PATH

    df = carregar_csv(csv_path)
    validar_colunas(df, REQUIRED_COLUMNS)
    df = limpar_dados(df)
    
    # Adicionar features derivadas APÓS limpeza
    df = _adicionar_features_derivadas(df)
    df = _adicionar_features_temporais_janela(df)
    df = _aplicar_features_avancadas(df, enable=use_advanced_features)

    thresholds = carregar_risk_thresholds() if use_saved_thresholds else None
    # Só persiste se não estamos reutilizando thresholds existentes.
    should_persist = persist_thresholds and thresholds is None
    df = classificar_risco(
        df, 
        thresholds=thresholds, 
        persist_thresholds=should_persist,
        use_semantic_thresholds=use_semantic_thresholds,
    )

    # Atualizar features para incluir as derivadas
    num_features_final = NUM_FEATURES.copy()  # Mantém FRP
    num_features_final.extend(
        ['Mes_sin', 'Mes_cos', 'Periodo_Critico', 'Indice_Seca', 'Precipitacao_ma7', 'DiaSemChuva_ma7']
    )

    # Tier 1: features físico-climáticas avançadas (SPI, KBDI proxy, lags, etc.)
    # Adicionadas apenas quando presentes no DataFrame (geradas em
    # _aplicar_features_avancadas ou já presentes no CSV pré-enriquecido).
    for feat in ADVANCED_FEATURES_NUM:
        if feat in df.columns and feat not in num_features_final:
            num_features_final.append(feat)
    logger.info(
        "Features avançadas (Tier 1) incluídas: %d / %d",
        sum(1 for f in ADVANCED_FEATURES_NUM if f in df.columns),
        len(ADVANCED_FEATURES_NUM),
    )
    
    # Adicionar Umidade apenas se existir e tiver dados válidos (não todos NaN)
    if 'Umidade' in df.columns and df['Umidade'].notna().sum() > 0:
        num_features_final.append('Umidade')
        logger.info("Umidade incluída como feature (dados reais disponíveis: %d valores não-nulos)", 
                   df['Umidade'].notna().sum())
    else:
        logger.info("Umidade não incluída como feature (dados não disponíveis ou todos NaN)")
    
    # Adicionar features de histórico de incêndios se existirem
    features_historico = [
        'Incendios_Ultimos_7_Dias',
        'Incendios_Ultimos_30_Dias',
        'Dias_Desde_Ultimo_Incendio',
        'Media_FRP_Ultimos_7_Dias',
        'Max_FRP_Ultimos_7_Dias'
    ]
    for feat in features_historico:
        if feat in df.columns:
            num_features_final.append(feat)
            logger.info("Feature de histórico incluída: %s", feat)
    
    cat_features_final = CAT_FEATURES.copy()
    cat_features_final.extend(['Estacao', 'Periodo_Dia'])

    X = df.drop(columns=['RiscoFogo', 'RiscoClassificado'], errors='ignore')
    # Colunas auxiliares que não devem entrar no modelo
    for aux in ('_Data',):
        if aux in X.columns:
            X = X.drop(columns=[aux])
    y = df['RiscoClassificado']
    
    # Garantir que apenas colunas que existem sejam usadas
    num_features_final = [f for f in num_features_final if f in X.columns]
    cat_features_final = [f for f in cat_features_final if f in X.columns]

    logger.info("Features numéricas finais: %d", len(num_features_final))
    logger.info("Features categóricas finais: %d", len(cat_features_final))

    return X, y, cat_features_final, num_features_final, df