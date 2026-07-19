"""
Script para adicionar features de histórico de incêndios ao dataset.

Features criadas:
- Incendios_Ultimos_7_Dias: Contagem de focos nos últimos 7 dias (raio ~10km)
- Incendios_Ultimos_30_Dias: Contagem de focos nos últimos 30 dias
- Dias_Desde_Ultimo_Incendio: Dias desde último foco na região
- Media_FRP_Ultimos_7_Dias: Média de FRP nos últimos 7 dias
- Max_FRP_Ultimos_7_Dias: Máximo FRP nos últimos 7 dias

Essas features são calculadas a partir do próprio dataset, sem necessidade de APIs externas.
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional
import argparse

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR.parent / 'base_de_dados.csv'
OUTPUT_PATH = BASE_DIR.parent / 'base_de_dados_com_historico.csv'

# Raio de busca para considerar "mesma região" (em graus)
# ~0.1 grau ≈ 11km na latitude da Amazônia
RAIO_REGIAO = 0.1


def calcular_distancia_haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calcula distância entre dois pontos usando fórmula de Haversine.
    Retorna distância em graus (aproximada).
    """
    # Simplificação: usar distância euclidiana em graus (suficiente para raio pequeno)
    return np.sqrt((lat1 - lat2)**2 + (lon1 - lon2)**2)


def adicionar_historico_incendios(
    input_path: Path,
    output_path: Path,
    raio_km: float = 10.0,
    dias_janela_curta: int = 7,
    dias_janela_longa: int = 30
) -> pd.DataFrame:
    """
    Adiciona features de histórico de incêndios ao dataset.
    
    Args:
        input_path: Caminho para CSV de entrada
        output_path: Caminho para CSV de saída
        raio_km: Raio em km para considerar "mesma região" (padrão: 10km)
        dias_janela_curta: Janela de dias para features de curto prazo (padrão: 7)
        dias_janela_longa: Janela de dias para features de longo prazo (padrão: 30)
    
    Returns:
        DataFrame com features de histórico adicionadas
    """
    logger.info("Carregando dataset: %s", input_path)
    df = pd.read_csv(input_path)
    
    logger.info("Dataset original: %d linhas, %d colunas", df.shape[0], df.shape[1])
    
    # Verificar colunas necessárias
    colunas_necessarias = ['Latitude', 'Longitude', 'Ano', 'Mes', 'Dia', 'FRP']
    faltando = [c for c in colunas_necessarias if c not in df.columns]
    if faltando:
        raise ValueError(f"Colunas necessárias não encontradas: {faltando}")
    
    # Converter raio de km para graus (aproximado: 1 grau ≈ 111km)
    raio_graus = raio_km / 111.0
    
    # Criar coluna de data para facilitar cálculos temporais
    # Verificar se as colunas existem e têm valores válidos
    if not all(col in df.columns for col in ['Ano', 'Mes', 'Dia']):
        raise ValueError("Colunas 'Ano', 'Mes' ou 'Dia' não encontradas no dataset")
    
    # Criar data usando string formatada (mais robusto)
    df['Data'] = pd.to_datetime(
        df['Ano'].astype(str) + '-' + 
        df['Mes'].astype(str).str.zfill(2) + '-' + 
        df['Dia'].astype(str).str.zfill(2),
        format='%Y-%m-%d',
        errors='coerce'  # Converter valores inválidos para NaT
    )
    
    # Verificar se há datas inválidas
    datas_invalidas = df['Data'].isna().sum()
    if datas_invalidas > 0:
        logger.warning(
            "%d registros com datas inválidas serão removidos ou terão features de histórico como 0",
            datas_invalidas
        )
    
    # Ordenar por data para facilitar cálculos de janela deslizante
    df = df.sort_values(['Data', 'Latitude', 'Longitude']).reset_index(drop=True)
    
    # Inicializar novas colunas
    df['Incendios_Ultimos_7_Dias'] = 0
    df['Incendios_Ultimos_30_Dias'] = 0
    df['Dias_Desde_Ultimo_Incendio'] = np.nan
    df['Media_FRP_Ultimos_7_Dias'] = 0.0
    df['Max_FRP_Ultimos_7_Dias'] = 0.0
    
    logger.info("Calculando features de histórico de incêndios...")
    logger.info("Isso pode levar alguns minutos para datasets grandes...")
    logger.info("Usando otimização com agrupamento espacial...")
    
    # OTIMIZAÇÃO: Agrupar por células espaciais para reduzir cálculos
    # Criar células de grade (grid) para agrupar coordenadas próximas
    grid_size = raio_graus * 2  # Tamanho da célula = 2x o raio
    df['Grid_Lat'] = (df['Latitude'] / grid_size).astype(int)
    df['Grid_Lon'] = (df['Longitude'] / grid_size).astype(int)
    
    total = len(df)
    logger.info("Processando %d registros...", total)
    
    # Usar apply com otimização para processar em lotes
    def calcular_historico_para_registro(row):
        """Calcula histórico para um registro específico."""
        lat = row['Latitude']
        lon = row['Longitude']
        data_atual = row['Data']
        grid_lat = row['Grid_Lat']
        grid_lon = row['Grid_Lon']
        
        # Janelas temporais
        data_inicio_curta = data_atual - timedelta(days=dias_janela_curta)
        data_inicio_longa = data_atual - timedelta(days=dias_janela_longa)
        
        # Filtrar por células adjacentes (otimização)
        mask_grid = (
            (df['Grid_Lat'] >= grid_lat - 1) & (df['Grid_Lat'] <= grid_lat + 1) &
            (df['Grid_Lon'] >= grid_lon - 1) & (df['Grid_Lon'] <= grid_lon + 1)
        )
        
        # Filtrar por região exata e data (apenas anteriores)
        mask_regiao = (
            mask_grid &
            (df['Latitude'] >= lat - raio_graus) &
            (df['Latitude'] <= lat + raio_graus) &
            (df['Longitude'] >= lon - raio_graus) &
            (df['Longitude'] <= lon + raio_graus) &
            (df['Data'] < data_atual)
        )
        
        df_temp = df[mask_regiao].copy()
        
        if len(df_temp) == 0:
            return {
                'Incendios_Ultimos_7_Dias': 0,
                'Incendios_Ultimos_30_Dias': 0,
                'Dias_Desde_Ultimo_Incendio': 365,
                'Media_FRP_Ultimos_7_Dias': 0.0,
                'Max_FRP_Ultimos_7_Dias': 0.0
            }
        
        # Filtrar por distância exata (Haversine) - apenas se necessário
        df_temp['Distancia'] = np.sqrt(
            (df_temp['Latitude'] - lat)**2 + (df_temp['Longitude'] - lon)**2
        )
        df_temp = df_temp[df_temp['Distancia'] <= raio_graus]
        
        resultado = {
            'Incendios_Ultimos_7_Dias': 0,
            'Incendios_Ultimos_30_Dias': 0,
            'Dias_Desde_Ultimo_Incendio': 365,
            'Media_FRP_Ultimos_7_Dias': 0.0,
            'Max_FRP_Ultimos_7_Dias': 0.0
        }
        
        # Features de curto prazo (últimos 7 dias)
        mask_curta = (df_temp['Data'] >= data_inicio_curta) & (df_temp['Data'] < data_atual)
        df_curta = df_temp[mask_curta]
        
        if len(df_curta) > 0:
            # Contar focos (FRP > 0 indica fogo)
            focos_curta = (df_curta['FRP'] > 0).sum()
            resultado['Incendios_Ultimos_7_Dias'] = focos_curta
            
            # Média e máximo de FRP
            frp_curta = df_curta[df_curta['FRP'] > 0]['FRP']
            if len(frp_curta) > 0:
                resultado['Media_FRP_Ultimos_7_Dias'] = float(frp_curta.mean())
                resultado['Max_FRP_Ultimos_7_Dias'] = float(frp_curta.max())
        
        # Features de longo prazo (últimos 30 dias)
        mask_longa = (df_temp['Data'] >= data_inicio_longa) & (df_temp['Data'] < data_atual)
        df_longa = df_temp[mask_longa]
        
        if len(df_longa) > 0:
            focos_longa = (df_longa['FRP'] > 0).sum()
            resultado['Incendios_Ultimos_30_Dias'] = focos_longa
        
        # Dias desde último incêndio
        df_incendios = df_temp[df_temp['FRP'] > 0]
        if len(df_incendios) > 0:
            ultimo_incendio = df_incendios['Data'].max()
            dias_desde = (data_atual - ultimo_incendio).days
            resultado['Dias_Desde_Ultimo_Incendio'] = dias_desde
        
        return resultado
    
    # Processar em lotes para mostrar progresso
    batch_size = 10000
    resultados = []
    
    for i in range(0, total, batch_size):
        batch_end = min(i + batch_size, total)
        logger.info("Processando lote: %d-%d de %d (%.1f%%)", 
                   i + 1, batch_end, total, (batch_end / total) * 100)
        
        batch_df = df.iloc[i:batch_end]
        batch_resultados = batch_df.apply(calcular_historico_para_registro, axis=1)
        resultados.extend(batch_resultados.tolist())
    
    # Aplicar resultados ao DataFrame
    df['Incendios_Ultimos_7_Dias'] = [r['Incendios_Ultimos_7_Dias'] for r in resultados]
    df['Incendios_Ultimos_30_Dias'] = [r['Incendios_Ultimos_30_Dias'] for r in resultados]
    df['Dias_Desde_Ultimo_Incendio'] = [r['Dias_Desde_Ultimo_Incendio'] for r in resultados]
    df['Media_FRP_Ultimos_7_Dias'] = [r['Media_FRP_Ultimos_7_Dias'] for r in resultados]
    df['Max_FRP_Ultimos_7_Dias'] = [r['Max_FRP_Ultimos_7_Dias'] for r in resultados]
    
    # Remover colunas auxiliares
    df.drop(columns=['Grid_Lat', 'Grid_Lon'], inplace=True)
    
    # Remover coluna auxiliar de data
    df.drop(columns=['Data'], inplace=True)
    
    logger.info("Features de histórico calculadas!")
    logger.info("Estatísticas:")
    logger.info("  Incendios_Ultimos_7_Dias: média=%.2f, max=%d", 
               df['Incendios_Ultimos_7_Dias'].mean(), df['Incendios_Ultimos_7_Dias'].max())
    logger.info("  Incendios_Ultimos_30_Dias: média=%.2f, max=%d",
               df['Incendios_Ultimos_30_Dias'].mean(), df['Incendios_Ultimos_30_Dias'].max())
    logger.info("  Dias_Desde_Ultimo_Incendio: média=%.2f, min=%.0f, max=%.0f",
               df['Dias_Desde_Ultimo_Incendio'].mean(),
               df['Dias_Desde_Ultimo_Incendio'].min(),
               df['Dias_Desde_Ultimo_Incendio'].max())
    
    # Salvar
    df.to_csv(output_path, index=False)
    logger.info("Dataset com histórico salvo em: %s", output_path)
    
    return df


def main():
    """Função principal."""
    parser = argparse.ArgumentParser(
        description='Adiciona features de histórico de incêndios ao dataset'
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
        help='Caminho para CSV de saída (padrão: base_de_dados_com_historico.csv)'
    )
    parser.add_argument(
        '--raio',
        type=float,
        default=10.0,
        help='Raio em km para considerar "mesma região" (padrão: 10.0)'
    )
    parser.add_argument(
        '--dias-curta',
        type=int,
        default=7,
        help='Janela de dias para features de curto prazo (padrão: 7)'
    )
    parser.add_argument(
        '--dias-longa',
        type=int,
        default=30,
        help='Janela de dias para features de longo prazo (padrão: 30)'
    )
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    if not input_path.exists():
        logger.error("Arquivo de entrada não encontrado: %s", input_path)
        return
    
    adicionar_historico_incendios(
        input_path=input_path,
        output_path=output_path,
        raio_km=args.raio,
        dias_janela_curta=args.dias_curta,
        dias_janela_longa=args.dias_longa
    )


if __name__ == '__main__':
    main()

