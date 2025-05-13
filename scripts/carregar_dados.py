import os
import pandas as pd
import numpy as np

def carregar_csv(caminho):
    if not os.path.exists(caminho):
        raise FileNotFoundError(f"Arquivo não encontrado: {caminho}")
    return pd.read_csv(caminho)

def limpar_dados(df):
    df = df.drop(columns=['Satelite', 'Pais', 'Bioma'], errors='ignore')
    df.replace(-999, np.nan, inplace=True)
    filtro_risco_fogo = df['RiscoFogo'].notnull() & (df['RiscoFogo'] >= 0)
    
    # Contando as linhas que atendem à condição
    print(filtro_risco_fogo.sum())
    df = df[df['RiscoFogo'].notnull() & (df['RiscoFogo'] >= 0)]
    return df

def classificar_risco(df):
    quantis = df['RiscoFogo'].quantile([0.33, 0.66]).values
    def categorize(risco):
        if pd.isna(risco): return np.nan
        if risco > quantis[1]: return 'Muito Alto'
        elif risco > quantis[0]: return 'Moderado'
        else: return 'Baixo'
    df['RiscoClassificado'] = df['RiscoFogo'].apply(categorize)
    print(quantis)
    return df

def carregar_dados():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, '..', 'base_de_dados.csv')
    
    df = carregar_csv(csv_path)
    df = limpar_dados(df)
    df = classificar_risco(df)

    cat_features = ['Estado', 'Municipio']
    num_features = ['DiaSemChuva', 'Precipitacao', 'Latitude', 'Longitude', 'FRP', 'Ano', 'Mes', 'Dia', 'Hora']
    
    X = df.drop(columns=['RiscoFogo', 'RiscoClassificado'])
    y = df['RiscoClassificado']

    return X, y, cat_features, num_features, df