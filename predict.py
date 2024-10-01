import numpy as np
import pandas as pd
from sklearn.calibration import LabelEncoder
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Carregar os dados
arquivo_csv = 'base_de_dados.csv'
df = pd.read_csv(arquivo_csv)

# Exibir as primeiras linhas dos dados

# 1. Limpeza de Dados
# Remover linhas com valores ausentes (se desejado)
df = df.dropna()

# Remover duplicatas
df = df.drop_duplicates()

# Converter colunas de data e hora para o formato datetime
#df['data'] = pd.to_datetime(df[['Ano', 'Mes', 'Dia']])
#df['hora'] = pd.to_datetime(df['Hora'], format='%H:%M').dt.hour

# 2. Normalização e Codificação
# Definir colunas numéricas e categóricas
colunas_numericas = ['DiaSemChuva', 'Precipitacao', 'FRP', 'Latitude', 'Longitude']
colunas_categoricas = ['Pais', 'Estado', 'Municipio', 'Bioma']

# Preprocessador
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),  # Imputar valores ausentes
            ('scaler', StandardScaler())  # Normalização
        ]), colunas_numericas),
        ('cat', OneHotEncoder(handle_unknown='ignore'), colunas_categoricas)
    ])

# Aplicar transformação
X = preprocessor.fit_transform(df)
#y = df['RiscoFogo']
#risco_fogo = np.array(df['RiscoFogo'])
#limiar = 0.5

#bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
#labels = ['Baixo Risco', 'Médio Baixo Risco', 'Médio Alto Risco', 'Alto Risco', 'Muito Alto Risco']
#y_categorico = pd.cut(risco_fogo, bins=bins, labels=labels)

#print("Rótulos categóricos:", y_categorico)

#y_binario = np.where(risco_fogo > limiar, 1, 0)

#print(y_binario)

def categorize_risco(risco):
    if risco > 0.8:
        return 'Muito Alto'
    elif risco > 0.4:
        return 'Moderado'
    else:
        return 'Baixo'

df['RiscoCategoria'] = df['RiscoFogo'].apply(categorize_risco)

label_encoder = LabelEncoder()
df['RiscoCategoriaCodificado'] = label_encoder.fit_transform(df['RiscoCategoria'])

y = df['RiscoCategoriaCodificado']  # Rótulo

print(df.head())

# Salvar o DataFrame tratado em um novo arquivo CSV (opcional)
df.to_csv('dados_incendio_tratados.csv', index=False)

print("Transformação concluída!")

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# Dividir os dados em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Inicializar e treinar o modelo
modelo = DecisionTreeClassifier()
modelo.fit(X_train, y_train)

# Fazer previsões
y_pred = modelo.predict(X_test)

# Avaliar o modelo
print("Acurácia do modelo:", accuracy_score(y_test, y_pred))
print("Relatório de Classificação:")
print(classification_report(y_test, y_pred))

# Exemplo de previsão com novos dados
# Supondo que você tenha novos dados já processados
novos_dados = pd.DataFrame({
    'DiaSemChuva': [1.0],  # Exemplo de valor
    'Precipitacao': [0.2],
    'Satelite': ['AQUA_M-T'],
    'FRP': [0.5],
    'Latitude': [0.4],
    'Longitude': [0.3],
    'Pais': ['Brasil'],
    'Estado': ['AMAZONAS'],
    'Municipio': ['MANAUS'],
    'Bioma': ['Amazônia'],
    'Mes': ['2'],
    'Dia': ['21'],
    'Ano': ['2024'],
})

# Unnamed: 0,Satelite,Pais,Estado,Municipio,Bioma,DiaSemChuva,Precipitacao,RiscoFogo,Latitude,Longitude,FRP,Ano,Mes,Dia,Hora,RiscoCategoria,RiscoCategoriaCodificado
# 377808,AQUA_M-T,Brasil,MARANHÃO,AMAPÁ DO MARANHÃO,Amazônia,0.0,0.0,0.6,-1.645,-45.934,50.9,2018,1,1,16,Moderado,1

novos_dados2 = pd.DataFrame({
    'DiaSemChuva': [0.0],  # Exemplo de valor
    'Precipitacao': [0.0],
    'Satelite': ['AQUA_M-T'],
    'FRP': [50.9],
    'Latitude': [-1.645],
    'Longitude': [-45.934],
    'Pais': ['Brasil'],
    'Estado': ['MARANHÃO'],
    'Municipio': ['AMAPÁ DO MARANHÃO'],
    'Bioma': ['Amazônia'],
    'Mes': ['1'],
    'Dia': ['1'],
    'Ano': ['2018'],
    'Hora': ['16']
})
# Transformar novos dados com o mesmo preprocessor
novos_dados_transformados = preprocessor.transform(novos_dados)
novos_dados_transformados2 = preprocessor.transform(novos_dados2)
previsao = modelo.predict(novos_dados_transformados)
previsao2 = modelo.predict(novos_dados_transformados2)
#print(classification_report(y_test, y_pred, target_names=['Não Fogo', 'Fogo']))
print("Previsão para novos dados:", previsao)
print("Previsão para novos dados:", previsao2)