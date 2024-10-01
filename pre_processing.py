import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Carregar os dados
df = pd.read_csv('dados_incendio.csv')

# Exibir as primeiras linhas dos dados
print(df.head())

# 1. Limpeza de Dados
# Remover linhas com valores ausentes (se desejado)
df = df.dropna()

# Remover duplicatas
df = df.drop_duplicates()

# Converter colunas de data e hora para o formato datetime
df['data'] = pd.to_datetime(df[['Ano', 'Mes', 'Dia']])
df['hora'] = pd.to_datetime(df['Hora'], format='%H:%M').dt.hour

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
y = df['RiscoFogo']

# Salvar o DataFrame tratado em um novo arquivo CSV (opcional)
df.to_csv('dados_incendio_tratados.csv', index=False)

print("Transformação concluída!")
