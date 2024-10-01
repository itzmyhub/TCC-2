import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Função para categorizar o risco
def categorize_risco(risco):
    if risco > 0.8:
        return 'Muito Alto'
    elif risco > 0.4:
        return 'Moderado'
    else:
        return 'Baixo'

# Carregar os dados
df = pd.read_csv('base_de_dados.csv')

# 1. Remover atributos com valor constante
df = df.drop(columns=['Satelite', 'Pais', 'Bioma'])

# 2. Tratar valores ausentes e inconsistentes
df.replace(-999, np.nan, inplace=True)

# Aplicar a função de categorização
df['RiscoClassificado'] = df['RiscoFogo'].apply(categorize_risco)

# Definir os atributos categóricos e numéricos
categorical_features = ['Estado', 'Municipio']
numeric_features = ['DiaSemChuva', 'Precipitacao', 'Latitude', 'Longitude', 'FRP', 'Ano', 'Mes', 'Dia', 'Hora']

# Imputar valores ausentes para atributos numéricos
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Codificação dos atributos categóricos
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Criar o pré-processador
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# Dividir os dados em conjunto de treino e teste
X = df.drop(columns=['RiscoFogo', 'RiscoClassificado'])
y = df['RiscoClassificado']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Criar o pipeline completo com RandomForestClassifier
pipeline_rf = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Treinar o modelo
pipeline_rf.fit(X_train, y_train)

# Avaliar o modelo
y_pred_rf = pipeline_rf.predict(X_test)
accuracy_rf = pipeline_rf.score(X_test, y_test)
print(f'Accuracy (Random Forest): {accuracy_rf:.2f}')

# Exibir relatório de classificação
print(classification_report(y_test, y_pred_rf))

# Dados para previsão (substitua com dados reais)
dados_para_previsao = pd.DataFrame({
    'DiaSemChuva': [0, 5, 10, 2],
    'Precipitacao': [0.0, 1.2, 0.5, 0.0],
    'Latitude': [-7.0, -8.5, -6.0, -7.5],
    'Longitude': [-56.0, -55.5, -57.0, -56.5],
    'FRP': [10.0, 30.0, 5.0, 0.0],
    'Ano': [2022, 2023, 2024, 2022],
    'Mes': [6, 7, 8, 9],
    'Dia': [15, 20, 25, 10],
    'Hora': [16, 17, 18, 15],
    'Estado': ['Estado1', 'Estado2', 'Estado1', 'Estado3'],
    'Municipio': ['Municipio1', 'Municipio2', 'Municipio3', 'Municipio4']
})

novo_dado = pd.DataFrame({
    'DiaSemChuva': [0.0],
    'Precipitacao': [8.0],
    'Latitude': [-3.858],
    'Longitude': [-51.087],
    'FRP': [0.3],
    'Ano': [2015],
    'Mes': [1],
    'Dia': [1],
    'Hora': [17],
    'Estado': ['PARÁ'],
    'Municipio': ['PACAJÁ']
})

dados_para_previsao = pd.concat([dados_para_previsao, novo_dado], ignore_index=True)

# Verificar os dados antes da previsão
print(dados_para_previsao)

previsoes_rf = pipeline_rf.predict(dados_para_previsao)

# Mostrar as previsões
print("Previsões para os dados fornecidos com Random Forest:")
for i, previsao in enumerate(previsoes_rf):
    print(f"Dados {i+1}: {previsao}")
