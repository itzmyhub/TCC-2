import os
import pandas as pd
import glob
import sweetviz as sv

from pandas_profiling import ProfileReport

#list all csv files only
csv_files = glob.glob('*.{}'.format('csv'))
csv_files

df_append = pd.DataFrame()
#append all files together
for file in csv_files:
            df_temp = pd.read_csv(file)
            df_append = df_append._append(df_temp, ignore_index=True)
data = df_append

# Visualizar as primeiras linhas dos dados
print(data)

# Tratamento de dados
# Convertendo a coluna DATA E HORA para o tipo datetime
data['DataHora'] = pd.to_datetime(data['DataHora'])


# Extraindo características da data/hora
data['Ano'] = data['DataHora'].dt.year
data['Mes'] = data['DataHora'].dt.month
data['Dia'] = data['DataHora'].dt.day
data['Hora'] = data['DataHora'].dt.hour

# Remover a coluna DATA E HORA original
data.drop(columns=['DataHora'], inplace=True)

profile = ProfileReport(data, title="Relatório de Perfil", explorative=True)

# Salvar o relatório em um arquivo HTML
profile.to_file("relatorio.html")
# Tratamento de valores ausentes
# Imputação de valores ausentes para variáveis numéricas com a média
num_cols = ['Latitude', 'Longitude', 'DiaSemChuva', 'Precipitacao', 'FRP']
data[num_cols] = data[num_cols].fillna(data[num_cols].mean())

# Imputação de valores ausentes para variáveis categóricas com o valor mais frequente
cat_cols = ['Satelite', 'Municipio', 'Estado', 'Pais', 'Bioma']
data[cat_cols] = data[cat_cols].fillna(data[cat_cols].mode().iloc[0])

# Codificação de variáveis categóricas
#label_encoders = {}
#for col in cat_cols:
#    le = LabelEncoder()
#    data[col] = le.fit_transform(data[col])
#    label_encoders[col] = le

# Separar variáveis independentes e dependentes
#X = data.drop(columns=['RiscoFogo'])  # Supondo que RISCO_FOGO seja a variável alvo
#y = data['RiscoFogo']

# Normalização dos dados numéricos
#scaler = StandardScaler()
#X[num_cols] = scaler.fit_transform(X[num_cols])

# Dividir os dados em conjunto de treino e teste
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#print("Dados preparados com sucesso!")
#print("Tamanho do conjunto de treinamento:", X_train.shape)
#print("Tamanho do conjunto de teste:", X_test.shape)

# Exemplo de como os dados normalizados e preparados estão prontos para modelos de aprendizado

# Visualizar as primeiras linhas do conjunto de treinamento
#print(X_train.head())
#print(y_train.head())

# Verificar a presença de valores nulos
#print(X_train.isnull().sum())
#print(y_train.isnull().sum())


#print(X_train.describe())
