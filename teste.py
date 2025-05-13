import pandas as pd
import numpy as np
import folium
from folium.plugins import MarkerCluster
import os
import webbrowser
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('base_de_dados.csv')

df = df.drop(columns=['Satelite', 'Pais', 'Bioma'])

df.replace(-999, np.nan, inplace=True)

df = df[df['RiscoFogo'].notnull() & (df['RiscoFogo'] >= 0)]

quantis = df['RiscoFogo'].quantile([0.33, 0.66]).values
def categorize_risco(risco):
    if risco > quantis[1]:
        return 'Muito Alto'
    elif risco > quantis[0]:
        return 'Moderado'
    else:
        return 'Baixo'

df['RiscoClassificado'] = df['RiscoFogo'].apply(categorize_risco)

categorical_features = ['Estado', 'Municipio']
numeric_features = ['DiaSemChuva', 'Precipitacao', 'Latitude', 'Longitude', 'FRP', 'Ano', 'Mes', 'Dia', 'Hora']

numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

X = df.drop(columns=['RiscoFogo', 'RiscoClassificado'])
y = df['RiscoClassificado']

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

pipeline_lr = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=500, solver='lbfgs', multi_class='multinomial'))
])

pipeline_lr.fit(X_train, y_train)

y_pred = pipeline_lr.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred))

conf_mat = confusion_matrix(y_test, y_pred, labels=['Baixo', 'Moderado', 'Muito Alto'])
sns.heatmap(conf_mat, annot=True, fmt='d', xticklabels=['Baixo', 'Moderado', 'Muito Alto'], yticklabels=['Baixo', 'Moderado', 'Muito Alto'])
plt.title("Matriz de Confusão")
plt.show()

print("aaaaaaaaa")

exemplos = pd.DataFrame({
    'DiaSemChuva': [0, 7, 15],
    'Precipitacao': [0.0, 0.3, 2.5],
    'Latitude': [-3.5, -6.0, -4.2],
    'Longitude': [-60.0, -56.5, -51.0],
    'FRP': [5.0, 20.0, 0.0],
    'Ano': [2023, 2024, 2022],
    'Mes': [8, 9, 6],
    'Dia': [10, 15, 22],
    'Hora': [16, 17, 18],
    'Estado': ['AMAZONAS', 'PARÁ', 'RONDÔNIA'],
    'Municipio': ['MANAUS', 'ALTAMIRA', 'PORTO VELHO']
})

previsoes_exemplos = pipeline_lr.predict(exemplos)
exemplos['RiscoPrevisto'] = previsoes_exemplos


print("Previsões para os exemplos:")
print(exemplos[['Latitude', 'Longitude', 'RiscoPrevisto']])

try:
    exemplos
except NameError:
    print("O DataFrame 'exemplos' com os dados de previsão não existe.")
    exit()

mapa = folium.Map(location=[-5, -60], zoom_start=4)
cluster = MarkerCluster().add_to(mapa)

cor_risco = {'Baixo': 'green', 'Moderado': 'orange', 'Muito Alto': 'red'}

for _, row in exemplos.iterrows():
    folium.Marker(
        location=[row['Latitude'], row['Longitude']],
        popup=f"{row['Municipio']} - Risco: {row['RiscoPrevisto']}",
        icon=folium.Icon(color=cor_risco.get(row['RiscoPrevisto'], 'blue'))
    ).add_to(cluster)

caminho_mapa = os.path.abspath("mapa_risco_amazonia_com_previsoes.html")

mapa.save(caminho_mapa)

print(f"✅ Mapa salvo com sucesso em:\n{caminho_mapa}")

webbrowser.open("file://" + caminho_mapa)