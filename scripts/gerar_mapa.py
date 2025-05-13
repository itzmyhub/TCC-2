import pandas as pd
import joblib
import folium
from folium.plugins import MarkerCluster
import os
import webbrowser
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
import time

geolocator = Nominatim(user_agent="tcc-mapeamento")

def geocodificar_municipio(municipio, estado):
    """Tenta obter latitude e longitude de um município usando geopy"""
    try:
        localizacao = geolocator.geocode(f"{municipio}, {estado}, Brasil", timeout=10)
        if localizacao:
            return localizacao.latitude, localizacao.longitude
        else:
            return None, None
    except GeocoderTimedOut:
        time.sleep(1)
        return geocodificar_municipio(municipio, estado)

base_dir = os.path.dirname(os.path.abspath(__file__))

modelos_dir = os.path.join(base_dir, '..', 'modelos')

os.makedirs(modelos_dir, exist_ok=True)

modelos_disponiveis = [f for f in os.listdir(modelos_dir) if f.endswith('.pkl')]

if not modelos_disponiveis:
    print("⚠️ Nenhum modelo encontrado na pasta 'modelos'. Execute 'treinar_modelos.py' primeiro.")
    exit()

print("📦 Modelos disponíveis:")
for i, nome in enumerate(modelos_disponiveis):
    print(f"[{i}] {nome}")

while True:
    try:
        escolha = int(input("\nDigite o número do modelo que deseja usar: "))
        modelo_nome = modelos_disponiveis[escolha]
        break
    except (ValueError, IndexError):
        print("Entrada inválida. Tente novamente.")

modelo_path = os.path.join(modelos_dir, modelo_nome)
modelo = joblib.load(modelo_path)
print(f"✅ Modelo '{modelo_nome}' carregado.\n")

exemplos = pd.DataFrame({
    'DiaSemChuva': [3, 10, 8, 12, 0, 5],
    'Precipitacao': [0.2, 2.1, 0.0, 4.5, 0.1, 3.8],
    'FRP': [5.0, 12.0, 0.0, 50.0, 1.5, 2.0],
    'Ano': [2023, 2024, 2025, 2023, 2022, 2021],
    'Mes': [8, 7, 9, 5, 6, 12],
    'Dia': [17, 22, 14, 3, 18, 25],
    'Hora': [16, 18, 17, 15, 19, 16],
    'Estado': ['AMAZONAS', 'PARÁ', 'RONDÔNIA', 'ACRE', 'MARANHÃO', 'TOCANTINS'],
    'Municipio': ['MANAUS', 'BELÉM', 'PORTO VELHO', 'RIO BRANCO', 'CUIABÁ', 'PALMAS']
})

latitudes = []
longitudes = []

for _, row in exemplos.iterrows():
    lat, lon = geocodificar_municipio(row['Municipio'], row['Estado'])
    latitudes.append(lat)
    longitudes.append(lon)

exemplos['Latitude'] = latitudes
exemplos['Longitude'] = longitudes

exemplos = exemplos.dropna(subset=['Latitude', 'Longitude'])

exemplos['RiscoPrevisto'] = modelo.predict(exemplos)
print("🔍 Previsões:")
print(exemplos[['Municipio', 'RiscoPrevisto']])

mapa = folium.Map(location=[-5, -60], zoom_start=4)
cluster = MarkerCluster().add_to(mapa)
cor_risco = {'Baixo': 'green', 'Moderado': 'orange', 'Muito Alto': 'red'}

for _, row in exemplos.iterrows():
    folium.Marker(
        location=[row['Latitude'], row['Longitude']],
        popup=f"{row['Municipio']} - Risco: {row['RiscoPrevisto']}",
        icon=folium.Icon(color=cor_risco.get(row['RiscoPrevisto'], 'blue'))
    ).add_to(cluster)

mapa_path = os.path.abspath("mapa_risco_amazonia_com_previsoes.html")
mapa.save(mapa_path)
webbrowser.open("file://" + mapa_path)
print(f"🗺️ Mapa salvo em: {mapa_path}")


