"""
Script de teste rápido para validar busca de umidade da NASA POWER.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from enriquecer_dados_umidade import buscar_umidade_nasa_power
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

# Testar com algumas coordenadas e datas
testes = [
    (-3.0, -60.0, datetime(2024, 1, 1)),  # Data recente
    (-3.0, -60.0, datetime(2020, 8, 15)),  # Data intermediária
    (-3.0, -60.0, datetime(2015, 7, 1)),   # Data mais antiga
    (-3.0, -60.0, datetime(1985, 1, 1)),   # Data próxima do limite
    (-3.0, -60.0, datetime(1980, 1, 1)),   # Data antes do limite (pode não ter dados)
]

print("Testando busca de umidade da NASA POWER...")
print("=" * 60)

for lat, lon, data in testes:
    print(f"\nTeste: lat={lat}, lon={lon}, data={data.strftime('%Y-%m-%d')}")
    resultado = buscar_umidade_nasa_power(lat, lon, data)
    if resultado is not None:
        print(f"  [OK] Sucesso: {resultado:.2f}%")
    else:
        print(f"  [ERRO] Nao encontrado")

print("\n" + "=" * 60)
print("Teste concluído!")

