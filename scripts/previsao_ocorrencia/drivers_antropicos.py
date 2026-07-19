"""Drivers ANTRÓPICOS de ignição — proxy leve (distância a cidades).

Motivação: o P5 mostrou que clima real + FWI NÃO agregam sobre o histórico de
fogo na escala mensal (ganho nulo). A hipótese desta etapa é que o sinal que
falta é o de **ignição humana** — o fogo na Amazônia é majoritariamente antrópico
(Aragão et al. 2018). Variáveis de pressão humana trazem informação NÃO redundante
com o histórico de fogo (capturam *por que* humanos ateiam fogo ali).

"Proxy leve primeiro": começa com a fonte de menor atrito — distância da célula
à cidade/povoado mais próximo (Natural Earth, download estático único, sem
rate-limit) —, proxy clássico de acessibilidade humana. Se houver ganho, escala-se
para fontes pesadas (MapBiomas uso do solo, PRODES/DETER desmatamento, estradas OSM).

Saída: ``dataset_ocorrencia_antropico.csv`` (dataset de ocorrência + dist_cidade_km).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from pyproj import Transformer
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "dataset_ocorrencia_mensal.csv"
CACHE = ROOT / ".cache_umidade" / "ne_populated_places.geojson"
NE_URL = ("https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/"
          "geojson/ne_10m_populated_places_simple.geojson")
# bbox aproximada da Amazônia Legal
BBOX = (-75, -19, -43, 6)  # lon_min, lat_min, lon_max, lat_max


def carregar_cidades() -> np.ndarray:
    if CACHE.exists():
        j = json.loads(CACHE.read_text(encoding="utf-8"))
    else:
        r = requests.get(NE_URL, timeout=60)
        r.raise_for_status()
        CACHE.write_text(r.text, encoding="utf-8")
        j = r.json()
    pts = []
    for f in j["features"]:
        lon, lat = f["geometry"]["coordinates"][:2]
        if BBOX[0] < lon < BBOX[2] and BBOX[1] < lat < BBOX[3]:
            pts.append((lon, lat))
    print(f"  cidades/povoados na bbox Amazônia: {len(pts)}")
    return np.array(pts)


def dist_para_cidades(cells: pd.DataFrame, cidades: np.ndarray) -> np.ndarray:
    """Distância (km) de cada célula à cidade mais próxima, em CRS métrico."""
    tr = Transformer.from_crs(4326, 5880, always_xy=True)  # SIRGAS 2000 / Brazil Polyconic
    cx, cy = tr.transform(cidades[:, 0], cidades[:, 1])
    tree = cKDTree(np.column_stack([cx, cy]))
    ex, ey = tr.transform(cells["LonBin"].to_numpy(), cells["LatBin"].to_numpy())
    d, _ = tree.query(np.column_stack([ex, ey]), k=1)
    return d / 1000.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--saida", type=str, default=str(ROOT / "dataset_ocorrencia_antropico.csv"))
    args = ap.parse_args()

    print("Carregando dataset e cidades (Natural Earth)...")
    df = pd.read_csv(DATASET)
    cidades = carregar_cidades()

    cells = df[["LatBin", "LonBin"]].drop_duplicates().reset_index(drop=True)
    cells["dist_cidade_km"] = dist_para_cidades(cells, cidades)
    print(f"  células: {len(cells)} | dist_cidade_km: "
          f"min={cells['dist_cidade_km'].min():.1f} "
          f"mediana={cells['dist_cidade_km'].median():.1f} "
          f"max={cells['dist_cidade_km'].max():.1f} km")

    out = df.merge(cells, on=["LatBin", "LonBin"], how="left")
    out.to_csv(args.saida, index=False)
    print(f"  feature 'dist_cidade_km' adicionada a {len(out):,} linhas")
    print(f"  salvo em: {args.saida}")


if __name__ == "__main__":
    main()
