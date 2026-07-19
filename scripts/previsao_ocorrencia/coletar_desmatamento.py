"""Coleta de DESMATAMENTO (PRODES/INPE) por célula 0,25° e ano — driver antrópico dinâmico.

Fonte: WFS do TerraBrasilis (camada ``prodes-legal-amz:yearly_deforestation``),
polígonos anuais de incremento de desmatamento com atributos ``year`` e ``area_km``.
Diferente do clima (P5) e da distância-a-cidade (estático), o desmatamento é
**dinâmico e precede a primeira queimada** de uma célula — é a causa proximal
antrópica do fogo na Amazônia (Aragão et al. 2018). Deve agregar valor justamente
no regime de NOVA IGNIÇÃO, onde a persistência do histórico de fogo é inútil.

Estratégia (leve em requisições, ~80 chamadas, ~4 min, sem rate-limit):
  • por ano (2014–2023), pagina o WFS (count=5000, startIndex);
  • computa o centróide de cada polígono → célula 0,25°;
  • acumula ``area_km`` por (célula, ano).

Saída: ``desmatamento_celula_ano.csv`` (LatBin, LonBin, ano, defor_km).
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from shapely.geometry import shape

ROOT = Path(__file__).resolve().parents[2]
WFS = "http://terrabrasilis.dpi.inpe.br/geoserver/prodes-legal-amz/ows"
LAYER = "prodes-legal-amz:yearly_deforestation"
CELL = 0.25
SAIDA = ROOT / "scripts" / "previsao_ocorrencia" / "desmatamento_celula_ano.csv"


def coletar_ano(ano: int, page: int = 5000) -> dict:
    """Retorna {(LatBin,LonBin): area_km_total} para o ano."""
    acc: dict = {}
    start = 0
    while True:
        params = {"service": "WFS", "version": "2.0.0", "request": "GetFeature",
                  "typeName": LAYER, "CQL_FILTER": f"year={ano}", "count": page,
                  "startIndex": start, "outputFormat": "application/json",
                  "propertyName": "year,area_km,geom", "srsName": "EPSG:4326"}
        for tent in range(4):
            try:
                r = requests.get(WFS, params=params, timeout=120)
                r.raise_for_status()
                j = r.json()
                break
            except Exception as e:
                if tent == 3:
                    raise
                time.sleep(2 * (tent + 1))
        feats = j.get("features", [])
        if not feats:
            break
        for f in feats:
            g = f.get("geometry")
            if not g:
                continue
            c = shape(g).centroid
            lat = round(round(c.y / CELL) * CELL, 3)
            lon = round(round(c.x / CELL) * CELL, 3)
            acc[(lat, lon)] = acc.get((lat, lon), 0.0) + float(f["properties"].get("area_km") or 0.0)
        start += len(feats)
        if len(feats) < page:
            break
    return acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ano_ini", type=int, default=2014)
    ap.add_argument("--ano_fim", type=int, default=2023)
    args = ap.parse_args()

    linhas = []
    t0 = time.time()
    for ano in range(args.ano_ini, args.ano_fim + 1):
        acc = coletar_ano(ano)
        for (lat, lon), km in acc.items():
            linhas.append({"LatBin": lat, "LonBin": lon, "ano": ano, "defor_km": round(km, 4)})
        print(f"  ano {ano}: {len(acc):,} células com desmatamento | "
              f"total {sum(acc.values()):.0f} km² | {(time.time()-t0)/60:.1f} min")

    df = pd.DataFrame(linhas)
    df.to_csv(SAIDA, index=False)
    print(f"\nConcluído: {len(df):,} registros (célula,ano) | salvo em {SAIDA}")
    print(f"  desmatamento total {args.ano_ini}-{args.ano_fim}: {df['defor_km'].sum():.0f} km²")


if __name__ == "__main__":
    main()
