"""Coleta DETER (alertas de desmatamento quase em tempo real, INPE) por célula/MÊS.

Aprofunda o achado do desmatamento (PRODES anual → +0,025 PR-AUC na nova ignição):
o DETER tem **data do alerta** (`view_date`), permitindo features de desmatamento
com **resolução mensal** — causalmente muito mais precisas que o PRODES anual para
prever fogo no mês seguinte.

IMPORTANTE: exclui a classe ``CICATRIZ_DE_QUEIMADA`` (cicatriz de fogo) — usá-la
seria vazamento do alvo. Mantém as classes de desmatamento/degradação/mineração.

Fonte: WFS TerraBrasilis ``deter-amz:deter_amz`` [prodes-terrabrasilis].
Saída: ``deter_celula_mes.csv`` (LatBin, LonBin, ym, deter_km).
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import requests
from shapely.geometry import shape

ROOT = Path(__file__).resolve().parents[2]
WFS = "http://terrabrasilis.dpi.inpe.br/geoserver/deter-amz/ows"
LAYER = "deter-amz:deter_amz"
CELL = 0.25
SAIDA = ROOT / "scripts" / "previsao_ocorrencia" / "deter_celula_mes.csv"


def coletar_ano(ano: int, page: int = 5000) -> dict:
    """Retorna {(LatBin,LonBin,ym): area_km} para o ano (exclui cicatriz de queimada)."""
    acc: dict = {}
    cql = (f"view_date >= '{ano}-01-01' AND view_date <= '{ano}-12-31' "
           f"AND classname <> 'CICATRIZ_DE_QUEIMADA'")
    start = 0
    while True:
        params = {"service": "WFS", "version": "2.0.0", "request": "GetFeature",
                  "typeName": LAYER, "CQL_FILTER": cql, "count": page, "startIndex": start,
                  "outputFormat": "application/json", "srsName": "EPSG:4326",
                  # DETER não tem chave primária: a paginação exige um sortBy estável.
                  "sortBy": "gid", "propertyName": "classname,view_date,areamunkm,geom"}
        for tent in range(4):
            try:
                r = requests.get(WFS, params=params, timeout=180)
                r.raise_for_status()
                j = r.json()
                break
            except Exception:
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
            ym = str(f["properties"]["view_date"])[:7]
            km = float(f["properties"].get("areamunkm") or 0.0)
            acc[(lat, lon, ym)] = acc.get((lat, lon, ym), 0.0) + km
        start += len(feats)
        if len(feats) < page:
            break
    return acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ano_ini", type=int, default=2016)
    ap.add_argument("--ano_fim", type=int, default=2023)
    ap.add_argument("--saida", type=Path, default=SAIDA)
    args = ap.parse_args()

    linhas = []
    t0 = time.time()
    for ano in range(args.ano_ini, args.ano_fim + 1):
        try:
            acc = coletar_ano(ano)
        except Exception as e:
            print(f"  ano {ano}: FALHA ({repr(e)[:80]}) — pulado", flush=True)
            continue
        for (lat, lon, ym), km in acc.items():
            linhas.append({"LatBin": lat, "LonBin": lon, "ym": ym, "deter_km": round(km, 4)})
        print(f"  ano {ano}: {len(acc):,} (célula,mês) | "
              f"{sum(acc.values()):.0f} km² | {(time.time()-t0)/60:.1f} min", flush=True)

    df = pd.DataFrame(linhas)
    df.to_csv(args.saida, index=False)
    print(f"\nConcluído: {len(df):,} registros (célula,mês) | salvo em {args.saida}")
    print(f"  desmatamento DETER total {args.ano_ini}-{args.ano_fim}: {df['deter_km'].sum():.0f} km²")


if __name__ == "__main__":
    main()
