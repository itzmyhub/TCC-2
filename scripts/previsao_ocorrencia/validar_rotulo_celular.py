"""Validação CÉLULA-A-CÉLULA do rótulo contra o raster do MapBiomas Fogo.

Fecha a lacuna deixada pela validação por estado/ano: confronta, no nível da célula
0,25° por ano, a presença de foco do INPE (que define o rótulo) com a presença de
cicatriz de área queimada do MapBiomas Fogo (Landsat 30 m). Quantifica omissão e
comissão dos focos com Cohen's kappa, precisão/revocação e concordância.

Acesso sem Earth Engine: os rasters anuais do MapBiomas são COGs públicos no GCS;
lê-se cada ano via /vsicurl/ em um \emph{overview} grosseiro com reamostragem por
máximo (preserva a presença de queimada), restrito à bbox da Amazônia --- uma única
leitura por ano, sem baixar gigabytes. Os COGs públicos cobrem 1985--2020, então a
validação usa 2015--2020 (overlap com o dataset).
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import Affine
from rasterio.windows import from_bounds
from sklearn.metrics import (cohen_kappa_score, confusion_matrix,
                             precision_recall_fscore_support)

ROOT = Path(__file__).resolve().parents[2]
REL_DIR = ROOT / "modelos" / "relatorios"
URL = ("/vsicurl/https://storage.googleapis.com/mapbiomas-public/brasil/fire/"
       "coverage-annual-1/brasil_fire_coverage_annual_{ano}.tif")
CELL = 0.25
BBOX = (-75.0, -19.0, -43.0, 6.0)  # lon_min, lat_min, lon_max, lat_max
ANOS = list(range(2015, 2021))     # COGs públicos vão até 2020
DECIM = 16                          # ~480 m: rápido e preserva presença com Resampling.max


def focos_por_celula_ano() -> pd.DataFrame:
    parts = []
    for a in sorted(glob.glob(str(ROOT / "focos_qmd_inpe_*.csv"))):
        d = pd.read_csv(a, usecols=["DataHora", "Latitude", "Longitude"]).dropna()
        d["ano"] = pd.to_datetime(d["DataHora"], format="%Y/%m/%d %H:%M:%S", errors="coerce").dt.year
        d["LatBin"] = (np.round(d["Latitude"] / CELL) * CELL).round(3)
        d["LonBin"] = (np.round(d["Longitude"] / CELL) * CELL).round(3)
        parts.append(d[["LatBin", "LonBin", "ano"]])
    f = pd.concat(parts, ignore_index=True)
    f = f[f["ano"].isin(ANOS)]
    return f.groupby(["LatBin", "LonBin", "ano"]).size().rename("n_focos").reset_index()


def celulas_queimadas_mapbiomas(ano: int) -> set:
    """Conjunto de células 0,25° com cicatriz de queimada no ano (MapBiomas)."""
    with rasterio.open(URL.format(ano=ano)) as ds:
        win = from_bounds(BBOX[0], BBOX[1], BBOX[2], BBOX[3], ds.transform)
        out_h, out_w = int(win.height // DECIM), int(win.width // DECIM)
        # média: para valores não-negativos, >0 indica qualquer pixel queimado no bloco
        arr = ds.read(1, window=win, out_shape=(out_h, out_w), resampling=Resampling.average)
        wt = ds.window_transform(win)
        t = wt * Affine.scale(win.width / out_w, win.height / out_h)
    ys, xs = np.where(arr > 0)
    # transform afim: x = a*col + b*row + c ; y = d*col + e*row + f
    lon = t.a * (xs + 0.5) + t.b * (ys + 0.5) + t.c
    lat = t.d * (xs + 0.5) + t.e * (ys + 0.5) + t.f
    latbin = np.round(np.round(lat / CELL) * CELL, 3)
    lonbin = np.round(np.round(lon / CELL) * CELL, 3)
    return set(map(tuple, np.unique(np.column_stack([latbin, lonbin]), axis=0)))


def main():
    print("Lendo focos do INPE por célula/ano...")
    fo = focos_por_celula_ano()
    universo = fo[["LatBin", "LonBin"]].drop_duplicates()  # células fire-prone
    print(f"  células fire-prone: {len(universo):,} | anos: {ANOS[0]}–{ANOS[-1]}")

    # presença de foco por (célula, ano)
    focos_set = set(map(tuple, fo[["LatBin", "LonBin", "ano"]].values))

    registros = []
    for ano in ANOS:
        print(f"  lendo raster MapBiomas {ano} (overview, Resampling.average)...", flush=True)
        queimadas = celulas_queimadas_mapbiomas(ano)
        for lat, lon in universo.itertuples(index=False):
            f_pres = int((lat, lon, ano) in focos_set)
            m_pres = int((lat, lon) in queimadas)
            registros.append((lat, lon, ano, f_pres, m_pres))
    df = pd.DataFrame(registros, columns=["LatBin", "LonBin", "ano", "focos", "mapbiomas"])

    y_f = df["focos"].to_numpy()
    y_m = df["mapbiomas"].to_numpy()
    cm = confusion_matrix(y_m, y_f)  # linhas=MapBiomas(ref), colunas=focos
    kappa = float(cohen_kappa_score(y_m, y_f))
    # tratando MapBiomas como referência, focos como "preditor" de área queimada
    prec, rec, f1, _ = precision_recall_fscore_support(y_m, y_f, average="binary", zero_division=0)
    concord = float((y_f == y_m).mean())

    res = {
        "fonte": "MapBiomas Fogo Coleção 3 (raster anual 30 m, via COG/vsicurl)",
        "nivel": "célula 0,25° x ano",
        "anos": ANOS, "n_celula_ano": int(len(df)),
        "focos_present_frac": float(y_f.mean()), "mapbiomas_present_frac": float(y_m.mean()),
        "concordancia": concord, "cohen_kappa": kappa,
        "focos_vs_mapbiomas": {"precisao": float(prec), "revocacao": float(rec), "f1": float(f1)},
        "matriz_confusao": {"mb0_focos0": int(cm[0, 0]), "mb0_focos1": int(cm[0, 1]),
                            "mb1_focos0": int(cm[1, 0]), "mb1_focos1": int(cm[1, 1])},
    }
    REL_DIR.mkdir(parents=True, exist_ok=True)
    (REL_DIR / "validacao_rotulo_celular.json").write_text(
        json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n== Concordância célula-a-célula (n={len(df):,} célula-ano, 2015–2020) ==")
    print(f"  focos presentes: {y_f.mean():.1%} | MapBiomas queimado: {y_m.mean():.1%}")
    print(f"  concordância: {concord:.3f} | Cohen's kappa: {kappa:.3f}")
    print(f"  focos vs área queimada — precisão {prec:.3f}, revocação {rec:.3f}, F1 {f1:.3f}")
    print(f"  matriz [MB x focos]: [[{cm[0,0]},{cm[0,1]}],[{cm[1,0]},{cm[1,1]}]]")
    print(f"  salvo em: {REL_DIR / 'validacao_rotulo_celular.json'}")


if __name__ == "__main__":
    main()
