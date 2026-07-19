"""P5 — Monta o dataset de ocorrência com CLIMA GRIDADO REAL + FWI.

Consome o cache de ``coletar_clima_gridded.py`` (agregados mensais reais por
célula) e o funde ao ``dataset_ocorrencia_mensal.csv``, substituindo as colunas
de clima antes imputadas por climatologia e adicionando o **FWI real** como
feature causal (clima do mês t prevê fogo em t+1).

Saída: ``dataset_ocorrencia_gridded.csv``.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "dataset_ocorrencia_mensal.csv"
CACHE = ROOT / ".cache_umidade" / "gridded_mensal"

# Novas features de clima real (todas referentes ao mês t → causais p/ alvo t+1)
CLIMA_REAL = ["temp_mean", "rh_mean", "wind_mean", "precip_sum",
              "dias_secos", "fwi_mean", "fwi_max", "dias_fwi_gt10"]


def carregar_cache() -> pd.DataFrame:
    arqs = sorted(glob.glob(str(CACHE / "cell_*.json")))
    if not arqs:
        raise FileNotFoundError("Cache gridded vazio — rode coletar_clima_gridded.py antes.")
    linhas = []
    for a in arqs:
        j = json.loads(Path(a).read_text())
        for m in j["meses"]:
            m2 = {"LatBin": j["LatBin"], "LonBin": j["LonBin"], **m}
            linhas.append(m2)
    g = pd.DataFrame(linhas)
    print(f"  cache: {len(arqs):,} células | {len(g):,} registros célula-mês")
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--saida", type=str, default=str(ROOT / "dataset_ocorrencia_gridded.csv"))
    args = ap.parse_args()

    print("Carregando dataset de ocorrência e cache gridado...")
    df = pd.read_csv(DATASET)
    grid = carregar_cache()

    # ym no dataset está como 'YYYY-MM'; no cache idem.
    antes = len(df)
    out = df.merge(grid, on=["LatBin", "LonBin", "ym"], how="left",
                   suffixes=("", "_grid"))
    cobertura = out["fwi_mean"].notna().mean()
    print(f"  linhas: {antes:,} | cobertura de clima real: {cobertura:.1%}")

    # Restringe às linhas com clima real (descarta cell-months sem cobertura)
    out = out[out["fwi_mean"].notna()].copy()
    print(f"  linhas com clima real: {len(out):,} | prevalência alvo: {out['alvo'].mean():.3f}")
    out.to_csv(args.saida, index=False)
    print(f"  features de clima real adicionadas: {CLIMA_REAL}")
    print(f"  salvo em: {args.saida}")


if __name__ == "__main__":
    main()
