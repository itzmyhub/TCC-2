"""P5 — Coleta de clima GRIDADO (NASA POWER) por célula 0,25°, com FWI real.

Remove a limitação central dos pipelines P1/P3: em vez de imputar o clima das
ausências por climatologia, busca a **reanálise diária real** (NASA POWER:
T2M, RH2M, WS2M, PRECTOTCORR) para CADA célula *fire-prone* ao longo de
2014–2023, computa o **FWI canadense dia-a-dia** (`fwi.py`) e agrega ao mês.

Características:
  • Concorrente (ThreadPoolExecutor) — ~20 min para ~4.168 células @8 threads.
  • Resumível — cache por célula em .cache_umidade/gridded_mensal/; reexecuções
    pulam células já coletadas (pode rodar em background e ser reinvocado).
  • Robusto — 1 retry por célula; falhas persistentes são puladas e logadas.

Saída: um JSON por célula com os agregados mensais (clima real + FWI). O
``montar_dataset_gridded.py`` consome esse cache.
"""
from __future__ import annotations

import argparse
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import fwi as fwi_mod
import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "dataset_ocorrencia_mensal.csv"
CACHE = ROOT / ".cache_umidade" / "gridded_mensal"
NASA_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"

_lock = threading.Lock()
_contador = {"ok": 0, "falha": 0, "cache": 0}


def _cell_path(lat, lon) -> Path:
    return CACHE / f"cell_{lat:.2f}_{lon:.2f}.json"


def _fetch_daily(lat, lon, start, end) -> pd.DataFrame:
    params = {"parameters": "T2M,RH2M,WS2M,PRECTOTCORR", "community": "AG",
              "longitude": lon, "latitude": lat, "start": start, "end": end, "format": "JSON"}
    # Backoff respeitando 429 (rate-limit do NASA POWER): a coleta full com alta
    # concorrência é bloqueada; aqui aguardamos Retry-After / backoff exponencial.
    for tent in range(5):
        r = requests.get(NASA_URL, params=params, timeout=90)
        if r.status_code == 429:
            espera = float(r.headers.get("Retry-After", 2 ** tent * 5))
            time.sleep(min(espera, 60))
            continue
        r.raise_for_status()
        break
    else:
        raise RuntimeError("429 persistente após retries")
    p = r.json()["properties"]["parameter"]
    rows = []
    for d in sorted(p["T2M"].keys()):
        t, rh, ws, pr = p["T2M"][d], p["RH2M"][d], p["WS2M"][d], p["PRECTOTCORR"][d]
        if min(t, rh, ws, pr) <= -900:
            continue
        rows.append({"data": pd.Timestamp(d), "temp": t,
                     "rh": float(np.clip(rh, 0, 100)),
                     "wind": max(0.0, ws) * 3.6, "rain": max(0.0, pr)})
    return pd.DataFrame(rows)


def _agregar_mensal(serie: pd.DataFrame) -> list:
    serie["ym"] = serie["data"].dt.to_period("M").astype(str)
    g = serie.groupby("ym")
    out = g.agg(
        temp_mean=("temp", "mean"), rh_mean=("rh", "mean"), wind_mean=("wind", "mean"),
        precip_sum=("rain", "sum"), fwi_mean=("fwi", "mean"), fwi_max=("fwi", "max"),
    ).reset_index()
    out["dias_secos"] = g["rain"].apply(lambda s: int((s < 1.0).sum())).values
    out["dias_fwi_gt10"] = g["fwi"].apply(lambda s: int((s > 10.0).sum())).values
    return out.round(3).to_dict(orient="records")


def processar_celula(lat, lon, start, end) -> str:
    fp = _cell_path(lat, lon)
    if fp.exists():
        with _lock:
            _contador["cache"] += 1
        return "cache"
    for tentativa in (1, 2):
        try:
            daily = _fetch_daily(lat, lon, start, end)
            if daily.empty:
                raise ValueError("série vazia")
            serie = fwi_mod.fwi_serie(daily, lat=lat)
            registros = _agregar_mensal(serie)
            fp.write_text(json.dumps({"LatBin": lat, "LonBin": lon, "meses": registros}))
            with _lock:
                _contador["ok"] += 1
            return "ok"
        except Exception as e:
            if tentativa == 2:
                with _lock:
                    _contador["falha"] += 1
                return f"falha: {e}"
            time.sleep(1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3,
                    help="Use poucos workers (3) — o NASA POWER aplica rate-limit (HTTP 429) sob alta concorrência.")
    ap.add_argument("--start", type=str, default="20140101")
    ap.add_argument("--end", type=str, default="20231231")
    ap.add_argument("--limite", type=int, default=0, help="0 = todas as células.")
    args = ap.parse_args()

    CACHE.mkdir(parents=True, exist_ok=True)
    cells = pd.read_csv(DATASET, usecols=["LatBin", "LonBin"]).drop_duplicates().reset_index(drop=True)
    if args.limite:
        cells = cells.head(args.limite)
    total = len(cells)
    pendentes = [(r.LatBin, r.LonBin) for r in cells.itertuples()
                 if not _cell_path(r.LatBin, r.LonBin).exists()]
    print(f"Células: {total} | já em cache: {total - len(pendentes)} | a coletar: {len(pendentes)}")
    if not pendentes:
        print("Tudo em cache. Nada a fazer.")
        return

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(processar_celula, lat, lon, args.start, args.end): (lat, lon)
                for lat, lon in pendentes}
        for i, _ in enumerate(as_completed(futs), 1):
            if i % 100 == 0 or i == len(pendentes):
                dt = time.time() - t0
                print(f"  {i}/{len(pendentes)} | ok={_contador['ok']} "
                      f"falha={_contador['falha']} | {dt/60:.1f} min "
                      f"| ~{dt/i*(len(pendentes)-i)/60:.1f} min restantes")
    cached_total = len(list(CACHE.glob("cell_*.json")))
    print(f"\nConcluído. ok={_contador['ok']} falha={_contador['falha']} "
          f"| total em cache: {cached_total}/{total}")


if __name__ == "__main__":
    main()
