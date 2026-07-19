"""Sistema Fire Weather Index (FWI) canadense — van Wagner & Pickett (1985).

Implementação fiel dos seis componentes (FFMC, DMC, DC, ISI, BUI, FWI), o índice
físico de perigo de incêndio usado operacionalmente pelo Copernicus EFFIS/GEFF
[digiuseppe2024geff] e a base do "Risco de Fogo" do INPE. Serve de **baseline
físico** para o benchmark P3 (o modelo de ML precisa superá-lo sobre fogo
observado para "agregar valor").

O FWI é um sistema SEQUENCIAL diário: FFMC/DMC/DC carregam estado dia-a-dia.
Entradas diárias ao meio-dia: temperatura (°C), umidade relativa (%),
vento (km/h) e chuva acumulada 24 h (mm).

Validação: ``self_test()`` reproduz os valores de referência publicados
(temp=17, rh=42, vento=25, chuva=0; FFMC₀=85, DMC₀=6, DC₀=15, mês=abril) →
FFMC≈87.69, DMC≈8.55, DC≈19.01, ISI≈10.85, BUI≈8.49, FWI≈10.10.

LIMITAÇÃO para a Amazônia: os fatores de comprimento-do-dia (Le/Lf) das tabelas
originais são do hemisfério norte; aplicamos o deslocamento de 6 meses para
latitudes do hemisfério sul. Para a faixa equatorial o comprimento do dia é
~constante; tratamos por sinal da latitude e documentamos o caveat.
"""
from __future__ import annotations

import math
from typing import Dict, List

import numpy as np
import pandas as pd

# Fator de comprimento do dia para o DMC (Le) — hemisfério norte, jan..dez
_LE_NH = [6.5, 7.5, 9.0, 12.8, 13.9, 13.9, 12.4, 10.9, 9.4, 8.0, 7.0, 6.0]
# Fator de comprimento do dia para o DC (Lf) — hemisfério norte, jan..dez
_LF_NH = [-1.6, -1.6, -1.6, 0.9, 3.8, 5.8, 6.4, 5.0, 2.4, 0.4, -1.6, -1.6]


def _le(month: int, lat: float) -> float:
    idx = (month - 1)
    if lat < 0:  # hemisfério sul: desloca 6 meses
        idx = (idx + 6) % 12
    return _LE_NH[idx]


def _lf(month: int, lat: float) -> float:
    idx = (month - 1)
    if lat < 0:
        idx = (idx + 6) % 12
    return _LF_NH[idx]


def _ffmc(temp, rh, wind, rain, ffmc_prev):
    mo = 147.2 * (101.0 - ffmc_prev) / (59.5 + ffmc_prev)
    if rain > 0.5:
        rf = rain - 0.5
        if mo <= 150.0:
            mr = mo + 42.5 * rf * math.exp(-100.0 / (251.0 - mo)) * (1.0 - math.exp(-6.93 / rf))
        else:
            mr = (mo + 42.5 * rf * math.exp(-100.0 / (251.0 - mo)) * (1.0 - math.exp(-6.93 / rf))
                  + 0.0015 * (mo - 150.0) ** 2 * math.sqrt(rf))
        mo = min(mr, 250.0)
    ed = (0.942 * rh ** 0.679 + 11.0 * math.exp((rh - 100.0) / 10.0)
          + 0.18 * (21.1 - temp) * (1.0 - math.exp(-0.115 * rh)))
    if mo > ed:
        ko = 0.424 * (1.0 - (rh / 100.0) ** 1.7) + 0.0694 * math.sqrt(wind) * (1.0 - (rh / 100.0) ** 8)
        kd = ko * 0.581 * math.exp(0.0365 * temp)
        m = ed + (mo - ed) * 10.0 ** (-kd)
    else:
        ew = (0.618 * rh ** 0.753 + 10.0 * math.exp((rh - 100.0) / 10.0)
              + 0.18 * (21.1 - temp) * (1.0 - math.exp(-0.115 * rh)))
        if mo < ew:
            kl = (0.424 * (1.0 - ((100.0 - rh) / 100.0) ** 1.7)
                  + 0.0694 * math.sqrt(wind) * (1.0 - ((100.0 - rh) / 100.0) ** 8))
            kw = kl * 0.581 * math.exp(0.0365 * temp)
            m = ew - (ew - mo) * 10.0 ** (-kw)
        else:
            m = mo
    return max(0.0, min(101.0, 59.5 * (250.0 - m) / (147.2 + m)))


def _dmc(temp, rh, rain, dmc_prev, le):
    t = max(temp, -1.1)
    rk = 1.894 * (t + 1.1) * (100.0 - rh) * le * 1e-4
    if rain > 1.5:
        re = 0.92 * rain - 1.27
        mo = 20.0 + math.exp(5.6348 - dmc_prev / 43.43)
        if dmc_prev <= 33.0:
            b = 100.0 / (0.5 + 0.3 * dmc_prev)
        elif dmc_prev <= 65.0:
            b = 14.0 - 1.3 * math.log(dmc_prev)
        else:
            b = 6.2 * math.log(dmc_prev) - 17.2
        mr = mo + 1000.0 * re / (48.77 + b * re)
        pr = 244.72 - 43.43 * math.log(mr - 20.0)
        dmc_prev = max(0.0, pr)
    return max(0.0, dmc_prev + rk)


def _dc(temp, rain, dc_prev, lf):
    t = max(temp, -2.8)
    pe = max(0.0, (0.36 * (t + 2.8) + lf) / 2.0)
    if rain > 2.8:
        rd = 0.83 * rain - 1.27
        qo = 800.0 * math.exp(-dc_prev / 400.0)
        qr = qo + 3.937 * rd
        dr = max(0.0, 400.0 * math.log(800.0 / qr))
        dc_prev = dr
    return dc_prev + pe


def _isi(ffmc, wind):
    fw = math.exp(0.05039 * wind)
    m = 147.2 * (101.0 - ffmc) / (59.5 + ffmc)
    ff = 91.9 * math.exp(-0.1386 * m) * (1.0 + m ** 5.31 / 4.93e7)
    return 0.208 * fw * ff


def _bui(dmc, dc):
    if dmc == 0 and dc == 0:
        return 0.0
    if dmc <= 0.4 * dc:
        bui = 0.8 * dmc * dc / (dmc + 0.4 * dc)
    else:
        bui = dmc - (1.0 - 0.8 * dc / (dmc + 0.4 * dc)) * (0.92 + (0.0114 * dmc) ** 1.7)
    return max(0.0, bui)


def _fwi(isi, bui):
    if bui <= 80.0:
        fd = 0.626 * bui ** 0.809 + 2.0
    else:
        fd = 1000.0 / (25.0 + 108.64 * math.exp(-0.023 * bui))
    b = 0.1 * isi * fd
    if b > 1.0:
        return math.exp(2.72 * (0.434 * math.log(b)) ** 0.647)
    return b


def fwi_um_dia(temp, rh, wind, rain, month, lat,
               ffmc_prev=85.0, dmc_prev=6.0, dc_prev=15.0) -> Dict[str, float]:
    """Computa os 6 componentes para um dia, dado o estado do dia anterior."""
    ffmc = _ffmc(temp, rh, wind, rain, ffmc_prev)
    dmc = _dmc(temp, rh, rain, dmc_prev, _le(month, lat))
    dc = _dc(temp, rain, dc_prev, _lf(month, lat))
    isi = _isi(ffmc, wind)
    bui = _bui(dmc, dc)
    fwi = _fwi(isi, bui)
    return {"ffmc": ffmc, "dmc": dmc, "dc": dc, "isi": isi, "bui": bui, "fwi": fwi}


def fwi_serie(daily: pd.DataFrame, lat: float) -> pd.DataFrame:
    """Aplica o sistema a uma série diária ordenada.

    daily: colunas ['data','temp','rh','wind','rain'] (vento em km/h).
    Retorna o DataFrame com colunas ffmc/dmc/dc/isi/bui/fwi por dia.
    """
    daily = daily.sort_values("data").reset_index(drop=True)
    ffmc_p, dmc_p, dc_p = 85.0, 6.0, 15.0
    out = []
    for _, r in daily.iterrows():
        month = pd.Timestamp(r["data"]).month
        res = fwi_um_dia(r["temp"], r["rh"], r["wind"], r["rain"], month, lat,
                         ffmc_p, dmc_p, dc_p)
        ffmc_p, dmc_p, dc_p = res["ffmc"], res["dmc"], res["dc"]
        out.append(res)
    return pd.concat([daily, pd.DataFrame(out)], axis=1)


def self_test() -> bool:
    """Reproduz os valores de referência de van Wagner & Pickett (1985)."""
    r = fwi_um_dia(temp=17.0, rh=42.0, wind=25.0, rain=0.0, month=4, lat=45.0,
                   ffmc_prev=85.0, dmc_prev=6.0, dc_prev=15.0)
    esperado = {"ffmc": 87.69, "dmc": 8.55, "dc": 19.01, "isi": 10.85, "bui": 8.49, "fwi": 10.10}
    ok = True
    print("Componente   calculado   referência   ok")
    for k, exp in esperado.items():
        got = r[k]
        passou = abs(got - exp) < 0.1
        ok = ok and passou
        print(f"  {k:6s}     {got:9.3f}   {exp:9.2f}    {'OK' if passou else 'FALHA'}")
    print(f"\nself_test: {'PASSOU' if ok else 'FALHOU'}")
    return ok


if __name__ == "__main__":
    self_test()
