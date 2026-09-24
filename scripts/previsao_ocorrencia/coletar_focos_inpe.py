"""Coleta os focos de fogo do INPE (dados abertos) no formato ``focos_qmd_inpe_*.csv``.

Reconstrói a entrada bruta de ``construir_ocorrencia.py`` direto do servidor de dados
abertos do Programa Queimadas [INPE24]:

  • anos fechados → ``anual/Brasil_todos_sats/focos_br_todos-sats_<ANO>.zip``;
  • ano corrente  → ``mensal/Brasil/focos_mensal_br_<AAAAMM>.csv|.zip``.

Mantém o mesmo recorte da base original do TCC: satélite de referência ``AQUA_M-T``
e bioma ``Amazônia`` (verificado: as células de 2019 caem 100% no domínio antigo).
Grava ``focos_qmd_inpe_<ANO>.csv`` na raiz com o cabeçalho original:
``DataHora, Satelite, Pais, Estado, Municipio, Bioma, DiaSemChuva, Precipitacao,
RiscoFogo, Latitude, Longitude, FRP``.

Uso:
  python coletar_focos_inpe.py                  # 2014 até o mês corrente
  python coletar_focos_inpe.py --ano_fim 2023   # só o período do TCC
  python coletar_focos_inpe.py --so_corrente    # re-baixa apenas o ano corrente
"""
from __future__ import annotations

import argparse
import datetime as dt
import io
import re
import time
import zipfile
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[2]
BASE = "https://dataserver-coids.inpe.br/queimadas/queimadas/focos/csv"
CACHE = ROOT / ".cache_focos"
SATELITE = "AQUA_M-T"
BIOMA = "Amazônia"
COLS_SAIDA = ["DataHora", "Satelite", "Pais", "Estado", "Municipio", "Bioma",
              "DiaSemChuva", "Precipitacao", "RiscoFogo", "Latitude", "Longitude", "FRP"]
# Nomes do dado aberto (anual usa latitude/data_pas; mensal usa lat/data_hora_gmt)
RENOMEAR = {"latitude": "Latitude", "lat": "Latitude", "longitude": "Longitude", "lon": "Longitude",
            "data_pas": "DataHora", "data_hora_gmt": "DataHora", "satelite": "Satelite",
            "pais": "Pais", "estado": "Estado", "municipio": "Municipio", "bioma": "Bioma",
            "numero_dias_sem_chuva": "DiaSemChuva", "precipitacao": "Precipitacao",
            "risco_fogo": "RiscoFogo", "frp": "FRP"}


def baixar(url: str, destino: Path) -> Path:
    """Baixa ``url`` para ``destino`` (com cache e novas tentativas)."""
    if destino.exists() and destino.stat().st_size > 0:
        return destino
    destino.parent.mkdir(parents=True, exist_ok=True)
    for tent in range(4):
        try:
            with requests.get(url, stream=True, timeout=300) as r:
                r.raise_for_status()
                tmp = destino.with_suffix(destino.suffix + ".part")
                with open(tmp, "wb") as f:
                    for bloco in r.iter_content(1 << 20):
                        f.write(bloco)
            tmp.replace(destino)
            return destino
        except Exception:
            if tent == 3:
                raise
            time.sleep(5 * (tent + 1))
    return destino


def filtrar(arquivo: Path) -> pd.DataFrame:
    """Lê um CSV (ou o CSV dentro de um .zip) em blocos, mantendo AQUA_M-T/Amazônia."""
    if arquivo.suffix == ".zip":
        with zipfile.ZipFile(arquivo) as z:
            nome = next(n for n in z.namelist() if n.endswith(".csv"))
            fonte = io.BytesIO(z.read(nome))
    else:
        fonte = arquivo
    partes = []
    for bloco in pd.read_csv(fonte, chunksize=500_000, skipinitialspace=True, low_memory=False):
        bloco.columns = [c.strip() for c in bloco.columns]
        bloco = bloco.rename(columns=RENOMEAR)
        bloco["Satelite"] = bloco["Satelite"].astype(str).str.strip()
        bloco["Bioma"] = bloco["Bioma"].astype(str).str.strip()
        partes.append(bloco[(bloco["Satelite"] == SATELITE) & (bloco["Bioma"] == BIOMA)])
    df = pd.concat(partes, ignore_index=True)
    df["DataHora"] = pd.to_datetime(df["DataHora"], errors="coerce").dt.strftime("%Y/%m/%d %H:%M:%S")
    return df[COLS_SAIDA]


def arquivos_mensais(ano: int) -> list[str]:
    """Lista os arquivos mensais publicados para ``ano`` (csv ou zip)."""
    html = requests.get(f"{BASE}/mensal/Brasil/", timeout=60).text
    return sorted(set(re.findall(rf'href="(focos_mensal_br_{ano}\d\d\.(?:csv|zip))"', html)))


def coletar_ano_fechado(ano: int) -> pd.DataFrame:
    nome = f"focos_br_todos-sats_{ano}.zip"
    return filtrar(baixar(f"{BASE}/anual/Brasil_todos_sats/{nome}", CACHE / nome))


def coletar_ano_corrente(ano: int) -> pd.DataFrame:
    partes = []
    hoje = dt.date.today().strftime("%Y%m")
    for nome in arquivos_mensais(ano):
        destino = CACHE / nome
        if hoje in nome and destino.exists():
            destino.unlink()  # mês em andamento: sempre re-baixa
        partes.append(filtrar(baixar(f"{BASE}/mensal/Brasil/{nome}", destino)))
        print(f"    {nome}: {len(partes[-1]):,} focos")
    return pd.concat(partes, ignore_index=True)


def anos_fechados_publicados() -> set[int]:
    html = requests.get(f"{BASE}/anual/Brasil_todos_sats/", timeout=60).text
    return {int(a) for a in re.findall(r"focos_br_todos-sats_(\d{4})\.zip", html)}


def main():
    ano_atual = dt.date.today().year
    ap = argparse.ArgumentParser(description="Coleta focos INPE (AQUA_M-T, Amazônia).")
    ap.add_argument("--ano_ini", type=int, default=2014)
    ap.add_argument("--ano_fim", type=int, default=ano_atual)
    ap.add_argument("--so_corrente", action="store_true", help="re-baixa só o ano corrente")
    args = ap.parse_args()

    fechados = anos_fechados_publicados()
    if args.so_corrente:
        # No início do ano o anual do ano anterior pode ainda não ter sido publicado.
        anos = [a for a in (ano_atual - 1, ano_atual) if a == ano_atual or a not in fechados]
    else:
        anos = range(args.ano_ini, args.ano_fim + 1)
    for ano in anos:
        t0 = time.time()
        print(f"  {ano}: {'anual' if ano in fechados else 'mensal'}...", flush=True)
        df = coletar_ano_fechado(ano) if ano in fechados else coletar_ano_corrente(ano)
        saida = ROOT / f"focos_qmd_inpe_{ano}.csv"
        df.to_csv(saida, index=False)
        print(f"  {ano}: {len(df):,} focos -> {saida.name} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
