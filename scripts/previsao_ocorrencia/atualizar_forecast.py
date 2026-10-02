"""Atualização OPERACIONAL do forecast de ocorrência de fogo (tempo real).

Encadeia o pipeline com os dados mais recentes publicados pelo INPE:

  1. focos do ano corrente (dados abertos do Programa Queimadas);
  2. alertas DETER do ano corrente (TerraBrasilis), mesclados ao histórico;
  3. features até o último mês COMPLETO (t);
  4. P(fogo em t+1) por célula com o modelo salvo -> CSV + mapa.

Por padrão NÃO re-treina (usa ``modelos/modelo_ocorrencia.pkl``). Com
``--retreinar``, re-treina com todo o histórico antes de prever — recomendado
uma vez por ano, quando o INPE publica o arquivo anual consolidado.

Uso:
  python atualizar_forecast.py              # atualiza dados + prevê
  python atualizar_forecast.py --retreinar  # idem, re-treinando o modelo
  python atualizar_forecast.py --sem_download  # só recalcula com os dados locais
"""
from __future__ import annotations

import argparse
import datetime as dt
import subprocess
import sys
from pathlib import Path

import pandas as pd

AQUI = Path(__file__).resolve().parent
ROOT = AQUI.parents[1]
DETER_ATUAL = AQUI / "deter_celula_mes_atual.csv"


def rodar(script: str, *args: str) -> None:
    cmd = [sys.executable, "-u", str(AQUI / script), *args]
    print(f"\n$ {script} {' '.join(args)}", flush=True)
    subprocess.run(cmd, check=True)


def atualizar_deter(ano: int) -> None:
    """Re-coleta o DETER do ano corrente e substitui essas linhas no histórico."""
    tmp = AQUI / f"deter_celula_mes_{ano}.tmp.csv"
    try:
        rodar("coletar_deter.py", "--ano_ini", str(ano), "--ano_fim", str(ano), "--saida", str(tmp))
    except subprocess.CalledProcessError:
        # TerraBrasilis instável: segue com o DETER já salvo em vez de abortar.
        ultimo = pd.read_csv(DETER_ATUAL)["ym"].max() if DETER_ATUAL.exists() else "nenhum"
        print(f"AVISO: falha ao coletar o DETER {ano}; usando o arquivo existente (até {ultimo}).")
        return
    novo = pd.read_csv(tmp)
    if DETER_ATUAL.exists():
        hist = pd.read_csv(DETER_ATUAL)
        hist = hist[~hist["ym"].astype(str).str.startswith(str(ano))]
        novo = pd.concat([hist, novo], ignore_index=True)
    novo.to_csv(DETER_ATUAL, index=False)
    tmp.unlink()
    print(f"DETER atualizado até {novo['ym'].max()} ({len(novo):,} registros)")


def main():
    ap = argparse.ArgumentParser(description="Atualiza dados e o forecast de ocorrência.")
    ap.add_argument("--retreinar", action="store_true")
    ap.add_argument("--sem_download", action="store_true")
    ap.add_argument("--ate_mes", type=str, default=None,
                    help="último mês de features (default: último mês completo)")
    args = ap.parse_args()

    hoje = dt.date.today()
    ate_mes = args.ate_mes or str(pd.Period(hoje, freq="M") - 1)
    print(f"Forecast operacional: features até {ate_mes} -> previsão para "
          f"{pd.Period(ate_mes, freq='M') + 1}")

    if not args.sem_download:
        rodar("coletar_focos_inpe.py", "--so_corrente")
        atualizar_deter(hoje.year)

    rodar("construir_ocorrencia.py", "--ate_mes", ate_mes)
    extra = [] if args.retreinar else ["--so_prever"]
    rodar("gerar_forecast_ocorrencia.py", "--deter", str(DETER_ATUAL), *extra)


if __name__ == "__main__":
    main()
