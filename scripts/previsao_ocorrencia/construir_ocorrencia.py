"""Construção do dataset de OCORRÊNCIA de fogo (P1 + P2).

Reorienta o problema do TCC: em vez de discretizar o índice RiscoFogo do INPE
(alvo = saída de outro modelo, base *presence-only*), este script constrói um
dataset de **ocorrência observada de fogo** com **classe negativa real** e
**features causais** para um problema de PREVISÃO com horizonte explícito.

Desenho (fundamentado em deandrade2024lstmgru — focos mensais na Amazônia; e no
paradigma SeasFire/FireCastNet de previsão de ocorrência sobre grade 0,25°):

  • Unidade espaço-temporal: (célula 0,25°, mês).
  • Universo: células *fire-prone* — toda célula com ≥1 foco em 2014–2023.
    (Restrição de domínio honesta: "dado que a célula é propensa a fogo,
     haverá fogo no próximo mês?". Evita o trivial 'floresta densa nunca queima'.)
  • Presença/ausência: cell-month com ≥1 foco → presença; sem foco → AUSÊNCIA
    (a classe negativa que faltava na base presence-only).
  • Alvo (forecast): y = 1 se houver foco no mês t+H, senão 0. H configurável.
  • Features: SOMENTE informação ≤ t (causais — corrige o vazamento temporal
    de −12,8 pp documentado pela validação rolling-origin do projeto).

LIMITAÇÃO DOCUMENTADA: o clima mês-a-mês (DiaSemChuva, Precipitacao) só existe
nos cell-months que tiveram foco (os CSVs do INPE são presence-only). Para os
cell-months de ausência usamos a **climatologia da célula por mês-calendário**
(média dos meses observados), com fallback célula→global. A solução plena
(reanálise NASA POWER/ERA5 por célula-mês para todas as células) é a melhoria
P5 e exige download massivo; aqui o sinal preditivo vem sobretudo do
**histórico de fogo defasado** + sazonalidade + climatologia local.

Saída: ``dataset_ocorrencia_mensal.csv`` na raiz do projeto.
"""
from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CELL = 0.25  # graus (~27 km), mesma granularidade do MERRA-2/SeasFire


def carregar_focos() -> pd.DataFrame:
    """Lê e concatena todos os CSVs de focos do INPE."""
    arquivos = sorted(glob.glob(str(ROOT / "focos_qmd_inpe_*.csv")))
    if not arquivos:
        raise FileNotFoundError("Nenhum focos_qmd_inpe_*.csv encontrado na raiz.")
    cols = ["DataHora", "Estado", "DiaSemChuva", "Precipitacao",
            "RiscoFogo", "Latitude", "Longitude", "FRP"]
    partes = []
    for a in arquivos:
        df = pd.read_csv(a, usecols=lambda c: c in cols)
        partes.append(df)
        print(f"  lido {os.path.basename(a)}: {len(df):,} focos")
    focos = pd.concat(partes, ignore_index=True)
    # Tipagem e limpeza de sentinelas (-999 = ausente no padrão INPE)
    focos["DataHora"] = pd.to_datetime(focos["DataHora"], format="%Y/%m/%d %H:%M:%S",
                                       errors="coerce")
    focos = focos.dropna(subset=["DataHora", "Latitude", "Longitude"])
    for c in ["DiaSemChuva", "Precipitacao", "RiscoFogo", "FRP"]:
        focos[c] = pd.to_numeric(focos[c], errors="coerce")
        focos.loc[focos[c] <= -999, c] = np.nan
    focos["FRP"] = focos["FRP"].fillna(0.0)
    print(f"  total de focos válidos: {len(focos):,}")
    return focos


def agregar_celula_mes(focos: pd.DataFrame) -> pd.DataFrame:
    """Agrega focos por (célula 0,25°, ano-mês)."""
    focos["LatBin"] = (np.round(focos["Latitude"] / CELL) * CELL).round(3)
    focos["LonBin"] = (np.round(focos["Longitude"] / CELL) * CELL).round(3)
    focos["ym"] = focos["DataHora"].dt.to_period("M")
    g = focos.groupby(["LatBin", "LonBin", "ym"]).agg(
        n_focos=("DataHora", "size"),
        DiaSemChuva=("DiaSemChuva", "mean"),
        Precipitacao=("Precipitacao", "mean"),
        RiscoFogo_inpe=("RiscoFogo", "mean"),
        FRP=("FRP", "mean"),
    ).reset_index()
    return g


def construir_grade(agg: pd.DataFrame) -> pd.DataFrame:
    """Expande para grade completa (célula × todos os meses) preenchendo
    cell-months sem foco com n_focos=0 (a classe negativa/ausência)."""
    meses = pd.period_range(agg["ym"].min(), agg["ym"].max(), freq="M")
    celulas = agg[["LatBin", "LonBin"]].drop_duplicates()
    print(f"  células fire-prone: {len(celulas):,} | meses: {len(meses)}")
    # produto cartesiano células × meses
    grade = celulas.assign(key=1).merge(
        pd.DataFrame({"ym": meses, "key": 1}), on="key").drop(columns="key")
    full = grade.merge(agg, on=["LatBin", "LonBin", "ym"], how="left")
    full["n_focos"] = full["n_focos"].fillna(0).astype(int)
    full["FRP"] = full["FRP"].fillna(0.0)
    full["fogo"] = (full["n_focos"] > 0).astype(int)
    return full


def imputar_clima_climatologia(full: pd.DataFrame) -> pd.DataFrame:
    """Preenche clima dos cell-months de ausência com a climatologia da célula
    por mês-calendário (média dos meses observados); fallback célula→global.

    CAVEAT (vazamento temporal leve, DOCUMENTADO): a climatologia é calculada
    com ``groupby().transform("mean")`` sobre o dataset COMPLETO (todos os anos),
    antes do split temporal feito em ``treinar_validar.py``. Logo, a média
    sazonal de cada célula usa também meses de anos futuros ao fold de teste.
    Impacto prático NULO: experimentos controlados (P5) mostram que o clima é
    redundante frente ao histórico de fogo (ganho +0,001 PR-AUC), e estas 3
    colunas são justamente as de menor importância. A correção estrita
    (climatologia estimada só no treino de cada fold) é encaminhamento futuro;
    mantém-se aqui a versão simples por reprodutibilidade dos números do TCC.
    """
    full["mes"] = full["ym"].dt.month
    for col in ["DiaSemChuva", "Precipitacao", "RiscoFogo_inpe"]:
        clim_cel_mes = full.groupby(["LatBin", "LonBin", "mes"])[col].transform("mean")
        clim_cel = full.groupby(["LatBin", "LonBin"])[col].transform("mean")
        glob_ = full[col].mean()
        full[col] = full[col].fillna(clim_cel_mes).fillna(clim_cel).fillna(glob_)
    return full


def adicionar_features_causais(full: pd.DataFrame, horizonte: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Features que usam SOMENTE dados ≤ t, e alvo = fogo em t+H.

    Retorna (linhas com alvo conhecido, linhas do último mês sem alvo). As
    últimas são a entrada do forecast operacional: features de t, alvo em t+H
    ainda não observado.
    """
    full = full.sort_values(["LatBin", "LonBin", "ym"]).reset_index(drop=True)
    g = full.groupby(["LatBin", "LonBin"], sort=False)

    # Histórico de fogo defasado (tudo ≤ t)
    for lag in (1, 2, 3):
        full[f"focos_lag{lag}"] = g["n_focos"].shift(lag).fillna(0)
        full[f"fogo_lag{lag}"] = g["fogo"].shift(lag).fillna(0)
    # Somas móveis do passado (shift(1) garante exclusão do mês corrente futuro)
    for w in (3, 6, 12):
        full[f"focos_roll{w}"] = (
            g["n_focos"].shift(1).rolling(w, min_periods=1).sum().reset_index(drop=True)
        )
        full[f"fogo_roll{w}"] = (
            g["fogo"].shift(1).rolling(w, min_periods=1).sum().reset_index(drop=True)
        )
    # FRP defasado e clima do mês corrente (conhecido ao fim de t; ≤ t p/ alvo t+H)
    full["FRP_lag1"] = g["FRP"].shift(1).fillna(0)

    # Meses desde o último fogo (vetorizado por célula)
    def meses_desde(s: pd.Series) -> pd.Series:
        out = np.empty(len(s), dtype=float)
        contador = 99
        for i, v in enumerate(s.to_numpy()):
            out[i] = contador
            contador = 0 if v > 0 else min(contador + 1, 99)
        return pd.Series(out, index=s.index)
    full["meses_desde_fogo"] = g["fogo"].transform(meses_desde)

    # Sazonalidade (estação seca da Amazônia ~ jul–out)
    full["mes_sin"] = np.sin(2 * np.pi * full["mes"] / 12)
    full["mes_cos"] = np.cos(2 * np.pi * full["mes"] / 12)
    full["estacao_seca"] = full["mes"].isin([7, 8, 9, 10]).astype(int)

    # ALVO: houve foco no mês t+H?
    full["alvo"] = g["fogo"].shift(-horizonte)
    # Descarta os 12 primeiros meses por célula (lags/rolls imaturos)
    full["ord"] = g.cumcount()
    full = full[full["ord"] >= 12].drop(columns="ord")
    ultimo = full[full["alvo"].isna() & (full["ym"] == full["ym"].max())].copy()
    full = full.dropna(subset=["alvo"]).copy()
    full["alvo"] = full["alvo"].astype(int)
    return full, ultimo


FEATURES = [
    "LatBin", "LonBin", "mes_sin", "mes_cos", "estacao_seca",
    "DiaSemChuva", "Precipitacao", "RiscoFogo_inpe",
    "focos_lag1", "focos_lag2", "focos_lag3",
    "fogo_lag1", "fogo_lag2", "fogo_lag3",
    "focos_roll3", "focos_roll6", "focos_roll12",
    "fogo_roll3", "fogo_roll6", "fogo_roll12",
    "FRP_lag1", "meses_desde_fogo",
]


def main():
    ap = argparse.ArgumentParser(description="Constrói dataset de ocorrência de fogo (P1+P2).")
    ap.add_argument("--horizonte", type=int, default=1, help="Horizonte de previsão em meses (default 1).")
    ap.add_argument("--saida", type=str, default=str(ROOT / "dataset_ocorrencia_mensal.csv"))
    ap.add_argument("--saida_ultimo", type=str, default=str(ROOT / "dataset_ocorrencia_ultimo_mes.csv"),
                    help="Features do último mês (sem alvo) — entrada do forecast operacional.")
    ap.add_argument("--ate_mes", type=str, default=None,
                    help="Último mês incluído (AAAA-MM). Use o último mês COMPLETO no modo "
                         "operacional; omitido = todos os focos disponíveis.")
    args = ap.parse_args()

    print("[1/5] Lendo focos do INPE...")
    focos = carregar_focos()
    print("[2/5] Agregando por (célula 0,25°, mês)...")
    agg = agregar_celula_mes(focos)
    if args.ate_mes:
        agg = agg[agg["ym"] <= pd.Period(args.ate_mes, freq="M")]
    print("[3/5] Expandindo grade e gerando classe negativa (ausências)...")
    full = construir_grade(agg)
    full = imputar_clima_climatologia(full)
    print(f"[4/5] Features causais + alvo (horizonte={args.horizonte} mês)...")
    full, ultimo = adicionar_features_causais(full, args.horizonte)

    cols_out = ["LatBin", "LonBin", "ym", "ano", "mes", "n_focos", "fogo"] + \
               [c for c in FEATURES if c not in ("LatBin", "LonBin")] + ["alvo"]
    for df_ in (full, ultimo):
        df_["ano"] = df_["ym"].dt.year
        df_["ym"] = df_["ym"].astype(str)
    out = full[cols_out].copy()
    out.to_csv(args.saida, index=False)
    ultimo[cols_out[:-1]].to_csv(args.saida_ultimo, index=False)
    print(f"  último mês ({ultimo['ym'].max()}): {len(ultimo):,} células -> {args.saida_ultimo}")

    prev = out["alvo"].mean()
    print("[5/5] Concluído.")
    print(f"  linhas: {len(out):,} | features: {len(FEATURES)}")
    print(f"  prevalência do alvo (fogo em t+{args.horizonte}): {prev:.3f} "
          f"({out['alvo'].sum():,} positivos / {len(out):,})")
    print(f"  período: {out['ano'].min()}–{out['ano'].max()}")
    print(f"  salvo em: {args.saida}")


if __name__ == "__main__":
    main()
