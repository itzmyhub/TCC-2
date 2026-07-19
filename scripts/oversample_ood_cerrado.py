"""Oversampling estratificado para pares (Estado, Mês) sub-representados.

Lê `modelos/relatorios/ood_pares_alvo.json` (gerado por
`scripts/analise_ood_cerrado.py`) e gera um dataset aumentado em
`base_de_dados_oversampled.csv`. O oversampling:

1. Identifica linhas no dataset enriquecido que pertencem a algum par alvo.
2. Replica cada linha alvo `floor(fator)` vezes (fator de 1.5x a 5x).
3. Adiciona jitter gaussiano leve em Latitude (σ=0,02º ≈ 2 km),
   Longitude (σ=0,02º), Hora (σ=1h) e FRP (σ=5% do valor) para evitar
   duplicação exata e dar variabilidade.
4. **Não modifica** Estado, Municipio, Mes (preservam a semântica do par)
   nem as features Tier 1 (serão recalculadas no `carregar_dados`).

Boas práticas:
- O oversampling **só deve ser usado no conjunto de TREINO** para evitar
  inflar artificialmente as métricas de avaliação. O script de retreino
  `scripts/retreinar_com_oversample.py` se encarrega disso fazendo o
  split antes do oversampling.
- Apesar disso, este script gera o CSV completo aumentado por
  conveniência de exploração; o retreino aplica a separação correta.

Uso:
    python scripts/oversample_ood_cerrado.py
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("oversample_ood")

ROOT = Path(__file__).resolve().parent.parent
ENRICHED_PATH = ROOT / "base_de_dados_enriquecido.csv"
OOD_ALVOS_PATH = ROOT / "modelos" / "relatorios" / "ood_pares_alvo.json"
OUT_PATH = ROOT / "base_de_dados_oversampled.csv"

# Sigmas de jitter
SIGMA_LAT = 0.02   # ≈ 2,2 km
SIGMA_LON = 0.02
SIGMA_HORA = 1.0    # ±1h
SIGMA_FRP_REL = 0.05  # ±5% do FRP

SEED = 42


def carregar_alvos() -> List[Dict]:
    if not OOD_ALVOS_PATH.exists():
        raise FileNotFoundError(
            f"Arquivo de alvos não encontrado: {OOD_ALVOS_PATH}. "
            "Rode antes: python scripts/analise_ood_cerrado.py"
        )
    with OOD_ALVOS_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("pares_alvo", [])


def _norm_estado(s) -> str:
    return s.strip().upper() if isinstance(s, str) else ""


def _aplicar_jitter(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    df = df.copy()
    n = len(df)
    if "Latitude" in df.columns:
        df["Latitude"] = df["Latitude"].astype(float) + rng.normal(0.0, SIGMA_LAT, size=n)
    if "Longitude" in df.columns:
        df["Longitude"] = df["Longitude"].astype(float) + rng.normal(0.0, SIGMA_LON, size=n)
    if "Hora" in df.columns:
        h_jitter = df["Hora"].astype(float) + rng.normal(0.0, SIGMA_HORA, size=n)
        df["Hora"] = np.clip(np.round(h_jitter).astype(int), 0, 23)
    if "FRP" in df.columns:
        frp = df["FRP"].astype(float).values
        df["FRP"] = np.maximum(
            0.0, frp * (1.0 + rng.normal(0.0, SIGMA_FRP_REL, size=n))
        )
    return df


def oversample(
    df: pd.DataFrame,
    alvos: List[Dict],
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Gera dataframe aumentado (original + cópias jittered dos pares-alvo)."""
    if "Estado" not in df.columns or "Mes" not in df.columns:
        raise ValueError("Dataset não tem colunas Estado/Mes — verificar entrada.")
    df["Estado"] = df["Estado"].map(_norm_estado)

    chunks_extras = []
    total_adicionado = 0
    for alvo in alvos:
        est = alvo["estado"]
        mes = int(alvo["mes"])
        fator = float(alvo["fator_oversample_sugerido"])
        n_copias = max(1, int(round(fator - 1.0)))  # 5x => 4 cópias extras
        mask = (df["Estado"] == est) & (df["Mes"] == mes)
        subset = df.loc[mask]
        if subset.empty:
            continue
        logger.info(
            "  %s mês=%d  n=%d  fator=%.1fx  +%d cópias jittered",
            est, mes, len(subset), fator, n_copias,
        )
        for _ in range(n_copias):
            chunks_extras.append(_aplicar_jitter(subset, rng))
        total_adicionado += len(subset) * n_copias

    if not chunks_extras:
        logger.warning("Nenhum par-alvo casou com o dataset; retornando original.")
        return df

    extras = pd.concat(chunks_extras, ignore_index=True)
    logger.info("Total adicionado: %d linhas (de %d originais)", total_adicionado, len(df))
    aumentado = pd.concat([df, extras], ignore_index=True)
    # Shuffle para evitar ordenação artificial
    aumentado = aumentado.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    return aumentado


def main() -> int:
    if not ENRICHED_PATH.exists():
        raise FileNotFoundError(f"Dataset enriquecido não encontrado: {ENRICHED_PATH}")

    alvos = carregar_alvos()
    if not alvos:
        logger.info("Nenhum alvo OOD encontrado — nada a fazer.")
        return 0
    logger.info("%d pares-alvo carregados de %s", len(alvos), OOD_ALVOS_PATH)

    logger.info("Carregando %s ...", ENRICHED_PATH)
    df = pd.read_csv(ENRICHED_PATH, low_memory=False)
    logger.info("Dataset original: %d linhas, %d cols", len(df), len(df.columns))

    rng = np.random.default_rng(SEED)
    aumentado = oversample(df, alvos, rng)

    logger.info("Dataset aumentado: %d linhas (+%.1f%%)",
                len(aumentado), 100 * (len(aumentado) - len(df)) / len(df))
    aumentado.to_csv(OUT_PATH, index=False)
    logger.info("Salvo em: %s (%.1f MB)", OUT_PATH, OUT_PATH.stat().st_size / 1024 / 1024)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
