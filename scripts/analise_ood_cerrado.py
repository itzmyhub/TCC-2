"""Análise da distribuição (Estado, Mês, Risco) no dataset de treino.

Objetivo: identificar pares (Estado, Mês) onde:
1. Há **focos reais documentados** (FRP > 0 ou classe Muito Alto), MAS
2. A representatividade no dataset é baixa,
3. E o modelo tende a errar (vide caso Pium-TO em maio).

A saída orienta o oversampling (scripts/oversample_ood_cerrado.py).

Saída:
- `modelos/relatorios/ood_distribuicao_estado_mes.json`: estatísticas
- `modelos/relatorios/ood_pares_alvo.json`: pares (Estado, Mês) recomendados
  para oversampling, com fator sugerido.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("analise_ood")

ROOT = Path(__file__).resolve().parent.parent
ENRICHED_PATH = ROOT / "base_de_dados_enriquecido.csv"
RELATORIOS_DIR = ROOT / "modelos" / "relatorios"
RELATORIOS_DIR.mkdir(parents=True, exist_ok=True)

DIST_JSON = RELATORIOS_DIR / "ood_distribuicao_estado_mes.json"
ALVO_JSON = RELATORIOS_DIR / "ood_pares_alvo.json"

# Estados de transição Cerrado-Amazônia (alvos prioritários)
ESTADOS_TRANSICAO = {"TO", "MA", "MT", "PA", "BA", "PI", "DF", "GO"}
TRANSICAO_TO_FULL = {
    "TO": "TOCANTINS",
    "MA": "MARANHÃO",
    "MT": "MATO GROSSO",
    "PA": "PARÁ",
    "BA": "BAHIA",
    "PI": "PIAUÍ",
    "DF": "DISTRITO FEDERAL",
    "GO": "GOIÁS",
}

# Meses de transição onde o modelo é provavelmente sub-treinado
MESES_TRANSICAO = (5, 6, 7)


def _norm_state(s) -> str:
    if not isinstance(s, str):
        return ""
    return s.strip().upper()


def carregar_dataset() -> pd.DataFrame:
    if not ENRICHED_PATH.exists():
        raise FileNotFoundError(f"Dataset enriquecido não encontrado: {ENRICHED_PATH}")
    logger.info("Carregando %s ...", ENRICHED_PATH)
    # Lê só as colunas necessárias para economia de memória
    cols_alvo = ["Estado", "Mes", "FRP", "Precipitacao_ma7", "DiaSemChuva_ma7",
                 "KBDI_proxy", "Risco_Incendio"]
    df0 = pd.read_csv(ENRICHED_PATH, nrows=0)
    cols_existentes = [c for c in cols_alvo if c in df0.columns]
    if "Risco_Incendio" not in cols_existentes:
        # Pode estar em outra coluna; tenta detectar
        for cand in ["Risco", "Classe", "y", "target"]:
            if cand in df0.columns:
                cols_existentes.append(cand)
                break
    df = pd.read_csv(ENRICHED_PATH, usecols=cols_existentes, low_memory=False)
    df["Estado"] = df["Estado"].apply(_norm_state)
    df["Mes"] = pd.to_numeric(df["Mes"], errors="coerce").fillna(0).astype(int)
    if "Risco_Incendio" not in df.columns:
        # se veio com outro nome, renomeia para "Risco_Incendio"
        for cand in ["Risco", "Classe", "y", "target"]:
            if cand in df.columns:
                df = df.rename(columns={cand: "Risco_Incendio"})
                break
    df["FRP"] = pd.to_numeric(df["FRP"], errors="coerce").fillna(0.0)
    logger.info("Dataset: %d linhas, %d cols", len(df), len(df.columns))
    return df


def analisar(df: pd.DataFrame) -> Dict:
    """Calcula distribuição por (Estado, Mês, Risco) + total por par."""
    total = len(df)
    df_focos = df[df["FRP"] > 0]
    logger.info("Total focos com FRP > 0: %d (%.1f%%)", len(df_focos), 100 * len(df_focos) / total)

    # Distribuição global por par (Estado, Mês)
    pares = df.groupby(["Estado", "Mes"]).size().reset_index(name="n_total")
    focos_por_par = df_focos.groupby(["Estado", "Mes"]).size().reset_index(name="n_focos")
    pares = pares.merge(focos_por_par, on=["Estado", "Mes"], how="left")
    pares["n_focos"] = pares["n_focos"].fillna(0).astype(int)

    # Distribuição por Risco
    if "Risco_Incendio" in df.columns:
        risco_dist = df.groupby(["Estado", "Mes", "Risco_Incendio"]).size().unstack(fill_value=0)
        risco_dist = risco_dist.reset_index()
        pares = pares.merge(risco_dist, on=["Estado", "Mes"], how="left")

    # Frequência relativa de focos por estado (média sobre meses)
    media_focos_por_estado = pares.groupby("Estado")["n_focos"].mean().to_dict()
    pares["media_focos_estado"] = pares["Estado"].map(media_focos_por_estado)
    pares["razao_vs_media_estado"] = pares["n_focos"] / pares["media_focos_estado"].replace(0, 1)

    return {
        "n_total": int(total),
        "n_focos_globais": int(len(df_focos)),
        "pares": pares,
    }


N_MIN_PARA_ALVO = 30   # n_total mínimo para considerar oversampling
RAZAO_MAX_PARA_ALVO = 0.7  # razão de focos vs média estadual abaixo da qual marca alvo


def identificar_alvos(pares: pd.DataFrame) -> List[Dict]:
    """Identifica pares (Estado, Mês) candidatos a oversampling.

    Critério (calibrado para o caso Pium-TO e similares):
      - Estado em ESTADOS_TRANSICAO (Cerrado-Amazônia)
      - Mês em MESES_TRANSICAO (maio-julho, transição chuvoso-seco)
      - Razão de focos vs média do estado < 0.7 (sub-representado)
      - n_total >= 30 (evita pares com dados muito escassos)
    """
    estados_alvo_set = {TRANSICAO_TO_FULL[s] for s in ESTADOS_TRANSICAO}
    alvos = []
    for _, row in pares.iterrows():
        if row["Estado"] not in estados_alvo_set:
            continue
        if row["Mes"] not in MESES_TRANSICAO:
            continue
        if row.get("n_total", 0) < N_MIN_PARA_ALVO:
            continue
        razao = float(row.get("razao_vs_media_estado", 0))
        if razao >= RAZAO_MAX_PARA_ALVO:
            continue
        # Fator de oversampling sugerido (inverso da razão, limitado a 5x)
        fator = min(5.0, max(1.5, 1.0 / max(razao, 0.1)))
        muito_alto = int(row.get("Muito Alto", 0)) if "Muito Alto" in row else 0
        moderado = int(row.get("Moderado", 0)) if "Moderado" in row else 0
        baixo = int(row.get("Baixo", 0)) if "Baixo" in row else 0
        alvos.append({
            "estado": row["Estado"],
            "mes": int(row["Mes"]),
            "n_total": int(row.get("n_total", 0)),
            "n_focos": int(row.get("n_focos", 0)),
            "razao_vs_media_estado": round(razao, 3),
            "fator_oversample_sugerido": round(fator, 2),
            "dist_classes": {
                "Baixo": baixo,
                "Moderado": moderado,
                "Muito Alto": muito_alto,
            },
        })
    # Ordenar do mais carente para o menos
    alvos.sort(key=lambda x: x["razao_vs_media_estado"])
    return alvos


def main() -> int:
    df = carregar_dataset()
    analise = analisar(df)
    pares = analise["pares"]

    # Salvar distribuição completa
    pares_para_salvar = pares.copy()
    # Converter colunas categóricas (Baixo/Moderado/Muito Alto) para int se existirem
    for c in ("Baixo", "Moderado", "Muito Alto"):
        if c in pares_para_salvar.columns:
            pares_para_salvar[c] = pares_para_salvar[c].fillna(0).astype(int)
    DIST_JSON.write_text(
        json.dumps({
            "n_total": analise["n_total"],
            "n_focos_globais": analise["n_focos_globais"],
            "pares_total": len(pares_para_salvar),
            "pares": pares_para_salvar.to_dict(orient="records"),
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("Distribuição salva em: %s", DIST_JSON)

    # Identificar alvos
    alvos = identificar_alvos(pares)
    ALVO_JSON.write_text(
        json.dumps({
            "criterios": {
                "estados_transicao": sorted(TRANSICAO_TO_FULL.values()),
                "meses_transicao": list(MESES_TRANSICAO),
                "razao_max_para_alvo": RAZAO_MAX_PARA_ALVO,
                "n_min_para_alvo": N_MIN_PARA_ALVO,
                "fator_oversample_min": 1.5,
                "fator_oversample_max": 5.0,
            },
            "n_alvos": len(alvos),
            "pares_alvo": alvos,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("Alvos para oversampling: %d", len(alvos))
    logger.info("Relatório de alvos: %s", ALVO_JSON)

    if alvos:
        logger.info("Top 10 pares com maior fator sugerido:")
        for a in sorted(alvos, key=lambda x: -x["fator_oversample_sugerido"])[:10]:
            logger.info(
                "  %s mês=%d  n=%d  focos=%d  razão=%.2f  fator=%.1fx",
                a["estado"], a["mes"], a["n_total"], a["n_focos"],
                a["razao_vs_media_estado"], a["fator_oversample_sugerido"],
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
