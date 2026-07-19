"""
explainer.py — Explicabilidade local rápida para o app interativo.

Objetivo
--------
Para um ponto consultado no mapa, identificar quais features mais
contribuíram para a previsão de risco, em linguagem amigável ao
usuário leigo (e tecnicamente rigorosa o suficiente para discussão no TCC).

Por que não usar `shap` em tempo real?
--------------------------------------
- TreeExplainer do SHAP exige acesso direto ao classificador final
  (não-pipeline). No app, o modelo é um Pipeline (preprocessor + classifier),
  às vezes empacotado em CalibratedClassifierCV. Extrair o estimador base
  consistentemente é frágil.
- SHAP-em-tempo-real adiciona ~200-800 ms por consulta — pesado para UI.
- O usuário-alvo do TCC (gestor ambiental) não precisa de SHAP value exato;
  precisa do "ranking + sentido" das features (alto/baixo, normal/anômalo).

Estratégia adotada
------------------
Calculamos uma "contribuição aproximada" baseada em:

    contrib_i = importance_i × z_i × sinal_i

onde:
- `importance_i` é a **importância global** da feature (vinda do JSON
  `modelos/relatorios/shap_feature_importance.json`, gerado offline via
  Gini/Permutation importance).
- `z_i` é a **anormalidade** do valor da feature no ponto (z-score
  normalizado clipado em [-3, 3]).
- `sinal_i` é o "sentido" físico esperado (+1 se feature aumenta risco,
  -1 se reduz). Definido manualmente por feature com base em literatura
  e na natureza física do problema.

Resultado: ordenação consistente das Top-N features mais influentes
para o ponto, com explicação curta e legível.
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sinal físico esperado por feature (+1 aumenta risco, -1 reduz)
# Baseado em literatura: Seager 2015, Forests 2024, npj Natural Hazards 2025.
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Importâncias-padrão para features Tier 1 (usadas se não existir entrada no
# JSON SHAP). Necessário porque o SHAP global atual foi calculado antes das
# features Tier 1 entrarem no pipeline. Os valores foram calibrados para
# serem comparáveis a `DiaSemChuva` (≈0.10) — features de seca dominantes — e
# escalonados conforme a literatura (Forests 2024: KBDI > SPI > VPD > MAs
# longas > acumulados longos).
# ---------------------------------------------------------------------------
DEFAULT_IMPORTANCE_TIER1: Dict[str, float] = {
    # Camada 1 — Médias móveis estendidas
    "Precipitacao_ma14": 0.045,
    "Precipitacao_ma30": 0.040,
    "Precipitacao_ma90": 0.035,
    "DiaSemChuva_ma14": 0.060,
    "DiaSemChuva_ma30": 0.055,
    "DiaSemChuva_ma90": 0.050,
    # Camada 2 — Acumulados
    "Precipitacao_acum_30d": 0.040,
    "Precipitacao_acum_90d": 0.035,
    "Precipitacao_acum_180d": 0.025,
    "Precipitacao_acum_365d": 0.020,
    # Camada 3 — SPI (literatura: forte preditor de área queimada)
    "SPI_1m": 0.055,
    "SPI_3m": 0.060,
    "SPI_6m": 0.045,
    # Camada 4 — Anomalia
    "Anomalia_Precipitacao": 0.050,
    "Anomalia_Precipitacao_rel": 0.045,
    # Camada 5 — KBDI / De Martonne / VPD (Forests 2024: KBDI top performer)
    "Temp_Climatologica": 0.025,
    "KBDI_proxy": 0.075,
    "Aridez_DeMartonne": 0.030,
    "VPD_proxy": 0.060,
    # Camada 6 — Histórico estendido
    "Incendios_Ultimos_90_Dias": 0.030,
    "Incendios_Ultimos_180_Dias": 0.025,
    "Incendios_Ultimos_365_Dias": 0.020,
    "Media_FRP_Celula_30d": 0.020,
    # Camada 7 — Estação seca acumulada
    "Dias_Secos_90d": 0.050,
}


SINAL_FISICO: Dict[str, int] = {
    # Secura e déficit hídrico → AUMENTAM risco
    "DiaSemChuva": +1,
    "Indice_Seca": +1,
    "DiaSemChuva_ma7": +1,
    "DiaSemChuva_ma14": +1,
    "DiaSemChuva_ma30": +1,
    "DiaSemChuva_ma90": +1,
    "KBDI_proxy": +1,
    "VPD_proxy": +1,
    "Temp_Climatologica": +1,
    "Dias_Secos_90d": +1,
    "Periodo_Critico": +1,
    # Precipitação alta / acumulado / aridez baixa → REDUZEM risco
    "Precipitacao": -1,
    "Precipitacao_ma7": -1,
    "Precipitacao_ma14": -1,
    "Precipitacao_ma30": -1,
    "Precipitacao_ma90": -1,
    "Precipitacao_acum_30d": -1,
    "Precipitacao_acum_90d": -1,
    "Precipitacao_acum_180d": -1,
    "Precipitacao_acum_365d": -1,
    "SPI_1m": -1,  # SPI positivo = chuvoso = MENOS risco
    "SPI_3m": -1,
    "SPI_6m": -1,
    "Anomalia_Precipitacao": -1,
    "Anomalia_Precipitacao_rel": -1,
    "Aridez_DeMartonne": -1,
    # Histórico de fogo → AUMENTA risco (mais focos = região seca/desmatada)
    "Incendios_Ultimos_7_Dias": +1,
    "Incendios_Ultimos_30_Dias": +1,
    "Incendios_Ultimos_90_Dias": +1,
    "Incendios_Ultimos_180_Dias": +1,
    "Incendios_Ultimos_365_Dias": +1,
    "Media_FRP_Ultimos_7_Dias": +1,
    "Max_FRP_Ultimos_7_Dias": +1,
    "Media_FRP_Celula_30d": +1,
    "FRP": +1,
    # Tempo desde último incêndio → MAIOR = menos risco recente
    "Dias_Desde_Ultimo_Incendio": -1,
}

# ---------------------------------------------------------------------------
# Explicações em linguagem natural por feature
# ---------------------------------------------------------------------------
DESCRICAO_FEATURE: Dict[str, str] = {
    "DiaSemChuva": "Dias consecutivos sem chuva",
    "Indice_Seca": "Índice de seca (DiaSemChuva / (Precipitação + 0.1))",
    "DiaSemChuva_ma7": "Média de dias sem chuva (7 dias)",
    "DiaSemChuva_ma14": "Média de dias sem chuva (14 dias)",
    "DiaSemChuva_ma30": "Média de dias sem chuva (30 dias)",
    "DiaSemChuva_ma90": "Média de dias sem chuva (90 dias)",
    "Precipitacao": "Precipitação observada (mm)",
    "Precipitacao_ma7": "Precipitação média (7 dias, mm)",
    "Precipitacao_ma14": "Precipitação média (14 dias, mm)",
    "Precipitacao_ma30": "Precipitação média (30 dias, mm)",
    "Precipitacao_ma90": "Precipitação média (90 dias, mm)",
    "Precipitacao_acum_30d": "Precipitação acumulada nos últimos 30 dias",
    "Precipitacao_acum_90d": "Precipitação acumulada nos últimos 90 dias",
    "Precipitacao_acum_180d": "Precipitação acumulada nos últimos 180 dias",
    "Precipitacao_acum_365d": "Precipitação acumulada no último ano",
    "SPI_1m": "SPI-1 mês (índice de chuva padronizado)",
    "SPI_3m": "SPI-3 meses (seca de médio prazo)",
    "SPI_6m": "SPI-6 meses (seca de longo prazo)",
    "Anomalia_Precipitacao": "Chuva atual vs. média do município no mês (mm)",
    "Anomalia_Precipitacao_rel": "Desvio relativo da chuva vs. média",
    "Temp_Climatologica": "Temperatura média mensal (climatologia do estado)",
    "KBDI_proxy": "Proxy do KBDI (déficit hídrico × temperatura)",
    "VPD_proxy": "Proxy do déficit de pressão de vapor (T² × secura)",
    "Aridez_DeMartonne": "Índice de aridez de De Martonne",
    "Incendios_Ultimos_7_Dias": "Focos detectados na região (últimos 7 dias)",
    "Incendios_Ultimos_30_Dias": "Focos detectados na região (últimos 30 dias)",
    "Incendios_Ultimos_90_Dias": "Focos na célula (últimos 90 dias)",
    "Incendios_Ultimos_180_Dias": "Focos na célula (últimos 180 dias)",
    "Incendios_Ultimos_365_Dias": "Focos na célula (último ano)",
    "Dias_Desde_Ultimo_Incendio": "Dias desde o último incêndio na região",
    "Media_FRP_Ultimos_7_Dias": "FRP médio (potência radiativa) últimos 7 dias",
    "Max_FRP_Ultimos_7_Dias": "FRP máximo nos últimos 7 dias",
    "Media_FRP_Celula_30d": "FRP médio na célula (últimos 30 dias)",
    "Dias_Secos_90d": "Dias secos (P<5mm) nos últimos 90 dias",
    "Periodo_Critico": "Está no período crítico (Jul-Set)?",
    "FRP": "Potência radiativa de fogo (ativo agora?)",
    "Mes_sin": "Sazonalidade (componente seno)",
    "Mes_cos": "Sazonalidade (componente cosseno)",
    "Latitude": "Latitude geográfica",
    "Longitude": "Longitude geográfica",
    "Ano": "Ano da consulta",
    "Mes": "Mês da consulta",
    "Dia": "Dia da consulta",
    "Hora": "Hora da consulta",
}


class _Explainer:
    """Encapsula a importância global pré-computada das features.

    Sempre carrega ``shap_feature_importance.json`` (importância global agregada).
    Se ``shap_per_class.json`` existir na mesma pasta, também carrega a
    importância **por classe** — usada preferencialmente quando o risco predito
    está disponível, garantindo explicações classe-específicas (ref. Lundberg
    et al., Nat. Mach. Intell. 2020).
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._loaded = False
        self._importance: Dict[str, float] = {}  # global agregado
        self._importance_per_class: Dict[str, Dict[str, float]] = {}  # cls → {feat: imp}
        self._stats: Dict[str, Dict[str, float]] = {}  # mean, std por feature
        self._json_path: Optional[Path] = None
        self._has_per_class: bool = False

    def _ensure_loaded(
        self,
        importance_path: Path,
        dataset_path: Path,
    ) -> None:
        with self._lock:
            if self._loaded and self._json_path == importance_path:
                return
            self._load(importance_path, dataset_path)
            self._loaded = True
            self._json_path = importance_path

    @staticmethod
    def _limpar_nome(raw_name: str) -> Tuple[str, bool]:
        """Remove prefixos `num__` / `cat__`. Retorna (nome_limpo, eh_categorica)."""
        for pfx in ("num__", "cat__"):
            if raw_name.startswith(pfx):
                return raw_name[len(pfx):], pfx == "cat__"
        return raw_name, False

    def _acumular_importance(
        self,
        target: Dict[str, float],
        raw_name: str,
        val: float,
    ) -> None:
        nome_limpo, eh_cat = self._limpar_nome(raw_name)
        if eh_cat and "_" in nome_limpo:
            # cat__Estado_AMAZONAS → "Estado" (agrega ao longo das categorias)
            base = nome_limpo.split("_", 1)[0]
            target[base] = target.get(base, 0.0) + val
        else:
            if val > target.get(nome_limpo, 0.0):
                target[nome_limpo] = val

    def _load(self, importance_path: Path, dataset_path: Path) -> None:
        if importance_path.exists():
            try:
                with importance_path.open("r", encoding="utf-8") as fp:
                    data = json.load(fp)
                for item in data.get("feature_importance", []):
                    raw_name = item.get("feature", "")
                    val = float(item.get("mean_abs_shap", 0.0) or 0.0)
                    self._acumular_importance(self._importance, raw_name, val)
                logger.info(
                    "Importance global carregada: %d features.",
                    len(self._importance),
                )
            except Exception as exc:
                logger.warning("Falha ao ler %s: %s", importance_path, exc)

        # Preenche importâncias-padrão para Tier 1 que faltarem (modelo
        # antigo de SHAP não tinha essas colunas). Não sobrescreve valores
        # já carregados do JSON.
        for k, v in DEFAULT_IMPORTANCE_TIER1.items():
            self._importance.setdefault(k, v)

        # ----- SHAP por classe (preferencial quando disponível) -----------
        per_class_path = importance_path.parent / "shap_per_class.json"
        if per_class_path.exists():
            try:
                with per_class_path.open("r", encoding="utf-8") as fp:
                    pcdata = json.load(fp)
                for cls, items in (pcdata.get("shap_per_class") or {}).items():
                    bucket: Dict[str, float] = {}
                    for it in items:
                        raw_name = it.get("feature", "")
                        val = float(it.get("mean_abs_shap", 0.0) or 0.0)
                        self._acumular_importance(bucket, raw_name, val)
                    # Defaults Tier 1 (usados apenas se ausentes neste class)
                    for k, v in DEFAULT_IMPORTANCE_TIER1.items():
                        bucket.setdefault(k, v)
                    self._importance_per_class[str(cls)] = bucket
                self._has_per_class = bool(self._importance_per_class)
                if self._has_per_class:
                    logger.info(
                        "SHAP por classe carregado: %d classes (%s).",
                        len(self._importance_per_class),
                        ", ".join(sorted(self._importance_per_class.keys())),
                    )
            except Exception as exc:
                logger.warning("Falha ao ler %s: %s", per_class_path, exc)
        # ------------------------------------------------------------------

        # Estatísticas (mean/std) para z-score — usa subamostra do dataset
        if not dataset_path.exists():
            return
        try:
            # Carrega 50k linhas para estatística (rápido e representativo)
            df = pd.read_csv(dataset_path, nrows=50_000, low_memory=False)
            todas_features = set(self._importance.keys()) | set(SINAL_FISICO.keys())
            for bucket in self._importance_per_class.values():
                todas_features.update(bucket.keys())
            for c in todas_features:
                if c not in df.columns:
                    continue
                serie = pd.to_numeric(df[c], errors="coerce").dropna()
                if len(serie) < 100:
                    continue
                self._stats[c] = {
                    "mean": float(serie.mean()),
                    "std": float(serie.std() or 1.0),
                    "p10": float(serie.quantile(0.10)),
                    "p50": float(serie.quantile(0.50)),
                    "p90": float(serie.quantile(0.90)),
                }
            logger.info("Estatísticas por feature prontas: %d features.", len(self._stats))
        except Exception as exc:
            logger.warning("Falha ao calcular estatísticas: %s", exc)

    def _importance_para(self, feat: str, risco_predito: Optional[str]) -> float:
        """Devolve a importância da feature: prefere SHAP class-specific quando
        ``risco_predito`` mapeia a uma classe carregada, com fallback no global.
        """
        if (
            risco_predito
            and self._has_per_class
            and risco_predito in self._importance_per_class
        ):
            return float(self._importance_per_class[risco_predito].get(feat, 0.0))
        return float(self._importance.get(feat, 0.0))

    # -------------------------------------------------------------------
    def explicar(
        self,
        valores_features: Dict[str, float],
        risco_predito: str,
        top_n: int = 6,
    ) -> List[Dict[str, Any]]:
        """Retorna lista ordenada de contribuições por feature.

        Cada item: {feature, descricao, valor, p50, p10, p90, importance,
                    z_score, sinal_fisico, contribuicao, sentido_texto,
                    direcao_risco}
        """
        if not self._loaded:
            return []

        contribs: List[Dict[str, Any]] = []
        usou_per_class = bool(
            risco_predito
            and self._has_per_class
            and risco_predito in self._importance_per_class
        )
        for feat, valor in valores_features.items():
            if valor is None:
                continue
            imp = self._importance_para(feat, risco_predito)
            if imp <= 0:
                continue
            stats = self._stats.get(feat)
            if stats:
                std = stats["std"] or 1.0
                z = float((valor - stats["mean"]) / std)
                z = float(np.clip(z, -3.0, 3.0))
            else:
                z = 0.0
            sinal = SINAL_FISICO.get(feat, 0)
            contrib = float(imp * z * sinal)

            sentido = "neutro"
            direcao = "neutra"
            if sinal != 0 and abs(z) > 0.2:
                if contrib > 0:
                    sentido = "favorece risco maior"
                    direcao = "aumenta"
                else:
                    sentido = "favorece risco menor"
                    direcao = "reduz"

            contribs.append({
                "feature": feat,
                "descricao": DESCRICAO_FEATURE.get(feat, feat),
                "valor": float(valor),
                "p10": stats.get("p10") if stats else None,
                "p50": stats.get("p50") if stats else None,
                "p90": stats.get("p90") if stats else None,
                "importance": imp,
                "z_score": z,
                "sinal_fisico": sinal,
                "contribuicao": contrib,
                "sentido_texto": sentido,
                "direcao_risco": direcao,
                "fonte_importance": "shap_per_class" if usou_per_class else "shap_global",
            })

        # Ordena por |contribuição| decrescente
        contribs.sort(key=lambda d: abs(d["contribuicao"]), reverse=True)
        return contribs[:top_n]

    def metadata(self) -> Dict[str, Any]:
        """Para o app expor ao usuário/dev qual fonte de importância foi usada."""
        return {
            "loaded": self._loaded,
            "n_features_global": len(self._importance),
            "tem_shap_per_class": self._has_per_class,
            "classes_per_class": sorted(self._importance_per_class.keys()),
        }


_EXPLAINER = _Explainer()


def explicar_predicao(
    valores_features: Dict[str, float],
    risco_predito: str,
    importance_path: Path,
    dataset_path: Path,
    top_n: int = 6,
) -> List[Dict[str, Any]]:
    """API pública. Inicializa o explainer no primeiro uso."""
    _EXPLAINER._ensure_loaded(importance_path, dataset_path)
    return _EXPLAINER.explicar(valores_features, risco_predito, top_n=top_n)


def metadata_explainer(
    importance_path: Path,
    dataset_path: Path,
) -> Dict[str, Any]:
    _EXPLAINER._ensure_loaded(importance_path, dataset_path)
    return _EXPLAINER.metadata()


__all__ = [
    "explicar_predicao",
    "metadata_explainer",
    "DESCRICAO_FEATURE",
    "SINAL_FISICO",
]
