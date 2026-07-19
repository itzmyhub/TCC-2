"""Indicadores leves de incerteza para a saída operacional (sem retreinar o modelo).

Inspirado em práticas de *uncertainty quantification* em Earth observation: quando
as duas classes mais prováveis estão próximas, o decisor deve tratar o rótulo
como menos confiável — análogo a *prediction sets* reduzidos a 2 classes sem
calibração conformal explícita (evita dependência de conjunto de calibração).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple


def multiclass_operational_uncertainty(
    probabilidades: Optional[Dict[str, float]],
    *,
    gap_alerta: float = 0.12,
) -> Dict[str, Any]:
    """A partir das probabilidades calibradas, resume ambiguidade top-2.

    Args:
        probabilidades: mapa classe → prob (ex.: Baixo, Moderado, Muito Alto).
        gap_alerta: se p1−p2 < este valor, marca ``incerteza_alta``.

    Returns:
        dict com gap_top2, classes_ordenadas, set_sugerido (1 ou 2 classes),
        incerteza_alta, nota explicativa curta.
    """
    if not probabilidades:
        return {
            "gap_top2": None,
            "classes_ordenadas": [],
            "set_sugerido": [],
            "incerteza_alta": None,
            "nota": "Probabilidades indisponíveis.",
        }

    items: List[Tuple[str, float]] = sorted(
            probabilidades.items(), key=lambda kv: (-float(kv[1]), kv[0]))
    if len(items) == 1:
        c0, p0 = items[0]
        return {
            "gap_top2": None,
            "classes_ordenadas": [c0],
            "set_sugerido": [c0],
            "incerteza_alta": False,
            "nota": "Apenas uma classe com massa de probabilidade.",
        }

    (c1, p1), (c2, p2) = items[0], items[1]
    gap = float(p1) - float(p2)
    incerteza = gap < gap_alerta
    set_sug = [c1, c2] if incerteza else [c1]

    nota = (
        f"Gap entre 1ª e 2ª classe = {gap:.3f}. "
        + (
            "Incerteza alta: considere o conjunto de classes sugerido antes de "
            "decisões binárias de despacho."
            if incerteza
            else "Classe dominante relativamente clara."
        )
    )

    return {
        "gap_top2": round(gap, 4),
        "classes_ordenadas": [c for c, _ in items],
        "set_sugerido": set_sug,
        "incerteza_alta": incerteza,
        "nota": nota,
    }


def fire_weather_proxy_heuristic(
    *,
    dia_sem_chuva_ma7: float,
    umidade_pct: Optional[float],
    vento_ms_ma7: Optional[float],
) -> float:
    """Proxy escalar 0–∞ combinando seca média na semana, aridez (1−RH) e vento.

    Não substitui FWI canadense; serve como indicador interpretável na UI/TCC,
    alinhado à literatura que combina umidade, sequência seca e vento para
    *fire weather*.
    """
    rh = float(umidade_pct) if umidade_pct is not None else 55.0
    rh = min(100.0, max(0.0, rh))
    ws = float(vento_ms_ma7) if vento_ms_ma7 is not None else 0.0
    ws = min(35.0, max(0.0, ws))
    seca = float(dia_sem_chuva_ma7) / 7.0
    aridez = (100.0 - rh) / 100.0 + 0.05
    vento_f = 1.0 + ws / 12.0
    return round(max(0.0, seca * aridez * vento_f), 4)
