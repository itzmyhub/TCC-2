"""Auditoria operacional de previsões.

Cada chamada a `/api/predict` é registrada em um arquivo JSONL append-only
em `logs/auditoria_predicoes.jsonl`. Posteriormente o script
`scripts/auditar_predicoes.py` cruza essas entradas com detecções FIRMS
para calcular F1 operacional em janelas móveis (7/14/30 dias).

Design:
- JSONL append-only (uma predição por linha): tolerante a falhas, fácil de
  rotacionar e ler em streaming.
- Schema versionado (`schema_version`): permite migração futura.
- Rotação automática por tamanho (`MAX_LOG_BYTES`) — gera arquivos
  `auditoria_predicoes.jsonl.1`, `.2` etc., como `logrotate`.
- Captura `prediction_id` (uuid4) curto para correlacionar com logs do app.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Diretório de logs (mesmo lugar dos outros logs do app)
_LOGS_DIR = Path(__file__).resolve().parent.parent / "logs"
AUDIT_LOG_PATH = _LOGS_DIR / "auditoria_predicoes.jsonl"

# Rotação a cada 50 MB (suficiente para ~100k predições)
MAX_LOG_BYTES = 50 * 1024 * 1024
MAX_ROTATIONS = 5

SCHEMA_VERSION = 1

# Lock para serializar escrita concorrente (Flask multi-thread)
_write_lock = threading.Lock()


def _ensure_dir() -> None:
    _LOGS_DIR.mkdir(parents=True, exist_ok=True)


def _rotate_if_needed(path: Path) -> None:
    """Rotação simples: renomeia .jsonl -> .1, .1 -> .2, etc., descartando o último."""
    try:
        if not path.exists() or path.stat().st_size < MAX_LOG_BYTES:
            return
    except OSError:
        return
    try:
        for i in range(MAX_ROTATIONS, 0, -1):
            src = path.with_suffix(path.suffix + (f".{i-1}" if i > 1 else ""))
            dst = path.with_suffix(path.suffix + f".{i}")
            if i == MAX_ROTATIONS and dst.exists():
                dst.unlink(missing_ok=True)
            if src.exists():
                src.replace(dst)
        # Após rotação, abrirá novo arquivo no append
    except OSError as exc:
        logger.warning("Falha ao rotacionar audit log: %s", exc)


def log_prediction(
    *,
    lat: float,
    lon: float,
    estado: Optional[str],
    municipio: Optional[str],
    ref_data_iso: str,
    risco: str,
    confianca: Optional[float],
    probabilidades: Optional[Dict[str, float]],
    modelo: str,
    fonte_clima: Optional[str],
    fonte_clima_detalhe: Optional[Dict[str, Any]] = None,
    granularidade_tier1: Optional[str] = None,
    features_chave: Optional[Dict[str, float]] = None,
    thresholds_aplicados: Optional[Dict[str, Any]] = None,
    prediction_id: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> str:
    """Registra uma predição em JSONL append-only.

    Retorna o `prediction_id` (gerado se não fornecido) para o chamador
    devolver na resposta HTTP e correlacionar com auditoria futura.
    """
    pid = prediction_id or uuid.uuid4().hex[:12]
    entry: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "prediction_id": pid,
        "logged_at_utc": datetime.now(timezone.utc).isoformat(),
        "ref_data": ref_data_iso,
        "lat": round(float(lat), 6),
        "lon": round(float(lon), 6),
        "estado": estado,
        "municipio": municipio,
        "risco_predito": risco,
        "confianca": float(confianca) if confianca is not None else None,
        "probabilidades": (
            {str(k): float(v) for k, v in probabilidades.items()}
            if probabilidades else None
        ),
        "modelo": modelo,
        "fonte_clima": fonte_clima,
        "fonte_clima_detalhe": fonte_clima_detalhe,
        "granularidade_tier1": granularidade_tier1,
        "features_chave": (
            {k: float(v) for k, v in features_chave.items() if v is not None}
            if features_chave else None
        ),
        "thresholds_aplicados": thresholds_aplicados,
    }
    if extra:
        entry["extra"] = extra

    try:
        _ensure_dir()
        with _write_lock:
            _rotate_if_needed(AUDIT_LOG_PATH)
            with AUDIT_LOG_PATH.open("a", encoding="utf-8") as f:
                json.dump(entry, f, ensure_ascii=False, default=str)
                f.write("\n")
                f.flush()
                try:
                    os.fsync(f.fileno())
                except (OSError, ValueError):
                    pass
    except Exception as exc:
        logger.warning("Falha ao gravar auditoria (predição segue normalmente): %s", exc)

    return pid


def iter_audit_entries(path: Optional[Path] = None):
    """Iterador sobre as entradas do log (útil para auditoria offline)."""
    p = path or AUDIT_LOG_PATH
    if not p.exists():
        return
    with p.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                logger.warning("Linha %d do audit log inválida: %s", line_no, exc)
                continue
