# -*- coding: utf-8 -*-
"""Gera fragmentos .tex coláveis em tcc_tex/conteudo_colar/ a partir de tcc.tex + 07."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "tcc_tex" / "tcc.tex"
OUT = ROOT / "tcc_tex" / "conteudo_colar"
ESTUDO = ROOT / "tcc_tex" / "07_estudo_caso_pium_to.tex"


def clean_chunk(s: str) -> str:
    s = s.replace("\\ce{CO2}", "CO$_{2}$")
    s = s.replace("\\ce{CO_2}", "CO$_{2}$")
    return s


def slice_lines(lines: list[str], start: int, end: int) -> str:
    """1-based inclusive line numbers."""
    return clean_chunk("".join(lines[start - 1 : end]))


def main() -> None:
    text = SRC.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)

    OUT.mkdir(parents=True, exist_ok=True)

    chunks = [
        ("Z00_resumo_abstract.tex", 144, 158),
        ("cap01_introducao.tex", 175, 228),
        # cap02 termina antes de \chapter{Fundamentação Teórica}; cap03 inclui esse \chapter.
        ("cap02_estado_da_arte.tex", 229, 261),
        ("cap03_fundamentacao.tex", 262, 608),
        ("cap04_metodologia.tex", 609, 989),
        ("cap05_resultados.tex", 990, 1380),
        ("cap06_conclusao.tex", 1384, 1446),
    ]

    for name, a, b in chunks:
        (OUT / name).write_text(slice_lines(lines, a, b), encoding="utf-8")

    if ESTUDO.exists():
        (OUT / "cap05_estudo_caso_pium_to.tex").write_text(
            clean_chunk(ESTUDO.read_text(encoding="utf-8")), encoding="utf-8"
        )
    else:
        raise SystemExit(f"Missing {ESTUDO}")

    print("Written to", OUT)


if __name__ == "__main__":
    main()
