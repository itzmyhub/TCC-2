"""Gera figura comparativa "Antes vs Depois" das três grandes iterações de melhoria
de modelo (sem Tier 1 → Tier 1 → Tier 1 + Optuna XGB/LGBM + Meta-learner GBM).

Saída: modelos/relatorios/evolucao_modelos.png e .json (dados brutos).

Cada figura mostra três barras agrupadas por iteração, uma cor por métrica
(acurácia, F1-macro, F1-Moderado). Útil para o capítulo de resultados do TCC
(citável como Figura X em §4.10).
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
REPORT_DIR = BASE_DIR.parent / 'modelos' / 'relatorios'


def main() -> None:
    iteracoes = [
        {
            'rotulo': 'Stacking\n(pré-Tier 1)',
            'sub': '04/2026',
            'accuracy': 80.49,
            'f1_macro': 74.92,
            'f1_moderado': 54.98,
        },
        {
            'rotulo': 'Stacking\nTier 1',
            'sub': '10/05/2026',
            'accuracy': 83.60,
            'f1_macro': 78.92,
            'f1_moderado': 61.68,
        },
        {
            'rotulo': 'Stacking GBM\nTier 1 + Optuna',
            'sub': '12/05/2026',
            'accuracy': 84.61,
            'f1_macro': 79.95,
            'f1_moderado': 62.96,
        },
        {
            'rotulo': 'Stacking GBM\n+ thresholds',
            'sub': '12/05/2026',
            'accuracy': 83.88,
            'f1_macro': 79.81,
            'f1_moderado': 63.75,
        },
    ]
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as exc:  # pragma: no cover
        logger.error("matplotlib indisponível: %s", exc)
        sys.exit(1)

    metricas = [
        ('accuracy', 'Acurácia (%)', '#1f77b4'),
        ('f1_macro', 'F₁-macro (%)', '#ff7f0e'),
        ('f1_moderado', 'F₁-Moderado (%)', '#2ca02c'),
    ]
    labels = [f"{it['rotulo']}\n{it['sub']}" for it in iteracoes]
    x = np.arange(len(iteracoes))
    largura = 0.26

    fig, ax = plt.subplots(figsize=(11, 6))
    for i, (chave, rotulo, cor) in enumerate(metricas):
        valores = [it[chave] for it in iteracoes]
        offset = (i - 1) * largura
        bars = ax.bar(x + offset, valores, largura, label=rotulo, color=cor, edgecolor='white')
        for bar, val in zip(bars, valores):
            ax.annotate(
                f'{val:.2f}',
                xy=(bar.get_x() + bar.get_width() / 2, val),
                xytext=(0, 3),
                textcoords='offset points',
                ha='center',
                va='bottom',
                fontsize=9,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel('Pontuação (%)', fontsize=11)
    ax.set_title(
        'Evolução das métricas do TCC — três iterações de melhoria + thresholds\n'
        'Hold-out estratificado (n=184 862), base_de_dados_enriquecido.csv',
        fontsize=12,
    )
    ax.set_ylim(45, 92)
    ax.grid(axis='y', linestyle=':', alpha=0.5)
    ax.legend(loc='upper left', fontsize=10)

    # Anotação total
    ganho_acc = iteracoes[-2]['accuracy'] - iteracoes[0]['accuracy']
    ganho_f1mod = iteracoes[-1]['f1_moderado'] - iteracoes[0]['f1_moderado']
    ax.text(
        len(iteracoes) - 0.5,
        46,
        f'Δ acumulado: +{ganho_acc:.2f} pp acurácia · +{ganho_f1mod:.2f} pp F₁-Moderado (vs Stacking pré-Tier 1)',
        ha='right',
        va='bottom',
        fontsize=10,
        style='italic',
        color='#444',
    )
    plt.tight_layout()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out_png = REPORT_DIR / 'evolucao_modelos.png'
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Figura salva em %s", out_png)

    out_json = REPORT_DIR / 'evolucao_modelos.json'
    with out_json.open('w', encoding='utf-8') as fp:
        json.dump(iteracoes, fp, ensure_ascii=False, indent=2)
    logger.info("Dados salvos em %s", out_json)


if __name__ == '__main__':
    main()
