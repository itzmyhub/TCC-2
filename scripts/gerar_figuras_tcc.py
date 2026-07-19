"""Gerador de figuras de alta resolução para o TCC.

Produz:
    - modelos/relatorios/matriz_confusao_stacking_gbm.png
    - modelos/relatorios/shap_per_classe_barras.png
    - modelos/relatorios/permutation_importance_por_classe.png
    - modelos/relatorios/validacao_temporal_comparativo.png
    - modelos/relatorios/distribuicao_classes.png

Uso:
    python scripts/gerar_figuras_tcc.py

Notas técnicas:
    - DPI = 300 para impressão.
    - Estilo "seaborn-v0_8-whitegrid" + paleta acessível (deuteranopia-friendly).
    - Acentuação preservada (UTF-8), fontes Helvetica/Arial.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RELATORIOS = ROOT / "modelos" / "relatorios"
DPI = 300

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.dpi": 150,
        "savefig.dpi": DPI,
        "savefig.bbox": "tight",
        "axes.grid": True,
        "grid.alpha": 0.3,
    }
)

PALETA_CLASSES = {
    "Baixo": "#2E7D32",
    "Moderado": "#F9A825",
    "Muito Alto": "#C62828",
}


def _carregar_json(caminho: Path) -> dict:
    with caminho.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def gerar_matriz_confusao() -> None:
    metrics = _carregar_json(RELATORIOS / "ensemble_stacking_gbm_metrics.json")
    classes = metrics["confusion_matrix"]["classes"]
    matriz = np.array(metrics["confusion_matrix"]["matrix"], dtype=float)
    matriz_norm = matriz / matriz.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))

    for ax, dados, fmt, titulo, cmap in (
        (axes[0], matriz, ",.0f", "Matriz de confusão — contagens absolutas", "Blues"),
        (axes[1], matriz_norm, ".2%", "Matriz de confusão — normalizada por linha", "Purples"),
    ):
        im = ax.imshow(dados, cmap=cmap, aspect="auto")
        ax.set_xticks(range(len(classes)))
        ax.set_yticks(range(len(classes)))
        ax.set_xticklabels(classes, rotation=0)
        ax.set_yticklabels(classes)
        ax.set_xlabel("Classe predita")
        ax.set_ylabel("Classe real")
        ax.set_title(titulo, pad=10)
        ax.grid(False)

        thresh = dados.max() / 2.0
        for i in range(len(classes)):
            for j in range(len(classes)):
                valor = dados[i, j]
                ax.text(
                    j,
                    i,
                    format(valor, fmt).replace(",", "."),
                    ha="center",
                    va="center",
                    color="white" if valor > thresh else "black",
                    fontsize=11,
                    fontweight="bold",
                )
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(
        "Matriz de confusão — ensemble_stacking_gbm  (n = 184.862; acurácia = 84,61%, F1-macro = 0,7995)",
        fontsize=12,
        y=1.02,
    )
    out = RELATORIOS / "matriz_confusao_stacking_gbm.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"OK  matriz de confusão -> {out.relative_to(ROOT)}")


def _limpar_nome_feature(nome: str) -> str:
    """Remove os prefixos do ColumnTransformer (``num__``, ``cat__``)."""
    for prefixo in ("num__", "cat__"):
        if nome.startswith(prefixo):
            return nome[len(prefixo):]
    return nome


def gerar_shap_per_classe() -> None:
    dados = _carregar_json(RELATORIOS / "shap_per_class.json")
    classes = list(PALETA_CLASSES.keys())
    top_n = 10
    shap_data = dados.get("shap_per_class", {})

    fig, axes = plt.subplots(1, 3, figsize=(13, 5.4), sharex=False)

    for ax, classe in zip(axes, classes):
        ranking = shap_data.get(classe, [])
        if not ranking:
            ax.text(0.5, 0.5, f"sem dados para {classe}", ha="center", va="center")
            ax.set_title(f"Classe {classe}")
            ax.grid(False)
            continue
        itens = ranking[:top_n]
        nomes = [_limpar_nome_feature(item["feature"]) for item in itens][::-1]
        valores = [item["mean_abs_shap"] for item in itens][::-1]

        cor = PALETA_CLASSES[classe]
        ax.barh(nomes, valores, color=cor, edgecolor="black", linewidth=0.4)
        ax.set_title(f"Top-{top_n} SHAP — classe {classe}", pad=8)
        ax.set_xlabel(r"$|$SHAP$|$ médio (RF leve $proxy$)")
        ax.tick_params(axis="y", labelsize=9)
        for i, v in enumerate(valores):
            ax.text(v, i, f"  {v:.4f}", va="center", fontsize=8.5)
        ax.grid(axis="x", alpha=0.3)

    fig.suptitle(
        "Importâncias SHAP por classe — Random Forest compacto proxy (n = 500 amostras)",
        fontsize=12,
        y=1.02,
    )
    fig.tight_layout()
    out = RELATORIOS / "shap_per_classe_barras.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"OK  SHAP por classe -> {out.relative_to(ROOT)}")


def gerar_permutation_importance() -> None:
    dados = _carregar_json(RELATORIOS / "permutation_importance_por_classe.json")

    fig, axes = plt.subplots(1, 4, figsize=(15, 5.4))
    eixos = ["global", "Baixo", "Moderado", "Muito Alto"]
    cores = {
        "global": "#1565C0",
        "Baixo": PALETA_CLASSES["Baixo"],
        "Moderado": PALETA_CLASSES["Moderado"],
        "Muito Alto": PALETA_CLASSES["Muito Alto"],
    }
    top_n = 10

    por_classe = dados.get("importancia_por_classe", {})

    for ax, eixo in zip(axes, eixos):
        if eixo == "global":
            ranking = dados.get("importancia_global", [])
        else:
            ranking = por_classe.get(eixo, [])

        itens = ranking[:top_n]
        nomes = [_limpar_nome_feature(item["feature"]) for item in itens][::-1]
        chave_delta = "delta_f1_macro" if eixo == "global" else "delta_f1"
        deltas = [item.get(chave_delta, 0.0) for item in itens][::-1]

        ax.barh(nomes, deltas, color=cores[eixo], edgecolor="black", linewidth=0.4)
        titulo = "Δ F1-macro" if eixo == "global" else f"Classe {eixo} (Δ F1)"
        ax.set_title(f"Top-{top_n} — {titulo}", pad=8)
        ax.set_xlabel("Queda de F1 ao permutar")
        ax.tick_params(axis="y", labelsize=8.5)
        for i, v in enumerate(deltas):
            ax.text(v, i, f"  {v:.4f}", va="center", fontsize=8)
        ax.grid(axis="x", alpha=0.3)

    fig.suptitle(
        "Permutation Importance sobre o ensemble_stacking_gbm (5.000 amostras, 3 repetições)",
        fontsize=12,
        y=1.02,
    )
    fig.tight_layout()
    out = RELATORIOS / "permutation_importance_por_classe.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"OK  Permutation Importance -> {out.relative_to(ROOT)}")


def gerar_validacao_temporal() -> None:
    rf = _carregar_json(RELATORIOS / "validacao_temporal.json")
    stk = _carregar_json(RELATORIOS / "validacao_temporal_stacking_gbm.json")

    def _extrair_folds(blob: dict) -> tuple[list[int], list[float], list[float], list[float]]:
        folds = blob.get("folds") or blob.get("resultados") or []
        anos = [f.get("ano") or f.get("test_year") or f.get("ano_teste") for f in folds]
        acc = [f.get("accuracy") or f.get("acuracia") for f in folds]
        f1m = [f.get("f1_macro") for f in folds]
        f1mod = [f.get("f1_moderado") or f.get("f1_Moderado") for f in folds]
        return anos, acc, f1m, f1mod

    anos_rf, acc_rf, f1m_rf, f1mod_rf = _extrair_folds(rf)
    anos_stk, acc_stk, f1m_stk, f1mod_stk = _extrair_folds(stk)

    metas = [("Acurácia", acc_rf, acc_stk), ("F1-macro", f1m_rf, f1m_stk), ("F1-Moderado", f1mod_rf, f1mod_stk)]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6), sharey=False)

    for ax, (titulo, valores_rf, valores_stk) in zip(axes, metas):
        x = np.arange(len(anos_rf or anos_stk))
        largura = 0.36
        ax.bar(
            x - largura / 2,
            valores_rf,
            largura,
            label="RF balanced (Tier 1)",
            color="#2E7D32",
            edgecolor="black",
            linewidth=0.4,
        )
        ax.bar(
            x + largura / 2,
            valores_stk,
            largura,
            label="Stacking GBM",
            color="#C62828",
            edgecolor="black",
            linewidth=0.4,
        )
        ax.set_xticks(x)
        ax.set_xticklabels([f"Fold {a}" for a in (anos_rf or anos_stk)])
        ax.set_title(titulo, pad=8)
        if titulo == "Acurácia":
            ax.set_ylabel("Valor da métrica")
            ax.set_ylim(0, 1.0)
        elif titulo == "F1-macro":
            ax.set_ylim(0, 1.0)
        else:
            ax.set_ylim(0, 0.6)
        ax.axhline(0.70, color="gray", linestyle="--", linewidth=0.7, label="Meta TCC ≥ 70%")
        ax.legend(loc="upper right", fontsize=8.5, framealpha=0.9)

        for xi, v_rf, v_stk in zip(x, valores_rf, valores_stk):
            ax.text(xi - largura / 2, v_rf + 0.01, f"{v_rf:.3f}", ha="center", fontsize=8)
            ax.text(xi + largura / 2, v_stk + 0.01, f"{v_stk:.3f}", ha="center", fontsize=8)

    fig.suptitle(
        "Validação temporal rolling-origin (folds 2021–2023) — RF balanced × Stacking GBM",
        fontsize=12,
        y=1.02,
    )
    fig.tight_layout()
    out = RELATORIOS / "validacao_temporal_comparativo.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"OK  Validação temporal -> {out.relative_to(ROOT)}")


def gerar_distribuicao_classes() -> None:
    suporte = {"Baixo": 67690, "Moderado": 31069, "Muito Alto": 86103}
    total = sum(suporte.values())

    fig, ax = plt.subplots(figsize=(8, 4.6))
    cores = [PALETA_CLASSES[c] for c in suporte.keys()]
    barras = ax.bar(
        list(suporte.keys()),
        list(suporte.values()),
        color=cores,
        edgecolor="black",
        linewidth=0.5,
    )
    for barra, valor in zip(barras, suporte.values()):
        pct = valor / total * 100
        ax.text(
            barra.get_x() + barra.get_width() / 2,
            valor + 1500,
            f"{valor:,}\n({pct:.1f}%)".replace(",", "."),
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    ax.set_ylabel("Quantidade de observações")
    ax.set_title(
        f"Distribuição estratificada das classes — hold-out de teste (n = {total:,})".replace(",", "."),
        pad=10,
    )
    ax.set_ylim(0, max(suporte.values()) * 1.18)
    ax.grid(axis="x", alpha=0)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    out = RELATORIOS / "distribuicao_classes.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"OK  Distribuição de classes -> {out.relative_to(ROOT)}")


def main() -> None:
    print("Gerando figuras para o TCC...")
    print(f"Diretório destino: {RELATORIOS.relative_to(ROOT)}")
    print("-" * 60)
    gerar_distribuicao_classes()
    gerar_matriz_confusao()
    gerar_shap_per_classe()
    gerar_permutation_importance()
    gerar_validacao_temporal()
    print("-" * 60)
    print("Concluído.")


if __name__ == "__main__":
    main()
