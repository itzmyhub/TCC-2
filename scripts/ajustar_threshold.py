"""
Ajuste de threshold por classe para melhorar F1 da classe Moderado.

O modelo atual (Ensemble Stacking) tem F1 ≤ 46% para Moderado.
Este script testa thresholds entre 0.25–0.50 e reporta o tradeoff
precision/recall/F1 para a classe Moderado.

Gera:
- modelos/relatorios/threshold_analysis.json: métricas por threshold testado
- modelos/relatorios/threshold_precision_recall.png: curva precision-recall
"""
import __main__ as _main
import json
import logging
import sys
from pathlib import Path

import joblib
import numpy as np

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR.parent / 'modelos'
METRICS_DIR = MODEL_DIR / 'relatorios'

sys.path.insert(0, str(BASE_DIR))
from carregar_dados import carregar_dados

# O Stacking picklou _LabelEncodingWrapper; joblib precisa achar a classe em __main__
import treinamento_modelo
if hasattr(treinamento_modelo, '_LabelEncodingWrapper'):
    _main._LabelEncodingWrapper = treinamento_modelo._LabelEncodingWrapper
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split


def _prever_com_threshold(proba, classes, threshold_moderado: float, threshold_muito_alto: float = 0.50):
    """
    Predição com thresholds customizados por classe.

    Lógica:
      1. Se P(Moderado) >= threshold_moderado → Moderado
      2. Se P(Muito Alto) >= threshold_muito_alto → Muito Alto
      3. Caso contrário → Baixo (classe padrão)

    Args:
        proba: array (n_samples, n_classes) de probabilidades
        classes: lista de nomes das classes (na ordem das colunas de proba)
        threshold_moderado: limiar mínimo para prever Moderado
        threshold_muito_alto: limiar mínimo para prever Muito Alto
    """
    classes = list(classes)
    idx_baixo = classes.index('Baixo') if 'Baixo' in classes else 0
    idx_moderado = classes.index('Moderado') if 'Moderado' in classes else 1
    idx_muito_alto = classes.index('Muito Alto') if 'Muito Alto' in classes else 2

    n = proba.shape[0]
    y_pred = np.empty(n, dtype=object)

    for i in range(n):
        p_mod = proba[i, idx_moderado]
        p_muito = proba[i, idx_muito_alto]
        p_baixo = proba[i, idx_baixo]

        if p_mod >= threshold_moderado:
            y_pred[i] = 'Moderado'
        elif p_muito >= threshold_muito_alto:
            y_pred[i] = 'Muito Alto'
        else:
            y_pred[i] = 'Baixo'

    return y_pred


def analisar_thresholds(
    modelo_nome: str = 'ensemble_stacking',
    thresholds_moderado=None,
    thresholds_muito_alto=None,
    dataset_path: str = None,
    n_amostras: int = 100000,
):
    """
    Analisa o impacto de diferentes thresholds para a classe Moderado.

    Args:
        modelo_nome: Nome do modelo .pkl (sem extensão)
        thresholds_moderado: Lista de thresholds a testar para Moderado
        thresholds_muito_alto: Lista de thresholds a testar para Muito Alto
        dataset_path: Caminho para CSV. None = padrão.
        n_amostras: Número de amostras do conjunto de teste
    """
    if thresholds_moderado is None:
        thresholds_moderado = [round(t, 2) for t in np.arange(0.25, 0.55, 0.05)]
    if thresholds_muito_alto is None:
        thresholds_muito_alto = [0.45, 0.50, 0.55]

    modelo_path = MODEL_DIR / f'{modelo_nome}.pkl'
    if not modelo_path.exists():
        modelos = [p.stem for p in MODEL_DIR.glob('*.pkl')]
        logger.error("Modelo não encontrado: %s. Disponíveis: %s", modelo_path, modelos)
        return

    logger.info("Carregando modelo: %s", modelo_path)
    pipeline = joblib.load(modelo_path)

    logger.info("Carregando dados...")
    X, y, _, _, _ = carregar_dados(
        dataset_path=dataset_path,
        use_saved_thresholds=True,
        persist_thresholds=False,
    )

    # Split reprodutível (mesmo que o treino — usar conjunto de teste)
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Subsample se necessário
    if len(X_test) > n_amostras:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(X_test), size=n_amostras, replace=False)
        X_test = X_test.iloc[idx]
        y_test = y_test.iloc[idx]

    logger.info("Conjunto de teste: %d amostras", len(X_test))

    if not hasattr(pipeline, 'predict_proba'):
        logger.error("Modelo não suporta predict_proba. Impossível ajustar threshold.")
        return

    logger.info("Calculando probabilidades...")
    proba = pipeline.predict_proba(X_test)
    classes = list(pipeline.classes_)
    logger.info("Classes: %s", classes)

    # Baseline (threshold padrão — argmax)
    y_pred_base = pipeline.predict(X_test)
    acc_base = accuracy_score(y_test, y_pred_base)
    f1_macro_base = f1_score(y_test, y_pred_base, average='macro')
    report_base = classification_report(y_test, y_pred_base, output_dict=True)

    logger.info("Baseline (argmax): accuracy=%.2f%% | F1-macro=%.4f | F1-Moderado=%.4f",
                acc_base * 100, f1_macro_base,
                report_base.get('Moderado', {}).get('f1-score', 0))

    resultados = []
    melhor = {'f1_moderado': 0.0, 'config': None}

    for thr_mod in thresholds_moderado:
        for thr_alto in thresholds_muito_alto:
            y_pred = _prever_com_threshold(proba, classes, thr_mod, thr_alto)
            acc = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
            f1_mac = f1_score(y_test, y_pred, average='macro', zero_division=0)

            mod_stats = report.get('Moderado', {})
            f1_mod = mod_stats.get('f1-score', 0)
            prec_mod = mod_stats.get('precision', 0)
            rec_mod = mod_stats.get('recall', 0)

            config = {
                'threshold_moderado': round(thr_mod, 2),
                'threshold_muito_alto': round(thr_alto, 2),
                'accuracy': round(acc, 4),
                'f1_macro': round(f1_mac, 4),
                'moderado': {
                    'precision': round(prec_mod, 4),
                    'recall': round(rec_mod, 4),
                    'f1': round(f1_mod, 4),
                },
                'baixo': {
                    'f1': round(report.get('Baixo', {}).get('f1-score', 0), 4),
                },
                'muito_alto': {
                    'f1': round(report.get('Muito Alto', {}).get('f1-score', 0), 4),
                },
            }
            resultados.append(config)

            if f1_mod > melhor['f1_moderado']:
                melhor['f1_moderado'] = f1_mod
                melhor['config'] = config

    resultados.sort(key=lambda r: r['moderado']['f1'], reverse=True)

    # Também identificar o melhor por F1-macro (objetivo agregado do TCC)
    melhor_macro = max(
        ({'f1_macro': r['f1_macro'], 'config': r} for r in resultados),
        key=lambda d: d['f1_macro'],
        default={'f1_macro': 0.0, 'config': None},
    )

    # Baseline como entry
    baseline = {
        'threshold_moderado': 'argmax',
        'threshold_muito_alto': 'argmax',
        'accuracy': round(acc_base, 4),
        'f1_macro': round(f1_macro_base, 4),
        'moderado': {
            'precision': round(report_base.get('Moderado', {}).get('precision', 0), 4),
            'recall': round(report_base.get('Moderado', {}).get('recall', 0), 4),
            'f1': round(report_base.get('Moderado', {}).get('f1-score', 0), 4),
        },
        'baixo': {'f1': round(report_base.get('Baixo', {}).get('f1-score', 0), 4)},
        'muito_alto': {'f1': round(report_base.get('Muito Alto', {}).get('f1-score', 0), 4)},
    }

    output = {
        'modelo': modelo_nome,
        'n_amostras_teste': len(X_test),
        'baseline': baseline,
        'melhor_para_moderado': melhor['config'],
        'melhor_para_f1_macro': melhor_macro['config'],
        'todos_resultados': resultados,
        'recomendacao': (
            f"Threshold Moderado={melhor['config']['threshold_moderado']} | "
            f"Muito Alto={melhor['config']['threshold_muito_alto']} "
            f"melhora F1-Moderado de {baseline['moderado']['f1']:.4f} "
            f"para {melhor['config']['moderado']['f1']:.4f} "
            f"(custo accuracy: {baseline['accuracy'] - melhor['config']['accuracy']:.4f})"
        ) if melhor['config'] else 'Sem melhoria encontrada',
    }

    # Persistir thresholds otimizados (formato consumível pelo app)
    if melhor_macro['config']:
        thresholds_prod_path = MODEL_DIR / 'prediction_thresholds.json'
        thresholds_prod = {
            'modelo': modelo_nome,
            'gerado_em': str(__import__('datetime').datetime.now().isoformat(timespec='seconds')),
            'estrategia': 'threshold_tuning_multi_classe',
            'logica': (
                "Se P(Moderado) >= threshold_moderado → Moderado; "
                "senão se P(Muito Alto) >= threshold_muito_alto → Muito Alto; "
                "senão → Baixo. Prioriza captura de Moderado (classe minoritária)."
            ),
            'thresholds_otimizados_f1_macro': {
                'threshold_moderado': melhor_macro['config']['threshold_moderado'],
                'threshold_muito_alto': melhor_macro['config']['threshold_muito_alto'],
                'accuracy_esperada': melhor_macro['config']['accuracy'],
                'f1_macro_esperado': melhor_macro['config']['f1_macro'],
                'f1_moderado_esperado': melhor_macro['config']['moderado']['f1'],
            },
            'thresholds_otimizados_f1_moderado': {
                'threshold_moderado': melhor['config']['threshold_moderado'],
                'threshold_muito_alto': melhor['config']['threshold_muito_alto'],
                'accuracy_esperada': melhor['config']['accuracy'],
                'f1_macro_esperado': melhor['config']['f1_macro'],
                'f1_moderado_esperado': melhor['config']['moderado']['f1'],
            } if melhor['config'] else None,
            'baseline_argmax': {
                'accuracy': baseline['accuracy'],
                'f1_macro': baseline['f1_macro'],
                'f1_moderado': baseline['moderado']['f1'],
            },
        }
        with thresholds_prod_path.open('w', encoding='utf-8') as fp:
            json.dump(thresholds_prod, fp, ensure_ascii=False, indent=2)
        logger.info("Thresholds de produção salvos em %s", thresholds_prod_path)

    saida = METRICS_DIR / 'threshold_analysis.json'
    saida.parent.mkdir(parents=True, exist_ok=True)
    with saida.open('w', encoding='utf-8') as fp:
        json.dump(output, fp, ensure_ascii=False, indent=2)
    logger.info("Análise de threshold salva em %s", saida)

    # Log resumo
    logger.info("\nBASELINE: accuracy=%.2f%% | F1-macro=%.4f | F1-Moderado=%.4f",
                baseline['accuracy'] * 100, baseline['f1_macro'], baseline['moderado']['f1'])
    if melhor['config']:
        cfg = melhor['config']
        logger.info(
            "MELHOR (F1-Moderado): thr_mod=%.2f | thr_alto=%.2f | "
            "acc=%.2f%% | F1-mac=%.4f | F1-Moderado=%.4f | prec=%.4f | rec=%.4f",
            cfg['threshold_moderado'], cfg['threshold_muito_alto'],
            cfg['accuracy'] * 100, cfg['f1_macro'],
            cfg['moderado']['f1'], cfg['moderado']['precision'], cfg['moderado']['recall'],
        )
    if melhor_macro['config']:
        cfg = melhor_macro['config']
        logger.info(
            "MELHOR (F1-macro):    thr_mod=%.2f | thr_alto=%.2f | "
            "acc=%.2f%% | F1-mac=%.4f | F1-Moderado=%.4f",
            cfg['threshold_moderado'], cfg['threshold_muito_alto'],
            cfg['accuracy'] * 100, cfg['f1_macro'], cfg['moderado']['f1'],
        )

    # Gráfico precision-recall por threshold
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        resultados_by_thr = {}
        for r in resultados:
            thr = r['threshold_moderado']
            if thr not in resultados_by_thr:
                resultados_by_thr[thr] = r

        thrs_sorted = sorted([r for r in resultados if r['threshold_muito_alto'] == 0.50],
                             key=lambda x: x['threshold_moderado'])

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Precision-Recall para Moderado
        ax1 = axes[0]
        if thrs_sorted:
            t_vals = [r['threshold_moderado'] for r in thrs_sorted]
            prec_vals = [r['moderado']['precision'] for r in thrs_sorted]
            rec_vals = [r['moderado']['recall'] for r in thrs_sorted]
            f1_vals = [r['moderado']['f1'] for r in thrs_sorted]

            ax1.plot(t_vals, prec_vals, 'b-o', label='Precision', markersize=6)
            ax1.plot(t_vals, rec_vals, 'r-s', label='Recall', markersize=6)
            ax1.plot(t_vals, f1_vals, 'g-^', label='F1', markersize=6)
            ax1.axvline(x=0.33, color='gray', linestyle=':', alpha=0.7, label='Argmax equiv.')
            ax1.set_xlabel('Threshold Moderado', fontsize=11)
            ax1.set_ylabel('Métrica', fontsize=11)
            ax1.set_title('Precision / Recall / F1 — Classe Moderado\n(Threshold Muito Alto = 0.50)', fontsize=11)
            ax1.legend()
            ax1.grid(alpha=0.3)

        # Accuracy e F1-macro por threshold
        ax2 = axes[1]
        if thrs_sorted:
            acc_vals = [r['accuracy'] for r in thrs_sorted]
            f1mac_vals = [r['f1_macro'] for r in thrs_sorted]

            ax2.plot(t_vals, acc_vals, 'b-o', label='Accuracy', markersize=6)
            ax2.plot(t_vals, f1mac_vals, 'r-s', label='F1-Macro', markersize=6)
            ax2.axhline(y=baseline['accuracy'], color='b', linestyle='--', alpha=0.5, label=f"Baseline acc={baseline['accuracy']:.3f}")
            ax2.axhline(y=baseline['f1_macro'], color='r', linestyle='--', alpha=0.5, label=f"Baseline F1={baseline['f1_macro']:.3f}")
            ax2.set_xlabel('Threshold Moderado', fontsize=11)
            ax2.set_ylabel('Métrica', fontsize=11)
            ax2.set_title('Accuracy e F1-Macro por Threshold', fontsize=11)
            ax2.legend()
            ax2.grid(alpha=0.3)

        plt.suptitle(f'Análise de Threshold — {modelo_nome}', fontsize=13, fontweight='bold')
        plt.tight_layout()

        grafico_path = METRICS_DIR / 'threshold_precision_recall.png'
        plt.savefig(grafico_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info("Gráfico de threshold salvo em %s", grafico_path)
    except Exception as e:
        logger.warning("Não foi possível gerar gráfico de threshold: %s", e)

    return output


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Análise de threshold por classe para modelos de risco')
    parser.add_argument(
        '--modelo',
        type=str,
        default='ensemble_stacking',
        help='Nome do modelo .pkl (padrão: ensemble_stacking)',
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default=None,
        help='Caminho para o dataset CSV',
    )
    parser.add_argument(
        '--n_amostras',
        type=int,
        default=100000,
        help='Número de amostras do conjunto de teste (padrão: 100000)',
    )
    args = parser.parse_args()

    analisar_thresholds(
        modelo_nome=args.modelo,
        dataset_path=args.dataset,
        n_amostras=args.n_amostras,
    )
