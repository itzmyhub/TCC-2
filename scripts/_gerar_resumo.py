"""Gera resumo comparativo de todos os modelos treinados.

Atualizado em 2026-05-10 para incluir CatBoost e modelos Tier 1
(features físico-climáticas avançadas, dataset base_de_dados_enriquecido.csv)."""
import json
import sys
from datetime import datetime
from pathlib import Path

RELATORIOS = Path(__file__).parent.parent / 'modelos' / 'relatorios'

modelos_info = [
    ('ensemble_stacking_gbm', 'Ensemble Stacking GBM (RF + LR + XGB Optuna + LGBM Optuna + CatBoost, meta=GradientBoosting) — Tier 1'),
    ('ensemble_stacking', 'Ensemble Stacking LR (RF + LR + XGB Optuna + LGBM Optuna + CatBoost, meta=LR) — Tier 1'),
    ('random_forest_balanced', 'RF Otimizado Optuna (n_est=346, max_depth=25) — Tier 1'),
    ('catboost_classifier', 'CatBoost classifier (auto_class_weights=Balanced) — Tier 1'),
    ('xgboost_classifier', 'XGBoost Optuna (n_est=455, depth=8, lr=0.19) — Tier 1'),
    ('lightgbm_classifier', 'LightGBM Optuna (n_est=410, depth=11, num_leaves=149) — Tier 1'),
    ('ensemble_voting_soft', 'Ensemble Voting Soft'),
    ('random_forest_smote', 'Random Forest SMOTE'),
    ('logistic_regression_balanced', 'Logistic Regression Balanced'),
    ('sgd_classifier', 'SGD Classifier'),
]

# Detectar dataset usado pelo split_metadata mais recente
SPLIT_META_PATH = Path(__file__).parent.parent / 'modelos' / 'split_metadata.json'
dataset_atual = 'base_de_dados_enriquecido.csv'  # default

resumo = {
    'meta': {
        'dataset': dataset_atual,
        'tier1_features_avancadas': True,
        'n_test_samples': 184862,
        'gerado_em': datetime.now().strftime('%Y-%m-%d'),
        'objetivo_accuracy': '>=70%',
        'nota': (
            'Tier 1: 24 features físico-climáticas adicionadas em '
            'scripts/features_avancadas.py (SPI, KBDI proxy, MAs estendidas, '
            'lags, anomalias, histórico estendido, dias secos).'
        ),
    },
    'modelos': [],
}

for nome, descricao in modelos_info:
    f = RELATORIOS / f'{nome}_metrics.json'
    if f.exists():
        m = json.loads(f.read_text(encoding='utf-8'))
        cr = m.get('classification_report', {})
        resumo['modelos'].append({
            'nome': nome,
            'descricao': descricao,
            'accuracy': round(m.get('accuracy', 0) * 100, 2),
            'f1_macro': round(m.get('f1_macro', 0) * 100, 2),
            'f1_baixo': round(cr.get('Baixo', {}).get('f1-score', 0) * 100, 2),
            'f1_moderado': round(cr.get('Moderado', {}).get('f1-score', 0) * 100, 2),
            'f1_muito_alto': round(cr.get('Muito Alto', {}).get('f1-score', 0) * 100, 2),
        })
    else:
        print(f'Arquivo nao encontrado: {f}')

resumo['modelos'].sort(key=lambda x: x['accuracy'], reverse=True)
resumo['melhor_modelo'] = resumo['modelos'][0]['nome']
resumo['melhor_accuracy'] = resumo['modelos'][0]['accuracy']

out = RELATORIOS / 'resumo_melhorias.json'
out.write_text(json.dumps(resumo, ensure_ascii=False, indent=2), encoding='utf-8')
print('Resumo salvo em', out)
print()
print(f'{"Modelo":<45} {"Accuracy":>9} {"F1-Macro":>9} {"F1-Mod":>8}')
print('-' * 75)
for model in resumo['modelos']:
    print(f'{model["nome"]:<45} {model["accuracy"]:>8.2f}% {model["f1_macro"]:>8.2f}% {model["f1_moderado"]:>7.2f}%')
