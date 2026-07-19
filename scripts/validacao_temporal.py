"""Validação temporal estrita por ano (rolling-origin) sobre o dataset enriquecido.

Motivação metodológica
----------------------
O split estratificado aleatório (`train_test_split(stratify=y)`) usado em
``treinamento_modelo.py`` mistura dados de todos os anos em treino e teste.
Como o pipeline carrega features de janela móvel (`Precipitacao_ma7/14/30/90`,
`DiaSemChuva_ma*`, `Precipitacao_acum_*`, `Incendios_Ultimos_*`, `SPI_*` etc.)
calculadas no conjunto **inteiro antes do split**, há risco de
**vazamento temporal**: linhas de teste podem ter recebido contribuição de
linhas que estão também no treino (vizinhança espacial + temporal).

Para quantificar esse viés, este script reavalia o pipeline em uma estratégia
**rolling-origin por ano**: para cada ano `Y` de avaliação:

* treino = `Ano < Y`;
* teste  = `Ano == Y`.

Isso simula um uso real: "treine com tudo que tenho até hoje, prediga o ano
seguinte". Comparamos as métricas (acurácia, F1-macro, F1-Moderado) com as do
split aleatório para estimar o ``Δ viés``.

Modelo padrão
-------------
Usamos ``random_forest_balanced`` (Optuna, Tier 1) — ele é o melhor *single
model* (82,93 %/78,66 % no split aleatório) e seu retreino em CPU dura
~5–10 minutos por fold (com subsample de 150 k amostras, vs 85 min do
Stacking GBM completo). Para 3 folds temporais totalizamos ~30 minutos.
O resultado é representativo: se o RF cair X pp sob validação temporal,
espera-se queda similar (ou menor, pelo efeito do ensemble) no Stacking.

**Nota sobre memória.** A codificação one-hot de Município (≈ 542 categorias)
expande o vetor de features para ≈ 600 colunas. Em float64, treinar com
todos os ~640 k exemplos exigiria ~3 GiB só para a matriz densa do
preprocessor — inviável em máquinas de desenvolvimento. Por isso, cada
fold de treino é subamostrado estratificadamente para ``max_train_samples``
(default 150 000), seguindo o mesmo padrão do Optuna XGB/LGBM em §8.7.
A quantidade de teste por fold (~75–100 k) é mantida intacta.

Saída
-----
``modelos/relatorios/validacao_temporal.json`` com a estrutura:

.. code-block:: json

    {
      "modelo": "random_forest_balanced",
      "estrategia": "rolling-origin (ano-a-ano)",
      "folds": [
        {"ano_teste": 2024, "n_treino": ..., "n_teste": ..., "accuracy": ..., "f1_macro": ..., "f1_moderado": ...},
        ...
      ],
      "media": {"accuracy": ..., "f1_macro": ..., "f1_moderado": ...},
      "baseline_split_aleatorio": {...},
      "delta_vies": {...}
    }

Referência: Bergmeir & Benítez (2012); Mosteiro et al. (2025, ECMWF/UKMO).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import Pipeline

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR.parent / 'modelos'
REPORT_DIR = MODEL_DIR / 'relatorios'
REPORT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(BASE_DIR))
from carregar_dados import carregar_dados  # noqa: E402
from pre_processor import PreProcessor  # noqa: E402
from treinamento_modelo import MODEL_LIBRARY_IMPROVED  # noqa: E402


def _metricas(y_true: pd.Series, y_pred: np.ndarray) -> dict:
    rep = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'f1_moderado': rep.get('Moderado', {}).get('f1-score', float('nan')),
        'f1_baixo': rep.get('Baixo', {}).get('f1-score', float('nan')),
        'f1_muito_alto': rep.get('Muito Alto', {}).get('f1-score', float('nan')),
    }


def _carregar_baseline_metricas(modelo_nome: str) -> Optional[dict]:
    path = REPORT_DIR / f'{modelo_nome}_metrics.json'
    if not path.exists():
        return None
    with path.open(encoding='utf-8') as fp:
        payload = json.load(fp)
    rep = payload.get('classification_report', {})
    return {
        'accuracy': payload.get('accuracy'),
        'f1_macro': payload.get('f1_macro'),
        'f1_moderado': rep.get('Moderado', {}).get('f1-score'),
        'f1_baixo': rep.get('Baixo', {}).get('f1-score'),
        'f1_muito_alto': rep.get('Muito Alto', {}).get('f1-score'),
    }


def validacao_temporal(
    modelo_nome: str = 'random_forest_balanced',
    dataset_path: str = 'base_de_dados_enriquecido.csv',
    anos_teste: Optional[List[int]] = None,
    saida_path: Optional[str] = None,
    calibrar: bool = True,
    max_train_samples: int = 150_000,
) -> dict:
    """Executa rolling-origin por ano e salva resultado em JSON."""

    logger.info("Carregando dados (com pipeline padrão de derivadas/avançadas)...")
    X, y, cat_features, num_features, df_raw = carregar_dados(
        dataset_path=dataset_path,
        use_saved_thresholds=True,
        persist_thresholds=False,
    )

    if 'Ano' not in X.columns:
        raise RuntimeError("Coluna 'Ano' ausente do DataFrame: validação temporal impossível.")

    anos_disponiveis = sorted(int(a) for a in X['Ano'].dropna().unique())
    logger.info("Anos disponíveis no dataset: %s", anos_disponiveis)

    if anos_teste is None:
        # default: os 3 últimos anos com >= 1% das observações
        contagem = X.groupby('Ano').size()
        limiar = max(int(0.01 * len(X)), 1000)
        anos_teste = [int(a) for a in anos_disponiveis if contagem.get(a, 0) >= limiar][-3:]
    logger.info("Anos de teste (rolling-origin): %s", anos_teste)

    if modelo_nome not in MODEL_LIBRARY_IMPROVED:
        raise KeyError(f"Modelo desconhecido: {modelo_nome}")

    folds = []
    t0 = time.time()
    for i, ano in enumerate(anos_teste, 1):
        mask_train = X['Ano'] < ano
        mask_test = X['Ano'] == ano
        n_train = int(mask_train.sum())
        n_test = int(mask_test.sum())
        if n_train < 1000 or n_test < 100:
            logger.warning("Fold ano=%s ignorado (n_train=%d, n_test=%d)", ano, n_train, n_test)
            continue

        logger.info(
            "Fold %d/%d — ano=%s | treino=%d (anos %s..%s) | teste=%d",
            i,
            len(anos_teste),
            ano,
            n_train,
            anos_disponiveis[0],
            ano - 1,
            n_test,
        )

        X_train = X.loc[mask_train]
        y_train = y.loc[mask_train]
        X_test = X.loc[mask_test]
        y_test = y.loc[mask_test]

        # OHE de Municipio (542 únicos) gera ~600 colunas → matriz densa ocupa
        # ~3 GiB para 640 k linhas. Para caber em memória do dev, fazemos
        # subsample estratificado do treino quando supera ``max_train_samples``.
        if len(X_train) > max_train_samples:
            from sklearn.model_selection import train_test_split as _tts
            _, X_train_sub, _, y_train_sub = _tts(
                X_train,
                y_train,
                test_size=max_train_samples / len(X_train),
                random_state=42,
                stratify=y_train,
            )
            logger.info(
                "  Subsample estratificado do treino: %d → %d (max_train_samples=%d)",
                n_train,
                len(X_train_sub),
                max_train_samples,
            )
            X_train = X_train_sub
            y_train = y_train_sub
            n_train = len(X_train)

        # Usa o ColumnTransformer interno do wrapper PreProcessor — esse sim
        # respeita a interface sklearn fit_transform(X, y) que o Pipeline exige.
        preproc_wrapper = PreProcessor(num_features=num_features, cat_features=cat_features)
        estim = MODEL_LIBRARY_IMPROVED[modelo_nome]
        try:
            from sklearn.base import clone
            estim = clone(estim)
        except Exception:  # pragma: no cover
            pass

        if calibrar:
            modelo = CalibratedClassifierCV(estim, method='isotonic', cv=3)
        else:
            modelo = estim

        pipeline = Pipeline([
            ('preprocessor', preproc_wrapper.preprocessor),
            ('modelo', modelo),
        ])

        t_fold = time.time()
        pipeline.fit(X_train, y_train)
        treino_s = time.time() - t_fold

        y_pred = pipeline.predict(X_test)
        metr = _metricas(y_test, y_pred)
        metr.update({
            'ano_teste': int(ano),
            'n_treino': n_train,
            'n_teste': n_test,
            'train_time_sec': round(treino_s, 1),
        })
        logger.info(
            "  → acc=%.4f | F1m=%.4f | F1-Mod=%.4f (treino %.0fs)",
            metr['accuracy'],
            metr['f1_macro'],
            metr['f1_moderado'],
            treino_s,
        )
        folds.append(metr)

    if not folds:
        raise RuntimeError("Nenhum fold válido foi avaliado.")

    media = {
        k: float(np.mean([f[k] for f in folds]))
        for k in ('accuracy', 'f1_macro', 'f1_moderado', 'f1_baixo', 'f1_muito_alto')
    }
    desvio = {
        k: float(np.std([f[k] for f in folds]))
        for k in ('accuracy', 'f1_macro', 'f1_moderado')
    }

    baseline = _carregar_baseline_metricas(modelo_nome)
    delta = None
    if baseline is not None:
        delta = {
            k: round(media[k] - baseline[k], 4) if baseline.get(k) is not None else None
            for k in ('accuracy', 'f1_macro', 'f1_moderado', 'f1_baixo', 'f1_muito_alto')
        }

    payload = {
        'modelo': modelo_nome,
        'dataset': dataset_path,
        'estrategia': 'rolling-origin (ano-a-ano, treino = Ano < Y, teste = Ano == Y)',
        'anos_teste': anos_teste,
        'folds': folds,
        'media': {k: round(v, 4) for k, v in media.items()},
        'desvio_padrao': {k: round(v, 4) for k, v in desvio.items()},
        'baseline_split_aleatorio': baseline,
        'delta_vies': delta,
        'tempo_total_segundos': round(time.time() - t0, 1),
        'calibrado': bool(calibrar),
    }

    saida = Path(saida_path) if saida_path else REPORT_DIR / 'validacao_temporal.json'
    saida.parent.mkdir(parents=True, exist_ok=True)
    with saida.open('w', encoding='utf-8') as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=2)
    logger.info("Resultado salvo em %s", saida)

    logger.info("==== RESUMO ====")
    logger.info("Média (rolling-origin):  acc=%.4f | F1m=%.4f | F1-Mod=%.4f",
                media['accuracy'], media['f1_macro'], media['f1_moderado'])
    if baseline is not None:
        logger.info(
            "Baseline split aleatório: acc=%.4f | F1m=%.4f | F1-Mod=%.4f",
            baseline.get('accuracy', float('nan')),
            baseline.get('f1_macro', float('nan')),
            baseline.get('f1_moderado', float('nan')),
        )
        logger.info(
            "Δ viés (temporal - aleatório): acc=%+.4f | F1m=%+.4f | F1-Mod=%+.4f",
            delta['accuracy'],
            delta['f1_macro'],
            delta['f1_moderado'],
        )

    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description='Validação temporal estrita (rolling-origin por ano)')
    parser.add_argument(
        '--modelo',
        default='random_forest_balanced',
        choices=list(MODEL_LIBRARY_IMPROVED.keys()),
    )
    parser.add_argument('--dataset', default='base_de_dados_enriquecido.csv')
    parser.add_argument(
        '--anos',
        nargs='*',
        type=int,
        default=None,
        help='Anos para servir como conjunto de teste (rolling-origin). Default: 3 últimos com massa suficiente.',
    )
    parser.add_argument('--saida', default=None)
    parser.add_argument('--no-calibrar', action='store_true', help='Desativa CalibratedClassifierCV (mais rápido).')
    parser.add_argument(
        '--max-train',
        type=int,
        default=150_000,
        help='Limite de amostras de treino por fold (subsample estratificado para caber em memória). Default 150 000.',
    )
    args = parser.parse_args()

    validacao_temporal(
        modelo_nome=args.modelo,
        dataset_path=args.dataset,
        anos_teste=args.anos,
        saida_path=args.saida,
        calibrar=not args.no_calibrar,
        max_train_samples=args.max_train,
    )


if __name__ == '__main__':
    main()
