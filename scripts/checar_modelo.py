import argparse
import json
import logging
from pathlib import Path
from typing import Optional

import joblib
from sklearn.metrics import classification_report, confusion_matrix

from carregar_dados import carregar_dados

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR.parent / 'modelos'
SPLIT_METADATA_PATH = MODEL_DIR / 'split_metadata.json'


def _carregar_indices_teste() -> list:
    if not SPLIT_METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Metadata de split não encontrada em {SPLIT_METADATA_PATH}. "
            "Execute o treinamento antes de checar o modelo."
        )
    with SPLIT_METADATA_PATH.open('r', encoding='utf-8') as fp:
        metadata = json.load(fp)
    return metadata['test_indices']


def carregar_teste():
    X, y, _, _, _ = carregar_dados(use_saved_thresholds=True, persist_thresholds=False)
    indices_teste = _carregar_indices_teste()
    X_test = X.loc[indices_teste]
    y_test = y.loc[indices_teste]
    return X_test, y_test


def escolher_modelo(nome_modelo: Optional[str]) -> Path:
    modelos = sorted(MODEL_DIR.glob('*.pkl'))
    if not modelos:
        raise FileNotFoundError(f"Nenhum modelo encontrado em {MODEL_DIR}")

    if nome_modelo:
        caminho = MODEL_DIR / f'{nome_modelo}.pkl'
        if not caminho.exists():
            raise FileNotFoundError(f"Modelo {nome_modelo} não encontrado em {caminho}")
        return caminho

    logger.info("Modelos disponíveis: %s", ", ".join(m.stem for m in modelos))
    return modelos[0]


def main(nome_modelo: Optional[str] = None, preview: int = 10):
    modelo_path = escolher_modelo(nome_modelo)
    pipeline = joblib.load(modelo_path)

    X_test, y_test = carregar_teste()
    y_pred = pipeline.predict(X_test)

    pares = list(zip(y_test, y_pred))
    logger.info("Exibindo as %d primeiras previsões para %s", min(preview, len(pares)), modelo_path.stem)
    for real, pred in pares[:preview]:
        print(f"Real: {real} → Previsto: {pred}")

    print(f"\nTotal de pares avaliados: {len(pares)}")
    print("\n📊 Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    print("\n🧩 Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Checar rapidamente o desempenho de um modelo salvo.")
    parser.add_argument('--modelo', type=str, help="Nome do arquivo de modelo (sem extensão .pkl).")
    parser.add_argument('--preview', type=int, default=10, help="Quantidade de pares real vs previsto a exibir.")
    args = parser.parse_args()
    main(nome_modelo=args.modelo, preview=args.preview)
