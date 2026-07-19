import json
import logging
from pathlib import Path

import joblib

from avaliador import AvaliadorModelos
from carregar_dados import carregar_dados, carregar_risk_thresholds


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR.parent / 'modelos'
RELATORIOS_DIR = MODEL_DIR / 'relatorios' / 'avaliacoes'
SPLIT_METADATA_PATH = MODEL_DIR / 'split_metadata.json'


def _carregar_indices_teste() -> list:
    if not SPLIT_METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Metadata de split não encontrada em {SPLIT_METADATA_PATH}. "
            "Execute o treinamento antes de avaliar."
        )
    with SPLIT_METADATA_PATH.open('r', encoding='utf-8') as fp:
        metadata = json.load(fp)
    return metadata['test_indices']


def _selecionar_conjunto_teste():
    X, y, _, _, _ = carregar_dados(use_saved_thresholds=True, persist_thresholds=False)
    indices_teste = _carregar_indices_teste()
    X_test = X.loc[indices_teste]
    y_test = y.loc[indices_teste]
    return X_test, y_test


def avaliar_modelos():
    RELATORIOS_DIR.mkdir(parents=True, exist_ok=True)
    X_test, y_test = _selecionar_conjunto_teste()
    thresholds = carregar_risk_thresholds()
    if thresholds:
        logger.info(
            "Avaliação utilizando thresholds de risco: moderado >= %.2f | muito alto >= %.2f",
            thresholds['moderate_min'],
            thresholds['very_high_min'],
        )

    for modelo_path in MODEL_DIR.glob('*.pkl'):
        nome_modelo = modelo_path.stem
        logger.info("Avaliação em andamento para %s", nome_modelo)
        pipeline = joblib.load(modelo_path)
        avaliador = AvaliadorModelos(pipeline, X_test, y_test, nome_modelo=nome_modelo)

        avaliador.relatorio_classificacao()
        avaliador.matriz_confusao()
        avaliador.confianca_media()

        destino = RELATORIOS_DIR / f'{nome_modelo}_evaluation.json'
        avaliador.exportar_metricas(destino)


if __name__ == '__main__':
    avaliar_modelos()
