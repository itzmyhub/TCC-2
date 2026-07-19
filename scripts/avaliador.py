import json
import logging
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import classification_report, confusion_matrix


logger = logging.getLogger(__name__)


def _softmax(scores: np.ndarray) -> np.ndarray:
    scores = np.atleast_2d(scores)
    scores = scores - scores.max(axis=1, keepdims=True)
    exp_scores = np.exp(scores)
    return exp_scores / exp_scores.sum(axis=1, keepdims=True)


class AvaliadorModelos:
    def __init__(self, modelo, X_test, y_test, nome_modelo: str = 'modelo'):
        self.modelo = modelo
        self.X_test = X_test
        self.y_test = y_test
        self.nome_modelo = nome_modelo

        self.y_pred = self.modelo.predict(self.X_test)
        self.classes = getattr(self.modelo, 'classes_', np.unique(self.y_test))
        self.probas = self._obter_probabilidades()
        self.metricas_basicas = self._calcular_metricas_basicas()

    def _obter_probabilidades(self) -> np.ndarray:
        if hasattr(self.modelo, "predict_proba"):
            logger.debug("Usando predict_proba para as probabilidades.")
            return self.modelo.predict_proba(self.X_test)

        if hasattr(self.modelo, "decision_function"):
            logger.debug("Usando decision_function para derivar probabilidades.")
            scores = self.modelo.decision_function(self.X_test)
            if scores.ndim == 1:
                scores = np.vstack([-scores, scores]).T
            return _softmax(scores)

        logger.warning(
            "Modelo %s não possui predict_proba ou decision_function. "
            "Probabilidades serão aproximadas via previsões determinísticas.",
            self.nome_modelo,
        )
        probas = np.zeros((len(self.y_pred), len(self.classes)))
        class_to_idx = {classe: idx for idx, classe in enumerate(self.classes)}
        for i, pred in enumerate(self.y_pred):
            probas[i, class_to_idx[pred]] = 1.0
        return probas

    def _calcular_metricas_basicas(self) -> Dict:
        matriz = confusion_matrix(self.y_test, self.y_pred, labels=self.classes)
        relatorio = classification_report(
            self.y_test,
            self.y_pred,
            labels=self.classes,
            output_dict=True,
            zero_division=0,
        )
        confiancas = self.probas.max(axis=1)
        metricas = {
            'classification_report': relatorio,
            'confusion_matrix': matriz.tolist(),
            'classes': list(map(str, self.classes)),
            'confidence_mean': float(np.mean(confiancas)),
            'confidence_std': float(np.std(confiancas)),
        }
        return metricas

    def relatorio_classificacao(self):
        logger.info("\n📄 Classification Report - %s", self.nome_modelo)
        texto = classification_report(
            self.y_test, self.y_pred, labels=self.classes, digits=3, zero_division=0
        )
        print(texto)
        return texto

    def matriz_confusao(self, normalizar: bool = True):
        logger.info("\n🧩 Confusion Matrix - %s", self.nome_modelo)
        matriz = confusion_matrix(
            self.y_test,
            self.y_pred,
            labels=self.classes,
            normalize='true' if normalizar else None,
        )
        print(matriz)
        return matriz

    def confianca_media(self):
        media = self.metricas_basicas['confidence_mean']
        logger.info("🔐 Confiança média das previsões: %.3f", media)
        return media

    def distribuicao_confianca(self):
        confiancas = self.probas.max(axis=1)
        plt.hist(confiancas, bins=10, color='skyblue', edgecolor='black')
        plt.title(f'Distribuição de Confiança - {self.nome_modelo}')
        plt.xlabel('Confiança da Previsão')
        plt.ylabel('Frequência')
        plt.grid(True)
        plt.show()

    def curva_calibracao(self, n_bins: int = 10):
        plt.figure(figsize=(8, 6))
        for i, classe in enumerate(self.classes):
            y_true_bin = (self.y_test == classe).astype(int)
            probas_classe = self.probas[:, i]
            true_prob, pred_prob = calibration_curve(y_true_bin, probas_classe, n_bins=n_bins)
            plt.plot(pred_prob, true_prob, marker='o', label=f'{classe}')

        plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
        plt.title(f'Curva de Calibração - {self.nome_modelo}')
        plt.xlabel('Confiança prevista')
        plt.ylabel('Frequência real')
        plt.legend()
        plt.grid(True)
        plt.show()

    def exportar_metricas(self, destino: Path) -> None:
        payload = dict(self.metricas_basicas)
        payload.update(
            {
                'modelo': self.nome_modelo,
            }
        )
        destino.parent.mkdir(parents=True, exist_ok=True)
        with destino.open('w', encoding='utf-8') as fp:
            json.dump(payload, fp, ensure_ascii=False, indent=2)
        logger.info("Métricas básicas exportadas para %s", destino)
