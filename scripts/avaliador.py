import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.calibration import calibration_curve

class AvaliadorModelos:
    def __init__(self, modelo, X_test, y_test, nome_modelo='modelo'):
        self.modelo = modelo
        self.X_test = X_test
        self.y_test = y_test
        self.nome_modelo = nome_modelo
        self.y_pred = self.modelo.predict(X_test)
        self.probas = self.modelo.predict_proba(X_test)
        self.classes = self.modelo.classes_

    def relatorio_classificacao(self):
        print(f"\n📄 Classification Report - {self.nome_modelo}")
        print(classification_report(self.y_test, self.y_pred, digits=3))

    def matriz_confusao(self):
        print(f"\n🧩 Confusion Matrix - {self.nome_modelo}")
        matriz = confusion_matrix(self.y_test, self.y_pred, normalize='true')
        print(matriz)

    def confianca_media(self):
        confiancas = self.probas.max(axis=1)
        media = np.mean(confiancas)
        print(f"\n🔐 Confiança média das previsões: {media:.3f}")

    def distribuicao_confianca(self):
        confiancas = self.probas.max(axis=1)
        plt.hist(confiancas, bins=10, color='skyblue', edgecolor='black')
        plt.title(f'Distribuição de Confiança - {self.nome_modelo}')
        plt.xlabel('Confiança da Previsão')
        plt.ylabel('Frequência')
        plt.grid(True)
        plt.show()

    def curva_calibracao(self):
        plt.figure(figsize=(8, 6))
        for i, classe in enumerate(self.classes):
            y_true_bin = (self.y_test == classe).astype(int)
            probas_classe = self.probas[:, i]
            true_prob, pred_prob = calibration_curve(y_true_bin, probas_classe, n_bins=10)
            plt.plot(pred_prob, true_prob, marker='o', label=f'{classe}')

        plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
        plt.title(f'Curva de Calibração - {self.nome_modelo}')
        plt.xlabel('Confiança prevista')
        plt.ylabel('Frequência real')
        plt.legend()
        plt.grid(True)
        plt.show()
