import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from carregar_dados import carregar_dados
from avaliador import AvaliadorModelos

X, y, cat_features, num_features, df = carregar_dados()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

modelos_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'modelos')

for nome_modelo in os.listdir(modelos_dir):
    if nome_modelo.endswith('.pkl'):
        modelo_path = os.path.join(modelos_dir, nome_modelo)
        pipeline = joblib.load(modelo_path)

        print(f"\n🔍 Avaliando modelo: {nome_modelo}")
        avaliador = AvaliadorModelos(pipeline, X_test, y_test, nome_modelo=nome_modelo)

        avaliador.relatorio_classificacao()
        avaliador.matriz_confusao()
        avaliador.confianca_media()
        avaliador.distribuicao_confianca()
        avaliador.curva_calibracao()
