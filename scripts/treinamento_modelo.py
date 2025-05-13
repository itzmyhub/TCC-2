import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import time

from pre_processor import PreProcessor
from carregar_dados import carregar_dados

X, y, cat_features, num_features, df = carregar_dados()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

preprocessor = PreProcessor(num_features=num_features, cat_features=cat_features)

modelos = {
    "sgd_classifier": SGDClassifier(loss='log_loss', random_state=42),  # muito rápido, ideal p/ grandes bases
}

model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'modelos')
os.makedirs(model_dir, exist_ok=True)

for nome_modelo, modelo in modelos.items():
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor.preprocessor),
        ('modelo', modelo)
    ])

    print(f"\n⏳ Treinando modelo: {nome_modelo}...")

    start = time.time()
    pipeline.fit(X_train, y_train)
    end = time.time()
    tempo_treino = end - start

    modelo_path = os.path.join(model_dir, f"{nome_modelo}.pkl")
    joblib.dump(pipeline, modelo_path)

    print(f"✅ Modelo '{nome_modelo}' treinado em {tempo_treino:.2f} segundos e salvo em: {modelo_path}")
