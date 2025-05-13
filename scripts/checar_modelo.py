import joblib
from sklearn.metrics import classification_report, confusion_matrix
from carregar_dados import carregar_dados
from sklearn.model_selection import train_test_split

# Carrega os dados
X, y, cat_features, num_features, df = carregar_dados()
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Carrega o pipeline salvo
pipeline = joblib.load(r'C:\tcc-2\TCC-2\modelos\random_forest.pkl')

# Faz as previsões
y_pred = pipeline.predict(X_test)

# Mostra as primeiras previsões e os rótulos reais correspondentes
for real, pred in list(zip(y_test, y_pred))[:10]:
    print(f"Real: {real} → Previsto: {pred}")

print(f"Total de pares: {len(list(zip(y_test, y_pred)))}")

# Avaliação completa
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))

print("\n🧩 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
