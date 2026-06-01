import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier  # <-- NUEVO ALGORITMO
from sklearn.metrics import accuracy_score, f1_score  # <-- NUEVA MÉTRICA
import numpy as np

# 1. Datos ficticios de Churn
X_train = np.array([[25, 50], [45, 110], [30, 80], [60, 200], [22, 40], [50, 150], [35, 90], [18, 30], [55, 120], [40, 95]])
y_train = np.array([0, 1, 0, 1, 0, 1, 0, 0, 1, 1])

# --- EXPERIMENTO ---
print("Entrenando segundo algoritmo: Random Forest Classifier...")
model_rf = RandomForestClassifier(n_estimators=10, random_state=42)
model_rf.fit(X_train, y_train)

# Evaluando el nuevo modelo
preds_rf = model_rf.predict(X_train)
acc_rf = accuracy_score(y_train, preds_rf)
f1_rf = f1_score(y_train, preds_rf)

print(f"Resultados del Experimento:")
print(f"- Nuevo Accuracy: {acc_rf:.2f}")
print(f"- Nuevo F1-Score: {f1_rf:.2f}")