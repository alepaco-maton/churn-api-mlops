import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np

# 1. Datos ficticios de Churn (10 clientes: características e indicador si se fue o no)
X_train = np.array([[25, 50], [45, 110], [30, 80], [60, 200], [22, 40], [50, 150], [35, 90], [18, 30], [55, 120], [40, 95]])
y_train = np.array([0, 1, 0, 1, 0, 1, 0, 0, 1, 1]) # 1 = Se fue (Churn), 0 = Se quedó

# 2. Entrenar Algoritmo 1: Regresión Logística
print("Entrenando modelo base: Regresión Logística...")
model = LogisticRegression()
model.fit(X_train, y_train)

# 3. Evaluar
preds = model.predict(X_train)
acc = accuracy_score(y_train, preds)
print(f"Métrica obtenida - Accuracy: {acc:.2f}")