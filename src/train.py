"""
Entrenamiento y persistencia del modelo de churn.

Genera:
  - models/modelo_churn.pkl          (formato del laboratorio)
  - models/modelo_churn_v1.joblib    (formato de la actividad)
  - models/modelo_churn_v1_metadata.json
  - docs/metricas_modelo.md
  - data/train.csv y data/test.csv
"""
from __future__ import annotations

import json
import pickle
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

AUTOR = "Alex J. Paco Surco"
MODEL_VERSION = "v1"
FEATURES = ["antiguedad", "cargo_mensual", "reclamos"]
TARGET = "churn"
RANDOM_STATE = 42

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "churn_clientes.csv"
TRAIN_PATH = BASE_DIR / "data" / "train.csv"
TEST_PATH = BASE_DIR / "data" / "test.csv"
MODELS_DIR = BASE_DIR / "models"
DOCS_DIR = BASE_DIR / "docs"

PKL_PATH = MODELS_DIR / "modelo_churn.pkl"
JOBLIB_PATH = MODELS_DIR / f"modelo_churn_{MODEL_VERSION}.joblib"
METADATA_PATH = MODELS_DIR / f"modelo_churn_{MODEL_VERSION}_metadata.json"
METRICS_PATH = DOCS_DIR / "metricas_modelo.md"


def cargar_datos() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"No se encontró {DATA_PATH}. Ejecuta primero: python src/generar_datos.py"
        )
    return pd.read_csv(DATA_PATH)


def entrenar_modelo(df: pd.DataFrame) -> tuple[RandomForestClassifier, dict[str, float]]:
    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    TRAIN_PATH.parent.mkdir(parents=True, exist_ok=True)
    train_df = X_train.copy()
    train_df[TARGET] = y_train
    test_df = X_test.copy()
    test_df[TARGET] = y_test
    train_df.to_csv(TRAIN_PATH, index=False)
    test_df.to_csv(TEST_PATH, index=False)

    modelo = RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        random_state=RANDOM_STATE,
    )
    modelo.fit(X_train, y_train)

    y_pred = modelo.predict(X_test)
    metricas = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
    }
    return modelo, metricas


def guardar_artefactos(modelo: RandomForestClassifier, metricas: dict[str, float]) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    entrenado_en = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    with PKL_PATH.open("wb") as archivo:
        pickle.dump(modelo, archivo)

    joblib.dump(modelo, JOBLIB_PATH)

    metadata = {
        "autor": AUTOR,
        "version": MODEL_VERSION,
        "algoritmo": "RandomForestClassifier",
        "variables": FEATURES,
        "variable_objetivo": TARGET,
        "entrenado_en": entrenado_en,
        "archivo_pkl": str(PKL_PATH.name),
        "archivo_joblib": str(JOBLIB_PATH.name),
        "metricas": metricas,
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    contenido_metricas = f"""# Métricas del modelo de churn

**Autor:** {AUTOR}  
**Versión:** {MODEL_VERSION}  
**Algoritmo:** RandomForestClassifier  
**Fecha de entrenamiento:** {entrenado_en}

## Variables de entrada

- `antiguedad` (1–72 meses)
- `cargo_mensual` (18–120 USD)
- `reclamos` (0–10)

## Resultados en conjunto de prueba

| Métrica | Valor |
|---------|-------|
| Accuracy | {metricas['accuracy']:.4f} |
| Precision | {metricas['precision']:.4f} |
| Recall | {metricas['recall']:.4f} |
| F1-Score | {metricas['f1_score']:.4f} |

## Artefactos generados

- `{PKL_PATH.name}` — serialización del laboratorio
- `{JOBLIB_PATH.name}` — serialización para la actividad
- `{METADATA_PATH.name}` — metadatos del modelo
"""
    METRICS_PATH.write_text(contenido_metricas, encoding="utf-8")


def main() -> None:
    print("=" * 50)
    print("ENTRENAMIENTO — Modelo de predicción de churn")
    print(f"Autor: {AUTOR}")
    print("=" * 50)

    df = cargar_datos()
    print(f"Registros cargados: {len(df)}")

    modelo, metricas = entrenar_modelo(df)
    guardar_artefactos(modelo, metricas)

    print("\nModelo entrenado y serializado correctamente.")
    print(f"  - {PKL_PATH}")
    print(f"  - {JOBLIB_PATH}")
    print(f"  - {METADATA_PATH}")
    print(f"  - {METRICS_PATH}")
    print(f"  - {TRAIN_PATH}")
    print(f"  - {TEST_PATH}")
    print("\nMétricas en test:")
    for nombre, valor in metricas.items():
        print(f"  - {nombre}: {valor:.4f}")


if __name__ == "__main__":
    main()
