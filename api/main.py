"""
API predictiva de churn — interfaz entre el cliente y el modelo entrenado.

Endpoints:
  GET  /        — bienvenida
  GET  /health  — estado del servicio y modelo
  POST /predict — predicción con validación de entrada
"""
from __future__ import annotations

import json
import pickle
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException

from api.schemas import ClienteChurnRequest, PredictResponse

AUTOR = "Alex J. Paco Surco"
UMBRAL_RIESGO = 0.5
FEATURES = ["antiguedad", "cargo_mensual", "reclamos"]

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "modelo_churn.pkl"
METADATA_PATH = BASE_DIR / "models" / "modelo_churn_v1_metadata.json"

modelo: Any = None
metadata: dict[str, Any] = {}


def cargar_modelo() -> None:
    global modelo, metadata

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No se encontró el modelo en {MODEL_PATH}. Ejecuta: python src/train.py"
        )

    with MODEL_PATH.open("rb") as archivo:
        modelo = pickle.load(archivo)

    if METADATA_PATH.exists():
        metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    cargar_modelo()
    yield


app = FastAPI(
    title="API Predictiva de Churn",
    description=(
        "Interfaz controlada para consumir el modelo de abandono de clientes "
        "sin acceso directo al código ni al archivo del modelo."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
def read_root() -> dict[str, str]:
    return {
        "mensaje": "API predictiva de churn en ejecución",
        "autor": AUTOR,
        "endpoints": "/health, /predict, /docs",
    }


@app.get("/health")
def health_check() -> dict[str, Any]:
    return {
        "status": "ok",
        "modelo_cargado": modelo is not None,
        "archivo_modelo": MODEL_PATH.name,
        "autor": AUTOR,
        "variables": FEATURES,
    }


def _clasificar_riesgo(probabilidad: float) -> str:
    return "alto_riesgo" if probabilidad >= UMBRAL_RIESGO else "bajo_riesgo"


@app.post("/predict", response_model=PredictResponse)
def predict(cliente: ClienteChurnRequest) -> PredictResponse:
    if modelo is None:
        raise HTTPException(status_code=503, detail="Modelo no disponible. Ejecuta el entrenamiento.")

    entrada = pd.DataFrame(
        [
            {
                "antiguedad": cliente.antiguedad,
                "cargo_mensual": cliente.cargo_mensual,
                "reclamos": cliente.reclamos,
            }
        ]
    )

    probabilidades = modelo.predict_proba(entrada)[0]
    indice_churn = list(modelo.classes_).index(1)
    probabilidad = round(float(probabilidades[indice_churn]), 2)

    return PredictResponse(
        prediccion=_clasificar_riesgo(probabilidad),
        probabilidad=probabilidad,
        autor=AUTOR,
    )
