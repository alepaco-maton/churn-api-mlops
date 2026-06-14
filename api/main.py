"""
API predictiva de churn — interfaz entre el cliente y el modelo entrenado.

Endpoints:
  GET  /        — bienvenida
  GET  /health  — estado del servicio y modelo
  GET  /info    — metadatos del modelo (mejora personal)
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
VERSION_SERVICIO = "1.0.0"
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
    version=VERSION_SERVICIO,
    lifespan=lifespan,
)


@app.get("/")
def read_root() -> dict[str, str]:
    return {
        "mensaje": "API predictiva de churn en ejecución",
        "autor": AUTOR,
        "endpoints": "/health, /info, /predict, /docs",
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


@app.get("/info")
def info_modelo() -> dict[str, Any]:
    """Mejora personal: expone versión, autor, variables y fecha de entrenamiento."""
    return {
        "autor": AUTOR,
        "version_servicio": VERSION_SERVICIO,
        "version_modelo": metadata.get("version", "v1"),
        "algoritmo": metadata.get("algoritmo", "RandomForestClassifier"),
        "variables": metadata.get("variables", FEATURES),
        "variable_objetivo": metadata.get("variable_objetivo", "churn"),
        "entrenado_en": metadata.get("entrenado_en"),
        "archivo_modelo": MODEL_PATH.name,
        "metricas": metadata.get("metricas", {}),
    }


def _clasificar_riesgo(probabilidad: float) -> str:
    return "alto_riesgo" if probabilidad >= UMBRAL_RIESGO else "bajo_riesgo"


def _generar_recomendacion(prediccion: str, reclamos: int) -> str:
    if prediccion == "alto_riesgo":
        if reclamos >= 3:
            return "Contactar al cliente con una oferta de retención y revisar reclamos abiertos."
        return "Monitorear de cerca y ofrecer un plan de fidelización personalizado."
    return "Mantener seguimiento rutinario; el riesgo de abandono es bajo."


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
    prediccion = _clasificar_riesgo(probabilidad)

    return PredictResponse(
        prediccion=prediccion,
        probabilidad=probabilidad,
        recomendacion=_generar_recomendacion(prediccion, cliente.reclamos),
        autor=AUTOR,
    )
