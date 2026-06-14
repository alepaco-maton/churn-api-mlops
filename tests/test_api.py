"""Pruebas de la API predictiva de churn."""
import pytest
from fastapi.testclient import TestClient

from api.main import app


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_root_incluye_autor(client):
    respuesta = client.get("/")
    assert respuesta.status_code == 200
    assert respuesta.json()["autor"] == "Alex J. Paco Surco"


def test_health_modelo_cargado(client):
    respuesta = client.get("/health")
    assert respuesta.status_code == 200
    cuerpo = respuesta.json()
    assert cuerpo["status"] == "ok"
    assert cuerpo["modelo_cargado"] is True


def test_predict_solicitud_valida(client):
    respuesta = client.post(
        "/predict",
        json={"antiguedad": 12, "cargo_mensual": 95.5, "reclamos": 3},
    )
    assert respuesta.status_code == 200
    cuerpo = respuesta.json()
    assert cuerpo["prediccion"] in ("alto_riesgo", "bajo_riesgo")
    assert 0.0 <= cuerpo["probabilidad"] <= 1.0
    assert cuerpo["autor"] == "Alex J. Paco Surco"
    assert "recomendacion" in cuerpo
    assert len(cuerpo["recomendacion"]) > 0


def test_info_endpoint(client):
    respuesta = client.get("/info")
    assert respuesta.status_code == 200
    cuerpo = respuesta.json()
    assert cuerpo["autor"] == "Alex J. Paco Surco"
    assert cuerpo["variables"] == ["antiguedad", "cargo_mensual", "reclamos"]
    assert "version_modelo" in cuerpo
    assert "entrenado_en" in cuerpo


def test_predict_campo_faltante(client):
    respuesta = client.post("/predict", json={"antiguedad": 12, "reclamos": 3})
    assert respuesta.status_code == 422


def test_predict_tipo_incorrecto(client):
    respuesta = client.post(
        "/predict",
        json={"antiguedad": "doce", "cargo_mensual": 95.5, "reclamos": 3},
    )
    assert respuesta.status_code == 422


def test_predict_valor_fuera_de_rango(client):
    respuesta = client.post(
        "/predict",
        json={"antiguedad": 12, "cargo_mensual": 95.5, "reclamos": -3},
    )
    assert respuesta.status_code == 422


def test_predict_valor_incoherente(client):
    respuesta = client.post(
        "/predict",
        json={"antiguedad": 3, "cargo_mensual": 95.5, "reclamos": 5},
    )
    assert respuesta.status_code == 422
