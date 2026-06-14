# Proyecto Churn MLOps — Alex J. Paco Surco

API predictiva de abandono de clientes (churn) con FastAPI. El modelo se entrena offline, se serializa en `models/modelo_churn.pkl` y la API lo carga para **inferencia** en cada solicitud.

## Estructura del proyecto

```
proyecto_churn_mlops/
├── api/
│   ├── main.py          # FastAPI: /, /health, /info, /predict
│   └── schemas.py       # Validación Pydantic
├── data/
│   ├── churn_clientes.csv
│   └── descripcion_dataset.md
├── docs/
│   └── metricas_modelo.md   # Generado por train.py
├── models/                  # Artefactos generados localmente
├── src/
│   ├── generar_datos.py
│   └── train.py
├── tests/
│   └── test_api.py
├── requirements.txt
└── INSTRUCCIONES_EJECUCION.md
```

## Instalación

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Flujo completo

```powershell
python src/generar_datos.py
python src/train.py
uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
```

## Endpoints

| Método | Ruta | Descripción |
|--------|------|-------------|
| GET | `/` | Bienvenida |
| GET | `/health` | Estado del servicio y modelo |
| GET | `/info` | Versión, autor, variables y métricas |
| POST | `/predict` | Predicción con validación |

Documentación interactiva: http://127.0.0.1:8000/docs

## Ejemplo de predicción

**Solicitud:**

```json
{
  "antiguedad": 12,
  "cargo_mensual": 95.5,
  "reclamos": 3
}
```

**Respuesta:**

```json
{
  "prediccion": "alto_riesgo",
  "probabilidad": 0.82,
  "recomendacion": "Contactar al cliente con una oferta de retención y revisar reclamos abiertos.",
  "autor": "Alex J. Paco Surco"
}
```

## Validaciones implementadas

| Caso | Ejemplo | HTTP |
|------|---------|------|
| Datos correctos | `antiguedad: 12` | 200 |
| Campo faltante | sin `cargo_mensual` | 422 |
| Tipo incorrecto | `antiguedad: "doce"` | 422 |
| Valor fuera de rango | `reclamos: -3` | 422 |
| Valor incoherente | `reclamos > antiguedad` | 422 |

## Pruebas automatizadas

```powershell
python -m pytest tests/test_api.py -v
```

## Mejoras personales

1. **Endpoint `/info`**: expone versión del modelo, autor, variables y fecha de entrenamiento.
2. **Recomendación en `/predict`**: incluye una acción sugerida según el nivel de riesgo.
3. **Validación de coherencia**: rechaza solicitudes donde `reclamos > antiguedad`.

## Autor

**Alex J. Paco Surco**

## Repositorio

[proyecto-churn-mlops-surco](https://github.com/alepaco-maton/churn-api-mlops)
