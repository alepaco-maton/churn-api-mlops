# Instrucciones de ejecución

## 1. Preparar entorno

```powershell
cd proyecto_churn_mlops
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Generar dataset

```powershell
python src/generar_datos.py
```

## 3. Entrenar modelo

```powershell
python src/train.py
```

Verificar que se crearon:

- `models/modelo_churn_v1.joblib`
- `models/modelo_churn_v1_metadata.json`
- `docs/metricas_modelo.md`

## 4. Ejecutar API

```powershell
uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
```

## 5. Probar endpoints

| Método | Ruta | Descripción |
|--------|------|-------------|
| GET | `/` | Mensaje de bienvenida |
| GET | `/health` | Estado del servicio y modelo |
| GET | `/info` | Mejora personal: metadatos del modelo |
| POST | `/predict` | Predicción de churn |

Documentación interactiva: http://127.0.0.1:8000/docs

## 6. Ejemplos de prueba con curl

```powershell
# Raíz
curl http://127.0.0.1:8000/

# Health
curl http://127.0.0.1:8000/health

# Predicción válida
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d "{\"tenure\": 12, \"monthly_charges\": 75.5, \"total_charges\": 900.0, \"senior_citizen\": 0, \"num_dependents\": 2}"

# Campo faltante
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d "{\"tenure\": 12, \"monthly_charges\": 75.5}"

# Tipo incorrecto
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d "{\"tenure\": \"doce\", \"monthly_charges\": 75.5, \"total_charges\": 900.0, \"senior_citizen\": 0, \"num_dependents\": 2}"

# Valor fuera de rango
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d "{\"tenure\": 999, \"monthly_charges\": 75.5, \"total_charges\": 900.0, \"senior_citizen\": 0, \"num_dependents\": 2}"
```
