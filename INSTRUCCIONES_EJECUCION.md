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

- `models/modelo_churn.pkl` (persistencia del laboratorio)
- `models/modelo_churn_v1.joblib` (actividad)
- `models/modelo_churn_v1_metadata.json`
- `docs/metricas_modelo.md`
- `data/train.csv` y `data/test.csv`

## 4. Ejecutar API

```powershell
uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
```

## 5. Probar endpoints

| Método | Ruta | Descripción |
|--------|------|-------------|
| GET | `/` | Mensaje de bienvenida |
| GET | `/health` | Estado del servicio y modelo |
| GET | `/info` | Mejora personal: versión, autor, variables y métricas |
| POST | `/predict` | Predicción de churn con validación y recomendación |

Documentación interactiva: http://127.0.0.1:8000/docs

## 6. Ejemplos de prueba con curl

```powershell
# Raíz
curl http://127.0.0.1:8000/

# Health (evidencia 4 del PDF)
curl http://127.0.0.1:8000/health

# Info — mejora personal (evidencia 8)
curl http://127.0.0.1:8000/info

# Predicción válida (evidencia 6)
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d "{\"antiguedad\": 12, \"cargo_mensual\": 95.5, \"reclamos\": 3}"

# Campo faltante (sin cargo_mensual)
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d "{\"antiguedad\": 12, \"reclamos\": 3}"

# Tipo incorrecto (antiguedad como texto)
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d "{\"antiguedad\": \"doce\", \"cargo_mensual\": 95.5, \"reclamos\": 3}"

# Valor fuera de rango (reclamos negativos)
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d "{\"antiguedad\": 12, \"cargo_mensual\": 95.5, \"reclamos\": -3}"

# Valor incoherente (reclamos > antiguedad)
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d "{\"antiguedad\": 3, \"cargo_mensual\": 95.5, \"reclamos\": 5}"
```
