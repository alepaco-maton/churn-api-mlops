# Proyecto Churn MLOps — Paco Surco

API predictiva de abandono de clientes (churn) con FastAPI, entrenamiento reproducible y despliegue local.

## Estructura del proyecto

```
proyecto_churn_mlops/
├── api/              # API FastAPI
├── data/             # Datasets y documentación
├── docs/             # Métricas del modelo (generadas)
├── models/           # Modelo serializado y metadatos
├── notebooks/        # Experimentación
├── src/              # Scripts de entrenamiento
├── tests/            # Pruebas automatizadas
├── requirements.txt
└── README.md
```

## Requisitos

- Python 3.10+
- Entorno virtual recomendado

## Instalación

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

## Uso rápido

```bash
# 1. Generar datos (si no existen)
python src/generar_datos.py

# 2. Entrenar modelo
python src/train.py

# 3. Levantar API
uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
```

## Autor

**Alex J. Paco Surco**

## Repositorio

[proyecto-churn-mlops-surco](https://github.com/alepaco-maton/churn-api-mlops)
