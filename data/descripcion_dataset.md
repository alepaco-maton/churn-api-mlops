# Dataset de predicción de churn

Dataset sintético de clientes de telecomunicaciones para entrenar un modelo de abandono (churn).

## Variables

| Variable | Tipo | Descripción | Rango |
|----------|------|-------------|-------|
| `tenure` | entero | Meses como cliente | 1 – 72 |
| `monthly_charges` | float | Cargo mensual en USD | 18 – 120 |
| `total_charges` | float | Cargo acumulado en USD | 0 – 8000 |
| `senior_citizen` | entero | Indicador de adulto mayor (0/1) | 0 – 1 |
| `num_dependents` | entero | Número de dependientes | 0 – 5 |
| `churn` | entero | Etiqueta objetivo (0 = permanece, 1 = abandona) | 0 – 1 |

## Archivos

- `churn_clientes.csv`: dataset completo.
- `train.csv` y `test.csv`: particiones generadas por el script de entrenamiento.

## Origen

Datos generados sintéticamente con `src/generar_datos.py` para fines académicos del laboratorio MLOps.
