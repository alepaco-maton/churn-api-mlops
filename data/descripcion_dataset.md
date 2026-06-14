# Dataset de predicción de churn

Dataset sintético de clientes para entrenar un modelo de abandono (churn), alineado con el laboratorio MLOps.

## Variables de entrada (usadas por la API)

| Variable | Tipo | Descripción | Rango |
|----------|------|-------------|-------|
| `antiguedad` | entero | Meses como cliente | 1 – 72 |
| `cargo_mensual` | float | Cargo mensual en USD | 18 – 120 |
| `reclamos` | entero | Número de reclamos registrados | 0 – 10 |

## Variable objetivo

| Variable | Tipo | Descripción |
|----------|------|-------------|
| `churn` | entero | 0 = permanece, 1 = abandona |

## Archivos

- `churn_clientes.csv`: dataset completo.
- `train.csv` y `test.csv`: particiones generadas por `src/train.py`.

## Ejemplo de solicitud a la API

```json
{
  "antiguedad": 12,
  "cargo_mensual": 95.5,
  "reclamos": 3
}
```

## Origen

Datos generados sintéticamente con `src/generar_datos.py` para fines académicos.
