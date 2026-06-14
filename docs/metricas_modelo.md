# Métricas del modelo de churn

**Autor:** Alex J. Paco Surco  
**Versión:** v1  
**Algoritmo:** RandomForestClassifier  
**Fecha de entrenamiento:** 2026-06-14T22:53:03Z

## Variables de entrada

- `antiguedad` (1–72 meses)
- `cargo_mensual` (18–120 USD)
- `reclamos` (0–10)

## Resultados en conjunto de prueba

| Métrica | Valor |
|---------|-------|
| Accuracy | 0.5667 |
| Precision | 0.5750 |
| Recall | 0.7188 |
| F1-Score | 0.6389 |

## Artefactos generados

- `modelo_churn.pkl` — serialización del laboratorio
- `modelo_churn_v1.joblib` — serialización para la actividad
- `modelo_churn_v1_metadata.json` — metadatos del modelo
