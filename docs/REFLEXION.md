## Reflexión final

¿Qué diferencia existe entre una API mínima que solo confirma que el servicio está activo y una API predictiva preparada para ser consumida por otro sistema?

Una API mínima —por ejemplo, una que responde “API funcionando correctamente” dentro de un contenedor Docker— solo verifica que el proceso está en ejecución; no entrega valor de Machine Learning ni puede integrarse con otros sistemas para tomar decisiones. En cambio, una API predictiva preparada para producción carga un modelo serializado, valida los datos de entrada, ejecuta inferencia y devuelve una respuesta estructurada (predicción, probabilidad y recomendación) que otro sistema puede consumir sin conocer el código, el notebook ni el archivo del modelo. Además incorpora contratos claros (esquemas, códigos de error, metadatos y documentación), lo que la hace trazable, reproducible y apta para integrarse en un flujo MLOps real.
