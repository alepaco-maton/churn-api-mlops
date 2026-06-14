from pydantic import BaseModel, Field, model_validator


class ClienteChurnRequest(BaseModel):
    """Datos de un nuevo cliente para solicitar una predicción."""

    antiguedad: int = Field(
        ...,
        ge=1,
        le=72,
        description="Meses como cliente (1–72)",
        examples=[12],
    )
    cargo_mensual: float = Field(
        ...,
        ge=18.0,
        le=120.0,
        description="Cargo mensual en USD (18–120)",
        examples=[95.5],
    )
    reclamos: int = Field(
        ...,
        ge=0,
        le=10,
        description="Número de reclamos registrados (0–10)",
        examples=[3],
    )

    @model_validator(mode="after")
    def validar_coherencia(self) -> "ClienteChurnRequest":
        if self.reclamos > self.antiguedad:
            raise ValueError(
                "Valor incoherente: los reclamos no pueden superar la antigüedad en meses."
            )
        return self


class PredictResponse(BaseModel):
    prediccion: str = Field(..., description="Nivel de riesgo: alto_riesgo o bajo_riesgo")
    probabilidad: float = Field(..., ge=0.0, le=1.0, description="Probabilidad de abandono")
    recomendacion: str = Field(..., description="Acción sugerida según el nivel de riesgo")
    autor: str
