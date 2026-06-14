"""Genera el dataset sintético de churn alineado con el laboratorio."""
from pathlib import Path

import numpy as np
import pandas as pd

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "churn_clientes.csv"


def generar_dataset(n_registros: int = 300, semilla: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(semilla)
    antiguedad = rng.integers(1, 73, size=n_registros)
    cargo_mensual = rng.uniform(18, 120, size=n_registros).round(2)
    reclamos = rng.integers(0, 11, size=n_registros)

    prob_churn = (
        0.12
        + (cargo_mensual / 120) * 0.22
        + (1 - antiguedad / 72) * 0.30
        + (reclamos / 10) * 0.28
    )
    churn = (rng.random(n_registros) < prob_churn).astype(int)

    return pd.DataFrame(
        {
            "antiguedad": antiguedad,
            "cargo_mensual": cargo_mensual,
            "reclamos": reclamos,
            "churn": churn,
        }
    )


if __name__ == "__main__":
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = generar_dataset()
    df.to_csv(DATA_PATH, index=False)
    print(f"Dataset generado: {DATA_PATH} ({len(df)} registros)")
