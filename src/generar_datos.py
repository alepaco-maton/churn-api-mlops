"""Genera el dataset sintético de churn si no existe."""
from pathlib import Path

import numpy as np
import pandas as pd

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "churn_clientes.csv"


def generar_dataset(n_registros: int = 300, semilla: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(semilla)
    tenure = rng.integers(1, 73, size=n_registros)
    monthly_charges = rng.uniform(18, 120, size=n_registros).round(2)
    total_charges = (tenure * monthly_charges * rng.uniform(0.85, 1.15, size=n_registros)).round(2)
    senior_citizen = rng.integers(0, 2, size=n_registros)
    num_dependents = rng.integers(0, 6, size=n_registros)

    prob_churn = (
        0.15
        + (monthly_charges / 120) * 0.25
        + (1 - tenure / 72) * 0.35
        + senior_citizen * 0.05
        - (num_dependents * 0.02)
    )
    churn = (rng.random(n_registros) < prob_churn).astype(int)

    return pd.DataFrame(
        {
            "tenure": tenure,
            "monthly_charges": monthly_charges,
            "total_charges": total_charges,
            "senior_citizen": senior_citizen,
            "num_dependents": num_dependents,
            "churn": churn,
        }
    )


if __name__ == "__main__":
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = generar_dataset()
    df.to_csv(DATA_PATH, index=False)
    print(f"Dataset generado: {DATA_PATH} ({len(df)} registros)")
