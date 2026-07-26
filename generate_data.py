"""Generate a reproducible, fully synthetic AML-like educational dataset."""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "synthetic_aml.csv"


def generate_synthetic_data(n_rows: int = 1200, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    age = np.clip(rng.normal(55, 18, n_rows), 18, 90).round().astype(int)
    wbc = np.clip(rng.lognormal(2.7, 0.9, n_rows), 0.4, 250).round(1)
    hemoglobin = np.clip(rng.normal(9.5, 1.8, n_rows), 4.5, 15.5).round(1)
    platelets = np.clip(rng.lognormal(4.2, 0.7, n_rows), 5, 500).round().astype(int)
    marrow_blasts = np.clip(rng.beta(2.2, 2.5, n_rows) * 85 + 20, 20, 99).round(1)

    complex_karyotype = rng.binomial(1, 0.18, n_rows)
    monosomal_karyotype = rng.binomial(
        1, np.clip(0.08 + 0.32 * complex_karyotype, 0, 0.75)
    )
    tp53_abnormality = rng.binomial(
        1,
        np.clip(
            0.035 + 0.20 * complex_karyotype + 0.12 * monosomal_karyotype,
            0,
            0.70,
        ),
    )
    flt3_itd = rng.binomial(1, 0.22, n_rows)
    npm1_mutation = rng.binomial(1, 0.28, n_rows)

    # Transparent simulation—not a clinical risk equation.
    logit = (
        -2.4
        + 0.025 * (age - 50)
        + 1.55 * complex_karyotype
        + 1.30 * monosomal_karyotype
        + 1.75 * tp53_abnormality
        + 0.65 * flt3_itd
        - 0.70 * npm1_mutation
        + 0.007 * (marrow_blasts - 45)
        + 0.003 * (wbc - 25)
    )
    probability = 1 / (1 + np.exp(-logit))
    adverse_risk = rng.binomial(1, probability)

    return pd.DataFrame(
        {
            "age": age,
            "wbc_10e9_l": wbc,
            "hemoglobin_g_dl": hemoglobin,
            "platelets_10e9_l": platelets,
            "marrow_blast_percent": marrow_blasts,
            "complex_karyotype": complex_karyotype,
            "monosomal_karyotype": monosomal_karyotype,
            "tp53_abnormality": tp53_abnormality,
            "flt3_itd": flt3_itd,
            "npm1_mutation": npm1_mutation,
            "adverse_risk": adverse_risk,
        }
    )


def main() -> None:
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    data = generate_synthetic_data()
    data.to_csv(DATA_PATH, index=False)
    print(f"Created {len(data):,} synthetic records at {DATA_PATH}")


if __name__ == "__main__":
    main()

