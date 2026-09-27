from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = [
    "Mix Name",
    "Material Source",
    "Cement (kg/m3)",
    "Fly Ash (kg/m3)",
    "Slag (kg/m3)",
    "Water (kg/m3)",
    "HRWR (kg/m3)",
    "Fine Aggregate (kg/m3)",
    "Coarse Aggregates (kg/m3)",
    "Temp (C)",
    "Time",
    "GWP",
    "Strength1 (psi)",
    "Strength2 (psi)",
    "Strength3 (psi)",
    "Strength (Mean)",
    "Strength (Std)",
    "# of measurements",
    "Slump (in)",
]

NUMERIC_COLUMNS = [
    "Material Source",
    "Cement (kg/m3)",
    "Fly Ash (kg/m3)",
    "Slag (kg/m3)",
    "Water (kg/m3)",
    "HRWR (kg/m3)",
    "Fine Aggregate (kg/m3)",
    "Coarse Aggregates (kg/m3)",
    "Temp (C)",
    "Time",
    "GWP",
    "Strength1 (psi)",
    "Strength2 (psi)",
    "Strength3 (psi)",
    "Strength (Mean)",
    "Strength (Std)",
    "# of measurements",
    "Slump (in)",
]

MATERIAL_COLUMNS = [
    "Cement (kg/m3)",
    "Fly Ash (kg/m3)",
    "Slag (kg/m3)",
    "Water (kg/m3)",
    "HRWR (kg/m3)",
    "Fine Aggregate (kg/m3)",
    "Coarse Aggregates (kg/m3)",
]

STRENGTH_REPLICATES = [
    "Strength1 (psi)",
    "Strength2 (psi)",
    "Strength3 (psi)",
]

ALLOWED_TIMES = {1, 3, 5, 14, 28}

# The CSV stores rounded mean/std values.
MEAN_TOLERANCE = 1.0
STD_TOLERANCE = 1.0


class ValidationError(Exception):
    """Raised when the BOxCrete dataset fails validation."""


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise ValidationError(f"Dataset not found: {path}")

    try:
        df = pd.read_csv(path)
    except Exception as exc:
        raise ValidationError(f"Could not read CSV: {exc}") from exc

    return df


def validate_schema(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]

    if missing:
        raise ValidationError(
            f"Missing required columns: {missing}"
        )


def validate_numeric_columns(df: pd.DataFrame) -> None:
    for column in NUMERIC_COLUMNS:
        if not pd.api.types.is_numeric_dtype(df[column]):
            raise ValidationError(
                f"Column '{column}' must be numeric."
            )


def validate_missing_values(df: pd.DataFrame) -> None:
    allowed_missing = {"Slump (in)"}

    for column in REQUIRED_COLUMNS:
        missing_count = int(df[column].isna().sum())

        if missing_count and column not in allowed_missing:
            raise ValidationError(
                f"Unexpected missing values in '{column}': "
                f"{missing_count}"
            )


def validate_basic_constraints(df: pd.DataFrame) -> None:
    if df["Mix Name"].isna().any():
        raise ValidationError("Mix Name contains missing values.")

    if (df["Mix Name"].astype(str).str.strip() == "").any():
        raise ValidationError("Mix Name contains empty values.")

    invalid_times = set(df["Time"].dropna().unique()) - ALLOWED_TIMES

    if invalid_times:
        raise ValidationError(
            f"Invalid Time values: {sorted(invalid_times)}"
        )

    if not df["# of measurements"].eq(3).all():
        raise ValidationError(
            "# of measurements contains values other than 3."
        )

    for column in MATERIAL_COLUMNS + ["GWP"]:
        if (df[column] < 0).any():
            raise ValidationError(
                f"Negative values found in '{column}'."
            )

    for column in STRENGTH_REPLICATES + [
        "Strength (Mean)",
        "Strength (Std)",
    ]:
        if (df[column] < 0).any():
            raise ValidationError(
                f"Negative values found in '{column}'."
            )

    if (df["Strength (Std)"] < 0).any():
        raise ValidationError("Negative strength standard deviation found.")


def validate_mix_consistency(df: pd.DataFrame) -> list[dict]:
    """
    Validate properties that should remain constant within a mix.

    Composition and material source are hard consistency requirements.
    GWP inconsistencies are recorded as data-quality anomalies because
    they may represent rounding or source-data inconsistencies.
    """

    hard_consistency_columns = [
        "Material Source",
        *MATERIAL_COLUMNS,
    ]

    anomalies = []

    for mix_name, group in df.groupby("Mix Name"):

        for column in hard_consistency_columns:
            if group[column].nunique(dropna=False) != 1:
                raise ValidationError(
                    f"Mix '{mix_name}' has inconsistent values "
                    f"for '{column}'."
                )

        if group["GWP"].nunique(dropna=False) != 1:
            anomalies.append(
                {
                    "mix_name": mix_name,
                    "issue": "inconsistent_gwp",
                    "values": sorted(
                        group["GWP"].dropna().unique().tolist()
                    ),
                    "times": sorted(
                        group.loc[group["GWP"].notna(), "Time"]
                        .unique()
                        .tolist()
                    ),
                }
            )

    return anomalies

def validate_strength_statistics(df: pd.DataFrame) -> None:
    calculated_mean = df[STRENGTH_REPLICATES].mean(axis=1)

    calculated_std = df[STRENGTH_REPLICATES].std(
        axis=1,
        ddof=1,
    )

    mean_difference = (
        calculated_mean - df["Strength (Mean)"]
    ).abs()

    std_difference = (
        calculated_std - df["Strength (Std)"]
    ).abs()

    if mean_difference.max() > MEAN_TOLERANCE:
        raise ValidationError(
            "Strength mean consistency check failed. "
            f"Maximum difference: {mean_difference.max():.4f}"
        )

    if std_difference.max() > STD_TOLERANCE:
        raise ValidationError(
            "Strength standard deviation consistency check failed. "
            f"Maximum difference: {std_difference.max():.4f}"
        )


def create_validation_report(df: pd.DataFrame) -> dict:
    slump_missing = int(df["Slump (in)"].isna().sum())

    return {
        "rows": len(df),
        "columns": len(df.columns),
        "unique_mixes": df["Mix Name"].nunique(),
        "material_sources": df["Material Source"].nunique(),
        "slump_observed": int(df["Slump (in)"].notna().sum()),
        "slump_missing": slump_missing,
        "curing_times": sorted(df["Time"].unique().tolist()),
        "duplicate_rows": int(df.duplicated().sum()),
    }


def validate_dataset(
    input_path: Path,
    output_path: Path,
) -> None:
    print("=" * 60)
    print("BOxCrete Dataset Validator")
    print("=" * 60)

    print(f"\nInput: {input_path}")

    df = load_dataset(input_path)

    print(f"Loaded {len(df)} rows.")

    print("\n[1/6] Schema validation...")
    validate_schema(df)
    print("      PASS")

    print("\n[2/6] Numeric type validation...")
    validate_numeric_columns(df)
    print("      PASS")

    print("\n[3/6] Missing-value validation...")
    validate_missing_values(df)
    print("      PASS")

    print("\n[4/6] Basic constraint validation...")
    validate_basic_constraints(df)
    print("      PASS")

    print("\n[5/6] Per-mix consistency validation...")
    anomalies = validate_mix_consistency(df)

    print("      PASS")

    if anomalies:
       print(f"      WARNING: {len(anomalies)} GWP anomaly/anomalies detected.")

       for anomaly in anomalies:
            print(
             f"      - {anomaly['mix_name']}: "
             f"GWP values={anomaly['values']}, "
             f"times={anomaly['times']}"
          )
    else:
            print("      No GWP anomalies detected.")

    print("\n[6/6] Strength statistics validation...")
    validate_strength_statistics(df)
    print("      PASS")

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Important:
    # No imputation or feature transformation happens here.
    # The validated dataset retains Slump NaN values.
    df.to_csv(output_path, index=False)

    report = create_validation_report(df)

    print("\n" + "=" * 60)
    print("VALIDATION SUCCESSFUL")
    print("=" * 60)

    print(f"Rows:              {report['rows']}")
    print(f"Columns:           {report['columns']}")
    print(f"Unique mixes:      {report['unique_mixes']}")
    print(f"Material sources:  {report['material_sources']}")
    print(f"Curing times:      {report['curing_times']}")
    print(f"Slump observed:    {report['slump_observed']}")
    print(f"Slump missing:     {report['slump_missing']}")
    print(f"Duplicate rows:    {report['duplicate_rows']}")

    print(f"\nValidated dataset:")
    print(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate BOxCrete CSV dataset."
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/raw/boxcrete_data.csv"),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/interim/boxcrete_validated.csv"),
    )

    args = parser.parse_args()

    try:
        validate_dataset(
            input_path=args.input,
            output_path=args.output,
        )
    except ValidationError as exc:
        print(f"\nVALIDATION FAILED:\n{exc}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
