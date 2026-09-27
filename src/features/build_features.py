from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

INPUT_DIR = ROOT / "data" / "processed"
OUTPUT_DIR = ROOT / "data" / "model_ready"


MATERIAL_COLUMNS = [
    "Cement (kg/m3)",
    "Fly Ash (kg/m3)",
    "Slag (kg/m3)",
    "Water (kg/m3)",
    "HRWR (kg/m3)",
    "Fine Aggregate (kg/m3)",
    "Coarse Aggregates (kg/m3)",
]


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add physically meaningful formulation ratios.

    No target-derived features are created here.
    """

    df = df.copy()

    binder = (
        df["Cement (kg/m3)"]
        + df["Fly Ash (kg/m3)"]
        + df["Slag (kg/m3)"]
    )

    total_aggregate = (
        df["Fine Aggregate (kg/m3)"]
        + df["Coarse Aggregates (kg/m3)"]
    )

    df["Binder (kg/m3)"] = binder

    df["Water_Binder_Ratio"] = (
        df["Water (kg/m3)"] / binder
    )

    df["FlyAsh_Binder_Ratio"] = (
        df["Fly Ash (kg/m3)"] / binder
    )

    df["Slag_Binder_Ratio"] = (
        df["Slag (kg/m3)"] / binder
    )

    df["HRWR_Binder_Ratio"] = (
        df["HRWR (kg/m3)"] / binder
    )

    df["FineAgg_Binder_Ratio"] = (
        df["Fine Aggregate (kg/m3)"] / binder
    )

    df["CoarseAgg_Binder_Ratio"] = (
        df["Coarse Aggregates (kg/m3)"] / binder
    )

    df["FineAgg_TotalAgg_Ratio"] = np.where(
        total_aggregate > 0,
        df["Fine Aggregate (kg/m3)"] / total_aggregate,
        np.nan,
    )

    df["CoarseAgg_TotalAgg_Ratio"] = np.where(
        total_aggregate > 0,
        df["Coarse Aggregates (kg/m3)"] / total_aggregate,
        np.nan,
    )

    return df


STRENGTH_FEATURES = [
    "Cement (kg/m3)",
    "Fly Ash (kg/m3)",
    "Slag (kg/m3)",
    "Water (kg/m3)",
    "HRWR (kg/m3)",
    "Fine Aggregate (kg/m3)",
    "Coarse Aggregates (kg/m3)",
    "Temp (C)",
    "Time",
    "Material Source",
    "Water_Binder_Ratio",
    "FlyAsh_Binder_Ratio",
    "Slag_Binder_Ratio",
    "HRWR_Binder_Ratio",
    "FineAgg_Binder_Ratio",
    "CoarseAgg_Binder_Ratio",
    "FineAgg_TotalAgg_Ratio",
    "CoarseAgg_TotalAgg_Ratio",
]

STRENGTH_TARGET = "Strength (Mean)"


GWP_FEATURES = [
    "Cement (kg/m3)",
    "Fly Ash (kg/m3)",
    "Slag (kg/m3)",
    "Water (kg/m3)",
    "HRWR (kg/m3)",
    "Fine Aggregate (kg/m3)",
    "Coarse Aggregates (kg/m3)",
    "Material Source",
    "Water_Binder_Ratio",
    "FlyAsh_Binder_Ratio",
    "Slag_Binder_Ratio",
    "HRWR_Binder_Ratio",
    "FineAgg_Binder_Ratio",
    "CoarseAgg_Binder_Ratio",
]

GWP_TARGET = "GWP"


SLUMP_FEATURES = [
    "Cement (kg/m3)",
    "Fly Ash (kg/m3)",
    "Slag (kg/m3)",
    "Water (kg/m3)",
    "HRWR (kg/m3)",
    "Fine Aggregate (kg/m3)",
    "Temp (C)",
    "Water_Binder_Ratio",
    "FlyAsh_Binder_Ratio",
    "Slag_Binder_Ratio",
    "HRWR_Binder_Ratio",
    "FineAgg_Binder_Ratio",
]

SLUMP_TARGET = "Slump (in)"


def validate_feature_schema(
    df: pd.DataFrame,
    features: list[str],
    target: str,
    model_name: str,
) -> None:

    missing = [
        column
        for column in features + [target]
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{model_name}: missing columns: {missing}"
        )

    overlap = set(features) & {target}

    if overlap:
        raise ValueError(
            f"{model_name}: target appears in features: {overlap}"
        )


def build_model_dataset(
    df: pd.DataFrame,
    features: list[str],
    target: str,
    model_name: str,
) -> pd.DataFrame:

    validate_feature_schema(
        df,
        features,
        target,
        model_name,
    )

    metadata = ["Mix Name"]

    result = df[
        metadata + features + [target]
    ].copy()

    return result


def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for split_name in [
        "train",
        "validation",
        "test",
    ]:

        input_path = INPUT_DIR / f"{split_name}.csv"

        if not input_path.exists():
            raise FileNotFoundError(input_path)

        df = pd.read_csv(input_path)

        df = add_engineered_features(df)

        strength = build_model_dataset(
            df,
            STRENGTH_FEATURES,
            STRENGTH_TARGET,
            "Strength",
        )

        gwp = build_model_dataset(
            df,
            GWP_FEATURES,
            GWP_TARGET,
            "GWP",
        )

        slump = build_model_dataset(
            df,
            SLUMP_FEATURES,
            SLUMP_TARGET,
            "Slump",
        )

        strength.to_csv(
            OUTPUT_DIR / f"strength_{split_name}.csv",
            index=False,
        )

        gwp.to_csv(
            OUTPUT_DIR / f"gwp_{split_name}.csv",
            index=False,
        )

        # Keep only observations with measured slump.
        slump = slump.dropna(subset=[SLUMP_TARGET])

        slump.to_csv(
            OUTPUT_DIR / f"slump_{split_name}.csv",
            index=False,
        )

        print(
            f"{split_name:10s} | "
            f"strength={len(strength):3d} | "
            f"gwp={len(gwp):3d} | "
            f"slump={len(slump):3d}"
        )


if __name__ == "__main__":
    main()