from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from src.boxcrete.strength_model import (
    fit_strength_v2,
    predict_strength,
)


ROOT = Path(__file__).resolve().parents[2]

TRAIN_PATH = (
    ROOT
    / "data"
    / "processed"
    / "train.csv"
)

VAL_PATH = (
    ROOT
    / "data"
    / "processed"
    / "validation.csv"
)

RAW_FEATURES = [
    "Cement (kg/m3)",
    "Fly Ash (kg/m3)",
    "Slag (kg/m3)",
    "Water (kg/m3)",
    "HRWR (kg/m3)",
    "Fine Aggregate (kg/m3)",
    "Coarse Aggregates (kg/m3)",
    "Material Source",
    "Temp (C)",
    "Time",
]


def dataframe_to_tensors(df):

    X = torch.tensor(
        df[RAW_FEATURES]
        .to_numpy(dtype="float64"),
        dtype=torch.float64,
    )

    y = torch.tensor(
        df["Strength (Mean)"]
        .to_numpy(dtype="float64")
        .reshape(-1, 1),
        dtype=torch.float64,
    )

    return X, y


def main():

    print("=" * 80)
    print("BOxCrete — Research Strength GP V2")
    print("=" * 80)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Device: {device}")

    train_df = pd.read_csv(
        TRAIN_PATH
    )

    val_df = pd.read_csv(
        VAL_PATH
    )

    X_train, y_train = (
        dataframe_to_tensors(
            train_df
        )
    )

    X_val, y_val = (
        dataframe_to_tensors(
            val_df
        )
    )

    X_train = X_train.to(device)
    y_train = y_train.to(device)

    X_val = X_val.to(device)
    y_val = y_val.to(device)

    print(
        f"Training rows:   {len(X_train)}"
    )

    print(
        f"Validation rows: {len(X_val)}"
    )

    print(
        f"Input dimensions: {X_train.shape[-1]}"
    )

    assert X_train.shape[-1] == 10

    print()
    print("Training V2...")
    print()

    model, likelihood, loo_loss = (
        fit_strength_v2(
            X=X_train,
            Y=y_train,
            seed=42,

            # Production objective
            mll_weight=0.5,
            block_loo_max_iter=150,
            block_loo_lr=0.1,

            # Keep False for the actual
            # combined-objective reproduction.
            mll_warmup=False,
        )
    )

    print()
    print(
        f"Final block-LOO loss: "
        f"{loo_loss:.6f}"
    )

    # ------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------

    mean, std = predict_strength(
        model,
        X_val,
    )

    mean_np = (
        mean.detach()
        .cpu()
        .numpy()
    )

    std_np = (
        std.detach()
        .cpu()
        .numpy()
    )

    y_np = (
        y_val.detach()
        .cpu()
        .numpy()
        .ravel()
    )

    mae = mean_absolute_error(
        y_np,
        mean_np,
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_np,
            mean_np,
        )
    )

    r2 = r2_score(
        y_np,
        mean_np,
    )

    lower = (
        mean_np
        - 1.96 * std_np
    )

    upper = (
        mean_np
        + 1.96 * std_np
    )

    coverage = np.mean(
        (y_np >= lower)
        & (y_np <= upper)
    )

    print()
    print("=" * 80)
    print("Validation Results")
    print("=" * 80)

    print(
        f"MAE  : {mae:.2f} psi"
    )

    print(
        f"RMSE : {rmse:.2f} psi"
    )

    print(
        f"R²   : {r2:.4f}"
    )

    print(
        f"Mean predictive σ : "
        f"{std_np.mean():.2f} psi"
    )

    print(
        f"95% coverage      : "
        f"{coverage:.4f}"
    )


if __name__ == "__main__":
    main()