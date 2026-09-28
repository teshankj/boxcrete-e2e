from pathlib import Path

import numpy as np
import pandas as pd
import torch

import json
import mlflow

from src.mlflow_config import configure_mlflow

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

ARTIFACT_DIR = ROOT / "artifacts" / "strength_v2"

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

    # ------------------------------------------------------------
    # MLflow
    # ------------------------------------------------------------

    tracking_uri = configure_mlflow()

    print(
        f"MLflow tracking: {tracking_uri}"
    )

    # ------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------

    ARTIFACT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ------------------------------------------------------------
    # Data
    # ------------------------------------------------------------

    train_df = pd.read_csv(
        TRAIN_PATH
    )

    val_df = pd.read_csv(
        VAL_PATH
    )

    X_train, y_train = dataframe_to_tensors(
        train_df
    )

    X_val, y_val = dataframe_to_tensors(
        val_df
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

    # ------------------------------------------------------------
    # MLflow Run
    # ------------------------------------------------------------

    with mlflow.start_run(
        run_name="strength_v2_block_loo"
    ):

        # --------------------------------------------------------
        # Tags
        # --------------------------------------------------------

        mlflow.set_tags(
            {
                "project": "BOxCrete",
                "task": "strength_prediction",
                "model": "strength_gp_v2",
                "architecture": (
                    "Multi-Matern+B''+F5_alllog+"
                    "gated_t+gated_noise"
                ),
                "training_objective": (
                    "block_loo_mll_combined"
                ),
                "dataset_split": (
                    "mix_grouped_70_15_15"
                ),
                "target_unit": "psi",
                "device": str(device),
            }
        )

        # --------------------------------------------------------
        # Parameters
        # --------------------------------------------------------

        mlflow.log_params(
            {
                "seed": 42,

                "training_rows": len(X_train),
                "validation_rows": len(X_val),

                "input_dimensions": X_train.shape[-1],

                "mll_weight": 0.5,
                "block_loo_max_iter": 150,
                "block_loo_lr": 0.1,

                "mll_warmup": False,

                "gate_tau": 0.1,

                "engineered_features": 7,

                "dtype": "float64",
            }
        )

        # --------------------------------------------------------
        # Training
        # --------------------------------------------------------

        print()
        print("Training V2...")
        print()

        model, likelihood, loo_loss = (
            fit_strength_v2(
                X=X_train,
                Y=y_train,
                seed=42,

                mll_weight=0.5,
                block_loo_max_iter=150,
                block_loo_lr=0.1,

                mll_warmup=False,
            )
        )

        print()
        print(
            f"Final block-LOO loss: "
            f"{loo_loss:.6f}"
        )

        mlflow.log_metric(
            "block_loo_loss",
            float(loo_loss),
        )

        # --------------------------------------------------------
        # Validation
        # --------------------------------------------------------

        mean, std = predict_strength(
            model,
            X_val,
        )

        mean_np = (
            mean.detach()
            .cpu()
            .numpy()
            .ravel()
        )

        std_np = (
            std.detach()
            .cpu()
            .numpy()
            .ravel()
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

        mean_sigma = float(
            std_np.mean()
        )

        median_sigma = float(
            np.median(std_np)
        )

        # --------------------------------------------------------
        # Print
        # --------------------------------------------------------

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
            f"{mean_sigma:.2f} psi"
        )

        print(
            f"Median predictive σ : "
            f"{median_sigma:.2f} psi"
        )

        print(
            f"95% coverage      : "
            f"{coverage:.4f}"
        )

        # --------------------------------------------------------
        # MLflow metrics
        # --------------------------------------------------------

        mlflow.log_metrics(
            {
                "val_mae_psi": float(mae),
                "val_rmse_psi": float(rmse),
                "val_r2": float(r2),
                "val_mean_predictive_sigma_psi": (
                    mean_sigma
                ),
                "val_median_predictive_sigma_psi": (
                    median_sigma
                ),
                "val_95_coverage": float(
                    coverage
                ),
            }
        )

        # --------------------------------------------------------
        # Save model
        # --------------------------------------------------------

        model_path = (
            ARTIFACT_DIR
            / "model_state.pt"
        )

        torch.save(
            model.state_dict(),
            model_path,
        )

        # --------------------------------------------------------
        # Validation predictions
        # --------------------------------------------------------

        predictions_df = pd.DataFrame(
            {
                "y_true_psi": y_np,
                "y_pred_psi": mean_np,
                "predictive_std_psi": std_np,
                "lower_95_psi": lower,
                "upper_95_psi": upper,
            }
        )

        predictions_path = (
            ARTIFACT_DIR
            / "validation_predictions.csv"
        )

        predictions_df.to_csv(
            predictions_path,
            index=False,
        )

        # --------------------------------------------------------
        # Metadata
        # --------------------------------------------------------

        metadata = {
            "model": "BOxCrete Strength GP V2",
            "variant": (
                "B''+F5_alllog+gated_t+"
                "gated_noise+maxscale_zeromean"
            ),
            "seed": 42,
            "training_rows": len(X_train),
            "validation_rows": len(X_val),
            "input_dimensions": 10,
            "target": "Strength (psi)",

            "objective": {
                "mll_weight": 0.5,
                "block_loo_max_iter": 150,
                "block_loo_lr": 0.1,
                "mll_warmup": False,
            },

            "metrics": {
                "block_loo_loss": float(
                    loo_loss
                ),
                "mae_psi": float(mae),
                "rmse_psi": float(rmse),
                "r2": float(r2),
                "mean_predictive_sigma_psi": (
                    mean_sigma
                ),
                "median_predictive_sigma_psi": (
                    median_sigma
                ),
                "coverage_95": float(
                    coverage
                ),
            },

            "raw_features": RAW_FEATURES,
        }

        metadata_path = (
            ARTIFACT_DIR
            / "metadata.json"
        )

        with open(
            metadata_path,
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                metadata,
                f,
                indent=2,
            )

        # --------------------------------------------------------
        # MLflow artifacts
        # --------------------------------------------------------

        mlflow.log_artifact(
            str(model_path),
            artifact_path="strength_v2",
        )

        mlflow.log_artifact(
            str(predictions_path),
            artifact_path="strength_v2",
        )

        mlflow.log_artifact(
            str(metadata_path),
            artifact_path="strength_v2",
        )

    print()
    print("=" * 80)
    print("V2 training + MLflow logging complete")
    print("=" * 80)

    print()
    print(
        f"Model artifact:"
        f"\n  {model_path}"
    )

    print(
        f"\nMLflow tracking:"
        f"\n  {tracking_uri}"
    )

if __name__ == "__main__":
    main()