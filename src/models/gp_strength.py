from __future__ import annotations

import json
import random
from pathlib import Path

import gpytorch
import mlflow
import numpy as np
import pandas as pd
import torch

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.preprocessing import StandardScaler

from src.features.build_features import (
    STRENGTH_FEATURES,
    STRENGTH_TARGET,
)
from src.mlflow_config import configure_mlflow


# ============================================================================
# Configuration
# ============================================================================

SEED = 42

TRAINING_ITERATIONS = 150
LEARNING_RATE = 0.05

NUM_SOURCES = 3
SOURCE_VALUES = [0, 1, 2]

ROOT = Path(__file__).resolve().parents[2]

TRAIN_PATH = (
    ROOT
    / "data"
    / "model_ready"
    / "strength_train.csv"
)

VALIDATION_PATH = (
    ROOT
    / "data"
    / "model_ready"
    / "strength_validation.csv"
)

ARTIFACT_DIR = (
    ROOT
    / "artifacts"
    / "gp_strength"
)

ARTIFACT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================================
# Reproducibility
# ============================================================================

def set_seed(seed: int = SEED) -> None:
    """Set random seeds for reproducible training."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================================
# Source-aware Gaussian Process
# ============================================================================

class SourceAwareExactGP(
    gpytorch.models.ExactGP
):
    """
    Exact GP for concrete/mortar strength.

    Continuous variables:
        Matern 5/2 kernel

    Material Source:
        Categorical IndexKernel

    Combined covariance:
        Matern 5/2 × IndexKernel
    """

    def __init__(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        likelihood: gpytorch.likelihoods.GaussianLikelihood,
        n_continuous_features: int,
        num_sources: int = NUM_SOURCES,
    ):
        super().__init__(
            train_x,
            train_y,
            likelihood,
        )

        self.mean_module = (
            gpytorch.means.ConstantMean()
        )

        # ------------------------------------------------------------
        # Continuous feature dimensions
        # ------------------------------------------------------------

        continuous_dims = torch.arange(
            n_continuous_features,
            dtype=torch.long,
        )

        # ------------------------------------------------------------
        # Material Source dimension
        #
        # Feature layout:
        #
        # [continuous_1, ..., continuous_n, Material Source]
        # ------------------------------------------------------------

        source_dim = torch.tensor(
            [n_continuous_features],
            dtype=torch.long,
        )

        # ------------------------------------------------------------
        # Continuous kernel
        # ------------------------------------------------------------

        continuous_kernel = (
            gpytorch.kernels.MaternKernel(
                nu=2.5,
                ard_num_dims=n_continuous_features,
                active_dims=continuous_dims,
            )
        )

        # ------------------------------------------------------------
        # Categorical Material Source kernel
        # ------------------------------------------------------------

        source_kernel = (
            gpytorch.kernels.IndexKernel(
                num_tasks=num_sources,
                rank=min(2, num_sources),
                active_dims=source_dim,
            )
        )

        # ------------------------------------------------------------
        # Combined source-aware covariance
        # ------------------------------------------------------------

        self.covar_module = (
            gpytorch.kernels.ScaleKernel(
                continuous_kernel * source_kernel
            )
        )

    def forward(
        self,
        x: torch.Tensor,
    ):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)

        return gpytorch.distributions.MultivariateNormal(
            mean_x,
            covar_x,
        )


# ============================================================================
# Data preparation
# ============================================================================

def load_data():
    """
    Load train and validation datasets.

    IMPORTANT:
        The test set is deliberately never loaded here.
    """

    train = pd.read_csv(TRAIN_PATH)
    validation = pd.read_csv(
        VALIDATION_PATH
    )

    # ------------------------------------------------------------
    # Continuous features
    # ------------------------------------------------------------

    continuous_features = [
        feature
        for feature in STRENGTH_FEATURES
        if feature != "Material Source"
    ]

    # ------------------------------------------------------------
    # Validate source values
    # ------------------------------------------------------------

    train_sources = set(
        train["Material Source"]
        .astype(int)
        .unique()
    )

    validation_sources = set(
        validation["Material Source"]
        .astype(int)
        .unique()
    )

    allowed_sources = set(
        SOURCE_VALUES
    )

    if not train_sources.issubset(
        allowed_sources
    ):
        raise ValueError(
            "Unexpected Material Source "
            f"values in training data: "
            f"{sorted(train_sources)}"
        )

    if not validation_sources.issubset(
        allowed_sources
    ):
        raise ValueError(
            "Unexpected Material Source "
            f"values in validation data: "
            f"{sorted(validation_sources)}"
        )

    # ------------------------------------------------------------
    # Continuous inputs
    # ------------------------------------------------------------

    X_train_cont = (
        train[continuous_features]
        .astype("float64")
    )

    X_val_cont = (
        validation[continuous_features]
        .astype("float64")
    )

    # ------------------------------------------------------------
    # Categorical source
    # ------------------------------------------------------------

    source_train = (
        train["Material Source"]
        .astype(int)
        .to_numpy()
    )

    source_val = (
        validation["Material Source"]
        .astype(int)
        .to_numpy()
    )

    # ------------------------------------------------------------
    # Target
    # ------------------------------------------------------------

    y_train = (
        train[STRENGTH_TARGET]
        .astype("float64")
        .to_numpy()
    )

    y_val = (
        validation[STRENGTH_TARGET]
        .astype("float64")
        .to_numpy()
    )

    # ------------------------------------------------------------
    # Scale only continuous features
    # ------------------------------------------------------------

    x_scaler = StandardScaler()

    X_train_cont = (
        x_scaler.fit_transform(
            X_train_cont
        )
    )

    X_val_cont = (
        x_scaler.transform(
            X_val_cont
        )
    )

    # ------------------------------------------------------------
    # Scale target
    # ------------------------------------------------------------

    y_scaler = StandardScaler()

    y_train_scaled = (
        y_scaler.fit_transform(
            y_train.reshape(-1, 1)
        )
        .ravel()
    )

    # ------------------------------------------------------------
    # Reconstruct feature matrix
    #
    # [continuous features, Material Source]
    # ------------------------------------------------------------

    X_train = np.column_stack(
        [
            X_train_cont,
            source_train,
        ]
    )

    X_val = np.column_stack(
        [
            X_val_cont,
            source_val,
        ]
    )

    return (
        train,
        validation,
        X_train,
        X_val,
        y_train_scaled,
        y_val,
        x_scaler,
        y_scaler,
        continuous_features,
    )


# ============================================================================
# MLflow dataset logging
# ============================================================================

def log_dataset_information(
    train: pd.DataFrame,
    validation: pd.DataFrame,
) -> None:
    """
    Register train/validation datasets in MLflow.

    This makes them visible under the MLflow
    run's Dataset section.
    """

    train_dataset = mlflow.data.from_pandas(
        train,
        source=str(TRAIN_PATH),
        name="boxcrete_strength_train",
        targets=STRENGTH_TARGET,
    )

    validation_dataset = (
        mlflow.data.from_pandas(
            validation,
            source=str(VALIDATION_PATH),
            name="boxcrete_strength_validation",
            targets=STRENGTH_TARGET,
        )
    )

    mlflow.log_input(
        train_dataset,
        context="training",
    )

    mlflow.log_input(
        validation_dataset,
        context="validation",
    )


# ============================================================================
# Training
# ============================================================================

def train_gp(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    n_continuous_features: int,
    device: torch.device,
):
    """Train the source-aware Exact GP."""

    likelihood = (
        gpytorch.likelihoods.GaussianLikelihood()
    )

    model = SourceAwareExactGP(
        train_x,
        train_y,
        likelihood,
        n_continuous_features=n_continuous_features,
        num_sources=NUM_SOURCES,
    )

    model = model.to(device)
    likelihood = likelihood.to(device)

    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    mll = (
        gpytorch.mlls
        .ExactMarginalLogLikelihood(
            likelihood,
            model,
        )
    )

    for iteration in range(
        1,
        TRAINING_ITERATIONS + 1,
    ):

        optimizer.zero_grad()

        output = model(train_x)

        loss = -mll(
            output,
            train_y,
        )

        loss.backward()

        optimizer.step()

        if (
            iteration == 1
            or iteration % 25 == 0
        ):
            print(
                f"Iteration "
                f"{iteration:03d}/"
                f"{TRAINING_ITERATIONS} "
                f"| Loss={loss.item():.6f} "
                f"| Noise="
                f"{likelihood.noise.item():.6f}"
            )

    return model, likelihood


# ============================================================================
# Evaluation
# ============================================================================

def evaluate_gp(
    model,
    likelihood,
    X_val,
    y_val_original,
    y_scaler,
    device,
):
    """Generate validation predictions and uncertainty."""

    model.eval()
    likelihood.eval()

    val_x = torch.tensor(
        X_val,
        dtype=torch.float32,
        device=device,
    )

    with (
        torch.no_grad(),
        gpytorch.settings.fast_pred_var(),
    ):

        prediction = likelihood(
            model(val_x)
        )

        mean_scaled = (
            prediction.mean
            .cpu()
            .numpy()
        )

        std_scaled = (
            prediction.stddev
            .cpu()
            .numpy()
        )

    # ------------------------------------------------------------
    # Convert prediction back to psi
    # ------------------------------------------------------------

    mean = (
        y_scaler.inverse_transform(
            mean_scaled.reshape(-1, 1)
        )
        .ravel()
    )

    target_std = float(
        y_scaler.scale_[0]
    )

    std = (
        std_scaled
        * target_std
    )

    mae = mean_absolute_error(
        y_val_original,
        mean,
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_val_original,
            mean,
        )
    )

    r2 = r2_score(
        y_val_original,
        mean,
    )

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "mean": mean,
        "std": std,
    }


# ============================================================================
# GP diagnostics
# ============================================================================

def calculate_diagnostics(
    validation: pd.DataFrame,
    y_val: np.ndarray,
    results: dict,
) -> dict:
    """
    Calculate uncertainty and per-source diagnostics.
    """

    predicted = results["mean"]
    predictive_std = results["std"]

    sources = (
        validation["Material Source"]
        .astype(int)
        .to_numpy()
    )

    # ------------------------------------------------------------
    # 95% Gaussian predictive interval
    # ------------------------------------------------------------

    lower_95 = (
        predicted
        - 1.96 * predictive_std
    )

    upper_95 = (
        predicted
        + 1.96 * predictive_std
    )

    coverage_95 = np.mean(
        (y_val >= lower_95)
        & (y_val <= upper_95)
    )

    # ------------------------------------------------------------
    # Absolute error
    # ------------------------------------------------------------

    absolute_error = np.abs(
        y_val - predicted
    )

    # ------------------------------------------------------------
    # Overall uncertainty diagnostics
    # ------------------------------------------------------------

    diagnostics = {
        "mean_predictive_std": float(
            np.mean(predictive_std)
        ),
        "median_predictive_std": float(
            np.median(predictive_std)
        ),
        "min_predictive_std": float(
            np.min(predictive_std)
        ),
        "max_predictive_std": float(
            np.max(predictive_std)
        ),
        "mean_absolute_error": float(
            np.mean(absolute_error)
        ),
        "95_interval_coverage": float(
            coverage_95
        ),
        "lower_95": lower_95,
        "upper_95": upper_95,
        "absolute_error": absolute_error,
        "sources": sources,
    }

    # ------------------------------------------------------------
    # Per-source metrics
    # ------------------------------------------------------------

    source_metrics = {}

    for source in SOURCE_VALUES:

        mask = (
            sources == source
        )

        if not np.any(mask):
            continue

        source_mae = (
            mean_absolute_error(
                y_val[mask],
                predicted[mask],
            )
        )

        source_rmse = np.sqrt(
            mean_squared_error(
                y_val[mask],
                predicted[mask],
            )
        )

        # R² requires at least two samples
        if mask.sum() >= 2:
            source_r2 = r2_score(
                y_val[mask],
                predicted[mask],
            )
        else:
            source_r2 = float("nan")

        source_metrics[source] = {
            "n": int(mask.sum()),
            "mae": float(source_mae),
            "rmse": float(source_rmse),
            "r2": float(source_r2),
            "mean_predictive_std": float(
                predictive_std[mask].mean()
            ),
            "95_interval_coverage": float(
                np.mean(
                    (y_val[mask] >= lower_95[mask])
                    & (
                        y_val[mask]
                        <= upper_95[mask]
                    )
                )
            ),
        }

    diagnostics[
        "source_metrics"
    ] = source_metrics

    return diagnostics


# ============================================================================
# Source kernel diagnostics
# ============================================================================

def extract_source_covariance(
    model,
) -> np.ndarray:
    """
    Extract the learned Material Source covariance
    from the GPyTorch IndexKernel.

    GPyTorch/LinearOperator versions may return a
    LinearOperator rather than a dense Tensor, so
    to_dense() is used here.
    """

    product_kernel = (
        model.covar_module.base_kernel
    )

    source_kernel = (
        product_kernel.kernels[1]
    )

    covariance_operator = (
        source_kernel.covar_matrix
    )

    covariance_tensor = (
        covariance_operator
        .to_dense()
        .detach()
        .cpu()
    )

    return covariance_tensor.numpy()

# ============================================================================
# Artifact generation
# ============================================================================

def save_artifacts(
    model,
    likelihood,
    x_scaler,
    y_scaler,
    continuous_features,
    diagnostics,
    train,
    validation,
    y_val,
) -> dict:
    """Save reproducible model and diagnostic artifacts."""

    # ------------------------------------------------------------
    # Model checkpoint
    # ------------------------------------------------------------

    model_path = (
        ARTIFACT_DIR
        / "model_state.pt"
    )

    likelihood_path = (
        ARTIFACT_DIR
        / "likelihood_state.pt"
    )

    torch.save(
        model.state_dict(),
        model_path,
    )

    torch.save(
        likelihood.state_dict(),
        likelihood_path,
    )

    # ------------------------------------------------------------
    # Scalers
    # ------------------------------------------------------------

    scaler_path = (
        ARTIFACT_DIR
        / "scalers.npz"
    )

    np.savez(
        scaler_path,
        x_mean=x_scaler.mean_,
        x_scale=x_scaler.scale_,
        y_mean=y_scaler.mean_,
        y_scale=y_scaler.scale_,
    )

    # ------------------------------------------------------------
    # Feature/model configuration
    # ------------------------------------------------------------

    config = {
        "model": "source_aware_exact_gp",
        "kernel": "Matern52 * IndexKernel",
        "matern_nu": 2.5,
        "index_kernel_rank": 2,
        "num_sources": NUM_SOURCES,
        "source_values": SOURCE_VALUES,
        "categorical_feature": "Material Source",
        "continuous_features": continuous_features,
        "all_features": STRENGTH_FEATURES,
        "target": STRENGTH_TARGET,
        "training_iterations": TRAINING_ITERATIONS,
        "learning_rate": LEARNING_RATE,
        "seed": SEED,
    }

    config_path = (
        ARTIFACT_DIR
        / "model_config.json"
    )

    with open(
        config_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            config,
            file,
            indent=2,
        )

    # ------------------------------------------------------------
    # Source covariance
    # ------------------------------------------------------------

    source_covariance = (
        extract_source_covariance(
            model
        )
    )

    covariance_path = (
        ARTIFACT_DIR
        / "source_covariance.csv"
    )

    covariance_df = pd.DataFrame(
        source_covariance,
        index=[
            f"source_{s}"
            for s in SOURCE_VALUES
        ],
        columns=[
            f"source_{s}"
            for s in SOURCE_VALUES
        ],
    )

    covariance_df.to_csv(
        covariance_path
    )

    # ------------------------------------------------------------
    # Validation predictions
    # ------------------------------------------------------------

    prediction_df = pd.DataFrame(
        {
            "Mix Name": validation[
                "Mix Name"
            ].to_numpy(),

            "Time": validation[
                "Time"
            ].to_numpy(),

            "Material Source": validation[
                "Material Source"
            ].to_numpy(),

            "true_strength_psi": y_val,

            "predicted_strength_psi": (
                diagnostics[
                    "predicted"
                ]
            ),

            "predictive_std_psi": (
                diagnostics[
                    "predictive_std"
                ]
            ),

            "lower_95_psi": (
                diagnostics[
                    "lower_95"
                ]
            ),

            "upper_95_psi": (
                diagnostics[
                    "upper_95"
                ]
            ),

            "absolute_error_psi": (
                diagnostics[
                    "absolute_error"
                ]
            ),
        }
    )

    predictions_path = (
        ARTIFACT_DIR
        / "validation_predictions.csv"
    )

    prediction_df.to_csv(
        predictions_path,
        index=False,
    )

    # ------------------------------------------------------------
    # Dataset/model manifest
    # ------------------------------------------------------------

    manifest = {
        "dataset": {
            "train_path": str(TRAIN_PATH),
            "validation_path": str(
                VALIDATION_PATH
            ),
            "train_rows": int(len(train)),
            "validation_rows": int(len(validation)),
            "test_used": False,
        },
        "model": config,
        "artifacts": [
            model_path.name,
            likelihood_path.name,
            scaler_path.name,
            config_path.name,
            covariance_path.name,
            predictions_path.name,
        ],
    }

    manifest_path = (
        ARTIFACT_DIR
        / "run_manifest.json"
    )

    with open(
        manifest_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            manifest,
            file,
            indent=2,
        )

    return {
        "model": model_path,
        "likelihood": likelihood_path,
        "scalers": scaler_path,
        "config": config_path,
        "source_covariance": covariance_path,
        "predictions": predictions_path,
        "manifest": manifest_path,
    }


# ============================================================================
# Main
# ============================================================================

def main():

    print("=" * 70)
    print(
        "BOxCrete Source-Aware Gaussian Process — Strength"
    )
    print("=" * 70)

    set_seed()

    # ------------------------------------------------------------
    # Device
    # ------------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    # ------------------------------------------------------------
    # MLflow
    # ------------------------------------------------------------

    tracking_uri = (
        configure_mlflow()
    )

    print(
        f"Device: {device}"
    )

    print(
        f"MLflow tracking: "
        f"{tracking_uri}"
    )

    # ------------------------------------------------------------
    # Data
    # ------------------------------------------------------------

    (
        train,
        validation,
        X_train,
        X_val,
        y_train,
        y_val,
        x_scaler,
        y_scaler,
        continuous_features,
    ) = load_data()

    n_train = len(train)
    n_validation = len(
        validation
    )

    n_continuous_features = len(
        continuous_features
    )

    print(
        f"Training rows:   "
        f"{n_train}"
    )

    print(
        f"Validation rows: "
        f"{n_validation}"
    )

    print(
        f"Continuous features: "
        f"{n_continuous_features}"
    )

    print(
        "Categorical feature: "
        "Material Source"
    )

    print(
        "Source categories: "
        "[0, 1, 2]"
    )

    print(
        f"Target: "
        f"{STRENGTH_TARGET}"
    )

    # ------------------------------------------------------------
    # Tensor conversion
    # ------------------------------------------------------------

    train_x = torch.tensor(
        X_train,
        dtype=torch.float32,
        device=device,
    )

    train_y = torch.tensor(
        y_train,
        dtype=torch.float32,
        device=device,
    )

    # ------------------------------------------------------------
    # MLflow run
    # ------------------------------------------------------------

    with mlflow.start_run(
        run_name="strength_source_aware_gp"
    ):

        # ========================================================
        # Tags
        # ========================================================

        mlflow.set_tags(
            {
                "project": "BOxCrete",
                "task": "strength_prediction",
                "model": "source_aware_exact_gp",
                "dataset_split": (
                    "mix_grouped_70_15_15"
                ),
                "categorical_feature": (
                    "Material Source"
                ),
                "target": STRENGTH_TARGET,
                "random_state": str(SEED),
                "test_set_used": "false",
                "framework": "GPyTorch",
            }
        )

        # ========================================================
        # Dataset information
        # ========================================================

        train_mix_count = (
            train["Mix Name"]
            .nunique()
        )

        validation_mix_count = (
            validation["Mix Name"]
            .nunique()
        )

        train_source_counts = (
            train["Material Source"]
            .value_counts()
            .sort_index()
            .to_dict()
        )

        validation_source_counts = (
            validation[
                "Material Source"
            ]
            .value_counts()
            .sort_index()
            .to_dict()
        )

        # ========================================================
        # Parameters
        # ========================================================

        mlflow.log_params(
            {
                "training_rows": n_train,
                "validation_rows": (
                    n_validation
                ),
                "training_mixes": (
                    train_mix_count
                ),
                "validation_mixes": (
                    validation_mix_count
                ),
                "n_features": len(
                    STRENGTH_FEATURES
                ),
                "n_continuous_features": (
                    n_continuous_features
                ),
                "categorical_features": (
                    "Material Source"
                ),
                "target": STRENGTH_TARGET,
                "num_sources": NUM_SOURCES,
                "source_values": str(
                    SOURCE_VALUES
                ),
                "kernel": (
                    "Matern52 * IndexKernel"
                ),
                "matern_nu": 2.5,
                "index_kernel_rank": 2,
                "training_iterations": (
                    TRAINING_ITERATIONS
                ),
                "learning_rate": (
                    LEARNING_RATE
                ),
                "device": str(device),
                "seed": SEED,
            }
        )

        # Source distributions
        mlflow.log_params(
            {
                "train_source_0": (
                    train_source_counts.get(
                        0,
                        0,
                    )
                ),
                "train_source_1": (
                    train_source_counts.get(
                        1,
                        0,
                    )
                ),
                "train_source_2": (
                    train_source_counts.get(
                        2,
                        0,
                    )
                ),
                "validation_source_0": (
                    validation_source_counts.get(
                        0,
                        0,
                    )
                ),
                "validation_source_1": (
                    validation_source_counts.get(
                        1,
                        0,
                    )
                ),
                "validation_source_2": (
                    validation_source_counts.get(
                        2,
                        0,
                    )
                ),
            }
        )

        # ========================================================
        # Dataset registration in MLflow
        # ========================================================

        log_dataset_information(
            train,
            validation,
        )

        # ========================================================
        # Train GP
        # ========================================================

        model, likelihood = train_gp(
            train_x,
            train_y,
            n_continuous_features,
            device,
        )

        # ========================================================
        # Validation prediction
        # ========================================================

        results = evaluate_gp(
            model,
            likelihood,
            X_val,
            y_val,
            y_scaler,
            device,
        )

        # ========================================================
        # Diagnostics
        # ========================================================

        diagnostics = (
            calculate_diagnostics(
                validation,
                y_val,
                results,
            )
        )

        # Add arrays needed by artifact writer
        diagnostics[
            "predicted"
        ] = results["mean"]

        diagnostics[
            "predictive_std"
        ] = results["std"]

        # ========================================================
        # Console metrics
        # ========================================================

        print()

        print(
            f"Validation MAE  = "
            f"{results['mae']:.2f} psi"
        )

        print(
            f"Validation RMSE = "
            f"{results['rmse']:.2f} psi"
        )

        print(
            f"Validation R²   = "
            f"{results['r2']:.4f}"
        )

        print()

        print(
            "Predictive uncertainty"
        )

        print(
            "----------------------"
        )

        print(
            f"Mean predictive std   = "
            f"{diagnostics['mean_predictive_std']:.2f} psi"
        )

        print(
            f"Median predictive std = "
            f"{diagnostics['median_predictive_std']:.2f} psi"
        )

        print(
            f"Min predictive std    = "
            f"{diagnostics['min_predictive_std']:.2f} psi"
        )

        print(
            f"Max predictive std    = "
            f"{diagnostics['max_predictive_std']:.2f} psi"
        )

        print(
            f"95% interval coverage = "
            f"{diagnostics['95_interval_coverage']:.4f}"
        )

        # ========================================================
        # Per-source metrics
        # ========================================================

        print()

        print(
            "Per-source validation performance"
        )

        print(
            "---------------------------------"
        )

        for (
            source,
            metrics,
        ) in diagnostics[
            "source_metrics"
        ].items():

            print(
                f"Source {source} | "
                f"N={metrics['n']:2d} | "
                f"MAE={metrics['mae']:.2f} | "
                f"RMSE={metrics['rmse']:.2f} | "
                f"R²={metrics['r2']:.4f} | "
                f"Mean σ={metrics['mean_predictive_std']:.2f} | "
                f"Coverage={metrics['95_interval_coverage']:.4f}"
            )

        # ========================================================
        # MLflow metrics
        # ========================================================

        mlflow.log_metrics(
            {
                "val_mae": results["mae"],
                "val_rmse": results["rmse"],
                "val_r2": results["r2"],
                "val_mean_predictive_std": (
                    diagnostics[
                        "mean_predictive_std"
                    ]
                ),
                "val_median_predictive_std": (
                    diagnostics[
                        "median_predictive_std"
                    ]
                ),
                "val_min_predictive_std": (
                    diagnostics[
                        "min_predictive_std"
                    ]
                ),
                "val_max_predictive_std": (
                    diagnostics[
                        "max_predictive_std"
                    ]
                ),
                "val_95_interval_coverage": (
                    diagnostics[
                        "95_interval_coverage"
                    ]
                ),
            }
        )

        # Per-source MLflow metrics
        for (
            source,
            metrics,
        ) in diagnostics[
            "source_metrics"
        ].items():

            mlflow.log_metrics(
                {
                    f"source_{source}_mae": (
                        metrics["mae"]
                    ),
                    f"source_{source}_rmse": (
                        metrics["rmse"]
                    ),
                    f"source_{source}_r2": (
                        metrics["r2"]
                    ),
                    f"source_{source}_mean_std": (
                        metrics[
                            "mean_predictive_std"
                        ]
                    ),
                    f"source_{source}_coverage": (
                        metrics[
                            "95_interval_coverage"
                        ]
                    ),
                }
            )

        # ========================================================
        # Save artifacts
        # ========================================================

        artifacts = save_artifacts(
            model=model,
            likelihood=likelihood,
            x_scaler=x_scaler,
            y_scaler=y_scaler,
            continuous_features=(
                continuous_features
            ),
            diagnostics=diagnostics,
            train=train,
            validation=validation,
            y_val=y_val,
        )

        # ========================================================
        # Log artifacts
        # ========================================================

        for (
            artifact_name,
            artifact_path,
        ) in artifacts.items():

            mlflow.log_artifact(
                str(artifact_path),
                artifact_path=(
                    "gp_strength"
                ),
            )

        # ========================================================
        # Print source covariance
        # ========================================================

        source_covariance = (
            extract_source_covariance(
                model
            )
        )

        print()

        print(
            "Learned Material Source covariance"
        )

        print(
            "-----------------------------------"
        )

        print(
            pd.DataFrame(
                source_covariance,
                index=[
                    f"source_{s}"
                    for s in SOURCE_VALUES
                ],
                columns=[
                    f"source_{s}"
                    for s in SOURCE_VALUES
                ],
            ).to_string()
        )

    # ------------------------------------------------------------
    # Completion
    # ------------------------------------------------------------

    print()

    print("=" * 70)

    print(
        "GP training and MLflow logging complete"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()