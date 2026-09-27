from pathlib import Path

import mlflow
import mlflow.sklearn

import pandas as pd

from sklearn.ensemble import (
    RandomForestRegressor,
    HistGradientBoostingRegressor,
)
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


from src.mlflow_config import configure_mlflow
from src.features.build_features import (
    STRENGTH_FEATURES,
    STRENGTH_TARGET,
)


ROOT = Path(__file__).resolve().parents[2]

MODEL_READY = ROOT / "data" / "model_ready"

RANDOM_STATE = 42


MODELS = {
    "ridge": Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                Ridge(alpha=1.0),
            ),
        ]
    ),

    "random_forest": RandomForestRegressor(
        n_estimators=500,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    ),

    "hist_gradient_boosting": HistGradientBoostingRegressor(
        max_iter=300,
        learning_rate=0.05,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        random_state=RANDOM_STATE,
    ),
}


def load_strength_data(split):

    path = MODEL_READY / f"strength_{split}.csv"

    return pd.read_csv(path)


def evaluate(model, X, y):

    predictions = model.predict(X)

    mae = mean_absolute_error(
        y,
        predictions,
    )

    rmse = mean_squared_error(
        y,
        predictions,
    ) ** 0.5

    r2 = r2_score(
        y,
        predictions,
    )

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
    }


def run_experiment(model_name, model):

    train = load_strength_data("train")
    validation = load_strength_data("validation")

    X_train = train[STRENGTH_FEATURES].copy()
    X_val = validation[STRENGTH_FEATURES].copy()

    y_train = train[STRENGTH_TARGET]
    y_val = validation[STRENGTH_TARGET]

    # Explicit numeric representation for sklearn baseline models.
    X_train = X_train.astype("float64")
    X_val = X_val.astype("float64")

    with mlflow.start_run(
        run_name=f"strength_{model_name}"
    ):

        mlflow.set_tag(
            "project",
            "BOxCrete",
        )

        mlflow.set_tag(
            "task",
            "strength_prediction",
        )

        mlflow.set_tag(
            "dataset_split",
            "mix_grouped_70_15_15",
        )

        mlflow.set_tag(
            "random_state",
            str(RANDOM_STATE),
        )

        mlflow.log_param(
            "train_rows",
            len(train),
        )

        mlflow.log_param(
            "validation_rows",
            len(validation),
        )

        mlflow.log_param(
            "n_features",
            len(STRENGTH_FEATURES),
        )

        mlflow.log_param(
            "features",
            ",".join(STRENGTH_FEATURES),
        )

        model.fit(
            X_train,
            y_train,
        )

        metrics = evaluate(
            model,
            X_val,
            y_val,
        )

        mlflow.log_metrics(
            {
                "val_mae": metrics["mae"],
                "val_rmse": metrics["rmse"],
                "val_r2": metrics["r2"],
            }
        )

        if hasattr(model, "get_params"):
            params = model.get_params()

            clean_params = {}

            for key, value in params.items():

                if isinstance(
                    value,
                    (
                        str,
                        int,
                        float,
                        bool,
                    ),
                ) or value is None:

                    clean_params[key] = value

            mlflow.log_params(clean_params)

        trusted_types = [
            "sklearn.tree._tree.Tree",
        ] 

        if model_name == "hist_gradient_boosting":
            trusted_types.append(
                "sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor"
            )

        mlflow.sklearn.log_model(
            model,
            name="model",
            skops_trusted_types=trusted_types,
        )

        print(
            f"{model_name:24s} | "
            f"MAE={metrics['mae']:.2f} | "
            f"RMSE={metrics['rmse']:.2f} | "
            f"R²={metrics['r2']:.4f}"
        )

        return metrics


def main():

    configure_mlflow()

    # Let MLflow automatically capture sklearn model
    # parameters/model information.
    mlflow.sklearn.autolog(
        log_input_examples=False,
        log_model_signatures=True,
        log_models=False,
    )

    print("=" * 70)
    print("BOxCrete Strength Baseline Benchmark")
    print("=" * 70)

    for model_name, model in MODELS.items():

        run_experiment(
            model_name,
            model,
        )


if __name__ == "__main__":
    main()