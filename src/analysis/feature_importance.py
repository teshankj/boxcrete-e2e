from pathlib import Path

import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline

from src.features.build_features import (
    STRENGTH_FEATURES,
    GWP_FEATURES,
    SLUMP_FEATURES,
)



ROOT = Path(__file__).resolve().parents[2]

MODEL_READY = ROOT / "data/model_ready"
OUTPUT_DIR = MODEL_READY / "feature_importance"


RANDOM_STATE = 42


TARGETS = {
    "strength": "Strength (Mean)",
    "gwp": "GWP",
    "slump": "Slump (in)",
}


def load_dataset(name, split):
    path = MODEL_READY / f"{name}_{split}.csv"
    return pd.read_csv(path)



FEATURE_SCHEMAS = {
    "strength": STRENGTH_FEATURES,
    "gwp": GWP_FEATURES,
    "slump": SLUMP_FEATURES,
}


def get_features(model_name):
    return FEATURE_SCHEMAS[model_name]


def run_importance(
    model_name,
    train,
    validation,
    target,
):
    features = get_features(model_name)

    X_train = train[features]
    y_train = train[target]

    X_val = validation[features]
    y_val = validation[target]

    # Simple, robust baseline.
    model = RandomForestRegressor(
        n_estimators=500,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_val)

    mae = mean_absolute_error(
        y_val,
        predictions,
    )

    r2 = r2_score(
        y_val,
        predictions,
    )

    importance = permutation_importance(
        model,
        X_val,
        y_val,
        scoring="neg_mean_absolute_error",
        n_repeats=20,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    result = pd.DataFrame(
        {
            "feature": features,
            "importance_mean": importance.importances_mean,
            "importance_std": importance.importances_std,
        }
    )

    result["model"] = model_name
    result["validation_mae"] = mae
    result["validation_r2"] = r2

    result = result.sort_values(
        "importance_mean",
        ascending=False,
    )

    return result


def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    for model_name, target in TARGETS.items():

        train = load_dataset(
            model_name,
            "train",
        )

        validation = load_dataset(
            model_name,
            "validation",
        )

        result = run_importance(
            model_name,
            train,
            validation,
            target,
        )

        output = (
            OUTPUT_DIR
            / f"{model_name}_permutation_importance.csv"
        )

        result.to_csv(
            output,
            index=False,
        )

        print("\n" + "=" * 60)
        print(model_name.upper())
        print("=" * 60)

        print(
            f"Validation MAE: "
            f"{result['validation_mae'].iloc[0]:.4f}"
        )

        print(
            f"Validation R²: "
            f"{result['validation_r2'].iloc[0]:.4f}"
        )

        print("\nFeature importance:")
        print(
            result[
                [
                    "feature",
                    "importance_mean",
                    "importance_std",
                ]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    main()