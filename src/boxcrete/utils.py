DEFAULT_X_COLUMNS = [
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

import pandas as pd
import torch


def derive_bounds_from_X(X):

    return torch.stack(
        [
            X.min(dim=0).values,
            X.max(dim=0).values,
        ]
    )


def load_strength_csv(path):

    df = pd.read_csv(path)

    X = torch.tensor(
        df[DEFAULT_X_COLUMNS]
        .to_numpy(dtype="float64"),
        dtype=torch.float64,
    )

    Y = torch.tensor(
        df["Strength (Mean)"]
        .to_numpy(dtype="float64")
        .reshape(-1, 1),
        dtype=torch.float64,
    )

    bounds = derive_bounds_from_X(X)

    return X, Y, bounds