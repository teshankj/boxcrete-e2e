from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

RAW = ROOT / "data/raw/boxcrete_data.csv"
CORRECTED = ROOT / "data/interim/boxcrete_validated.csv"
SPLITS = ROOT / "data/processed/mix_splits.csv"

TRAIN = ROOT / "data/processed/train.csv"
VALIDATION = ROOT / "data/processed/validation.csv"
TEST = ROOT / "data/processed/test.csv"

MODEL_READY = ROOT / "data/model_ready"


EXPECTED_COLUMNS = [
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


def load(path):
    assert path.exists(), f"Missing dataset: {path}"
    return pd.read_csv(path)


def test_raw_schema():
    df = load(RAW)

    assert list(df.columns) == EXPECTED_COLUMNS
    assert len(df) == 670


def test_unique_mix_count():
    df = load(CORRECTED)

    assert df["Mix Name"].nunique() == 149


def test_material_source_counts():
    df = load(CORRECTED)

    counts = (
        df[["Mix Name", "Material Source"]]
        .drop_duplicates()["Material Source"]
        .value_counts()
        .sort_index()
    )

    assert counts.to_dict() == {
        0: 69,
        1: 27,
        2: 53,
    }


def test_coarse_aggregate_invariant():
    df = load(CORRECTED)

    mortar = df["Material Source"] == 0
    concrete = df["Material Source"].isin([1, 2])

    assert (
        df.loc[mortar, "Coarse Aggregates (kg/m3)"] == 0
    ).all()

    assert (
        df.loc[concrete, "Coarse Aggregates (kg/m3)"] > 0
    ).all()


def test_mix_name_convention():
    df = load(CORRECTED)

    mixes = (
        df[["Mix Name", "Material Source"]]
        .drop_duplicates()
    )

    mortar = mixes["Material Source"] == 0
    concrete = mixes["Material Source"].isin([1, 2])

    assert mixes.loc[mortar, "Mix Name"].str.startswith("M").all()
    assert mixes.loc[concrete, "Mix Name"].str.startswith("C").all()


def test_c42_raw_anomaly_is_preserved():
    raw = load(RAW)

    row = raw[
        (raw["Mix Name"] == "C42")
        & (raw["Time"] == 1)
    ]

    assert len(row) == 1
    assert row["GWP"].iloc[0] == 160.0


def test_c42_correction():
    df = load(CORRECTED)

    c42 = df[df["Mix Name"] == "C42"]

    assert len(c42) == 5
    assert c42["GWP"].tolist() == [
        161.0,
        161.0,
        161.0,
        161.0,
        161.0,
    ]


def test_split_mix_counts():
    splits = load(SPLITS)

    counts = splits["split"].value_counts().to_dict()

    assert counts == {
        "train": 104,
        "validation": 22,
        "test": 23,
    }


def test_split_source_distribution():
    splits = load(SPLITS)

    table = pd.crosstab(
        splits["split"],
        splits["Material Source"],
    )

    expected = {
        "train": {0: 48, 1: 19, 2: 37},
        "validation": {0: 10, 1: 4, 2: 8},
        "test": {0: 11, 1: 4, 2: 8},
    }

    for split, source_counts in expected.items():
        for source, count in source_counts.items():
            assert table.loc[split, source] == count


def test_no_mix_leakage():
    train = load(TRAIN)
    validation = load(VALIDATION)
    test = load(TEST)

    train_mixes = set(train["Mix Name"])
    validation_mixes = set(validation["Mix Name"])
    test_mixes = set(test["Mix Name"])

    assert not train_mixes & validation_mixes
    assert not train_mixes & test_mixes
    assert not validation_mixes & test_mixes

    assert (
        train_mixes
        | validation_mixes
        | test_mixes
    ) == set(load(CORRECTED)["Mix Name"])


def test_row_counts():
    assert len(load(TRAIN)) == 468
    assert len(load(VALIDATION)) == 99
    assert len(load(TEST)) == 103


def test_slump_counts():
    assert load(TRAIN)["Slump (in)"].notna().sum() == 220
    assert load(VALIDATION)["Slump (in)"].notna().sum() == 45
    assert load(TEST)["Slump (in)"].notna().sum() == 40


def test_strength_feature_leakage():
    df = load(MODEL_READY / "strength_train.csv")

    target = "Strength (Mean)"

    forbidden_features = {
        "Strength1 (psi)",
        "Strength2 (psi)",
        "Strength3 (psi)",
        "Strength (Std)",
        "# of measurements",
        "GWP",
        "Slump (in)",
    }

    metadata = {
        "Mix Name",
        "Material Source",
        "Time",
        target,
    }

    features = set(df.columns) - metadata

    assert target not in features
    assert not forbidden_features & features


def test_gwp_feature_leakage():
    df = load(MODEL_READY / "gwp_train.csv")

    target = "GWP"

    forbidden_features = {
        "Strength1 (psi)",
        "Strength2 (psi)",
        "Strength3 (psi)",
        "Strength (Mean)",
        "Strength (Std)",
        "# of measurements",
        "Slump (in)",
        "Time",
    }

    metadata = {
        "Mix Name",
        "Material Source",
        "Time",
        target,
    }

    features = set(df.columns) - metadata

    assert target not in features
    assert not forbidden_features & features


def test_slump_feature_leakage():
    df = load(MODEL_READY / "slump_train.csv")

    target = "Slump (in)"

    forbidden_features = {
        "Material Source",
        "Coarse Aggregates (kg/m3)",
        "GWP",
        "Strength1 (psi)",
        "Strength2 (psi)",
        "Strength3 (psi)",
        "Strength (Mean)",
        "Strength (Std)",
        "# of measurements",
    }

    metadata = {
        "Mix Name",
        "Material Source",
        "Time",
        target,
    }

    features = set(df.columns) - metadata

    assert target not in features
    assert not forbidden_features & features

def test_model_ready_split_sizes():
    expected = {
        "strength_train.csv": 468,
        "strength_validation.csv": 99,
        "strength_test.csv": 103,
        "gwp_train.csv": 468,
        "gwp_validation.csv": 99,
        "gwp_test.csv": 103,
        "slump_train.csv": 220,
        "slump_validation.csv": 45,
        "slump_test.csv": 40,
    }

    for filename, expected_rows in expected.items():
        df = load(MODEL_READY / filename)
        assert len(df) == expected_rows