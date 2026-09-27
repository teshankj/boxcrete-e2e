from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = ROOT / "data" / "interim" / "boxcrete_validated.csv"
OUTPUT_DIR = ROOT / "data" / "processed"

RANDOM_STATE = 42


def build_mix_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """Create one record per unique mix."""

    source_counts = df.groupby("Mix Name")["Material Source"].nunique()

    inconsistent = source_counts[source_counts != 1]

    if not inconsistent.empty:
        raise ValueError(
            "Some mixes contain multiple Material Source values:\n"
            f"{inconsistent}"
        )

    mix_meta = (
        df[["Mix Name", "Material Source"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    return mix_meta


def create_splits(mix_meta: pd.DataFrame) -> pd.DataFrame:
    """
    Split unique mixes into approximately:

        Train      70%
        Validation 15%
        Test       15%

    Stratification is performed using Material Source.
    """

    train, temp = train_test_split(
        mix_meta,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=mix_meta["Material Source"],
    )

    validation, test = train_test_split(
        temp,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=temp["Material Source"],
    )

    train = train.copy()
    validation = validation.copy()
    test = test.copy()

    train["split"] = "train"
    validation["split"] = "validation"
    test["split"] = "test"

    splits = pd.concat(
        [train, validation, test],
        ignore_index=True,
    )

    return splits


def validate_splits(
    df: pd.DataFrame,
    splits: pd.DataFrame,
) -> None:
    """Validate that the generated split is leakage-free."""

    expected_mixes = set(df["Mix Name"].unique())
    assigned_mixes = set(splits["Mix Name"])

    if expected_mixes != assigned_mixes:
        raise ValueError(
            "Split assignments do not match dataset mixes."
        )

    duplicate_assignments = splits["Mix Name"].duplicated()

    if duplicate_assignments.any():
        duplicates = splits.loc[
            duplicate_assignments, "Mix Name"
        ].tolist()

        raise ValueError(
            f"Mixes assigned to multiple splits: {duplicates}"
        )

    if splits["split"].isna().any():
        raise ValueError("Some mixes have no split assignment.")


def print_summary(
    full_df: pd.DataFrame,
    splits: pd.DataFrame,
) -> None:

    print("=" * 60)
    print("BOxCrete Group-Aware Dataset Split")
    print("=" * 60)

    print(f"\nTotal rows:  {len(full_df)}")
    print(f"Total mixes: {full_df['Mix Name'].nunique()}")

    print("\nMix counts:")
    print(
        splits["split"]
        .value_counts()
        .reindex(["train", "validation", "test"])
    )

    print("\nMix distribution by Material Source:")
    source_table = pd.crosstab(
        splits["split"],
        splits["Material Source"],
    )

    print(
        source_table.reindex(
            ["train", "validation", "test"]
        )
    )

    print("\nRow counts:")
    print(
        full_df["split"]
        .value_counts()
        .reindex(["train", "validation", "test"])
    )

    print("\nCuring-age distribution:")
    print(
        pd.crosstab(
            full_df["split"],
            full_df["Time"],
        ).reindex(
            ["train", "validation", "test"]
        )
    )


def main():

    df = pd.read_csv(INPUT_PATH)

    print(f"Loaded {len(df)} rows from:")
    print(INPUT_PATH)

    mix_meta = build_mix_metadata(df)

    print(
        f"\nDetected {len(mix_meta)} unique mixes."
    )

    splits = create_splits(mix_meta)

    validate_splits(df, splits)

    # Map split assignment back to every curing-age observation.
    split_map = splits.set_index("Mix Name")["split"]

    df["split"] = df["Mix Name"].map(split_map)

    if df["split"].isna().any():
        raise ValueError(
            "Some observations were not assigned to a split."
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Master mix-level assignment
    splits.sort_values(
        ["split", "Material Source", "Mix Name"]
    ).to_csv(
        OUTPUT_DIR / "mix_splits.csv",
        index=False,
    )

    # Complete dataset including split label
    df.to_csv(
        OUTPUT_DIR / "boxcrete_split.csv",
        index=False,
    )

    # Individual datasets
    for split_name in [
        "train",
        "validation",
        "test",
    ]:

        subset = df[
            df["split"] == split_name
        ].copy()

        subset.to_csv(
            OUTPUT_DIR / f"{split_name}.csv",
            index=False,
        )

    print_summary(df, splits)

    print("\nLeakage check: PASS")

    print("\nGenerated:")
    print("  data/processed/mix_splits.csv")
    print("  data/processed/boxcrete_split.csv")
    print("  data/processed/train.csv")
    print("  data/processed/validation.csv")
    print("  data/processed/test.csv")


if __name__ == "__main__":
    main()