from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

RAW_PATH = ROOT / "data" / "raw" / "boxcrete_data.csv"
CORRECTIONS_PATH = ROOT / "data" / "corrections" / "boxcrete_corrections.csv"
OUTPUT_PATH = ROOT / "data" / "interim" / "boxcrete_validated.csv"


def apply_corrections(
    df: pd.DataFrame,
    corrections: pd.DataFrame,
) -> pd.DataFrame:
    df = df.copy()

    for _, correction in corrections.iterrows():
        mix = correction["Mix Name"]
        time = int(correction["Time"])
        column = correction["Column"]

        original = correction["Original Value"]
        corrected = correction["Corrected Value"]

        mask = (
            (df["Mix Name"] == mix)
            & (df["Time"] == time)
        )

        if mask.sum() != 1:
            raise ValueError(
                f"Expected exactly one row for "
                f"{mix} at {time} days; found {mask.sum()}."
            )

        actual = df.loc[mask, column].iloc[0]

        if float(actual) != float(original):
            raise ValueError(
                f"Correction mismatch for {mix}, {time} days, {column}: "
                f"manifest expects {original}, but dataset contains {actual}."
            )

        df.loc[mask, column] = corrected

    return df


def main():
    df = pd.read_csv(RAW_PATH)
    corrections = pd.read_csv(CORRECTIONS_PATH)

    corrected_df = apply_corrections(df, corrections)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    corrected_df.to_csv(OUTPUT_PATH, index=False)

    print("=" * 60)
    print("BOxCrete Data Correction")
    print("=" * 60)

    for _, correction in corrections.iterrows():
        print(
            f"{correction['Mix Name']} | "
            f"{correction['Column']} | "
            f"{correction['Original Value']} -> "
            f"{correction['Corrected Value']}"
        )

    print(f"\nOutput: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()