"""Data loading, validation and transparent cleaning for Pennywise."""
from pathlib import Path
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = BASE_DIR / "data" / "sample_monthly_expenses.csv"
CATEGORY_COLUMNS = ["housing", "food", "transport", "books_supplies", "entertainment", "miscellaneous"]
REQUIRED_COLUMNS = ["student_id", "month", "monthly_income", "budget", *CATEGORY_COLUMNS, "total_expenses"]
MODEL_FEATURES = ["monthly_income", "budget", "month_number", "previous_month_expenses"]


def load_dataset(source=DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Read a CSV and reject files that do not contain the project schema."""
    df = pd.read_csv(source)
    missing = sorted(set(REQUIRED_COLUMNS) - set(df.columns))
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))
    return df


def clean_dataset(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Clean invalid category amounts and rebuild total from the six categories.

    Negative category expenses are invalid for this dataset, so they are replaced
    with the median of that category. Positive high values are NOT removed merely
    because they are statistical outliers.
    """
    df = raw.copy()
    report = {"rows_before": len(df), "duplicate_rows": int(df.duplicated().sum()), "negative_category_values": {}, "missing_values_before": df.isna().sum().to_dict()}
    df["month"] = pd.to_datetime(df["month"], errors="coerce")
    numeric_cols = ["monthly_income", "budget", *CATEGORY_COLUMNS, "total_expenses"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.drop_duplicates().copy()
    df = df.dropna(subset=["student_id", "month", "monthly_income", "budget"]).copy()
    for col in ["monthly_income", "budget"]:
        df.loc[df[col] < 0, col] = np.nan
        df[col] = df[col].fillna(df[col].median())
    for col in CATEGORY_COLUMNS:
        invalid = df[col].lt(0)
        report["negative_category_values"][col] = int(invalid.sum())
        df.loc[invalid, col] = np.nan
        df[col] = df[col].fillna(df[col].median())
    # Target is recalculated from cleaned components; this avoids retaining a
    # total that still includes an invalid negative category amount.
    df["total_expenses"] = df[CATEGORY_COLUMNS].sum(axis=1)
    report["missing_values_after_cleaning"] = int(df.isna().sum().sum())
    df = df.sort_values(["student_id", "month"]).reset_index(drop=True)
    df["month_number"] = df["month"].dt.month
    df["previous_month_expenses"] = df.groupby("student_id")["total_expenses"].shift(1)
    report["rows_after"] = len(df)
    report["rows_removed"] = report["rows_before"] - len(df)
    report["negative_values_fixed"] = sum(report["negative_category_values"].values())
    return df, report


def iqr_outlier_summary(df: pd.DataFrame, columns=None) -> pd.DataFrame:
    """Return IQR fences and counts; detection does not automatically delete data."""
    columns = columns or [*CATEGORY_COLUMNS, "total_expenses", "monthly_income", "budget"]
    rows = []
    for col in columns:
        values = pd.to_numeric(df[col], errors="coerce").dropna()
        if values.empty:
            continue
        q1, q3 = values.quantile([0.25, 0.75])
        iqr = q3 - q1
        low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        count = int(((values < low) | (values > high)).sum())
        rows.append({"Column": col, "Q1": q1, "Q3": q3, "IQR": iqr, "Lower fence": low, "Upper fence": high, "Outlier count": count})
    return pd.DataFrame(rows)


def make_model_frame(cleaned: pd.DataFrame) -> pd.DataFrame:
    """Return rows with all model features available (first month/student is omitted)."""
    return cleaned.dropna(subset=[*MODEL_FEATURES, "total_expenses"]).copy()
