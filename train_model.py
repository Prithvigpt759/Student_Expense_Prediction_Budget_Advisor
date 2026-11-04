"""Train and evaluate Pennywise's monthly expense regression model."""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from data_utils import BASE_DIR, DEFAULT_DATA_PATH, MODEL_FEATURES, clean_dataset, load_dataset, make_model_frame, iqr_outlier_summary

MODEL_PATH = BASE_DIR / "models" / "expense_model.joblib"
METRICS_PATH = BASE_DIR / "reports" / "model_metrics.json"


def train_and_save(data_path=DEFAULT_DATA_PATH):
    raw = load_dataset(data_path)
    cleaned, cleaning_report = clean_dataset(raw)
    frame = make_model_frame(cleaned)
    X, y, groups = frame[MODEL_FEATURES], frame["total_expenses"], frame["student_id"]
    if groups.nunique() < 5:
        raise ValueError("At least five different students are needed for a student-wise holdout evaluation.")
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    train_idx, test_idx = next(splitter.split(X, y, groups))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("regressor", RandomForestRegressor(n_estimators=250, min_samples_leaf=3, max_features=1.0, random_state=42, n_jobs=-1)),
    ])
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    baseline = np.repeat(y_train.mean(), len(y_test))
    metrics = {
        "model": "Random Forest Regressor",
        "features": MODEL_FEATURES,
        "target": "total_expenses (sum of cleaned expense categories)",
        "split_method": "GroupShuffleSplit by student_id (80% train groups / 20% test groups)",
        "train_rows": int(len(train_idx)), "test_rows": int(len(test_idx)),
        "train_students": int(groups.iloc[train_idx].nunique()), "test_students": int(groups.iloc[test_idx].nunique()),
        "mae": float(mean_absolute_error(y_test, predictions)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, predictions))),
        "r2": float(r2_score(y_test, predictions)),
        "baseline_mae": float(mean_absolute_error(y_test, baseline)),
        "baseline_rmse": float(np.sqrt(mean_squared_error(y_test, baseline))),
        "cleaning": cleaning_report,
        "interpretation": "Compare MAE/RMSE with the baseline. A negative R² means this model did not beat predicting the training-set mean on this holdout sample. Do not describe the model as highly accurate unless metrics support that claim."
    }
    MODEL_PATH.parent.mkdir(exist_ok=True)
    METRICS_PATH.parent.mkdir(exist_ok=True)
    joblib.dump({"model": model, "features": MODEL_FEATURES, "target": "total_expenses"}, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    iqr_outlier_summary(cleaned).to_csv(BASE_DIR / "reports" / "outlier_summary.csv", index=False)
    pd.DataFrame({"actual_expenses": y_test.to_numpy(), "predicted_expenses": predictions}).to_csv(BASE_DIR / "reports" / "test_predictions.csv", index=False)
    return model, metrics, cleaned, frame, (X_test, y_test, predictions)


if __name__ == "__main__":
    _, metrics, _, _, _ = train_and_save()
    print(json.dumps(metrics, indent=2))
