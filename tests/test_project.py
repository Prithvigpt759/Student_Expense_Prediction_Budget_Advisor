"""Small offline checks for the bundled dataset and prediction pipeline."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from data_utils import DEFAULT_DATA_PATH, CATEGORY_COLUMNS, clean_dataset, load_dataset, make_model_frame
from train_model import train_and_save

raw = load_dataset(DEFAULT_DATA_PATH)
assert raw.shape == (540, 11), raw.shape
cleaned, report = clean_dataset(raw)
assert report["negative_values_fixed"] == 6, report
assert (cleaned[CATEGORY_COLUMNS] >= 0).all().all()
assert np.allclose(cleaned["total_expenses"], cleaned[CATEGORY_COLUMNS].sum(axis=1))
assert cleaned["student_id"].nunique() == 30
frame = make_model_frame(cleaned)
assert len(frame) == 510, len(frame)
model, metrics, _, _, _ = train_and_save()
pred = model.predict([[15000, 12000, 6, 7400]])
assert len(pred) == 1 and np.isfinite(pred[0]) and pred[0] >= 0
assert metrics["train_students"] + metrics["test_students"] == 30
print("PASS: dataset schema, cleaning, total calculation, group split and new prediction")
print(f"Test MAE ₹{metrics['mae']:.2f}; baseline MAE ₹{metrics['baseline_mae']:.2f}; R² {metrics['r2']:.3f}")
