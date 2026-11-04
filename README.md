# Pennywise — Student Expense Prediction & Budget Advisor

A beginner-friendly Streamlit project for exploring monthly student expenses, reviewing data quality and IQR outliers, estimating a future month's expenses with a Random Forest Regressor, and planning a budget.

> **Dataset note:** `data/sample_monthly_expenses.csv` is synthetic sample data (30 student IDs across 18 months). Do not describe it as real student financial records.

## Run on Windows (VS Code terminal)

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
python train_model.py
python -m streamlit run app.py
```

The first run of the app also creates the model if it is missing. Running `python train_model.py` explicitly retrains and saves it.

## Project files

- `data/sample_monthly_expenses.csv` — bundled sample dataset.
- `data_utils.py` — schema checks, cleaning, feature creation and IQR outlier summaries.
- `train_model.py` — student-wise train/test split, Random Forest training, baseline comparison and saved metrics.
- `models/expense_model.joblib` — saved fitted pipeline.
- `reports/model_metrics.json` — reproducible test metrics and cleaning summary.
- `app.py` — Streamlit dashboard with overview, EDA, outlier review, prediction, budget planner and monthly outlook.
- `notebooks/expense_analysis.ipynb` — guided analysis notebook.

## Dataset and cleaning choices

Each row is one student-month. Six expense categories are used to calculate `total_expenses`. Negative category amounts are invalid and are replaced with that category's median. The total is then recalculated from the cleaned categories. Positive IQR outliers are flagged for review but are not automatically deleted because high spending can be genuine.

## Model design

The Random Forest predicts `total_expenses` using monthly income, planned budget, month number and the student's previous month's expenses. The first available month for each student has no previous-month feature and is omitted from model fitting. A `GroupShuffleSplit` separates students between training and test sets, reducing student-level leakage. MAE and RMSE are reported in rupees; R² may be negative. Results are compared with a simple training-mean baseline. Do not claim high accuracy unless the reported test results justify it.

The model is trained on the included sample data. An uploaded CSV changes the dataset/EDA views but does not silently retrain the deployed model. The interactive estimate sends modified values to the saved model; changing inputs does not retrain it. The budget planner is arithmetic, not ML.

## Limitations

The dataset is synthetic and relatively small. The model is educational, not suitable for real financial decisions. The monthly outlook uses a transparent recent-three-month average, not a learned forecasting model. More representative real data and further validation would be required for practical use.
