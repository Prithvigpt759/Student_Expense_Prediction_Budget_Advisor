"""Pennywise: student expense insights, estimates and budget planning."""
from io import BytesIO
import json
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from data_utils import DEFAULT_DATA_PATH, CATEGORY_COLUMNS, MODEL_FEATURES, clean_dataset, load_dataset, iqr_outlier_summary, make_model_frame
from train_model import MODEL_PATH, METRICS_PATH, train_and_save

st.set_page_config(page_title="Pennywise | Student Budget Advisor", page_icon="💸", layout="wide")
st.title("💸 Pennywise")
st.caption("Student Expense Prediction & Budget Advisor")
st.write("Understand your spending, estimate a future month's expenses and build a realistic budget.")

with st.sidebar:
    st.header("Data source")
    uploaded = st.file_uploader("Upload a CSV (optional)", type=["csv"])
    st.caption("The included dataset is synthetic sample data. Your upload is used for insights; the saved prediction model was trained on the included sample dataset.")
    st.divider()
    st.caption("For best results, use the same columns as sample_monthly_expenses.csv.")

try:
    raw = pd.read_csv(uploaded) if uploaded is not None else load_dataset(DEFAULT_DATA_PATH)
    cleaned, cleaning_report = clean_dataset(raw)
except Exception as exc:
    st.error(f"Could not read this dataset: {exc}")
    st.stop()

source_name = "Uploaded CSV" if uploaded is not None else "Included sample dataset"
st.info(f"**{source_name}** · {len(raw):,} original rows · {len(cleaned):,} cleaned rows")

# Train once if a user has removed the generated model file; normal app reruns load it.
if not MODEL_PATH.exists() or not METRICS_PATH.exists():
    with st.spinner("Preparing the sample prediction model for the first run..."):
        train_and_save()
model_bundle = joblib.load(MODEL_PATH)
model = model_bundle["model"]
metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))

pages = st.tabs([" Overview", " Dataset & EDA", " Outliers & Cleaning", " Model & Prediction", " Budget Planner", " Monthly Outlook"])

with pages[0]:
    st.subheader("Your spending snapshot")
    a, b, c, d = st.columns(4)
    a.metric("Students", f"{cleaned['student_id'].nunique():,}")
    b.metric("Monthly records", f"{len(cleaned):,}")
    c.metric("Average monthly expense", f"₹{cleaned['total_expenses'].mean():,.0f}")
    d.metric("Average budget", f"₹{cleaned['budget'].mean():,.0f}")
    st.subheader("Average spending by category")
    category_avg = cleaned[CATEGORY_COLUMNS].mean().sort_values(ascending=False)
    st.bar_chart(category_avg)
    st.subheader("Monthly spending trend")
    monthly = cleaned.groupby("month")["total_expenses"].mean().sort_index()
    st.line_chart(monthly)
    st.caption("Each point is the average of the available student records for that calendar month. These sample records are synthetic, not verified financial records.")

with pages[1]:
    st.subheader("Know your dataset")
    st.write("One row represents one student's recorded expenses for one month.")
    dictionary = pd.DataFrame([
        ("student_id", "Student identifier", "Category / ID"), ("month", "Month of the record", "Date"),
        ("monthly_income", "Student's monthly income in rupees", "Input feature"), ("budget", "Planned monthly budget in rupees", "Input feature"),
        ("housing", "Housing expense", "Expense category"), ("food", "Food expense", "Expense category"),
        ("transport", "Transport expense", "Expense category"), ("books_supplies", "Books and supplies expense", "Expense category"),
        ("entertainment", "Entertainment expense", "Expense category"), ("miscellaneous", "Other / miscellaneous expense", "Expense category"),
        ("total_expenses", "Sum of the six cleaned expense categories", "Prediction target")], columns=["Column", "Meaning", "Role"])
    st.dataframe(dictionary, use_container_width=True, hide_index=True)
    st.subheader("Dataset preview")
    st.dataframe(cleaned.head(20), use_container_width=True, hide_index=True)
    st.subheader("Basic data quality")
    x, y, z = st.columns(3)
    x.metric("Duplicate rows removed", cleaning_report["duplicate_rows"])
    y.metric("Negative category values fixed", cleaning_report["negative_values_fixed"])
    z.metric("Missing values after cleaning", cleaning_report["missing_values_after_cleaning"])
    st.subheader("Expense distributions")
    chosen = st.selectbox("Choose an expense category", CATEGORY_COLUMNS + ["total_expenses"])
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.hist(cleaned[chosen], bins=20, edgecolor="black")
    ax.set_xlabel("Amount (₹)"); ax.set_ylabel("Number of monthly records")
    ax.set_title(f"Distribution of {chosen.replace('_', ' ').title()}")
    st.pyplot(fig); plt.close(fig)
    st.subheader("Correlation between numeric variables")
    corr_cols = ["monthly_income", "budget", *CATEGORY_COLUMNS, "total_expenses"]
    fig, ax = plt.subplots(figsize=(10, 6))
    image = ax.imshow(cleaned[corr_cols].corr(), aspect="auto", vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(corr_cols)), [c.replace("_", " ") for c in corr_cols], rotation=70, ha="right")
    ax.set_yticks(range(len(corr_cols)), [c.replace("_", " ") for c in corr_cols])
    fig.colorbar(image, ax=ax, label="Correlation")
    fig.tight_layout(); st.pyplot(fig); plt.close(fig)
    st.caption("Correlation describes association, not proof that one variable causes another.")

with pages[2]:
    st.subheader("Outlier detection and data cleaning")
    st.write("The IQR rule flags unusually low or high values for review. A flagged value is not automatically an error.")
    outliers = iqr_outlier_summary(cleaned)
    st.dataframe(outliers.round(2), use_container_width=True, hide_index=True)
    st.subheader("How the source data was cleaned")
    st.markdown("""
    1. Parse the month column as a date and expense/income columns as numbers.
    2. Remove exact duplicate rows.
    3. Replace negative expense-category values with that category's median. Negative spending is invalid in this dataset.
    4. Recalculate `total_expenses` as the sum of the six cleaned categories.
    5. Keep positive IQR outliers for review rather than deleting potentially genuine high spending.
    """)
    st.write("Cleaning summary")
    st.json({"original_rows": cleaning_report["rows_before"], "rows_after": cleaning_report["rows_after"], "duplicate_rows": cleaning_report["duplicate_rows"], "negative_values_fixed_by_column": cleaning_report["negative_category_values"], "rows_removed": cleaning_report["rows_removed"]})
    selected_outlier_col = st.selectbox("Inspect values for a category", CATEGORY_COLUMNS + ["total_expenses"])
    q1, q3 = cleaned[selected_outlier_col].quantile([.25, .75]); iqr = q3 - q1
    low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    flagged = cleaned[(cleaned[selected_outlier_col] < low) | (cleaned[selected_outlier_col] > high)]
    st.write(f"IQR fences: ₹{low:,.2f} to ₹{high:,.2f} · flagged records: {len(flagged)}")
    st.dataframe(flagged[["student_id", "month", selected_outlier_col]], use_container_width=True, hide_index=True)

with pages[3]:
    st.subheader("How the prediction model works")
    st.markdown("""
    **Task:** estimate a student's next monthly total expenses.

    **Algorithm:** Random Forest Regressor. It combines many decision trees and averages their numerical predictions.

    **Input features:** monthly income, planned budget, calendar month number and that student's previous month's cleaned expenses.

    **Target:** total expenses, calculated as the sum of housing, food, transport, books/supplies, entertainment and miscellaneous expenses.

    **Validation:** students are split into separate training and test groups. This reduces the risk of the same student's records appearing in both sets.
    """)
    st.subheader("Held-out test results")
    a, b, c, d = st.columns(4)
    a.metric("MAE", f"₹{metrics['mae']:,.0f}")
    b.metric("RMSE", f"₹{metrics['rmse']:,.0f}")
    c.metric("R²", f"{metrics['r2']:.3f}")
    d.metric("Baseline MAE", f"₹{metrics['baseline_mae']:,.0f}")
    st.caption(f"Evaluation split: {metrics['split_method']}. Test students: {metrics['test_students']} · Test records: {metrics['test_rows']}.")
    if metrics["mae"] < metrics["baseline_mae"]:
        st.success("The model has lower MAE than the mean baseline on this test split.")
    else:
        st.warning("The model does not beat the mean baseline on MAE for this test split. Treat estimates as experimental, not highly accurate financial advice.")
    with st.expander("What do these metrics mean?"):
        st.write("**MAE:** average absolute prediction error in rupees. **RMSE:** similar to MAE but penalizes large errors more. **R²:** compares the model with predicting the test-set mean; it can be negative when the model performs poorly. **Baseline MAE:** error from a simple model that always predicts the average expense in the training data.")
    st.subheader("Try a new prediction")
    st.write("Change the inputs and Pennywise will send the modified values to the saved Random Forest model.")
    train_clean, _ = clean_dataset(load_dataset(DEFAULT_DATA_PATH))
    c1, c2 = st.columns(2)
    with c1:
        income = st.number_input("Monthly income (₹)", min_value=0.0, value=float(cleaned["monthly_income"].median()), step=500.0)
        budget = st.number_input("Monthly budget (₹)", min_value=0.0, value=float(cleaned["budget"].median()), step=500.0)
    with c2:
        month_num = st.selectbox("Month to estimate", list(range(1, 13)), index=0, format_func=lambda n: pd.Timestamp(2024, n, 1).strftime("%B"))
        previous = st.number_input("Previous month's expenses (₹)", min_value=0.0, value=float(train_clean["total_expenses"].median()), step=250.0)
    input_frame = pd.DataFrame([[income, budget, month_num, previous]], columns=MODEL_FEATURES)
    estimate = float(model.predict(input_frame)[0])
    st.metric("Estimated monthly expenses", f"₹{estimate:,.0f}")
    st.metric("Estimated amount left from budget", f"₹{budget-estimate:,.0f}")
    st.caption("This is a model estimate based on synthetic sample patterns. Changing inputs updates the prediction; it does not retrain the model.")
    if uploaded is not None:
        st.info("The model remains trained on the bundled sample dataset. Uploading a CSV changes the analysis views, not the saved model's training data.")

with pages[4]:
    st.subheader("Build a monthly budget")
    limit = st.number_input("Monthly budget limit (₹)", min_value=0.0, value=10000.0, step=500.0, key="budget_limit")
    budget_categories = ["Housing", "Food", "Transport", "Books & supplies", "Entertainment", "Miscellaneous"]
    defaults = [3000, 2200, 700, 400, 600, 500]
    amounts = {}
    left, right = st.columns(2)
    for i, (category, default) in enumerate(zip(budget_categories, defaults)):
        with (left if i % 2 == 0 else right):
            amounts[category] = st.number_input(f"{category} (₹)", min_value=0.0, value=float(default), step=100.0, key=f"plan_{category}")
    planned = sum(amounts.values()); remaining = limit - planned
    a, b, c = st.columns(3)
    a.metric("Planned expenses", f"₹{planned:,.0f}")
    b.metric("Budget remaining", f"₹{remaining:,.0f}")
    c.metric("Budget used", f"{planned / limit * 100:.1f}%" if limit else "—")
    if remaining < 0:
        st.error(f"Your plan exceeds the budget by ₹{abs(remaining):,.0f}.")
    elif limit and planned >= .85 * limit:
        st.warning("Your plan uses at least 85% of the budget. Keep a buffer for unexpected costs.")
    else:
        st.success("Your plan is within the budget.")
    st.bar_chart(pd.Series(amounts, name="Planned amount (₹)"))
    st.caption("This planner uses arithmetic rules. It is not an ML prediction.")

with pages[5]:
    st.subheader("Monthly spending outlook")
    monthly = cleaned.groupby("month")["total_expenses"].mean().sort_index()
    st.line_chart(monthly)
    if len(monthly) >= 6:
        horizon = st.slider("Months ahead to show", 1, 6, 3)
        # Simple, explainable moving-average forecast avoids implying a complex
        # seasonal model from a short synthetic history.
        recent_average = float(monthly.tail(3).mean())
        future_dates = pd.date_range(monthly.index.max() + pd.offsets.MonthBegin(1), periods=horizon, freq="MS")
        forecast = pd.Series([recent_average] * horizon, index=future_dates, name="Simple forecast")
        st.metric("Recent 3-month average", f"₹{recent_average:,.0f}")
        st.dataframe(pd.DataFrame({"Month": future_dates.strftime("%b %Y"), "Illustrative estimate (₹)": np.round(forecast.values, 0)}), use_container_width=True, hide_index=True)
        st.caption("This is a simple moving-average outlook, not the Random Forest prediction. It assumes recent average spending continues and is only a rough planning aid.")
    else:
        st.info("At least six distinct months are recommended before showing a basic outlook.")

st.divider()
st.caption("Pennywise · Educational budgeting tool. The bundled dataset is synthetic. Model estimates are not financial guarantees.")
