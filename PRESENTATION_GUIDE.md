# Pennywise — classroom presentation guide

## 60–90 second project introduction

"Pennywise is a student expense prediction and budget-advisor application built with Python, Pandas, Scikit-learn and Streamlit. It uses a synthetic dataset containing 540 student-month records for 30 student IDs over 18 months. The application first validates and cleans the data, then presents spending patterns and IQR-based outlier flags. A Random Forest Regressor estimates a future month's total expenses using monthly income, planned budget, month number and the student's previous month's expenses. A separate budget planner performs arithmetic checks and is not presented as an ML model."

## Demonstration order

1. **Overview:** point out the number of records/students, average expenses and monthly trend.
2. **Dataset & EDA:** explain that each row is one student for one month; show the data dictionary, histogram and correlation heatmap.
3. **Outliers & Cleaning:** explain why negative expense values are invalid, why their category median is used, and why positive IQR outliers are flagged rather than automatically deleted.
4. **Model & Prediction:** identify the four input features and target. Show MAE, RMSE, R² and the baseline comparison. Change one or more input values and show the estimate update.
5. **Budget Planner:** change category allocations and show remaining budget. Explain that this tab uses arithmetic rules, not the trained model.
6. **Monthly Outlook:** explain the recent-three-month moving average and its limitations.

## Key technical explanations

- **Regression:** predicts a continuous numerical value, here monthly expenses in rupees.
- **Random Forest:** trains multiple decision trees and averages their predictions to reduce the instability of a single tree.
- **IQR:** Q1 and Q3 are the 25th and 75th percentiles. IQR = Q3 − Q1. The usual review fences are Q1 − 1.5×IQR and Q3 + 1.5×IQR.
- **MAE:** mean absolute prediction error, in rupees; lower is better.
- **RMSE:** square root of mean squared error; it penalizes large errors more heavily than MAE.
- **R²:** compares the model with predicting the mean target. It can be negative when the model performs worse than that reference.
- **Group split:** all records belonging to a student are kept together in either training or test data, reducing the risk of student-level leakage.
- **Prediction vs retraining:** changing the form inputs calls the already-trained model again. It does not retrain the model.
- **Outlier vs invalid value:** an extreme positive expense can be genuine; a negative expense is invalid for these expense categories.

## Be honest about the result

The fixed student-wise test split currently gives approximately:

- MAE: ₹599
- RMSE: ₹724
- R²: −0.050
- Mean-baseline MAE: ₹581

The Random Forest does not beat the simple mean baseline on this test split. Explain this as a limitation of the small synthetic dataset and available predictors. Do not claim that the model is highly accurate. A future version should use more representative real student-month data, richer explanatory features and repeated group-based validation.

## Likely teacher questions

**Q: Why did you use Random Forest?**  
A: It can model non-linear relationships and interactions without requiring feature scaling. It combines many decision trees and averages their outputs.

**Q: Why is this a regression problem?**  
A: The output is a continuous amount of money, not a class label.

**Q: What is the target variable?**  
A: `total_expenses`, recalculated as the sum of six cleaned expense categories.

**Q: Why remove the first month for each student from training?**  
A: The previous-month-expense feature is unavailable for that first record.

**Q: Why not delete all outliers?**  
A: A high expense may be genuine. IQR flags values for investigation; it does not prove they are errors.

**Q: Why are negative expenses changed?**  
A: They are invalid for these expense categories. The category median is a transparent simple replacement, after which the total is recalculated.

**Q: Does changing the budget planner change the ML prediction?**  
A: No. The budget planner is a separate arithmetic tool. The ML prediction changes when its own four model inputs are modified.

**Q: What does negative R² mean?**  
A: On this test sample, the model performs worse than predicting the training-set mean for every test row.

**Q: Is this real student financial data?**  
A: No. The included dataset is synthetic and is used for educational demonstration.
