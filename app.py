import warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, f1_score, roc_auc_score, confusion_matrix
from statsmodels.tsa.holtwinters import ExponentialSmoothing

st.set_page_config(page_title="Pennywise | Student Money Planner", page_icon="💸", layout="wide")
st.markdown("""
<style>
.block-container {padding-top: 1.6rem; max-width: 1200px;}
[data-testid="stSidebar"] {background: #f7f8fc;}
div[data-testid="stMetric"] {background: #f7f8fc; border: 1px solid #e8eaf2; padding: 16px; border-radius: 14px;}
</style>
""", unsafe_allow_html=True)

st.title("💸 Pennywise")
st.caption("Your personal student money companion")
st.write("Plan your monthly spending, understand where your money goes, and get a helpful outlook for the months ahead.")

with st.sidebar:
    st.header("Your money space")
    uploaded = st.file_uploader("Upload your spending CSV (optional)", type=["csv"])
    st.caption("Sample figures are included so you can explore the app. Upload your own data for a personal analysis.")
    if uploaded:
        data = pd.read_csv(uploaded)
        source_label = "Your uploaded data"
    else:
        data = pd.read_csv(Path(__file__).parent / "sample_monthly_expenses.csv")
        source_label = "Sample figures"
    st.divider()
    st.caption("Tip: for predictions, your file should include monthly spending and budget information.")

st.info(f"Viewing **{source_label}** · {len(data):,} records")

# Friendly navigation rather than technical/course terminology.
pages = st.tabs(["🏠 Overview", "📊 Spending Insights", "🔮 Expense Estimate", "🔔 Spending Alerts", "🗓️ Monthly Outlook", "💰 Plan My Budget"])

with pages[0]:
    st.subheader("Your monthly snapshot")
    if "total_expenses" in data.columns:
        totals = pd.to_numeric(data["total_expenses"], errors="coerce").dropna()
        avg = totals.mean() if len(totals) else 0
        latest = totals.iloc[-1] if len(totals) else 0
        budget_avg = pd.to_numeric(data["budget"], errors="coerce").mean() if "budget" in data.columns else np.nan
        a,b,c = st.columns(3)
        a.metric("Average recorded spending", f"₹{avg:,.0f}")
        b.metric("Most recent recorded amount", f"₹{latest:,.0f}")
        c.metric("Average planned budget", f"₹{budget_avg:,.0f}" if pd.notna(budget_avg) else "Add budget data")
        st.subheader("Spending at a glance")
        if "month" in data.columns:
            chart_data=data.copy()
            chart_data["month"]=pd.to_datetime(chart_data["month"],errors="coerce")
            chart_data["total_expenses"]=pd.to_numeric(chart_data["total_expenses"],errors="coerce")
            monthly=chart_data.dropna(subset=["month"]).groupby("month")["total_expenses"].mean().sort_index()
            st.line_chart(monthly)
        else:
            st.bar_chart(totals.reset_index(drop=True))
    else:
        st.warning("To show your snapshot, upload a CSV containing a `total_expenses` column.")
    st.caption("The numbers shown depend on the dataset you upload. Sample figures are for trying out the app, not verified student records.")

with pages[1]:
    st.subheader("Understand your spending")
    st.write("Explore the amounts in your file and compare spending categories.")
    numeric=data.select_dtypes(include=np.number)
    if numeric.empty:
        st.warning("No numeric spending columns were found. Upload a CSV with amounts in numeric columns.")
    else:
        selected=st.selectbox("Choose a spending measure",numeric.columns)
        x=numeric[selected].dropna()
        a,b,c=st.columns(3)
        a.metric("Average",f"{x.mean():,.0f}")
        b.metric("Middle value",f"{x.median():,.0f}")
        c.metric("Highest value",f"{x.max():,.0f}")
        fig,ax=plt.subplots()
        ax.hist(x,bins=20)
        ax.set_xlabel(selected.replace("_"," ").title()); ax.set_ylabel("Records")
        ax.set_title("How this amount is distributed")
        st.pyplot(fig); plt.close(fig)
        if "total_expenses" in data.columns and selected != "total_expenses":
            fig,ax=plt.subplots()
            ax.scatter(pd.to_numeric(data[selected],errors="coerce"),pd.to_numeric(data["total_expenses"],errors="coerce"),alpha=.5)
            ax.set_xlabel(selected.replace("_"," ").title()); ax.set_ylabel("Total expenses")
            ax.set_title("How this category relates to total spending")
            st.pyplot(fig); plt.close(fig)
        st.subheader("Spending data")
        st.dataframe(data.head(30),use_container_width=True)
        with st.expander("Check data that may need attention"):
            st.write("Missing entries")
            st.dataframe(data.isna().sum().rename("Missing entries").to_frame())
            st.write(f"Repeated rows: {int(data.duplicated().sum())}")
            nums=data.select_dtypes(include=np.number)
            unusual={}
            for col in nums:
                q1,q3=nums[col].quantile([.25,.75]); spread=q3-q1
                unusual[col]=int(((nums[col] < q1-1.5*spread)|(nums[col] > q3+1.5*spread)).sum())
            st.write("Values far from the usual range")
            st.dataframe(pd.Series(unusual,name="Values to review").to_frame())

with pages[2]:
    st.subheader("Estimate your monthly expenses")
    st.write("Use your income, planned budget and previous spending to estimate a future month's total.")
    required={"total_expenses"}
    if not required.issubset(data.columns):
        st.warning("This estimate needs a `total_expenses` column. Upload a suitable CSV to continue.")
    else:
        work=data.copy()
        if "month" in work.columns:
            work["month"]=pd.to_datetime(work["month"],errors="coerce")
            work["month_number"]=work["month"].dt.month
            work=work.sort_values(["student_id","month"] if "student_id" in work.columns else ["month"])
            work["previous_month_expenses"]=work.groupby("student_id")["total_expenses"].shift(1) if "student_id" in work.columns else work["total_expenses"].shift(1)
        features=[c for c in ["monthly_income","budget","month_number","previous_month_expenses"] if c in work.columns]
        if not features:
            st.warning("Your CSV needs at least one of these columns: `monthly_income`, `budget`, or a usable month history.")
        else:
            model_data=work[features+["total_expenses"]].copy()
            for col in model_data.columns: model_data[col]=pd.to_numeric(model_data[col],errors="coerce")
            model_data=model_data.dropna(subset=["total_expenses"])
            X=model_data[features]; y=model_data["total_expenses"]
            if len(model_data)<20:
                st.warning("At least 20 usable records are recommended before making an estimate.")
            else:
                Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42)
                model=Pipeline([("fill",SimpleImputer(strategy="median")),("scale",StandardScaler()),("predict",RandomForestRegressor(n_estimators=150,max_depth=8,random_state=42))])
                model.fit(Xtr,ytr); pred=model.predict(Xte)
                a,b,c=st.columns(3)
                a.metric("Typical estimate difference",f"₹{mean_absolute_error(yte,pred):,.0f}")
                c.metric("Prediction fit",f"{r2_score(yte,pred):.2f}")
                b.metric("Larger-error indicator",f"₹{np.sqrt(mean_squared_error(yte,pred)):,.0f}")
                st.subheader("Recorded spending vs estimates")
                st.line_chart(pd.DataFrame({"Recorded":yte.reset_index(drop=True),"Estimated":pred}))
                st.caption("Estimates are based on the patterns in the available data and are not guaranteed future costs.")

with pages[3]:
    st.subheader("Check your spending risk")
    st.write("See whether records suggest spending may exceed the planned budget.")
    if not {"total_expenses","budget"}.issubset(data.columns):
        st.warning("This feature needs `total_expenses` and `budget` columns in your CSV.")
    else:
        work=data.copy()
        work["total_expenses"]=pd.to_numeric(work["total_expenses"],errors="coerce")
        work["budget"]=pd.to_numeric(work["budget"],errors="coerce")
        work["over_budget"]=(work["total_expenses"]>work["budget"]).astype(int)
        features=[c for c in ["monthly_income","budget"] if c in work.columns]
        if "month" in work.columns:
            work["month_number"]=pd.to_datetime(work["month"],errors="coerce").dt.month
            features.append("month_number")
        work=work.dropna(subset=["total_expenses","budget"])
        if len(features)==0 or len(work)<20 or work["over_budget"].nunique()<2:
            st.warning("More suitable records are needed, including examples both within and over budget.")
        else:
            X=work[features].apply(pd.to_numeric,errors="coerce")
            y=work["over_budget"]
            Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.25,random_state=42,stratify=y)
            model=Pipeline([("fill",SimpleImputer(strategy="median")),("scale",StandardScaler()),("predict",LogisticRegression(max_iter=1500,class_weight="balanced"))])
            model.fit(Xtr,ytr)
            probs=model.predict_proba(Xte)[:,1]
            cutoff=st.slider("When should an alert appear?",0.1,0.9,0.5,0.05)
            pred=(probs>=cutoff).astype(int)
            a,b=st.columns(2)
            a.metric("Alert performance",f"{f1_score(yte,pred,zero_division=0):.2f}")
            b.metric("Risk signal quality",f"{roc_auc_score(yte,probs):.2f}" if len(np.unique(yte))==2 else "N/A")
            st.write("Recorded results compared with alerts")
            st.dataframe(pd.DataFrame(confusion_matrix(yte,pred),index=["Actually within budget","Actually over budget"],columns=["Alert not shown","Alert shown"]))
            st.caption("This is a pattern-based estimate, not a guarantee. Check your current balance and upcoming bills too.")

with pages[4]:
    st.subheader("Your upcoming monthly spending")
    if not {"month","total_expenses"}.issubset(data.columns):
        st.warning("A dated history with `month` and `total_expenses` columns is needed to build a monthly outlook.")
    else:
        work=data.copy()
        work["month"]=pd.to_datetime(work["month"],errors="coerce")
        work["total_expenses"]=pd.to_numeric(work["total_expenses"],errors="coerce")
        monthly=work.dropna(subset=["month","total_expenses"]).groupby("month")["total_expenses"].mean().sort_index()
        st.line_chart(monthly)
        if len(monthly)<8:
            st.info("Add more months of genuine history for a more useful outlook. At least eight monthly points are recommended.")
        else:
            horizon=st.slider("How many months ahead?",1,12,3)
            try:
                model=ExponentialSmoothing(monthly,trend="add",seasonal=None,initialization_method="estimated").fit()
                forecast=model.forecast(horizon)
                future_idx=pd.date_range(monthly.index.max()+pd.offsets.MonthBegin(1),periods=horizon,freq="MS")
                result=pd.DataFrame({"Month":future_idx.strftime("%b %Y"),"Estimated spending (₹)":np.asarray(forecast).round(0)})
                st.dataframe(result,use_container_width=True)
                plot=pd.concat([monthly.rename("Recorded spending"),pd.Series(np.asarray(forecast),index=future_idx,name="Estimated spending")])
                st.line_chart(plot)
            except Exception as exc:
                st.warning(f"An outlook could not be created from this data: {exc}")

with pages[5]:
    st.subheader("Build your monthly plan")
    st.write("Set your monthly spending limit and adjust the amounts below.")
    limit=st.number_input("Your monthly budget (₹)",min_value=0.0,value=10000.0,step=500.0)
    categories=["Housing","Food","Transport","Books & supplies","Entertainment","Personal care","Technology","Health & wellness","Other"]
    defaults=[3000,2200,700,300,500,300,200,200,400]
    amounts={}
    left,right=st.columns(2)
    for i,(cat,default) in enumerate(zip(categories,defaults)):
        with (left if i%2==0 else right):
            amounts[cat]=st.number_input(f"{cat} (₹)",min_value=0.0,value=float(default),step=100.0,key="plan_"+cat)
    total=sum(amounts.values()); remaining=limit-total
    a,b,c=st.columns(3)
    a.metric("Planned spending",f"₹{total:,.0f}")
    b.metric("Money left",f"₹{remaining:,.0f}")
    c.metric("Budget used",f"{total/limit*100:.0f}%" if limit else "—")
    if remaining<0:
        st.error(f"Your plan is ₹{abs(remaining):,.0f} over budget.")
        flexible=amounts["Entertainment"]+amounts["Other"]
        st.write(f"Consider reviewing flexible spending first (entertainment and other: ₹{flexible:,.0f}).")
    elif limit and total>=limit*.85:
        st.warning("You're using most of your budget. Keep a little room for unexpected costs.")
    else:
        st.success("Your planned spending is within budget.")
    fig,ax=plt.subplots()
    ax.bar(list(amounts.keys()),list(amounts.values()))
    ax.set_ylabel("Amount (₹)"); ax.tick_params(axis="x",rotation=45)
    st.pyplot(fig); plt.close(fig)

st.divider()
st.caption("Pennywise · For planning and educational use. Sample data is synthetic; predictions depend on the data provided.")
