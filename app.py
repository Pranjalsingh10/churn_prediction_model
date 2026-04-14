import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

st.set_page_config(page_title="Churn Dashboard", layout="wide")

# ---------- THEME ----------
st.markdown("""
<style>
body {
    background-color: #0f172a;
    color: white;
}
.block-container {
    padding-top: 2rem;
}
.card {
    background: #1e293b;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0px 4px 25px rgba(0,0,0,0.4);
}
.metric {
    background: linear-gradient(135deg, #1e293b, #334155);
    padding: 15px;
    border-radius: 12px;
    text-align: center;
}
.section {
    font-size: 22px;
    font-weight: 600;
    margin-top: 20px;
}
</style>
""", unsafe_allow_html=True)

# ---------- HEADER ----------
st.title("📡 Customer Churn Dashboard")
st.caption("EDA • ML Model • Business Insights • Prediction")
st.markdown("---")

# ---------- PATHS ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(BASE_DIR, "data", "churn_data.csv")
MODEL_PATH = os.path.join(BASE_DIR, "model", "model.pkl")
COLUMNS_PATH = os.path.join(BASE_DIR, "model", "columns.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "model", "scaler.pkl")

EDA_CHURN = os.path.join(BASE_DIR, "EDA", "eda_churn_count.png")
EDA_CONTRACT = os.path.join(BASE_DIR, "EDA", "eda_contract.png")
EDA_TENURE = os.path.join(BASE_DIR, "EDA", "eda_tenure.png")
EDA_MONTHLY = os.path.join(BASE_DIR, "EDA", "eda_monthly_charges.png")
EDA_CORR = os.path.join(BASE_DIR, "EDA", "eda_correlation.png")
EDA_SUPPORT = os.path.join(BASE_DIR, "EDA", "eda_tech_support.png")

# ---------- LOAD ----------
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    return df.dropna()

df = load_data()

@st.cache_resource
def load_model():
    model = pickle.load(open(MODEL_PATH, "rb"))
    columns = pickle.load(open(COLUMNS_PATH, "rb"))
    scaler = pickle.load(open(SCALER_PATH, "rb"))
    return model, columns, scaler

model, columns, scaler = load_model()

# ---------- METRICS ----------
churn_df = df[df["Churn"] == "Yes"]
stay_df = df[df["Churn"] == "No"]

total_customers = len(df)
total_churned = len(churn_df)
total_stayed = len(stay_df)
churn_rate = total_churned / total_customers * 100

# ---------- TABS ----------
tab1, tab2 = st.tabs(["📊 Dashboard", "🎯 Prediction"])

# =========================
# 📊 DASHBOARD
# =========================
with tab1:

    st.markdown('<div class="section">📊 Business Overview</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f'<div class="metric"><h2>{total_customers}</h2><p>Total</p></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="metric"><h2 style="color:#ef4444;">{total_churned}</h2><p>Churned</p></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="metric"><h2 style="color:#22c55e;">{total_stayed}</h2><p>Retained</p></div>', unsafe_allow_html=True)
    c4.markdown(f'<div class="metric"><h2>{churn_rate:.1f}%</h2><p>Churn Rate</p></div>', unsafe_allow_html=True)

    st.markdown("---")

    # Section 1
    st.markdown("### 📉 Churn Distribution & Contracts")
    col1, col2 = st.columns(2)
    col1.image(EDA_CHURN, width="stretch")
    col2.image(EDA_CONTRACT, width="stretch")

    st.success("💡 Month-to-month customers have highest churn → push long-term plans")

    st.markdown("---")

    # Section 2
    st.markdown("### 💰 Tenure & Charges Impact")
    col1, col2 = st.columns(2)
    col1.image(EDA_TENURE, width="stretch")
    col2.image(EDA_MONTHLY, width="stretch")

    st.warning("⚠️ Low tenure + high charges = highest churn risk")

    st.markdown("---")

    # Section 3
    st.markdown("### 🧠 Behavioral Insights")
    col1, col2 = st.columns(2)
    col1.image(EDA_CORR, width="stretch")
    col2.image(EDA_SUPPORT, width="stretch")

    st.info("📞 Lack of tech support strongly correlates with churn")

# =========================
# 🎯 PREDICTION
# =========================
with tab2:

    st.markdown('<div class="section">🎯 Predict Customer Churn</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        tenure = st.slider("Tenure", 0, 72, 12)
        monthly = st.number_input("Monthly Charges", 0.0, 200.0, 65.0)
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])

    with col2:
        internet = st.selectbox("Internet", ["DSL", "Fiber optic", "No"])
        gender = st.selectbox("Gender", ["Male", "Female"])

    if st.button("🚀 Predict"):

        input_data = pd.DataFrame([np.zeros(len(columns))], columns=columns)

        # ---------- NUMERICAL ----------
        raw = pd.DataFrame([[tenure, monthly, tenure * monthly]],
                           columns=["tenure", "MonthlyCharges", "TotalCharges"])
        scaled = scaler.transform(raw)

        input_data["tenure"] = scaled[0][0]
        input_data["MonthlyCharges"] = scaled[0][1]
        input_data["TotalCharges"] = scaled[0][2]

        # ---------- GENDER ----------
        if "gender" in input_data.columns:
            input_data["gender"] = 1 if gender == "Female" else 0

        # ---------- CONTRACT ----------
        for col in columns:
            if "Contract_" in col:
                input_data[col] = 0

        contract_col = f"Contract_{contract}"
        if contract_col in input_data.columns:
            input_data[contract_col] = 1

        # ---------- INTERNET (FIXED) ----------
        for col in columns:
            if "InternetService_" in col:
                input_data[col] = 0

        internet_col = f"InternetService_{internet}"
        if internet_col in input_data.columns:
            input_data[internet_col] = 1

        # ---------- FINAL ALIGNMENT ----------
        input_data = input_data[columns]

        # ---------- PREDICT ----------
        pred = model.predict(input_data)[0]
        prob = model.predict_proba(input_data)[0][1] * 100

        st.markdown("---")

        if pred == 1:
            st.markdown(f"""
            <div class="card">
                <h2 style="color:#ef4444;">⚠️ High Churn Risk</h2>
                <h3>{prob:.1f}% probability</h3>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="card">
                <h2 style="color:#22c55e;">✅ Customer Likely to Stay</h2>
                <h3>{prob:.1f}% probability</h3>
            </div>
            """, unsafe_allow_html=True)