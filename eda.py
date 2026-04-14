import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ---------- LOAD DATA ----------
df = pd.read_csv("data/churn_data.csv")
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df = df.dropna()

# ---------- THEME ----------
theme = "plotly_dark"

# ==================================
# 1️⃣ CHURN DISTRIBUTION
# ==================================
fig = px.histogram(
    df,
    x="Churn",
    color="Churn",
    text_auto=True,
    color_discrete_map={"Yes": "#ef4444", "No": "#22c55e"},
    title="Customer Churn Distribution"
)
fig.update_layout(template=theme)
fig.show()

# ==================================
# 2️⃣ TENURE DISTRIBUTION
# ==================================
fig = px.histogram(
    df,
    x="tenure",
    color="Churn",
    marginal="box",
    nbins=30,
    color_discrete_map={"Yes": "#ef4444", "No": "#22c55e"},
    title="Tenure Distribution by Churn"
)
fig.update_layout(template=theme)
fig.show()

# ==================================
# 3️⃣ MONTHLY CHARGES
# ==================================
fig = px.box(
    df,
    x="Churn",
    y="MonthlyCharges",
    color="Churn",
    color_discrete_map={"Yes": "#ef4444", "No": "#22c55e"},
    title="Monthly Charges vs Churn"
)
fig.update_layout(template=theme)
fig.show()

# ==================================
# 4️⃣ CORRELATION HEATMAP
# ==================================
df["Churn_num"] = df["Churn"].map({"Yes": 1, "No": 0})

corr = df[["tenure", "MonthlyCharges", "TotalCharges", "Churn_num"]].corr()

fig = go.Figure(
    data=go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.columns,
        colorscale="RdBu",
        text=corr.round(2),
        texttemplate="%{text}",
        hovertemplate="Feature 1: %{y}<br>Feature 2: %{x}<br>Corr: %{z}<extra></extra>"
    )
)

fig.update_layout(title="Feature Correlation", template=theme)
fig.show()

# ==================================
# 5️⃣ CONTRACT VS CHURN
# ==================================
fig = px.histogram(
    df,
    x="Contract",
    color="Churn",
    barmode="group",
    color_discrete_map={"Yes": "#ef4444", "No": "#22c55e"},
    title="Contract Type vs Churn"
)
fig.update_layout(template=theme)
fig.show()

# ==================================
# 6️⃣ TECH SUPPORT VS CHURN
# ==================================
fig = px.histogram(
    df,
    x="TechSupport",
    color="Churn",
    barmode="group",
    color_discrete_map={"Yes": "#ef4444", "No": "#22c55e"},
    title="Tech Support vs Churn"
)
fig.update_layout(template=theme)
fig.show()

# ==================================
# 📊 INSIGHTS
# ==================================
churn_pct = df["Churn"].value_counts(normalize=True) * 100
print(f"\nChurn: {churn_pct['Yes']:.1f}% | No Churn: {churn_pct['No']:.1f}%")

avg_tenure = df.groupby("Churn")["tenure"].mean()
print(f"Avg tenure Churn: {avg_tenure['Yes']:.1f}, No Churn: {avg_tenure['No']:.1f}")

avg_charge = df.groupby("Churn")["MonthlyCharges"].mean()
print(f"Avg charges Churn: {avg_charge['Yes']:.2f}, No Churn: {avg_charge['No']:.2f}")

print("\nTop Insights:")
print("→ Month-to-month contracts churn the most")
print("→ Low tenure customers are high risk")
print("→ High charges increase churn")
print("→ No tech support increases churn")