import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def preprocess(path):
    df = pd.read_csv(path)

    # Convert TotalCharges
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df.dropna(inplace=True)

    # Drop ID column
    if "customerID" in df.columns:
        df.drop("customerID", axis=1, inplace=True)

    # Encode target
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    # Encode categorical
    cat_cols = df.select_dtypes(include="object").columns
    le = LabelEncoder()

    for col in cat_cols:
        df[col] = le.fit_transform(df[col])

    # Scale numeric
    scaler = StandardScaler()
    num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    df[num_cols] = scaler.fit_transform(df[num_cols])

    return df, scaler