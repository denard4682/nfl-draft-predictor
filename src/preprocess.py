import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess_data(df):
    df["Drafted"] = df["Pick"].apply(lambda x: 0 if pd.isnull(x) else 1)
    df.fillna(df.mean(numeric_only=True), inplace=True)
    le = LabelEncoder()
    df["Position"] = le.fit_transform(df["Position"])
    df["School"] = le.fit_transform(df["School"])
    df = df.drop(columns=["Player", "Team"])
    return df



