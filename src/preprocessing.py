
#preprocessing.py  Data cleaning and transformation

import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_data(path):
    df = pd.read_csv(path)

    # Drop useless columns
    df.drop(columns=['EmployeeID'], inplace=True, errors='ignore')

    # Handle missing values
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    # Encode target
    df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})

    # Binary encoding
    if 'OverTime' in df.columns:
        df['OverTime'] = df['OverTime'].map({'No': 0, 'Yes': 1})

    # One-hot encoding remaining categorical
    df = pd.get_dummies(df, drop_first=True)

    # Feature scaling for original numeric columns only
    scaler = StandardScaler()
    numeric_features = [col for col in num_cols if col != 'Attrition' and col in df.columns]
    if numeric_features:
        df[numeric_features] = scaler.fit_transform(df[numeric_features])

    return df