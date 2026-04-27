# preprocessing.py  Data cleaning and transformation

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessor(X: pd.DataFrame):
    numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_pipeline, numeric_cols),
            ('cat', categorical_pipeline, categorical_cols),
        ],
        remainder='drop',
    )

    return preprocessor, numeric_cols, categorical_cols


def preprocess_data(path: str):
    df = pd.read_csv(path)

    if 'EmployeeID' in df.columns:
        df.drop(columns=['EmployeeID'], inplace=True, errors='ignore')

    df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})

    X = df.drop(columns=['Attrition'])
    y = df['Attrition']

    preprocessor, numeric_cols, categorical_cols = build_preprocessor(X)
    X_transformed = preprocessor.fit_transform(X)

    feature_names = []
    if numeric_cols:
        feature_names.extend(numeric_cols)
    if categorical_cols:
        onehot = preprocessor.named_transformers_['cat']['onehot']
        feature_names.extend(onehot.get_feature_names_out(categorical_cols).tolist())

    processed_df = pd.DataFrame(X_transformed, columns=feature_names, index=df.index)
    processed_df['Attrition'] = y

    return processed_df, preprocessor, X, y