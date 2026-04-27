# model.py          ML model training and evaluation

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier


def build_pipeline(preprocessor, classifier):
    return Pipeline(
        steps=[
            ('preprocessor', preprocessor),
            ('classifier', classifier),
        ]
    )


def compute_metrics(y_test, y_pred):
    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'classification_report': classification_report(y_test, y_pred, output_dict=True, zero_division=0),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
    }


def train_pipeline(preprocessor, X, y, classifier):
    pipeline = build_pipeline(preprocessor, classifier)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    return pipeline, compute_metrics(y_test, y_pred)


def default_random_forest():
    return RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')


def default_xgboost():
    return XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

