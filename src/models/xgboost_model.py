from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from .evaluate import compute_metrics


def build_xgb_pipeline(preprocessor, classifier=None):
    if classifier is None:
        classifier = XGBClassifier(
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=42,
            scale_pos_weight=1,
        )
    return Pipeline(
        steps=[
            ('preprocessor', preprocessor),
            ('classifier', classifier),
        ]
    )


def train_xgb(preprocessor, X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    ratio = (y == 0).sum() / max((y == 1).sum(), 1)
    classifier = XGBClassifier(
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42,
        scale_pos_weight=ratio,
        n_estimators=100,
        learning_rate=0.1,
        verbosity=0,
    )
    pipeline = build_xgb_pipeline(preprocessor, classifier)
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    metrics = compute_metrics(y_test, y_pred)
    metrics['scale_pos_weight'] = ratio
    return pipeline, metrics
