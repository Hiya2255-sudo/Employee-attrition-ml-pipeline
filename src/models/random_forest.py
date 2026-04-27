from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from .evaluate import compute_metrics
from .tuning import run_grid_search


def build_rf_pipeline(preprocessor, classifier=None):
    if classifier is None:
        classifier = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight='balanced',
        )
    return Pipeline(
        steps=[
            ('preprocessor', preprocessor),
            ('classifier', classifier),
        ]
    )


def train_rf_baseline(preprocessor, X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    pipeline = build_rf_pipeline(preprocessor)
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    return pipeline, compute_metrics(y_test, y_pred)


def train_rf_tuned(preprocessor, X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    pipeline = build_rf_pipeline(preprocessor)
    param_grid = {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [None, 10, 20],
        'classifier__min_samples_split': [2, 5],
    }
    best_pipeline, best_params, best_score = run_grid_search(
        pipeline, param_grid, X_train, y_train, cv=3, scoring='f1'
    )
    y_pred = best_pipeline.predict(X_test)
    metrics = compute_metrics(y_test, y_pred)
    metrics['best_params'] = best_params
    metrics['best_cv_score'] = best_score
    return best_pipeline, metrics
