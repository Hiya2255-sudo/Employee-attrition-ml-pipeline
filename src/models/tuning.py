from sklearn.model_selection import GridSearchCV


def run_grid_search(pipeline, param_grid, X_train, y_train, cv=3, scoring='f1'):
    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        verbose=0,
    )
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_, search.best_score_
