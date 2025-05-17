import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

def train_model(cfg):
    X_train_path = cfg.data.x_train_path 
    y_train_path = cfg.data.y_train_path
    
    X_train = pd.read_csv(X_train_path)
    y_train = pd.read_csv(y_train_path).values.ravel()

    model_params = cfg.model.random_forest

    model_pipeline = Pipeline(steps=[
        ('clf', RandomForestClassifier(
            n_estimators=model_params.n_estimators,
            max_depth=model_params.max_depth,
            min_samples_split=model_params.min_samples_split,
            random_state=42
        ))
    ])

    param_grid = {
        'clf__n_estimators': [100, 200],
        'clf__max_depth': [None, 5, 10],
        'clf__min_samples_split': [2, 5],
    }

    grid_search = GridSearchCV(model_pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)

    model_only_path = cfg.model.model_only_path
    combined_model_path = cfg.model.combined_model_path

    joblib.dump(grid_search.best_estimator_, model_only_path)

    full_pipeline = Pipeline(steps=[
        ('preprocessor', joblib.load(cfg.data.preprocessor_pkl)),
        ('clf', grid_search.best_estimator_.named_steps['clf'])
    ])
    joblib.dump(full_pipeline, combined_model_path)

    print("Training complete.")
    print("Best Params:", grid_search.best_params_)
    print("Best CV Score:", grid_search.best_score_)
