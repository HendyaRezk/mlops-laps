import pandas as pd
import joblib
import hydra
from omegaconf import DictConfig
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
import os

def train_model(cfg: DictConfig):
    """Main training function with model training and serialization"""
    os.makedirs(os.path.dirname(cfg.model.model_only_path), exist_ok=True)
    
    print(f"Loading training data from {cfg.data.x_train_path}")
    X_train = pd.read_csv(cfg.data.x_train_path)
    y_train = pd.read_csv(cfg.data.y_train_path).values.ravel()

    model_params = cfg.model.random_forest
    print(f"Training model with parameters: {model_params}")
    
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

    print("Starting grid search...")
    grid_search = GridSearchCV(
        model_pipeline, 
        param_grid, 
        cv=5, 
        scoring='accuracy', 
        n_jobs=-1, 
        verbose=1
    )
    grid_search.fit(X_train, y_train)

    print(f"Saving model to {cfg.model.model_only_path}")
    joblib.dump(grid_search.best_estimator_, cfg.model.model_only_path)

    print("Creating full pipeline with preprocessor...")
    full_pipeline = Pipeline(steps=[
        ('preprocessor', joblib.load(cfg.data.preprocessor_pkl)),
        ('clf', grid_search.best_estimator_.named_steps['clf'])
    ])
    joblib.dump(full_pipeline, cfg.model.combined_model_path)

    print("\nTraining complete")
    print("═"*40)
    print(f"Best Parameters: {grid_search.best_params_}")
    print(f"Best CV Accuracy: {grid_search.best_score_:.4f}")
    print(f"Models saved to:\n- {cfg.model.model_only_path}\n- {cfg.model.combined_model_path}")

@hydra.main(config_path="../conf", config_name="config", version_base="1.1")
def main(cfg: DictConfig):
    """Entry point for DVC pipeline"""
    print("═"*40)
    print("Starting model training...")
    train_model(cfg)
    print("═"*40)

if __name__ == "__main__":
    main()