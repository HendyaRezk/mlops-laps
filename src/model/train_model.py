import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

BASE_PATH = r"/teamspace/studios/this_studio/mlops/data/processed"
X_TRAIN_CSV = f"{BASE_PATH}/X_train.csv"
Y_TRAIN_CSV = f"{BASE_PATH}/y_train.csv"
PREPROCESSOR_PKL = f"{BASE_PATH}/preprocessor.pkl"

MODEL_ONLY_PATH = r"/teamspace/studios/this_studio/mlops/src/model/model_only.pkl"
COMBINED_MODEL_PATH = r"/teamspace/studios/this_studio/mlops/src/model/full_pipeline_model.pkl"

X_train = pd.read_csv(X_TRAIN_CSV)
y_train = pd.read_csv(Y_TRAIN_CSV).values.ravel()

preprocessor = joblib.load(PREPROCESSOR_PKL)

model_pipeline = Pipeline(steps=[
    ('clf', RandomForestClassifier(random_state=42))
])

param_grid = {
    'clf__n_estimators': [100, 200],
    'clf__max_depth': [None, 5, 10],
    'clf__min_samples_split': [2, 5],
}

grid_search = GridSearchCV(model_pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

joblib.dump(grid_search.best_estimator_, MODEL_ONLY_PATH)

full_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('clf', grid_search.best_estimator_.named_steps['clf'])  
])

joblib.dump(full_pipeline, COMBINED_MODEL_PATH)

print("Best Params:", grid_search.best_params_)
print("Best CV Score:", grid_search.best_score_)
