import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from omegaconf import DictConfig

def evaluate(cfg):
    X_test_path = cfg.data.x_test_path
    y_test_path = cfg.data.y_test_path
    model_path = cfg.model.combined_model_path

    X_test = pd.read_csv(X_test_path)
    y_test = pd.read_csv(y_test_path).values.ravel()

    model = joblib.load(model_path)

    if not isinstance(X_test, pd.DataFrame):
        X_test = pd.DataFrame(X_test, columns=model.named_steps['preprocessor'].get_feature_names_out())

    y_pred = model.predict(X_test)

    print("\nEvaluation Metrics:")
    print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
    print(f"F1 Score:  {f1_score(y_test, y_pred):.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

