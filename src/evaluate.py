import pandas as pd
import joblib
import json
import os
from pathlib import Path
import hydra
from omegaconf import DictConfig
from sklearn.metrics import (accuracy_score, precision_score, 
                            recall_score, f1_score, classification_report)

def evaluate(cfg: DictConfig):
    print(f"\nLoading test data from {cfg.data.x_test_path}")
    X_test = pd.read_csv(cfg.data.x_test_path)
    y_test = pd.read_csv(cfg.data.y_test_path).values.ravel()
    
    print(f"Loading model from {cfg.model.combined_model_path}")
    model = joblib.load(cfg.model.combined_model_path)

    if not isinstance(X_test, pd.DataFrame):
        feature_names = model.named_steps['preprocessor'].get_feature_names_out()
        X_test = pd.DataFrame(X_test, columns=feature_names)

    print("Generating predictions...")
    y_pred = model.predict(X_test)

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred)
    }

    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 Score:  {metrics['f1']:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    metrics_path = Path("reports/metrics.json")
    metrics_path.parent.mkdir(exist_ok=True)
    
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to {metrics_path}")

@hydra.main(config_path="../conf", config_name="config", version_base="1.1")
def main(cfg: DictConfig):
    evaluate(cfg)

if __name__ == "__main__":
    main()