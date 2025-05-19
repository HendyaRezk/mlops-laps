import pandas as pd
import joblib
import mlflow
import dagshub
from mlflow.models.signature import infer_signature
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from omegaconf import DictConfig
import hydra


@hydra.main(config_path="../../conf", config_name="config", version_base="1.1")
def main(cfg: DictConfig):
    dagshub.init(repo_owner="HendyaRezk", repo_name="mlops-laps", mlflow=True)

    print(f"Loading model from {cfg.model.combined_model_path}")
    model = joblib.load(cfg.model.combined_model_path)

    print(f"Loading test data from {cfg.data.x_test_path}")
    X_test = pd.read_csv(cfg.data.x_test_path)
    y_test = pd.read_csv(cfg.data.y_test_path).values.ravel()

    X_test[X_test.select_dtypes("int").columns] = X_test.select_dtypes("int").astype(float)

    mlflow.set_experiment(cfg.mlflow.experiment_name)

    with mlflow.start_run() as run:
        y_pred = model.predict(X_test)

        signature = infer_signature(X_test, y_pred)

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=cfg.names.model_name,
            signature=signature,
            input_example=X_test.iloc[[0]]
        )

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division="warn"),
            "recall": recall_score(y_test, y_pred, zero_division="warn"),
            "f1": f1_score(y_test, y_pred, zero_division="warn")
        }

        try:
            metrics["roc_auc"] = roc_auc_score(y_test, y_pred)
        except ValueError:
            pass  

        mlflow.log_metrics(metrics)

        print("\nMetrics logged to MLflow:")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")

        model_uri = f"runs:/{run.info.run_id}/model"
        result = mlflow.register_model(model_uri=model_uri, name=cfg.names.model_name)

        print(f"\nModel registered: {result.name} (version {result.version})")

        from mlflow.tracking import MlflowClient
        client = MlflowClient()
        client.transition_model_version_stage(
            name=result.name,
            version=result.version,
            stage="Production"
        )
        client.set_model_version_tag(
            name=result.name,
            version=result.version,
            key="production",
            value="true"
        )
        print("Model promoted to Production.")


if __name__ == "__main__":
    main()
