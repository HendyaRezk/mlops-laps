import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

RAW_X_TEST_CSV = r"/teamspace/studios/this_studio/mlops/data/raw/X_test.csv"
RAW_Y_TEST_CSV = r"/teamspace/studios/this_studio/mlops/data/raw/y_test.csv"
MODEL_PATH = r"/teamspace/studios/this_studio/mlops/src/model/full_pipeline_model.pkl"

def evaluate():
    X_test = pd.read_csv(RAW_X_TEST_CSV)
    y_test = pd.read_csv(RAW_Y_TEST_CSV).values.ravel()

    model = joblib.load(MODEL_PATH)
    y_pred = model.predict(X_test)

    print("\n Evaluation Metrics:")
    print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
    print(f"F1 Score:  {f1_score(y_test, y_pred):.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    evaluate()
