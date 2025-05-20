import os
import joblib
import pandas as pd
from pydantic import ValidationError
from src.deployment.requests import InferenceRequest
import litserve as ls

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    df['Title'] = df['Title'].replace(
        ['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 
         'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    df['Title'] = df['Title'].replace({'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs'})
    df.drop(columns=['Name', 'Ticket', 'Cabin', 'PassengerId'], inplace=True, errors='ignore')
    return df

class InferenceAPI(ls.LitAPI):
    def setup(self, device="cpu"):
        # Load the trained model
        model_path = "/teamspace/studios/this_studio/mlops/src/model/full_pipeline_model.pkl"
        self._model = joblib.load(model_path)

    def decode_request(self, request):
        try:
            columns = request["dataframe_split"]["columns"]
            rows = request["dataframe_split"]["data"]
            inference_requests = []

            # Create InferenceRequest instances
            for row in rows:
                row_dict = dict(zip(columns, row))
                try:
                    inference_request = InferenceRequest(**row_dict)
                    inference_requests.append(inference_request)
                except ValidationError as e:
                    print(f"Validation error for row {row}: {e}")
                    return {"message": "Validation error", "data": str(e)}

            df = pd.DataFrame(rows, columns=columns)

            # Perform feature engineering
            df_fe = feature_engineering(df)

            # Apply preprocessing (scaling + encoding)
            return df_fe

        except Exception as e:
            print(f"Decoding error: {e}")
            return None

    def predict(self, x):
        if x is not None:
            return self._model.predict(x)
        else:
            return None

    def encode_response(self, output):
        if output is None:
            message = "Error Occurred"
        else:
            message = "Response Produced Successfully"
        response = {
            "message": message,
            "data": output.tolist() if output is not None else None,
        }
        return response
