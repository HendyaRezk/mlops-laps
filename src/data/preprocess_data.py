import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import hydra
from omegaconf import DictConfig

def feature_engineering(df):
    df = df.copy()
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    df['Title'] = df['Title'].replace(
        ['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 
         'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    df['Title'] = df['Title'].replace({'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs'})
    df.drop(columns=['Name', 'Ticket', 'Cabin', 'PassengerId'], inplace=True)
    return df

def preprocess_data(cfg: DictConfig):
    os.makedirs(os.path.dirname(cfg.data.interim_path), exist_ok=True)
    os.makedirs(cfg.data.processed_path, exist_ok=True)
    
    print(f"Loading data from {cfg.data.raw_path}")
    df = pd.read_csv(cfg.data.raw_path)
    df_fe = feature_engineering(df)
    
    print(f"Saving interim data to {cfg.data.interim_path}")
    df_fe.to_csv(cfg.data.interim_path, index=False)
    
    X = df_fe.drop("Survived", axis=1)
    y = df_fe["Survived"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    pd.DataFrame(X_test).to_csv(cfg.data.x_test_path, index=False)
    pd.DataFrame(y_test, columns=['Survived']).to_csv(cfg.data.y_test_path, index=False)
    
    numeric_features = ['Age', 'Fare', 'FamilySize']
    categorical_features = ['Sex', 'Embarked', 'Title', 'IsAlone']
    
    preprocessor = ColumnTransformer([
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), numeric_features),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ]), categorical_features)
    ])
    
    X_train_processed = preprocessor.fit_transform(X_train)
    
    ohe = preprocessor.named_transformers_['cat'].named_steps['encoder']
    cat_feature_names = ohe.get_feature_names_out(categorical_features)
    feature_names = np.concatenate([numeric_features, cat_feature_names])
    
    pd.DataFrame(X_train_processed, columns=feature_names).to_csv(
        f"{cfg.data.processed_path}/X_train.csv", index=False)
    pd.DataFrame(y_train, columns=['Survived']).to_csv(
        f"{cfg.data.processed_path}/y_train.csv", index=False)
    
    joblib.dump(preprocessor, cfg.data.preprocessor_pkl)
    print("✔ Preprocessing complete")

@hydra.main(config_path="../conf", config_name="config", version_base="1.1")
def main(cfg: DictConfig):
    """Entry point for DVC pipeline"""
    print("Starting data preprocessing...")
    preprocess_data(cfg)

if __name__ == "__main__":
    main()