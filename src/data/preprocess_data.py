import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

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

def preprocess_data(cfg):
    RAW_PATH = cfg.data.raw_path
    INTERIM_PATH = cfg.data.interim_path
    PREPROCESS_PATH = cfg.data.processed_path
    X_TEST_CSV = cfg.data.x_test_path
    Y_TEST_CSV = cfg.data.y_test_path
    df = pd.read_csv(RAW_PATH)
    df_fe = feature_engineering(df)
    df_fe.to_csv(INTERIM_PATH, index=False)
    X = df_fe.drop("Survived", axis=1)
    y = df_fe["Survived"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_test_df = pd.DataFrame(X_test)  
    y_test_df = pd.DataFrame({'Survived': y_test})  
    X_test_df.to_csv(X_TEST_CSV, index=False)
    y_test_df.to_csv(Y_TEST_CSV, index=False)

    numeric_features = ['Age', 'Fare', 'FamilySize']
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_features = ['Sex', 'Embarked', 'Title', 'IsAlone']
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

    X_train_processed = preprocessor.fit_transform(X_train)

    ohe = preprocessor.named_transformers_['cat'].named_steps['encoder']
    cat_feature_names = ohe.get_feature_names_out(categorical_features)
    processed_feature_names = np.concatenate([numeric_features, cat_feature_names])

    X_train_df = pd.DataFrame(X_train_processed, columns=processed_feature_names)
    y_train_df = pd.DataFrame({'Survived': y_train})
    X_train_df.to_csv(f"{PREPROCESS_PATH}/X_train.csv", index=False)
    y_train_df.to_csv(f"{PREPROCESS_PATH}/y_train.csv", index=False)

    joblib.dump(preprocessor, cfg.data.preprocessor_pkl)

    print("Processing complete")
