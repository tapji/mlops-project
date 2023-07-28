# Libraries to help with reading and manipulating data
import numpy as np
import pandas as pd
import pickle
import pathlib
import scipy, sklearn
import mlflow
import prefect 
from prefect import flow, task
import xgboost as xgb
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings("ignore")

@task(retries=3, retry_delay_seconds=3)
def read_dataframe(filename: str) -> pd.DataFrame:
    """Read data into DataFrame"""
    df = pd.read_csv(filename, encoding='iso-8859-1')

    # Remove '£' and ',' sign from 'Total Cost/ 10000 miles' column and convert to numeric values
    df['Annual fuel Cost 10000 Miles'] = df['Annual fuel Cost 10000 Miles'].str.replace('£', '')
    df['Annual fuel Cost 10000 Miles'] = df['Annual fuel Cost 10000 Miles'].str.replace(',', '').astype(int)
    df['Annual Electricity cost / 10000 miles'] = df['Annual Electricity cost / 10000 miles'].str.replace('£', '')
    df['Annual Electricity cost / 10000 miles'] = df['Annual Electricity cost / 10000 miles'].str.replace(',', '').astype(int)
    df['Total cost / 10000 miles'] = df['Total cost / 10000 miles'].str.replace('£', '')
    df['Total cost / 10000 miles'] = df['Total cost / 10000 miles'].str.replace(',', '').astype(int)

    # Dropping irrelevant features
    df.drop(['Manufacturer', 'Model', 'Description', 'Transmission', 'Engine Power (Kw)', 'Engine Power (PS)',
             'Electric energy consumption Miles/kWh', 'wh/km', 'Diesel VED Supplement', 'Testing Scheme', 'Euro Standard',
             'Maximum range (Miles)', 'WLTP Imperial Low', 'WLTP Imperial Medium', 'WLTP Imperial High',
             'WLTP Imperial Extra High', 'WLTP Imperial Combined', 'WLTP Imperial Combined (Weighted)',
             'WLTP Metric Low', 'WLTP Metric Medium', 'WLTP Metric High', 'WLTP Metric Extra High', 'WLTP Metric Combined',
             'WLTP Metric Combined (Weighted)', 'WLTP CO2 Weighted', 'Equivalent All Electric Range Miles',
             'Equivalent All Electric Range KM', 'THC Emissions [mg/km]', 'Electric Range City Miles', 'RDE NOx Urban',
             'Powertrain', 'Annual fuel Cost 10000 Miles', 'Electric Range City Km', 'Noise Level dB(A)', 'RDE NOx Combined',
             'Emissions CO [mg/km]', 'Emissions NOx [mg/km]', 'THC + NOx Emissions [mg/km]', 'Annual Electricity cost / 10000 miles',
             'Maximum range (Km)'], axis=1, inplace=True)

    # Inputting missing values
    columns_to_impute = ['WLTP CO2', 'Particulates [No.] [mg/km]']
    imputer = SimpleImputer(strategy='mean')
    df[columns_to_impute] = imputer.fit_transform(df[columns_to_impute])

    # Converting object data type to categorical
    for feature in df.columns:
        if df[feature].dtype == 'object':
            df[feature] = pd.Categorical(df[feature])  # Replace strings with an integer

    # Renaming columns
    df.rename(columns=lambda x: x.replace(' ', '_'), inplace=True)

    return df

@task(log_prints=True)
def data_prep(df_train: pd.DataFrame, df_test: pd.DataFrame) -> tuple:
    """Selecting categorical and numerical features"""

    categorical = ['Fuel_Type']
    numerical = ['Engine_Capacity', 'Total_cost_/_10000_miles', 'Particulates_[No.]_[mg/km]']

    dv = DictVectorizer()

    train_dicts = df_train[categorical + numerical].to_dict(orient='records')
    X_train = dv.fit_transform(train_dicts)

    test_dicts = df_test[categorical + numerical].to_dict(orient='records')
    X_test = dv.transform(test_dicts)

    y_train = df_train['WLTP_CO2'].values
    y_test = df_test['WLTP_CO2'].values

    return X_train, X_test, y_train, y_test

@task(log_prints=True)
def train_best_model(
    X_train: scipy.sparse._csr.csr_matrix,
    X_test: scipy.sparse._csr.csr_matrix,
    y_train: np.ndarray,
    y_test: np.ndarray,
    dv: sklearn.feature_extraction.DictVectorizer,
) -> None:
    """Training the best model and write everything out"""

    with mlflow.start_run():
        train = xgb.DMatrix(X_train, label=y_train)
        test = xgb.DMatrix(X_test, label=y_test)

        params = {
            'learning_rate': 0.0502324462543933,
            'max_depth':	75,
            'min_child_weight':	19.592017909776523,
            'objective':	'reg:linear',
            'reg_alpha':	0.36230511158752154,
            'reg_lambda':	0.2713344777035102,
            'seed':	42
        }

        mlflow.log_params(params)

        booster = xgb.train(
            params=params,
            dtrain=train,
            num_boost_round=1000,
            evals=[(test, 'validation')],
            early_stopping_rounds=50)
    
        y_pred = booster.predict(test)
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        mlflow.log_metric("rmse", rmse)

        pathlib.path('models').mkdir(exist_ok=True)
        with open("models/preprocessor.b", "wb") as f_out:
            pickle.dump(booster, f_out)
        mlflow.log_artifact("models/preprocessor.b", artifact_path="preprocessor")
        mlflow.xgboost.log_model(booster, artifact_path="models_mlflow")

    return None

@task(log_prints=True)
def predict_with_model(model: xgb.Booster, X_test: scipy.sparse.csr.csr_matrix) -> np.ndarray:
    """Use the trained model to make predictions on new data"""
    y_pred = model.predict(X_test)
    return y_pred

@task
def save_predictions(y_pred: np.ndarray, filename: str):
    """Save model predictions to a file"""
    np.savetxt(filename, y_pred)

# Define the main flow
@flow
def main_flow(
    train_path: str = "./data/emission_data_2022",
    test_path: str = "./data/emission_data_2022",
) -> None:
    """The main training pipeline"""
    # MLflow settings
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("mlops-project")

    # Load
    df_train = read_dataframe(train_path)
    df_test = read_dataframe(test_path)

    # Transform
    X_train, X_test, y_train, y_test = data_prep(df_train, df_test)

    # Train
    model = train_best_model(X_train, X_test, y_train, y_test)

    # Predict
    predictions = predict_with_model(model, X_test)

    # Save predictions
    save_predictions(predictions, "path/to/predictions.txt")

if __name__ == "__main__":
    main_flow()
