# Libraries to help with reading and manipulating data
import numpy as np
import pandas as pd
import pickle
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction import DictVectorizer
import pathlib
import scipy, sklearn
import mlflow
from prefect import flow, task

# for regression models
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso

# XGBoost
import xgboost as xgb
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.pyll import scope

# To help with data visualization
%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns

#performance
from sklearn.metrics import mean_squared_error

# To supress warnings
import warnings
warnings.filterwarnings("ignore")

@task(retries=3, retry_delay_seconds=3)
def read_dataframe(filename: str) -> pd.DataFrame:
    """Read data into Dataframe"""
    df = pd.read_csv(filename, encoding='iso-8859-1')
    
    # Remove '£' and ',' sign from 'Total Cost/ 10000 miles' column and convert to numeric values
    df['Annual fuel Cost 10000 Miles'] = df['Annual fuel Cost 10000 Miles'].str.replace('£', '')
    df['Annual fuel Cost 10000 Miles'] = df['Annual fuel Cost 10000 Miles'].str.replace(',', '').astype(int)
    df['Annual Electricity cost / 10000 miles'] = df['Annual Electricity cost / 10000 miles'].str.replace('£', '')
    df['Annual Electricity cost / 10000 miles'] = df['Annual Electricity cost / 10000 miles'].str.replace(',', '').astype(int)
    df['Total cost / 10000 miles'] = df['Total cost / 10000 miles'].str.replace('£', '')
    df['Total cost / 10000 miles'] = df['Total cost / 10000 miles'].str.replace(',', '').astype(int)
    # dropping irrelevant features
    df.drop(['Manufacturer', 'Model', 'Description','Transmission', 'Engine Power (Kw)', 'Engine Power (PS)',
      'Electric energy consumption Miles/kWh', 'wh/km', 'Diesel VED Supplement', 'Testing Scheme', 'Euro Standard', 'Maximum range (Miles)',
      'WLTP Imperial Low', 'WLTP Imperial Medium', 'WLTP Imperial High','WLTP Imperial Extra High', 'WLTP Imperial Combined',
      'WLTP Imperial Combined (Weighted)', 'WLTP Metric Low','WLTP Metric Medium', 'WLTP Metric High', 'WLTP Metric Extra High',
      'WLTP Metric Combined', 'WLTP Metric Combined (Weighted)','WLTP CO2 Weighted', 'Equivalent All Electric Range Miles', 'Equivalent All Electric Range KM',
      'THC Emissions [mg/km]', 'Electric Range City Miles', 'RDE NOx Urban', 'Powertrain', 'Annual fuel Cost 10000 Miles', 'Electric Range City Km', 'Noise Level dB(A)',
      'RDE NOx Combined', 'Emissions CO [mg/km]', 'Emissions NOx [mg/km]', 'THC + NOx Emissions [mg/km]', 'Annual Electricity cost / 10000 miles', 'Maximum range (Km)'], axis=1, inplace=True)

    # inputting missing values
    columns_to_impute = ['WLTP CO2', 'Particulates [No.] [mg/km]']
    imputer = SimpleImputer(strategy='mean')
    df[columns_to_impute] = imputer.fit_transform(df[columns_to_impute])

   # Converting object data type to categorical
    for feature in df.columns: 
      if df[feature].dtype == 'object': 
         df[feature] = pd.Categorical(df[feature])# Replace strings with an integer

    # Renaming columns
    df.rename(columns=lambda x: x.replace(' ', '_'), inplace=True)

    return df

@task
def data_prep(
    df_train: pd.DataFrame, df_test: pd.DataFrame
) -> tuple(
    [
        scipy.sparse._csr.csr_matrix,
        scipy.sparse._csr.csr_matrix,
        np.ndarray,
        np.ndarray,
        sklearn.feature_extraction.DictVectorizer,
    ]
):
    """Selecting categorical and numerical features"""
     
    categorical = ['Fuel_Type']
    numerical = ['Engine_Capacity', 'Total_cost_/_10000_miles', 'Particulates_[No.]_[mg/km]']
    dv = DictVectorizer()

    train_dicts = df_train[categorical + numerical].to_dict(orient='records')
    X_train = dv.fit_transform(train_dicts)
    
    test_dicts = X_test[categorical + numerical].to_dict(orient='records')
    X_test = dv.transform(test_dicts)

    y_train = df_train['WLTP_CO2'].values
    y_test = df_test['WLTP_CO2'].values

    return X_train, X_test, y_train, y_test