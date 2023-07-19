import os
import pickle
import click
import pandas as pd
from sklearn.impute import SimpleImputer

# for modelling
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# To supress warnings
import warnings
warnings.filterwarnings("ignore")

def dump_pickle(obj, filename: str):
    with open(filename, "wb") as f_out:
        return pickle.dump(obj, f_out)


def read_dataframe(filename: str):
    df = pd.read_csv(filename, encoding='iso-8859-1')
    
    # Remove '£' and ',' sign from 'Total Cost/ 10000 miles' column and convert to numeric values
    df['Annual fuel Cost 10000 Miles'] = df['Annual fuel Cost 10000 Miles'].str.replace('£', '')
    df['Annual fuel Cost 10000 Miles'] = df['Annual fuel Cost 10000 Miles'].str.replace(',', '').astype(int)
    df['Annual Electricity cost / 10000 miles'] = df['Annual Electricity cost / 10000 miles'].str.replace('£', '')
    df['Annual Electricity cost / 10000 miles'] = df['Annual Electricity cost / 10000 miles'].str.replace(',', '').astype(int)
    df['Total cost / 10000 miles'] = df['Total cost / 10000 miles'].str.replace('£', '')
    df['Total cost / 10000 miles'] = df['Total cost / 10000 miles'].str.replace(',', '').astype(int)

    # dropping irrelevant feautures
    df.drop(['Description','Transmission', 'Engine Power (Kw)', 'Engine Power (PS)',
       'Electric energy consumption Miles/kWh', 'wh/km', 'Maximum range (Km)',
       'WLTP Imperial Low', 'WLTP Imperial Medium', 'WLTP Imperial High',
       'WLTP Imperial Extra High', 'WLTP Imperial Combined',
       'WLTP Imperial Combined (Weighted)', 'Diesel VED Supplement'], axis = 1)

    # inputting missing values
    columns_to_impute = ['Maximum range (Miles)', 'WLTP Metric Low', 'WLTP Metric Medium',
       'WLTP Metric High', 'WLTP Metric Extra High', 'WLTP Metric Combined',
       'WLTP Metric Combined (Weighted)', 'WLTP CO2', 'WLTP CO2 Weighted',
       'Equivalent All Electric Range Miles', 'Equivalent All Electric Range KM', 'Electric Range City Miles',
       'Electric Range City Km', 'Emissions CO [mg/km]', 'THC Emissions [mg/km]',
       'Emissions NOx [mg/km]', 'THC + NOx Emissions [mg/km]',
       'Particulates [No.] [mg/km]', 'RDE NOx Urban', 'RDE NOx Combined']
    imputer = SimpleImputer(strategy='mean')
    df[columns_to_impute] = imputer.fit_transform(df[columns_to_impute])

    # Converting object datatype to categorical
    for feature in df.columns: # Loop through all columns in the dataframe
        if df[feature].dtype == 'object': # Only apply for columns with categorical strings
            df[feature] = pd.Categorical(df[feature])# Replace strings with an integer

    # Renaming columns
    for col in df.columns:
        if ' ' in col:
            new_col = col.replace(' ', '_')
            df.rename(columns={col: new_col}, inplace=True)
    return df

def perform_regression(df):
    # independent variables
    X = df.drop(["WLTP_CO2"], axis=1)
    # dependent variable 
    y = df["WLTP_CO2"]

    X = pd.get_dummies(X, drop_first=True)   # Adding intercept to the dataset
    X = sm.add_constant(X)
    
    # Splitting X and y into train and test sets in a 70:30 ratio
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=1
    )
    
    olsmod = sm.OLS(y_train, X_train)
    olsres = olsmod.fit()
    
    # return the regression summary as a string
    return olsres.summary().as_text()

@click.command()
@click.option(
    "--raw_data_path",
    help="Location where the emmission data was saved"
)
@click.option(
    "--dest_path",
    help="Location where the resulting files will be saved"
)
def run_data_prep(raw_data_path: str, dest_path: str, dataset: str = "data"):
    # Load csv files
    df_train = read_dataframe(
        os.path.join(raw_data_path, f"{dataset}_emission_data.csv")
    )
    df_val = read_dataframe(
        os.path.join(raw_data_path, f"{dataset}_emission_data.csv")
    )
    df_test = read_dataframe(
        os.path.join(raw_data_path, f"{dataset}_tripdata_2022-03.parquet")
    )

    # Extract the target
    target = 'tip_amount'
    y_train = df_train[target].values
    y_val = df_val[target].values
    y_test = df_test[target].values

    # Fit the DictVectorizer and preprocess data
    dv = DictVectorizer()
    X_train, dv = preprocess(df_train, dv, fit_dv=True)
    X_val, _ = preprocess(df_val, dv, fit_dv=False)
    X_test, _ = preprocess(df_test, dv, fit_dv=False)

    # Create dest_path folder unless it already exists
    os.makedirs(dest_path, exist_ok=True)

    # Save DictVectorizer and datasets
    dump_pickle(dv, os.path.join(dest_path, "dv.pkl"))
    dump_pickle((X_train, y_train), os.path.join(dest_path, "train.pkl"))
    dump_pickle((X_val, y_val), os.path.join(dest_path, "val.pkl"))
    dump_pickle((X_test, y_test), os.path.join(dest_path, "test.pkl"))


if __name__ == '__main__':
    run_data_prep()