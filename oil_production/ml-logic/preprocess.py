import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def preprocess_features():

    # Replace missing NAN with median
    for feature in df.columns:
        df[feature].replace(np.nan, df[feature].median(), inplace=True)

    train_start_date = pd.to_datetime('2013-01-01')
    train_end_date = pd.to_datetime('2014-11-30')
    val_start_date = pd.to_datetime('2014-12-01')
    val_end_date = pd.to_datetime('2014-12-31')
    test_start_date = pd.to_datetime('2019-01-01')
    test_end_date = pd.to_datetime('2022-12-31')
    test2_start_date = pd.to_datetime('2009-01-01')
    test2_end_date = pd.to_datetime('2009-01-21')

    df_scale_fit = df[(df['Date'] >= train_start_date)
                      & (df['Date'] <= val_end_date)]
    df_train = df[(df['Date'] >= train_start_date)
                  & (df['Date'] <= train_end_date)]
    df_val = df[(df['Date'] >= val_start_date) & (df['Date'] <= val_end_date)]
    df_test = df[(df['Date'] >= test_start_date)
                 & (df['Date'] <= test_end_date)]
    df_test2 = df[(df['Date'] >= test2_start_date)
                  & (df['Date'] <= test2_end_date)]

    # Instantiate MinMaxScaler
    min_max_scaler = MinMaxScaler()

    # Fit
    min_max_scaler.fit(df_scale_fit.drop(columns=['Date','Qoil MPFM']))
    X = df.drop(columns=['Date', 'Qoil MPFM'])
    X_train = df_train.drop(columns=['Date', 'Qoil MPFM'])
    X_val = df_val.drop(columns=['Date', 'Qoil MPFM'])
    X_test = df_test.drop(columns=['Date', 'Qoil MPFM'])
    X_test2 = df_test2.drop(columns=['Date', 'Qoil MPFM'])

    # Transform
    X_train_scaled = min_max_scaler.transform(X_train)
    X_val_scaled = min_max_scaler.transform(X_val)
    X_test_scaled = min_max_scaler.transform(X_test)
    X_test2_scaled = min_max_scaler.transform(X_test2)

    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_val_scaled = pd.DataFrame(X_val_scaled, columns=X_val.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
    X_test2_scaled = pd.DataFrame(X_test2_scaled, columns=X_test.columns)
