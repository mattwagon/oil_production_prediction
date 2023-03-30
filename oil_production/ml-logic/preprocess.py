import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def preprocess_features(df):

    # Replace missing NAN with median
    for feature in df.columns:
        df[feature].replace(np.nan, df[feature].median(), inplace=True)

    train_start_date = pd.to_datetime('2007-02-01')
    train_end_date = pd.to_datetime('2019-12-31')
    test_start_date = pd.to_datetime('2020-01-01')
    test_end_date = pd.to_datetime('2022-12-31')

    df_train = df[(df['Date']>=train_start_date) & (df['Date']<=train_end_date)]
    df_test = df[(df['Date']>=test_start_date) & (df['Date']<=test_end_date)]

    scaler_X = MinMaxScaler(feature_range=(0,1))
    scaler_y = MinMaxScaler(feature_range=(0,1))

    X_train = scaler_X.fit_transform(X_train.reshape(-1,X_train.shape[-1])).reshape(X_train.shape)
    X_test = scaler_X.transform(X_test.reshape(-1,X_test.shape[-1])).reshape(X_test.shape)
    y_train = scaler_y.fit_transform(y_train.reshape(-1,y_train.shape[-1])).reshape(y_train.shape)
    y_test = scaler_y.transform(y_test.reshape(-1,y_test.shape[1])).reshape(y_test.shape)
