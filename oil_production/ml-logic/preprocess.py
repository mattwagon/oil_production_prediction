import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def subsample_sequence(sequence, length, horizon):

    # Bounds of sampling
    last_possible = len(sequence) - length - horizon
    random_start = np.random.randint(0, last_possible)

    # Sample
    X = sequence.iloc[random_start:random_start + length, :].drop(columns=['Date'])
    y = sequence.iloc[random_start + length:random_start + length + horizon, :]['Qoil MPFM']

    return X, y

def get_X_y(sequence, length, horizon, number_of_samples):

        X, y = [], []

        # Do as many samples as specified
        for i in range(1, number_of_samples + 1):

            # Record sample X & y
            xi, yi = subsample_sequence(sequence, length, horizon)
            X.append(np.array(xi.values.T.tolist()).T) # Getting the right shape (sequences, observations,features)
            y.append(yi)

        return np.array(X), np.array(y)

def get_real_X_y(sequence, length, horizon):

    X, y = [], []

    # Do as many samples as specified
    for i in range(1,len(sequence)-length):

        # Record sample X & y
        xi = sequence.iloc[i:i + length, :].drop(columns=['Date'])
        yi = sequence.iloc[i + length:i + length + horizon, :]['Qoil MPFM']
        X.append(np.array(xi.values.T.tolist()).T) # Getting the right shape (sequences, observations,features)
        y.append(yi)

    return np.array(X), np.array(y)

def preprocess_features(df):

    X_train, y_train = get_X_y(sequence=df_train,
               length=500,
               horizon=1,
               number_of_samples=5000)

    X_test_real, y_test_real = get_real_X_y(sequence=df_test,
                                            length=500,
                                            horizon=1)

    scaler_X = MinMaxScaler(feature_range=(0,1))
    scaler_y = MinMaxScaler(feature_range=(0,1))

    X_train = scaler_X.fit_transform(X_train.reshape(-1,X_train.shape[-1])).reshape(X_train.shape)
    y_train = scaler_y.fit_transform(y_train.reshape(-1,y_train.shape[-1])).reshape(y_train.shape)

    X_test_real = scaler_X.transform(X_test_real.reshape(-1,X_test_real.shape[-1])).reshape(X_test_real.shape)

    return X_train, y_train
