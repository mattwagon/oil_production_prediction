import os
import pandas as pd
import oil_production.ml_logic

def preprocess_and_train():
    '''
    - Retrieve and concatonate csv data
    - Clean and preprocess data
    - Train a keras model
    '''

    from oil_production.ml_logic.data import clean_data
    from oil_production.ml_logic.preprocess import preprocess_train, preprocess_test
    from oil_production.ml_logic.model import initialize_model, fit_model

    path = os.path.join('..', 'raw_data')
    file_names = os.listdir(path)
    csv_files = [f for f in file_names if f.endswith('.csv')]
    csv_files.sort()

    df = pd.DataFrame()
    for file in csv_files:
        file_path = os.path.join(path, file)
        df_aux = pd.read_csv(file_path)
        df = pd.concat([df, df_aux], ignore_index=True)

    # Clean data using data.py
    df = clean_data(df)

    # Create (X_train, y_train) without data leaks

    train_start_date = pd.to_datetime('2007-02-01')
    train_end_date = pd.to_datetime('2019-12-31')
    test_start_date = pd.to_datetime('2020-01-01')
    test_end_date = pd.to_datetime('2022-12-31')

    df_train = df[(df['Date']>=train_start_date) & (df['Date']<=train_end_date)]
    df_test = df[(df['Date']>=test_start_date) & (df['Date']<=test_end_date)]

    X_train, y_train = preprocess_train(df_train)
    X_test, y_test = preprocess_test(df_test)

    # Initialize and fit model

    model = initialize_model(X_train, y_train)

    model, history = fit_model(model)

    return model, history
