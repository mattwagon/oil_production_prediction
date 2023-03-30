import os
import pandas as pd






def preprocess_and_train():

    path = os.path.join('..', 'raw_data')
    file_names = os.listdir(path)
    csv_files = [f for f in file_names if f.endswith('.csv')]
    csv_files.sort()

    df = pd.DataFrame()
    for file in csv_files:
        file_path = os.path.join(path, file)
        df_aux = pd.read_csv(file_path)
        df = pd.concat([df, df_aux], ignore_index=True)

    from oil_production.ml_logic.data import clean_data
