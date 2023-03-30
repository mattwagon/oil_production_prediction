from sklearn.linear_model import LinearRegression

def basic_mvp():

    train_start_date = pd.to_datetime('2007-02-01')
    train_end_date = pd.to_datetime('2019-12-31')
    test_start_date = pd.to_datetime('2020-01-01')
    test_end_date = pd.to_datetime('2022-12-31')
