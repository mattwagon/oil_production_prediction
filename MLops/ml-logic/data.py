import pandas as pd

def clean_data(df: pd.Dataframe) -> pd.Dataframe:

    remove = ["Sand Rate", "MPFM NTotal Count Rate", "MPFM N81 Count Rate", "MPFM N356 Count Rate", "MPFM N32 Count Rate", "MPFM GOR", "Downhole Gauge T", "Downhole Gauge P"]
    df.drop(columns=remove, inplace=True)

    corrected_drop = ['Qwat MPFM corrected', 'Qoil MPFM corrected', 'Qliq MPFM corrected', 'Qgas MPFM corrected']
    df.drop(columns=corrected_drop, inplace=True)

    choke_drop = ['Choke Opening Calc1', 'Choke Opening Calc2', 'Choke Measured', 'Choke Calculated', 'Choke CCR']
    df.drop(columns=choke_drop, inplace = True)

    # Cleaning datetime from Timestamp('2007-02-01 00:00:00+0100', tz='pytz.FixedOffset(60)') to Timestamp('2007-02-01 00:00:00')
    df['Date'] = pd.to_datetime(df['Date'])
    df['Date'] = df['Date'].dt.tz_convert(None)
    df['Date'] = pd.to_datetime(df['Date'].apply(lambda x: x + pd.DateOffset(hours=1)))

    # Drop columns where all values are missing
    # df = df.dropna(how='all', axis='columns')

    # Drop columns based on missing value threshold of >30%
    thresh = len(df) * 0.7
    df.dropna(thresh = thresh, axis = 1, inplace = True)

    # Drop features with std = 0
    null_std = df.iloc[:, 1:].loc[:, df.std(numeric_only=True) < .0000001].columns
    df.drop(columns=null_std)

    return df
