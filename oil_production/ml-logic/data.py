import pandas as pd

def clean_data(df: pd.Dataframe) -> pd.Dataframe:

    # Drop choke, corrected, correlated features and others based on domain knowledge
    remove = ["Sand Rate", "MPFM NTotal Count Rate", "MPFM N81 Count Rate", "MPFM N356 Count Rate", "MPFM N32 Count Rate", "MPFM GOR", "Downhole Gauge T", "Downhole Gauge P", 'Qwat PC', 'Qgas PC', 'Qoil PC', 'Tubing dP', 'MPFM P', 'Qwat MPFM corrected', 'Qoil MPFM corrected', 'Qliq MPFM corrected', 'Qgas MPFM corrected', 'Choke Opening Calc1', 'Choke Opening Calc2', 'Choke Measured', 'Choke Calculated', 'Choke CCR', 'MPFM CF GOR', 'MPFM CF GOR']

    df.drop(columns=remove, inplace=True)

    df = df.drop_duplicates()

    # Cleaning datetime from Timestamp('2007-02-01 00:00:00+0100', tz='pytz.FixedOffset(60)') to Timestamp('2007-02-01 00:00:00')
    df['Date'] = pd.to_datetime(df['Date'])
    df['Date'] = df['Date'].dt.tz_convert(None)
    df['Date'] = pd.to_datetime(df['Date'].apply(lambda x: x + pd.DateOffset(hours=1)))

    # Drop columns based on missing value threshold of >30%
    thresh = len(df) * 0.7
    df.dropna(thresh = thresh, axis = 1, inplace = True)

    # Drop features with std = 0
    null_std = df.iloc[:, 1:].loc[:, df.std(numeric_only=True) < .0000001].columns
    df.drop(columns=null_std)

    print("✅ data cleaned")

    return df
