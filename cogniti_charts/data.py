import pandas as pd, numpy as np
def load_prices(csv_path):
    df = pd.read_csv(csv_path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    return df
def label_patterns(df, lookback=60):
    ret = df["close"].pct_change().fillna(0.0)
    rolling_max = df["close"].rolling(lookback).max()
    breakout = (df["close"] >= rolling_max * 0.995)
    trend = ret.rolling(max(3, lookback//3)).mean().fillna(0.0)
    reversal = (trend.shift(1) * ret) < -0.0004
    vola = ret.rolling(max(5, lookback//4)).std().fillna(ret.std())
    consolidation = vola < vola.quantile(0.30)
    y = np.full(len(df), 1, dtype=int); y[consolidation]=1; y[reversal]=2; y[breakout]=0
    df=df.copy(); df["label"]=y; return df
def train_test_split_time(df, test_ratio=0.2, val_ratio=0.1):
    n=len(df); n_test=int(n*test_ratio); n_val=int(n*val_ratio)
    return df.iloc[: n-n_test-n_val], df.iloc[n-n_test-n_val: n-n_test], df.iloc[n-n_test:]
