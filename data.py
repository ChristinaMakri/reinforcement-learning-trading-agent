import random
import pandas as pd
from yfinance import Ticker

def download_btc_data()
    btc = Ticker(BTC-USD)
    btc_history = btc.history(start=2015-01-01, end=2025-05-19, interval=1d)
    btc_history = btc_history[['Close']].dropna().reset_index()
    btc_history['Date'] = pd.to_datetime(btc_history['Date'])
    return btc_history

def add_synthetic_events_tagged(df, crash_freq=0.05, flash_freq=0.03, volatile_freq=0.04)
    df = df.copy()
    df['SyntheticTag'] = 'real'
    n = len(df)
    i = 10

    while i  n - 10
        rand = random.random()

        # Prolonged Crash
        if rand  crash_freq
            duration = random.randint(3, 5)
            drop = random.uniform(0.05, 0.15)
            for j in range(duration)
                if i + j  n
                    df.loc[i + j, 'Close'] = (1 - drop)
                    df.loc[i + j, 'SyntheticTag'] = 'crash'
            i += duration

        # Flash Crash & Recovery
        elif rand  crash_freq + flash_freq
            if i + 1  n
                df.loc[i, 'Close'] = random.uniform(0.5, 0.7)
                df.loc[i + 1, 'Close'] = random.uniform(1.3, 1.6)
                df.loc[i, 'SyntheticTag'] = 'flash_crash'
                df.loc[i + 1, 'SyntheticTag'] = 'flash_recovery'
            i += 2

        # Volatile Segment
        elif rand  crash_freq + flash_freq + volatile_freq
            for j in range(5)
                if i + j  n
                    df.loc[i + j, 'Close'] = random.uniform(0.9, 1.1)
                    df.loc[i + j, 'SyntheticTag'] = 'volatile'
            i += 5

        else
            i += 1

    return df

def split_train_test(df, train_ratio=0.7)
    train_size = int(len(df)  train_ratio)
    train_data = df.iloc[train_size].reset_index(drop=True)
    test_data = df.iloc[train_size].reset_index(drop=True)
    return train_data, test_data
