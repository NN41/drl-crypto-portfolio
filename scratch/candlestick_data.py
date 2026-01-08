# %%

import numpy as np
import pandas as pd
import requests
from datetime import datetime, timezone
import time
import os

def datetime_to_unix(utc_datetime: datetime) -> int:
    """Converts a UTC-aware datetime object to a Unix timestamp (in ms)"""
    return int(utc_datetime.timestamp() * 1000)

def unix_to_datetime(unix_timestamp_ms: int) -> datetime:
    """Converts a Unix timestamp (in ms) to a UTC-aware datetime object"""
    return datetime.fromtimestamp(unix_timestamp_ms / 1000, tz=timezone.utc)

start_date = '2016-01-01'
end_date = '2026-01-01'
resolution = '30'
instrument_name = 'MATIC_USDC-PERPETUAL'

url = "https://www.deribit.com/api/v2/public/get_tradingview_chart_data"

start_dt = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
end_dt = datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)

start_timestamp = datetime_to_unix(start_dt)
end_timestamp = datetime_to_unix(end_dt)

resolution_minutes = 1440 if resolution == '1D' else int(resolution)
chunk_size_ms = 5000 * resolution_minutes * 60 * 1000

all_chunks = []
current_end_timestamp = end_timestamp
while current_end_timestamp > start_timestamp:
    current_start_timestamp = max(current_end_timestamp - chunk_size_ms, start_timestamp)
    print(f"Getting data for {unix_to_datetime(current_start_timestamp)} to {unix_to_datetime(current_end_timestamp)}...")
    params = {
        'instrument_name': instrument_name,
        'start_timestamp': current_start_timestamp,
        'end_timestamp': current_end_timestamp,
        'resolution': resolution
    }

    response = requests.get(url, params=params)
    response.raise_for_status()
    data = response.json()

    result = data['result']
    chunk_df = pd.DataFrame({
        'timestamp': result['ticks'],
        'open': result['open'],
        'high': result['high'],
        'low': result['low'],
        'close': result['close'],
        'volume': result['volume'],
        'cost': result['cost']
    })

    all_chunks.append(chunk_df)
    time.sleep(0.1)
    current_end_timestamp = current_start_timestamp

# OHLC at timestamp t covers [t, t+1), so the close of this interval and the open of the next interval should be roughly (if not exactly) equal
# 'volume' is the traded volume over the interval in terms of base currency, while 'cost' is in terms of quoted currency
df = pd.concat(all_chunks[::-1], ignore_index=True)
df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
df = df.sort_values(by='timestamp').reset_index(drop=True).drop_duplicates()

assert df['timestamp'].duplicated().sum() == 0
time_diffs = df['timestamp'].diff().dropna()
expected_diff = resolution_minutes * 60 * 1000
assert (time_diffs == expected_diff).all()

output_dir = './data/raw/ohlcv'
os.makedirs(output_dir, exist_ok=True)
filename = f"{instrument_name}_resolution_{resolution}.csv"
df.to_csv(os.path.join(output_dir, filename), index=False)

# %%