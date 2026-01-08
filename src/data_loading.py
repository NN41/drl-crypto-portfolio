import numpy as np
import pandas as pd
from datetime import timedelta
from pathlib import Path


def load_and_split_data(
    instrument_names,
    features,
    start_date_train,
    start_date_validation,
    start_date_test,
    end_date_test,
    resolution_minutes,
    data_dir='./data/raw/ohlcv'
):
    """
    Load OHLCV data for multiple instruments and split into train/validation/test sets,
    backfilling missing early data with the first available close price.

    Args:
        instrument_names: List of instrument names to load
        features: List of feature columns to extract (e.g., ['high', 'low', 'close'])
        start_date_train: Start date for training data
        start_date_validation: Start date for validation data
        start_date_test: Start date for test data
        end_date_test: End date for all data
        resolution_minutes: Time resolution in minutes
        data_dir: Directory containing OHLCV CSV files

    Returns:
        tuple: (train_prices, validation_prices, test_prices,
                n_train_periods, n_validation_periods, n_test_periods, all_datetimes)
            - *_prices: numpy arrays of shape (n_features, n_assets, n_periods)
            - n_*_periods: number of periods in each split
            - all_datetimes: numpy array of datetime64 values
    """
    assert resolution_minutes == 30, "Code is not ready for other resolutions"

    data_dir = Path(data_dir)
    date_range = pd.date_range(start_date_train, end_date_test, freq='30min', tz='UTC')
    dfs = {}
    for instrument in instrument_names:
        df = pd.read_csv(data_dir / f'{instrument}_resolution_30.csv').sort_values('timestamp')
        df['datetime'] = pd.to_datetime(df['datetime'], utc=True) + timedelta(minutes=resolution_minutes)
        first_valid_close = df['close'].dropna().iloc[0]
        df_full = df.set_index('datetime').reindex(date_range)
        df_full[features] = df_full[features].fillna(first_valid_close)
        dfs[instrument] = df_full.reset_index().rename(columns={'index': 'datetime'})

    # Calculate train/validation/test split indices
    all_datetimes = date_range.values
    val_split_idx = np.searchsorted(all_datetimes, pd.Timestamp(start_date_validation).to_datetime64())
    test_split_idx = np.searchsorted(all_datetimes, pd.Timestamp(start_date_test).to_datetime64())

    # Stack all assets into single array: shape (n_features, n_assets, n_periods)
    all_prices = np.stack([dfs[instrument][features].values for instrument in instrument_names]).transpose(2, 0, 1)
    train_prices = all_prices[:, :, :val_split_idx]
    validation_prices = all_prices[:, :, val_split_idx:test_split_idx]
    test_prices = all_prices[:, :, test_split_idx:]

    n_train_periods = train_prices.shape[-1]
    n_validation_periods = validation_prices.shape[-1]
    n_test_periods = test_prices.shape[-1]

    return train_prices, validation_prices, test_prices, n_train_periods, n_validation_periods, n_test_periods, all_datetimes