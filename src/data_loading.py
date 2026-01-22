# %%

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

    assert resolution_minutes == 30, "Code is not ready for a data resolution other than 30 minutes."

    data_dir = Path(data_dir)
    date_range = pd.date_range(start_date_train, end_date_test, freq='30min', tz='UTC') # includes end points

    all_datetimes = date_range.values

    n_periods = len(all_datetimes)
    n_non_cash_assets = len(instrument_names)
    n_features = len(features)

    all_prices = []
    all_availability_masks = []
    for instrument in instrument_names:
        df = pd.read_csv(data_dir / f'{instrument}_resolution_30.csv').sort_values('timestamp') # timestamp for Deribit OHLC data is concurrent with the open of the interval
        df['datetime'] = pd.to_datetime(df['datetime'], utc=True) + timedelta(minutes=resolution_minutes) # move the datetimes forward by 30 minutes to synchronize it with the close price

        first_valid_close = df['close'].dropna().iloc[0]
        df_full = df.set_index('datetime').reindex(date_range)[features]
        prices = df_full.fillna(first_valid_close).values # of shape (n_periods, n_features)
        all_prices.append(prices)

        availability_mask = np.logical_not(df_full['close'].isna().values) # True when data exists. Of shape (n_periods, )
        all_availability_masks.append(availability_mask)

    all_prices = np.stack(all_prices).transpose(2, 0, 1) # following paper: of shape (n_features, n_non_cash_assets, n_periods)
    all_availability_masks = np.stack(all_availability_masks).transpose(1, 0) # of shape (n_periods, n_non_cash_assets) - contains NO cash

    # Calculate train/validation/test split indices
    val_split_idx = np.searchsorted(all_datetimes, pd.Timestamp(start_date_validation).to_datetime64())
    test_split_idx = np.searchsorted(all_datetimes, pd.Timestamp(start_date_test).to_datetime64())

    train_prices = all_prices[:, :, :val_split_idx]
    validation_prices = all_prices[:, :, val_split_idx:test_split_idx]
    test_prices = all_prices[:, :, test_split_idx:]

    train_availability_mask = all_availability_masks[:val_split_idx, :]
    validation_availability_mask = all_availability_masks[val_split_idx:test_split_idx, :]
    test_availability_mask = all_availability_masks[test_split_idx:, :]

    n_train_periods = train_prices.shape[-1]
    n_validation_periods = validation_prices.shape[-1]
    n_test_periods = test_prices.shape[-1]

    assert n_train_periods + n_validation_periods + n_test_periods == n_periods
    assert all_prices.shape == (n_features, all_availability_masks.shape[1], all_availability_masks.shape[0])

    return train_prices, validation_prices, test_prices, n_train_periods, n_validation_periods, n_test_periods, all_datetimes, train_availability_mask, validation_availability_mask, test_availability_mask
