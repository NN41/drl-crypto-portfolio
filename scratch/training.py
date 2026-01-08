# %%

import numpy as np
import pandas as pd
from datetime import timedelta, datetime, timezone
from pathlib import Path
import time

import torch
from torch.utils.tensorboard import SummaryWriter

from src.policies import CNNPolicy
from src.train_utils import geometrically_sample_batch_start_indices, uniformly_sample_batch_start_indices, run_one_epoch
from src.evaluation import run_walk_forward_test, calculate_performance_metrics
from src.model_io import save_model

commission_rate = 0.0005 # 0.0005 = 5 bips
n_recent_periods = 50 # number of periods passed to the policy to choose a portfolio
batch_size = 50 # with 2 assets, do x5.5 to match the number of training data points used per update; number of actions in a single batch
n_online_batches = 30
n_osbl_update_steps = 30

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f"Using device {device}")

def seed_everything(seed=42):
    # Complete deterministic behavior on GPU operations is difficult due to CUDA optimizations.
    # The following gives a good balance between reproducibility and performance.
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

# %%

RESOLUTION_MINUTES = 30
START_DATE = datetime(2021, 10, 17, 17, 0, 0, tzinfo=timezone.utc)
END_DATE = datetime(2025, 10, 15, 0, 30, 0, tzinfo=timezone.utc)
START_TEST_DATE = datetime(2025, 8, 24, 21, 0, 0, tzinfo=timezone.utc)

# 11-asset list from inspect_data.py REDUCED_LIST
asset_symbols = [
    'ADA_USDC-PERPETUAL',
    'AVAX_USDC-PERPETUAL',
    'BTC-PERPETUAL',
    'BNB_USDC-PERPETUAL',
    'DOGE_USDC-PERPETUAL',
    'ETH-PERPETUAL',
    'LINK_USDC-PERPETUAL',
    'PAXG_USDC-PERPETUAL',
    'SOL_USDC-PERPETUAL',
    'TRUMP_USDC-PERPETUAL',
    'XRP_USDC-PERPETUAL',
]
assets = [s.split('_')[0].split('-')[0].lower() for s in asset_symbols]

features = ['high', 'low', 'close'] # follow the standard order of the OHLC acronym O-H-L-C

# Load all asset data with datetime processing
data_dir = Path('./data/raw/ohlcv')
dfs = {}
for symbol in asset_symbols:
    csv_file = data_dir / f'{symbol}_resolution_30.csv'
    df = pd.read_csv(csv_file).sort_values('timestamp')
    df['datetime'] = pd.to_datetime(df['datetime'], utc=True) + timedelta(minutes=RESOLUTION_MINUTES)
    dfs[symbol] = df

# Forward-fill missing early data for each asset
for symbol in asset_symbols:
    df = dfs[symbol]
    first_valid_close = df['close'].dropna().iloc[0]

    # Create continuous datetime index from START_DATE to END_DATE
    date_range = pd.date_range(START_DATE, END_DATE, freq='30min', tz='UTC')
    df_full = df.set_index('datetime').reindex(date_range)

    # Forward-fill OHLC with first available close for missing early data
    missing_mask = df_full[features].isna().any(axis=1)
    for col in features:
        df_full.loc[missing_mask, col] = first_valid_close
    df_full = df_full.reset_index().rename(columns={'index': 'datetime'})
    dfs[symbol] = df_full

# Verify all dataframes have same datetime index
common_dates = dfs[asset_symbols[0]]['datetime'].values
for symbol in asset_symbols[1:]:
    assert np.array_equal(dfs[symbol]['datetime'].values, common_dates), f"Date mismatch for {symbol}"

# Stack all assets into single array: shape (n_features, 11, n_periods) as in paper
all_prices = np.stack([dfs[symbol][features].values for symbol in asset_symbols]).transpose(2, 0, 1)
all_datetimes = common_dates

# Calculate train/test split based on START_TEST_DATE
split_idx = np.searchsorted(all_datetimes, pd.Timestamp(START_TEST_DATE).to_datetime64())
n_train_periods = split_idx # 32504
n_test_periods = len(all_datetimes) - n_train_periods # 2456

train_prices = all_prices[:, :, :n_train_periods]
test_prices = all_prices[:, :, n_train_periods:]

# %%
seed_everything(seed=42)

n_features, n_non_cash_assets, n_train_periods = train_prices.shape
learning_rate = 1e-4
weight_decay = 1e-8
n_epochs = 1000 * 2
n_epochs_per_validation = 10
n_batches_per_epoch = 2000
geometric_parameter = 2e-5 # instead of 5e-5

n_available_periods = train_prices.shape[-1]
prices_array = train_prices

portfolio_vector_memory = np.ones((n_available_periods, n_non_cash_assets + 1)) / (n_non_cash_assets + 1)
policy = CNNPolicy(n_features=n_features, n_recent_periods=n_recent_periods).to(device)
optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate, weight_decay=weight_decay)

# %%

run_timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
writer = SummaryWriter(log_dir=f'runs/run_{run_timestamp}')
training_start_time = time.time()
for epoch in range(n_epochs):
    print(f"Epoch {epoch+1} / {n_epochs}")

    batch_start_indices = geometrically_sample_batch_start_indices(
        n_samples=n_batches_per_epoch, 
        n_available_periods=n_available_periods, 
        batch_size=batch_size,
        geometric_parameter=geometric_parameter,
        n_recent_periods=n_recent_periods
    )

    epoch_avg_log_return = run_one_epoch(
        prices_array=prices_array,
        batch_start_indices=batch_start_indices,
        portfolio_vector_memory=portfolio_vector_memory,
        policy=policy,
        optimizer=optimizer,
        n_recent_periods=n_recent_periods,
        batch_size=batch_size,
        device=device,
        commission_rate=commission_rate
    )

    print(f"\tEpoch avg log-return: {epoch_avg_log_return:.9f}")

    writer.add_scalar('Train/AvgLogReturn', epoch_avg_log_return, epoch+1)
    if epoch % int(n_epochs * 0.1) == 0:
        for name, param in policy.named_parameters():
            writer.add_histogram(f'Weights/{name}', param, epoch+1)
            if param.grad is not None:
                writer.add_histogram(f'Gradients/{name}', param.grad, epoch+1)

    if epoch % n_epochs_per_validation == 0:
        print(f"\tRunning validation...")
        initial_portfolio = np.array([1] * (n_non_cash_assets + 1)) / (n_non_cash_assets + 1)
        validation_results = run_walk_forward_test(
            policy=policy,
            initial_portfolio_weights=initial_portfolio,
            initial_prices=train_prices,
            forward_prices=test_prices,
            all_datetimes=all_datetimes,
            assets=assets,
            n_recent_periods=n_recent_periods,
            commission_rate=commission_rate,
            device=device,
            use_osbl=False,
            n_osbl_update_steps=None,
            optimizer=None,
        )
        validation_metrics = calculate_performance_metrics(validation_results, RESOLUTION_MINUTES)
        writer.add_scalar('Validation/Final_Accumulated_Portfolio_Value_Multiplier', validation_metrics['fAPV'], epoch+1)
        writer.add_scalar('Validation/Annualized_Sharpe_Ratio', validation_metrics['SR'], epoch+1)
        writer.add_scalar('Validation/Maximum_Drawdown', validation_metrics['MDD'], epoch+1)
        print(f"\tVALIDATION RESULTS: fAPV={validation_metrics['fAPV']:.4f}, SR={validation_metrics['SR']:.4f}, MDD={validation_metrics['MDD']:.4f}")

training_elapsed = time.time() - training_start_time
print(f"\nTraining completed in {training_elapsed/3600:.2f} hours ({training_elapsed/60:.1f} minutes)")
writer.close()

# %%
save_model(policy, optimizer, save_dir='./models', filename=f'cnn_policy_{run_timestamp}.pt', n_epochs=n_epochs, commission_rate=commission_rate, learning_rate=learning_rate, weight_decay=weight_decay, n_features=n_features, n_recent_periods=n_recent_periods)

# %%
