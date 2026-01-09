# %%
import numpy as np
import torch
import importlib
from datetime import datetime, timezone

from src.policies import CNNPolicy
from src.data_loading import load_and_split_data
import src.evaluation as evaluation

# %%
# Setup (run once)
RESOLUTION_MINUTES = 30
START_DATE_TRAIN = datetime(2022, 4, 28, 0, 0, 0, tzinfo=timezone.utc)
START_DATE_VALIDATION = datetime(2025, 7, 17, 0, 0, 0, tzinfo=timezone.utc)
START_DATE_TEST = datetime(2025, 10, 9, 0, 0, 0, tzinfo=timezone.utc)
END_DATE_TEST = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)

instrument_names = [
    'BTC-PERPETUAL',
    'ETH-PERPETUAL',
    'SOL_USDC-PERPETUAL',
    'XRP_USDC-PERPETUAL',
    'DOGE_USDC-PERPETUAL',
    'PAXG_USDC-PERPETUAL',
    'ADA_USDC-PERPETUAL',
    'AVAX_USDC-PERPETUAL',
    'DOT_USDC-PERPETUAL',
    'BNB_USDC-PERPETUAL',
    'UNI_USDC-PERPETUAL',
]
assets = [s.split('_')[0].split('-')[0].lower() for s in instrument_names]
features = ['high', 'low', 'close']

train_prices, validation_prices, test_prices, n_train_periods, n_validation_periods, n_test_periods, all_datetimes = load_and_split_data(
    instrument_names, features, START_DATE_TRAIN, START_DATE_VALIDATION, START_DATE_TEST, END_DATE_TEST, RESOLUTION_MINUTES
)

n_features, n_non_cash_assets, _ = train_prices.shape
n_recent_periods = 50
commission_rate = 0.0005
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

policy = CNNPolicy(n_features=n_features, n_recent_periods=n_recent_periods).to(device)

# Use small subset for fast testing (e.g., first 100 periods of validation)
test_forward_prices = validation_prices[:, :, :100]

# %%
# Reload and test (run this cell repeatedly as you modify src/evaluation.py)
importlib.reload(evaluation)

initial_portfolio = np.array([1] * (n_non_cash_assets + 1)) / (n_non_cash_assets + 1)

results, wb, w = evaluation.run_walk_forward_test(
    policy=policy,
    initial_portfolio_weights=initial_portfolio,
    initial_prices=train_prices,
    forward_prices=test_forward_prices,
    all_datetimes=all_datetimes,
    assets=assets,
    n_recent_periods=n_recent_periods,
    commission_rate=commission_rate,
    device=device,
    use_osbl=False,
    n_osbl_update_steps=None,
    optimizer=None,
)


# %%

wb - w

