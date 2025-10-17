# %%

import gymnasium as gym
import numpy as np
import pandas as pd
from typing import Optional
import matplotlib.pyplot as plt
from datetime import timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

RESOLUTION_MINUTES = 30

df_btc = pd.read_csv('./data/raw/ohlcv/BTC-PERPETUAL_resolution_30.csv').sort_values('timestamp')
df_eth = pd.read_csv('./data/raw/ohlcv/ETH-PERPETUAL_resolution_30.csv').sort_values('timestamp')

df_btc['datetime'] = pd.to_datetime(df_btc['datetime'], utc=True) + timedelta(minutes=RESOLUTION_MINUTES)
df_eth['datetime'] = pd.to_datetime(df_eth['datetime'], utc=True) + timedelta(minutes=RESOLUTION_MINUTES)

assets = ['btc', 'eth']
features = ['high', 'low', 'close'] # follow the standard order of the OHLC acronym (O, H, L, C)

all_prices = np.stack([
    df_btc[features].values,
    df_eth[features].values
]).transpose(2, 0, 1) # of shape (n_features, n_non_cash_assets, n_periods) as in paper

all_close_datetimes = df_btc['datetime'].values

n_features, n_non_cash_assets, n_periods = all_prices.shape
n_recent_periods = 50
batch_size = 500

n_batch_periods = n_recent_periods + (batch_size - 1) + 1 # the total number of periods needed to run and evaluate the agent on a batch of consecutive timesteps
batch_start_idx = np.random.randint(0, n_periods - n_batch_periods + 1)
batch_end_idx = batch_start_idx + n_batch_periods - 1

batch_prices = all_prices[:, :, batch_start_idx:batch_end_idx+1] # +1 is crucial
assert batch_prices.shape[-1] == n_batch_periods, "Number of periods in batch price history is different from expected"

# %%

batch_normalized_price_histories = []
batch_price_relatives = []
batch_previous_weights = []
for idx in range(n_recent_periods, n_batch_periods):

    start_idx_price_history = idx - n_recent_periods
    end_idx_price_history = idx - 1
    price_history = batch_prices[:, :, start_idx_price_history:end_idx_price_history+1]
    latest_close_prices = price_history[-1:, :, -1:] # first -1 selects close price, second -1 selects most recent one
    normalized_price_history = price_history / latest_close_prices

    next_close_prices = batch_prices[-1:, :, end_idx_price_history + 1].reshape(1,2,1)
    price_relatives = next_close_prices / latest_close_prices

    previous_weights = np.random.rand(n_non_cash_assets + 1)
    previous_weights = previous_weights / previous_weights.sum()

    batch_normalized_price_histories.append(normalized_price_history)
    batch_price_relatives.append(price_relatives)
    batch_previous_weights.append(previous_weights)

batch_normalized_price_histories = np.stack(batch_normalized_price_histories)
batch_price_relatives = np.stack(batch_price_relatives)
batch_previous_weights = np.stack(batch_previous_weights)

# %%

import torch
from src.networks import CNNPolicy

device = 'cpu'

policy = CNNPolicy(n_features=n_features, n_recent_periods=n_recent_periods)
optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

# %%

policy.train()

batch_normalized_price_histories_tensor = torch.from_numpy(batch_normalized_price_histories).float().to(device)
batch_price_relatives_tensor = torch.from_numpy(batch_price_relatives).float().to(device)
batch_previous_weights_tensor = torch.from_numpy(batch_previous_weights).float().to(device)

batch_w = policy(batch_normalized_price_histories_tensor, batch_previous_weights_tensor)

batch_y = torch.cat([
    torch.ones((batch_size, 1)),
    batch_price_relatives_tensor.squeeze(1).squeeze(-1)
], dim=1)

batch_mu = torch.ones(batch_size)

batch_log_returns = torch.log(torch.sum(batch_y * batch_w, dim=1) * batch_mu)
average_log_return = torch.mean(batch_log_returns)
loss = -average_log_return
print(average_log_return.item())

optimizer.zero_grad()
loss.backward()
optimizer.step()

print(f"  Cash bias value: {policy.cash_bias.item():.6f}")
