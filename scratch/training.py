# %%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.networks import CNNPolicy

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f"Using device {device}")

def seed_everything(seed=42):
    # Complete deterministic behavior on GPU operations is difficult due to CUDA optimizations.
    # The following gives a good balance between reproducibility and performance.
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True 

seed_everything(seed=42)

# %%

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

# %%

def generate_consecutive_batch(batch_data_start_idx, n_recent_periods, batch_size, portfolio_vector_memory, all_prices):
    n_batch_periods = n_recent_periods + (batch_size - 1) + 1 # the total number of periods needed to run and evaluate the agent on a batch of consecutive timesteps
    batch_start_idx = batch_data_start_idx + n_recent_periods - 1 # at this index, we are at time t, and we have to choose w_t based on w_{t-1}
    batch_end_idx = batch_data_start_idx + n_batch_periods - 1

    batch_prices = all_prices[:, :, batch_data_start_idx:batch_end_idx+1] # +1 is crucial
    assert batch_prices.shape[-1] == n_batch_periods, "Number of periods in batch price history is different from expected"

    batch_normalized_price_histories = []
    for idx in range(n_recent_periods, n_batch_periods):
        start_idx_price_history = idx - n_recent_periods
        end_idx_price_history = idx - 1
        price_history = batch_prices[:, :, start_idx_price_history:end_idx_price_history+1]
        latest_close_prices = price_history[-1:, :, -1:] # first -1 selects close price, second -1 selects most recent one
        normalized_price_history = price_history / latest_close_prices
        batch_normalized_price_histories.append(normalized_price_history)
    batch_normalized_price_histories = np.stack(batch_normalized_price_histories)

    batch_current_close_prices = all_prices[-1:, :, batch_start_idx:batch_start_idx+batch_size]
    batch_next_close_prices = all_prices[-1:, :, batch_start_idx+1:batch_start_idx+batch_size+1]
    batch_price_relatives = np.expand_dims((batch_next_close_prices / batch_current_close_prices).transpose(2,0,1), axis=-1)

    batch_previous_weights = portfolio_vector_memory[batch_start_idx-1:batch_start_idx-1+batch_size]

    return {
        "data_start_idx": batch_data_start_idx,
        "first_decision_idx": batch_start_idx, # the index corresponding to decision time of the weights
        "normalized_price_histories": batch_normalized_price_histories,
        "price_relatives": batch_price_relatives,
        "previous_weights": batch_previous_weights
    }

def next_mu(mu, c_s, c_p, w_prime, w):
    return 1 / (1 - c_p * w[:, 0:1]) * (1 - c_p * w_prime[:, 0:1] - (c_s + c_p - c_s * c_p) * torch.sum(F.relu(w_prime[:, 1:] - mu * w[:, 1:]), dim=1, keepdim=True))

def approximate_mu(w_prev, y, w, commission_rate):
    w_prime = (w_prev * y) / torch.sum((w_prev * y), dim=1, keepdim=True)

    mu_0 = commission_rate * torch.sum(torch.abs(w_prime[:, 1:] - w[:, 1:]), dim=1, keepdim=True)
    mu_0 = torch.zeros_like(mu_0)
    all_mu = [mu_0]
    for i in range(10): # 10 is MORE than enough to get the consecutive change below 1e-7
        mu_prev = all_mu[-1]
        all_mu.append(next_mu(mu_prev, commission_rate, commission_rate, w_prime, w))

    mu = all_mu[-1]
    return mu

# %%

commission_rate = 0.0025 # 0.0005 = 5 bips
n_features, n_non_cash_assets, n_periods = all_prices.shape
n_recent_periods = 50
batch_size = 50

portfolio_vector_memory = np.ones((n_periods, n_non_cash_assets + 1)) / (n_non_cash_assets + 1)
valid_batch_data_start_indices = range(0, n_periods - n_recent_periods - batch_size + 1)

policy = CNNPolicy(n_features=n_features, n_recent_periods=n_recent_periods).to(device)
optimizer = torch.optim.Adam(policy.parameters(), lr=3e-5, weight_decay=1e-8)

policy.train()


# %%

n_epochs = 1000
n_batches_per_epoch = 2000
n_total_updates = n_epochs * n_batches_per_epoch

update_avg_log_returns = []

for epoch_number in range(n_epochs):
    print(f"Epoch {epoch_number+1} / {n_epochs}")
    batch_data_start_indices_for_epoch = np.random.choice(valid_batch_data_start_indices, size=n_batches_per_epoch)

    update_total_log_return = 0
    update_number_of_steps = 0

    for batch_number, batch_data_start_idx in enumerate(batch_data_start_indices_for_epoch):

        batch = generate_consecutive_batch(batch_data_start_idx, n_recent_periods, batch_size, portfolio_vector_memory, all_prices)
        batch_normalized_price_histories_tensor = torch.from_numpy(batch['normalized_price_histories']).float().to(device)
        batch_price_relatives_tensor = torch.from_numpy(batch['price_relatives']).float().to(device)
        batch_previous_weights_tensor = torch.from_numpy(batch['previous_weights']).float().to(device)

        batch_w = policy(batch_normalized_price_histories_tensor, batch_previous_weights_tensor)

        batch_y = torch.cat([
            torch.ones((batch_size, 1)).to(device),
            batch_price_relatives_tensor.squeeze(1).squeeze(-1)
        ], dim=1)

        # batch_mu = torch.ones(batch_size).to(device)
        batch_mu = approximate_mu(batch_previous_weights_tensor, batch_y, batch_w, commission_rate)

        batch_log_returns = torch.log(torch.sum(batch_y * batch_w, dim=1) * batch_mu)
        average_log_return = torch.mean(batch_log_returns)
        loss = -average_log_return

        update_total_log_return += np.sum(batch_log_returns.detach().cpu().numpy())
        update_number_of_steps += batch_log_returns.shape[0]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update portfolio memory vector
        batch_first_decision_idx = batch['first_decision_idx']
        batch_weights = batch_w.detach().cpu().numpy()
        portfolio_vector_memory[batch_first_decision_idx:batch_first_decision_idx+batch_size] = batch_weights

    print(f"Batch {(epoch_number+1)*n_batches_per_epoch} out of {n_total_updates}:")
    update_avg_log_return = update_total_log_return / update_number_of_steps
    update_avg_log_returns.append(update_avg_log_return)
    print(f"\tAvg log return since last update: {update_avg_log_return:.9f}")


# %%

fig, ax = plt.subplots()
ax.plot(update_avg_log_returns)
ax.plot(pd.Series(update_avg_log_returns).rolling(window=100).mean())
ax.set_ylim((0.0005,0.001))


# %%


