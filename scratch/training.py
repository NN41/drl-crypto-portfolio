# %%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta, datetime, timezone
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from src.networks import CNNPolicy

commission_rate = 0.0005 # 0.0005 = 5 bips
n_recent_periods = 50
batch_size = int(50 * 5.5) # x5.5 to match the number of training data points used per 
n_online_batches = 30
geometric_parameter = 5e-5

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

df_btc = pd.read_csv('./data/raw/ohlcv/BTC-PERPETUAL_resolution_30.csv').sort_values('timestamp')
df_eth = pd.read_csv('./data/raw/ohlcv/ETH-PERPETUAL_resolution_30.csv').sort_values('timestamp')

df_btc['datetime'] = pd.to_datetime(df_btc['datetime'], utc=True) + timedelta(minutes=RESOLUTION_MINUTES)
df_eth['datetime'] = pd.to_datetime(df_eth['datetime'], utc=True) + timedelta(minutes=RESOLUTION_MINUTES)

assets = ['btc', 'eth']
features = ['high', 'low', 'close'] # follow the standard order of the OHLC acronym (O, H, L, C)
n_train_periods = 32504
n_test_periods = 2456
n_total_periods = n_train_periods + n_test_periods

all_prices = np.stack([
    df_btc[features].values,
    df_eth[features].values
]).transpose(2, 0, 1) # of shape (n_features, n_non_cash_assets, n_periods) as in paper

test_train_prices = all_prices[:, :, -n_total_periods:]
train_prices = test_train_prices[:, :, :n_train_periods]
test_prices = test_train_prices[:, :, -n_test_periods:] # never seen before prices

# %%

def generate_consecutive_batch(batch_data_start_idx, n_recent_periods, batch_size, portfolio_vector_memory, prices_array):
    assert batch_data_start_idx >= 0, "Indexing must be non-negative (i.e. starting from the front, not from the back)"
    n_batch_periods = n_recent_periods + (batch_size - 1) + 1 # the total number of periods needed to run and evaluate the agent on a batch of consecutive timesteps
    batch_start_idx = batch_data_start_idx + n_recent_periods - 1 # at this index, we are at time t, and we have to choose w_t based on w_{t-1}
    batch_end_idx = batch_data_start_idx + n_batch_periods - 1

    batch_prices = prices_array[:, :, batch_data_start_idx:batch_end_idx+1] # +1 is crucial
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

    batch_current_close_prices = prices_array[-1:, :, batch_start_idx:batch_start_idx+batch_size]
    batch_next_close_prices = prices_array[-1:, :, batch_start_idx+1:batch_start_idx+batch_size+1]
    batch_price_relatives = np.expand_dims((batch_next_close_prices / batch_current_close_prices).transpose(2,0,1), axis=-1)

    batch_previous_weights = portfolio_vector_memory[batch_start_idx-1:batch_start_idx-1+batch_size]

    return {
        "data_start_idx": batch_data_start_idx,
        "first_decision_idx": batch_start_idx, # the index corresponding to decision time of the weights
        "last_decision_idx": batch_start_idx + batch_size - 1, # index corresponding to time of last decision
        "data_last_idx": batch_end_idx, # index of very last price datapoint needed for evaluation preceding decision
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

def perform_one_minibatch_update(batch, policy, optimizer, device=device, batch_size=batch_size, commission_rate=commission_rate):
    batch_normalized_price_histories_tensor = torch.from_numpy(batch['normalized_price_histories']).float().to(device)
    batch_price_relatives_tensor = torch.from_numpy(batch['price_relatives']).float().to(device)
    batch_previous_weights_tensor = torch.from_numpy(batch['previous_weights']).float().to(device)

    policy.train()
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

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return batch_log_returns, batch_w

def run_one_epoch(batch_data_start_indices_for_epoch, portfolio_vector_memory, train_prices, policy, optimizer, n_recent_periods=n_recent_periods, batch_size=batch_size, device=device, commission_rate=commission_rate):

    update_total_log_return = 0
    update_number_of_steps = 0

    for batch_number, batch_data_start_idx in enumerate(batch_data_start_indices_for_epoch):

        batch = generate_consecutive_batch(batch_data_start_idx, n_recent_periods, batch_size, portfolio_vector_memory, train_prices)
        batch_log_returns, batch_w = perform_one_minibatch_update(batch, policy, optimizer, device=device, batch_size=batch_size, commission_rate=commission_rate)

        update_total_log_return += np.sum(batch_log_returns.detach().cpu().numpy())
        update_number_of_steps += batch_log_returns.shape[0]

        # update portfolio memory vector
        batch_first_decision_idx = batch['first_decision_idx']
        batch_weights = batch_w.detach().cpu().numpy()
        portfolio_vector_memory[batch_first_decision_idx:batch_first_decision_idx+batch_size] = batch_weights

    update_avg_log_return = update_total_log_return / update_number_of_steps
    return update_avg_log_return

def sample_valid_online_batch_data_start_indices(current_train_prices, n_online_batches=n_online_batches, geometric_parameter=geometric_parameter):

    n_current_train_periods = current_train_prices.shape[-1]
    max_valid_index = n_current_train_periods - batch_size - (n_recent_periods - 1)

    valid_online_batch_data_start_indices = []
    while len(valid_online_batch_data_start_indices) < n_online_batches:
        geometric_sample = np.random.geometric(p=geometric_parameter, size=n_online_batches)
        indices_sample = max_valid_index - geometric_sample
        valid_indices_sample = indices_sample[indices_sample >= 0]
        valid_online_batch_data_start_indices.extend(valid_indices_sample)
    valid_online_batch_data_start_indices = np.array(valid_online_batch_data_start_indices)

    return valid_online_batch_data_start_indices

# %%
seed_everything(seed=42)

n_features, n_non_cash_assets, n_train_periods = train_prices.shape
learning_rate = 3e-5 #1e-4 # as opposed to the paper's 3e-5
n_epochs = 1000
n_batches_per_epoch = 2000
n_total_updates = n_epochs * n_batches_per_epoch

portfolio_vector_memory = np.ones((n_train_periods, n_non_cash_assets + 1)) / (n_non_cash_assets + 1)
valid_batch_data_start_indices = range(0, n_train_periods - n_recent_periods - batch_size + 1)

policy = CNNPolicy(n_features=n_features, n_recent_periods=n_recent_periods).to(device)
optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate, weight_decay=1e-8)


run_timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")

writer = SummaryWriter(log_dir=f'runs/experiment_{run_timestamp}')
for epoch in range(n_epochs):
    print(f"Epoch {epoch+1} / {n_epochs}")
    batch_data_start_indices_for_epoch = np.random.choice(valid_batch_data_start_indices, size=n_batches_per_epoch)
    epoch_avg_log_return = run_one_epoch(batch_data_start_indices_for_epoch, portfolio_vector_memory, train_prices, policy, optimizer)
    print(f"\tEpoch avg log-return: {epoch_avg_log_return:.9f}")

    writer.add_scalar('Train/AvgLogReturn', epoch_avg_log_return, epoch+1)
    for name, param in policy.named_parameters():
        writer.add_histogram(f'Weights/{name}', param, epoch+1)
        if param.grad is not None:
            writer.add_histogram(f'Gradients/{name}', param.grad, epoch+1)
writer.close()

# Save the trained model
save_dir = './models'
os.makedirs(save_dir, exist_ok=True)
model_path = os.path.join(save_dir, f'cnn_policy_{run_timestamp}.pt')
torch.save({
    'model_state_dict': policy.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'n_epochs': n_epochs,
    'commission_rate': commission_rate,
    'learning_rate': learning_rate,
    'n_features': n_features,
    'n_recent_periods': n_recent_periods
}, model_path)
print(f"Model saved to {model_path}")


# %%

n_online_epochs = test_prices.shape[-1]

remaining_next_prices = test_prices.copy()
current_train_prices = train_prices.copy()
current_portfolio_vector_memory = portfolio_vector_memory.copy()

latest_price_history = train_prices[:, :, -n_recent_periods:]
latest_close_prices = latest_price_history[-1:, :, -1:]
latest_normalized_price_history = (latest_price_history / latest_close_prices)[np.newaxis, :]

previous_weights = current_portfolio_vector_memory[np.newaxis, -2]

latest_normalized_price_history_tensor = torch.tensor(latest_normalized_price_history, dtype=torch.float32, device=device)
previous_weights_tensor = torch.tensor(previous_weights, dtype=torch.float32, device=device)

policy.eval()
with torch.no_grad():
    next_weights_tensor = policy(latest_normalized_price_history_tensor, previous_weights_tensor)

next_weights = next_weights_tensor.detach().cpu().numpy().squeeze()

current_portfolio_vector_memory[-1] = next_weights

# %%



online_epoch = 0
print(f"Online epoch {online_epoch+1} / {n_online_epochs}")
valid_online_batch_data_start_indices = sample_valid_online_batch_data_start_indices(current_train_prices)
online_epoch_avg_log_return = run_one_epoch(valid_online_batch_data_start_indices, current_portfolio_vector_memory, current_train_prices, policy, optimizer)
print(f"\tEpoch avg log-return: {online_epoch_avg_log_return:.9f}")
writer.add_scalar('TrainOnline/AvgLogReturn', online_epoch_avg_log_return, online_epoch+1)


# %%

# %%





# %%
current_portfolio_vector_memory = np.concat((current_portfolio_vector_memory, np.array([[0] * current_portfolio_vector_memory.shape[1]])))

# %%



next_prices = remaining_next_prices[:, :, 0:1]
remaining_next_prices = remaining_next_prices[:, :, 1:]

current_train_prices = np.concatenate([current_train_prices, next_prices], axis=2)

current_train_prices

# %%



n_current_train_periods = current_train_prices.shape[-1]
valid_batch_data_start_indices = range(0, n_current_train_periods - n_recent_periods - batch_size + 1)
valid_batch_data_start_indices

# %%


