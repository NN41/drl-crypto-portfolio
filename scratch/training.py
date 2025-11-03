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

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.policies import CNNPolicy, BuyAndHoldPolicy, EqualWeightPolicy

commission_rate = 0.0005 # 0.0005 = 5 bips
n_recent_periods = 50 # number of periods passed to the policy to choose a portfolio
batch_size = 50 # int(50 * 5.5) # x5.5 to match the number of training data points used per update; number of actions in a single batch
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
features = ['high', 'low', 'close'] # follow the standard order of the OHLC acronym O-H-L-C
n_train_periods = 32504
n_test_periods = 2456
n_total_periods = n_train_periods + n_test_periods

all_prices = np.stack([
    df_btc[features].values,
    df_eth[features].values
]).transpose(2, 0, 1) # of shape (n_features, n_non_cash_assets, n_periods) as in paper

test_train_prices = all_prices[:, :, -n_total_periods:]
train_prices = test_train_prices[:, :, :n_train_periods]
test_prices = test_train_prices[:, :, -n_test_periods:]

all_datetimes = df_btc['datetime'].values[-n_total_periods:] # datetimes synchronized with the close price of each period

# %%


def geometrically_sample_batch_start_indices(n_samples, n_available_periods, batch_size, geometric_parameter, n_recent_periods):
    max_batch_start_index = n_available_periods - batch_size - 1 # must have enough data for batch_size consecutive actions plus one final reward
    min_batch_start_index = n_recent_periods - 1 # must have enough data for the price history
    chosen_indices = []
    while len(chosen_indices) < n_samples:
        geometric_sample = np.random.geometric(p=geometric_parameter, size=n_samples)
        potential_indices = max_batch_start_index - (geometric_sample - 1)
        valid_indices = potential_indices[potential_indices >= min_batch_start_index]
        chosen_indices.extend(valid_indices)
    return np.array(chosen_indices[:n_samples])

def prepare_batch_of_consecutive_periods(prices_array, portfolio_vector_memory, batch_start_idx, batch_size, n_recent_periods):
    '''
    Prepares a mini-batch of training data, consisting of a consecutive sequence of price histories, previous portfolio vectors and price relatives.
    In other words, for an action at time t, we have X_t and w_{t-1} and y_{t+1}
    '''
    batch_normalized_price_histories = []
    for action_idx in range(batch_start_idx, batch_start_idx+batch_size):
        price_history = prices_array[:, :, action_idx-n_recent_periods+1:action_idx+1]
        latest_close_prices = price_history[-1:, :, -1:]
        normalized_price_history = price_history / latest_close_prices
        batch_normalized_price_histories.append(normalized_price_history)
    batch_normalized_price_histories = np.array(batch_normalized_price_histories) # shape (batch_size, n_features, n_non_cash_assets, n_recent_periods)

    batch_previous_weights = portfolio_vector_memory[batch_start_idx-1:batch_start_idx-1+batch_size] # shape (batch_size, n_non_cash_assets+1)

    batch_previous_close_prices = prices_array[-1, :, batch_start_idx-1:batch_start_idx-1+batch_size]
    batch_current_close_prices = prices_array[-1, :, batch_start_idx:batch_start_idx+batch_size]
    batch_next_close_prices = prices_array[-1, :, batch_start_idx+1:batch_start_idx+1+batch_size]

    batch_next_price_relatives = (batch_next_close_prices / batch_current_close_prices).transpose(1,0) # shape (batch_size, n_non_cash_assets)
    batch_current_price_relatives = (batch_current_close_prices / batch_previous_close_prices).transpose(1,0) # shape (batch_size, n_non_cash_assets)

    assert np.abs((1 / batch_normalized_price_histories[:, -1, :, -2]) - batch_current_price_relatives).sum() < 1e-9
    assert np.abs(batch_current_price_relatives[1:] - batch_next_price_relatives[:-1]).sum() < 1e-16
    assert batch_normalized_price_histories.shape[-1] == n_recent_periods
    assert batch_normalized_price_histories.shape[0] == batch_size
    assert batch_previous_weights.shape[0] == batch_size
    assert batch_next_price_relatives.shape[0] == batch_size

    return {
        "batch_start_idx": batch_start_idx, # index corresponding to time t at which X_t is observed and action w_t is taken
        "normalized_price_histories": batch_normalized_price_histories, # X_t
        "current_price_relatives": batch_current_price_relatives, # y_t, the price relatives observable at time t, based on v_t and v_{t-1}, necessary to compute mu_t
        "previous_weights": batch_previous_weights, # w_{t-1}
        "next_price_relatives": batch_next_price_relatives, # y_{t+1}, necessary for the reward of taking action w_t
    }

def next_mu(mu, c_s, c_p, w_prime, w):
    '''
    Calculates the next element in the sequence that converges to the transaction remainder factor (mu).
    Arguments: sell commission rate (c_s), purchase commission rate (c_p), weights before reallocation at time t (w_prime),
    weights after reallocation at time t (w). A commission rate of 0.0001 means 0.01%.
    '''
    sum_of_relus = torch.sum(F.relu(w_prime[:, 1:] - mu * w[:, 1:]), dim=1, keepdim=True)
    main_part = (1 - c_p * w_prime[:, :1] - (c_s + c_p - c_s * c_p) * sum_of_relus)
    final_multiplier = 1 / (1 - c_p * w[:, :1])
    return final_multiplier * main_part

def approximate_mu(w_prev, y, w, commission_rate, train_mode=False, return_w_prime=False):
    '''
    Approximates the transaction remainder factor mu.
    All input tensors must include cash component and must contain a batch dimension.
    '''
    steps_train_mode = 10 # 10 is MORE than enough to get the consecutive change below 1e-7
    max_steps_test_mode = 50
    threshold = 1e-9

    w_prime = (w_prev * y) / torch.sum((w_prev * y), dim=1, keepdim=True)

    assert w_prev.dim() == w.dim() == y.dim() == 2, "All inputs must have batch dimension"
    assert np.mean(np.abs(np.sum(w.detach().cpu().numpy(), axis=1) - 1)) < 1e-6
    assert np.mean(np.abs(np.sum(w_prime.detach().cpu().numpy(), axis=1) - 1)) < 1e-6

    mu_0 = 1 - commission_rate * torch.sum(torch.abs(w_prime[:, 1:] - w[:, 1:]), dim=1, keepdim=True)

    all_mu = [mu_0]
    step = 0
    while True:
        mu_prev = all_mu[-1]
        mu_next = next_mu(mu_prev, commission_rate, commission_rate, w_prime, w)
        all_mu.append(mu_next)

        step += 1
        if train_mode:
            # policy network is in train mode, so we iterate until we have a nested gradient graph of X layers deep
            if step >= steps_train_mode:
                break
        else:
            # policy network is in eval mode, so we dynamically approximate until the errors between two consecutive iterations are all small enough,
            # or until we have reached the max number of steps.
            consecutive_errors = np.abs((mu_prev - mu_next).detach().cpu().numpy())
            if np.all(consecutive_errors < threshold) or (step >= max_steps_test_mode):
                break

    final_mu = all_mu[-1]
    if return_w_prime:
        return final_mu, w_prime
    else:
        return final_mu

def perform_one_minibatch_update(batch, policy, optimizer, device, batch_size, commission_rate):

    batch_normalized_price_histories = torch.from_numpy(batch['normalized_price_histories']).float().to(device)
    batch_previous_weights = torch.from_numpy(batch['previous_weights']).float().to(device)
    batch_current_price_relatives = torch.from_numpy(batch['current_price_relatives']).float().to(device)
    batch_next_price_relatives = torch.from_numpy(batch['next_price_relatives']).float().to(device)

    policy.train()
    batch_current_weights = policy(batch_normalized_price_histories, batch_previous_weights)

    batch_current_price_relatives = torch.cat([
        torch.ones((batch_size, 1)).to(device),
        batch_current_price_relatives
    ], dim=1)

    batch_next_price_relatives = torch.cat([
        torch.ones((batch_size, 1)).to(device),
        batch_next_price_relatives
    ], dim=1)

    batch_mu = approximate_mu(
        w_prev=batch_previous_weights,
        y=batch_current_price_relatives,
        w=batch_current_weights,
        commission_rate=commission_rate,
        train_mode=policy.training,
        return_w_prime=False
    )

    batch_log_returns = torch.log(torch.sum(batch_next_price_relatives * batch_current_weights, dim=1, keepdim=True) * batch_mu)
    average_log_return = torch.mean(batch_log_returns)
    loss = -average_log_return

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return batch_log_returns, batch_current_weights

def run_one_epoch(prices_array, batch_start_indices, portfolio_vector_memory, policy, optimizer, n_recent_periods, batch_size, device, commission_rate):

    epoch_total_log_return = 0
    epoch_number_of_steps = 0

    for batch_start_idx in batch_start_indices:

        batch = prepare_batch_of_consecutive_periods(prices_array, portfolio_vector_memory, batch_start_idx, batch_size, n_recent_periods)
        batch_log_returns, batch_weights = perform_one_minibatch_update(batch=batch, policy=policy, optimizer=optimizer, device=device, batch_size=batch_size, commission_rate=commission_rate)

        epoch_total_log_return += batch_log_returns.sum().item()
        epoch_number_of_steps += batch_log_returns.shape[0]

        # update portfolio memory vector
        batch_first_action_idx = batch['batch_start_idx']
        batch_weights = batch_weights.detach().cpu().numpy()
        portfolio_vector_memory[batch_first_action_idx:batch_first_action_idx+batch_size] = batch_weights

    epoch_avg_log_return = epoch_total_log_return / epoch_number_of_steps
    return epoch_avg_log_return

# %%

seed_everything(seed=42)

n_features, n_non_cash_assets, n_train_periods = train_prices.shape
learning_rate = 1e-4 # as opposed to the paper's 3e-5
weight_decay = 1e-8
batch_size = int(50 * 5.5)
n_epochs = 1000
n_batches_per_epoch = 2000
n_total_updates = n_epochs * n_batches_per_epoch
geometric_parameter = 5e-5

n_available_periods = train_prices.shape[-1]
prices_array = train_prices

portfolio_vector_memory = np.ones((n_train_periods, n_non_cash_assets + 1)) / (n_non_cash_assets + 1)
policy = CNNPolicy(n_features=n_features, n_recent_periods=n_recent_periods).to(device)
optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate, weight_decay=weight_decay)

# %%

run_timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")

writer = SummaryWriter(log_dir=f'runs/experiment_{run_timestamp}')
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

writer.close()

# %%
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

# # Load the trained model
# model_path = './models/pretrained/cnn_policy_20251030_094203.pt'
# checkpoint = torch.load(model_path, map_location=device)
# policy = CNNPolicy(n_features=checkpoint['n_features'], n_recent_periods=checkpoint['n_recent_periods']).to(device)
# policy.load_state_dict(checkpoint['model_state_dict'])
# print(f"Loaded model from {model_path}")
# print(f"Model trained for {checkpoint['n_epochs']} epochs with commission_rate={checkpoint['commission_rate']}")

# %%

# def run_walk_forward_test(policy, initial_portfolio_weights, train_prices, test_prices, all_datetimes, assets, n_recent_periods, commission_rate, device):
#     """
#     Run walk-forward test on test data with given policy and initial portfolio allocation.

#     Args:
#         policy: Callable that takes (normalized_price_history, previous_weights) and returns new_weights
#         initial_portfolio_weights: np.array of shape (n_assets,) including cash
#         train_prices: Training prices of shape (n_features, n_non_cash_assets, n_train_periods)
#         test_prices: Test prices of shape (n_features, n_non_cash_assets, n_test_periods)
#         all_datetimes: Datetimes for entire period (train + test)
#         assets: List of asset names (e.g., ['btc', 'eth'])
#         n_recent_periods: Number of recent periods for price history
#         commission_rate: Transaction commission rate (decimal)
#         device: torch device

#     Returns:
#         pd.DataFrame with test results
#     """
#     n_train_periods = train_prices.shape[-1]
#     current_idx = n_train_periods - 1
#     seen_prices = train_prices.copy()
#     unseen_prices = test_prices.copy()
#     current_portfolio_vector_memory = np.tile(initial_portfolio_weights, (n_train_periods - 1, 1))

#     test_results = {
#         'log_returns': [], 
#         'datetimes': [], 
#         'indices': [], 
#         'relative_turnover': [], 
#         'transaction_remainder_factor': []
#     }
#     for asset in assets:
#         test_results[f'weight_before_{asset}'] = []
#         test_results[f'weight_{asset}'] = []
#         test_results[f'close_{asset}'] = []

#     test_results['indices'].append(current_idx)
#     test_results['datetimes'].append(all_datetimes[current_idx])
#     test_results['log_returns'].append(None)
#     test_results['relative_turnover'].append(None)
#     test_results['transaction_remainder_factor'].append(None)
#     for asset in assets:
#         test_results[f'weight_before_{asset}'].append(None)

#     while unseen_prices.shape[-1] > 0:
#         # At time t, at the start of Period t+1, we observe the price history up to and including time t.
#         price_history = seen_prices[:, :, -n_recent_periods:]
#         latest_close_prices = price_history[-1:, :, -1:]
#         normalized_price_history = price_history / latest_close_prices

#         previous_weights = current_portfolio_vector_memory[-1]

#         # At time t, we choose the weights for the coming period, Period t+1
#         with torch.no_grad():
#             normalized_price_tensor = torch.tensor(normalized_price_history, dtype=torch.float32, device=device)
#             previous_weights_tensor = torch.tensor(previous_weights, dtype=torch.float32, device=device)
#             new_weights = policy(normalized_price_tensor, previous_weights_tensor)
#             new_weights = new_weights.squeeze(0).cpu().numpy()

#         for i, asset in enumerate(assets):
#             test_results[f'weight_{asset}'].append(new_weights[i+1])
#             test_results[f'close_{asset}'].append(latest_close_prices.squeeze()[i])

#         # We take a step forward to time t+1, and we observe the new (close) prices and compute the price relatives
#         current_idx += 1
#         new_close = unseen_prices[-1, :, 0]
#         previous_close = latest_close_prices.squeeze()
#         price_relative = np.concatenate([[1.0], new_close / previous_close])

#         # At time t+1, we compute THIS IS WRONG WRONG WRONG. IT SHOULD BE DONE A STEP BACK. SEE EQUATION 21, 22
#         transaction_remainder_factor, weights_before_rebalancing = approximate_mu(
#             torch.from_numpy(previous_weights).unsqueeze(0), 
#             torch.from_numpy(price_relative).unsqueeze(0), 
#             torch.from_numpy(new_weights).unsqueeze(0), 
#             commission_rate, 
#             return_w_prime=True
#         )
#         transaction_remainder_factor = transaction_remainder_factor.item()
#         weights_before_rebalancing = weights_before_rebalancing.squeeze(0).numpy()
#         log_return = np.log(transaction_remainder_factor * (price_relative @ previous_weights)) # this is the wrong log return
#         relative_turnover = (1 - transaction_remainder_factor) / commission_rate

#         test_results['indices'].append(current_idx)
#         test_results['datetimes'].append(all_datetimes[current_idx])
#         test_results['log_returns'].append(log_return)
#         test_results['relative_turnover'].append(relative_turnover)
#         test_results['transaction_remainder_factor'].append(transaction_remainder_factor)
#         for i, asset in enumerate(assets):
#             test_results[f'weight_before_{asset}'].append(weights_before_rebalancing[i+1])

#         current_portfolio_vector_memory = np.concatenate([current_portfolio_vector_memory, new_weights[np.newaxis, :]], axis=0)
#         seen_prices = np.concatenate([seen_prices, unseen_prices[:, :, :1]], axis=-1)
#         unseen_prices = unseen_prices[:, :, 1:]

#     for i, asset in enumerate(assets):
#         test_results[f'weight_{asset}'].append(None)
#         test_results[f'close_{asset}'].append(new_close[i])

#     return pd.DataFrame(test_results)

# %%

initial_portfolio_weights = np.array([1.,0.,0.])
policy.eval()

seen_prices = train_prices.copy()
unseen_prices = test_prices.copy()

n_train_periods = seen_prices.shape[-1]
current_idx = n_train_periods - 1

current_portfolio_vector_memory = np.ones((n_train_periods - 1, n_non_cash_assets + 1)) / (n_non_cash_assets + 1)
current_portfolio_vector_memory[-1] = initial_portfolio_weights

test_results = {
    'log_returns': [], 
    'datetimes': [], 
    'indices': [], 
    'relative_turnover': [], 
    'transaction_remainder_factor': []
}
for i, asset in enumerate(assets):
    test_results[f'weight_before_{asset}'] = []
    test_results[f'weight_{asset}'] = []
    test_results[f'close_{asset}'] = []

previous_transaction_remainder_factor = 1.0

while unseen_prices.shape[-1] > 0:

    # At time t, at the start of Period t+1, we observe the price history up to and including time t.
    price_history = seen_prices[:, :, -n_recent_periods:]
    latest_close_prices = price_history[-1:, :, -1:]
    normalized_price_history = price_history / latest_close_prices

    previous_weights = current_portfolio_vector_memory[-1]

    price_relative = 1 / normalized_price_history[-1, :, -2]
    price_relative = np.insert(price_relative, 0, 1.0)

    log_return_prime = np.log(previous_transaction_remainder_factor * (previous_weights @ price_relative))

    test_results['indices'].append(current_idx)
    test_results['datetimes'].append(all_datetimes[current_idx])
    test_results['log_returns'].append(log_return_prime)

    # At time t, we choose the weights for the coming period, Period t+1
    with torch.no_grad():
        normalized_price_tensor = torch.tensor(normalized_price_history, dtype=torch.float32, device=device)
        previous_weights_tensor = torch.tensor(previous_weights, dtype=torch.float32, device=device)
        new_weights = policy(normalized_price_tensor, previous_weights_tensor)
        new_weights = new_weights.squeeze(0).cpu().numpy()

    transaction_remainder_factor, weights_before_rebalancing = approximate_mu(
        torch.from_numpy(previous_weights).unsqueeze(0), 
        torch.from_numpy(price_relative).unsqueeze(0), 
        torch.from_numpy(new_weights).unsqueeze(0), 
        commission_rate,
        train_mode=False, 
        return_w_prime=True
    )
    transaction_remainder_factor = transaction_remainder_factor.item()
    weights_before_rebalancing = weights_before_rebalancing.squeeze(0).numpy()
    relative_turnover = (1 - transaction_remainder_factor) / commission_rate

    test_results['relative_turnover'].append(relative_turnover)
    test_results['transaction_remainder_factor'].append(transaction_remainder_factor)
    for i, asset in enumerate(assets):
        test_results[f'weight_before_{asset}'].append(weights_before_rebalancing[i+1])
        test_results[f'weight_{asset}'].append(new_weights[i+1])
        test_results[f'close_{asset}'].append(latest_close_prices.squeeze()[i])

    previous_transaction_remainder_factor = transaction_remainder_factor
    current_portfolio_vector_memory = np.concatenate([current_portfolio_vector_memory, new_weights[np.newaxis, :]], axis=0)

    # take a step and observe new prices
    current_idx += 1
    seen_prices = np.concatenate([seen_prices, unseen_prices[:, :, :1]], axis=-1)
    unseen_prices = unseen_prices[:, :, 1:]

df_results = pd.DataFrame(test_results)

# %%


# def calculate_performance_metrics(df_results, resolution_minutes):
#     """Calculate performance metrics from walk-forward test results."""
#     step_log_returns = df_results['log_returns'].dropna().values
#     step_returns = np.exp(step_log_returns) - 1
#     running_log_returns = np.cumsum(step_log_returns)

#     apv_ratios = np.exp(running_log_returns)
#     portfolio_values = np.concatenate([[1.0], apv_ratios])

#     running_max = np.maximum.accumulate(portfolio_values)
#     running_drawdown = (running_max - portfolio_values) / running_max
#     running_max_drawdown = np.maximum.accumulate(running_drawdown)

#     periodic_sharpe = np.mean(step_returns) / np.sqrt(np.var(step_returns, ddof=1))
#     annualized_sharpe = np.sqrt(365 * 24 * 60 / resolution_minutes) * periodic_sharpe

#     return {
#         'fAPV': apv_ratios[-1],
#         'SR': annualized_sharpe,
#         'MDD': running_max_drawdown[-1],
#         'apv_ratios': apv_ratios,
#         'portfolio_values': portfolio_values,
#         'running_drawdown': running_drawdown,
#         'running_max_drawdown': running_max_drawdown
#     }

step_log_returns = df_results['log_returns'].values[1:]
step_returns = np.exp(step_log_returns) - 1
running_log_returns = np.cumsum(step_log_returns)







# %%


# Add performance metrics
step_log_returns = df_test_results['log_returns'].dropna().values
step_returns = np.exp(step_log_returns) - 1
running_log_returns = np.cumsum(step_log_returns)

portfolio_value_start = 1
apv_ratios = np.exp(running_log_returns)
portfolio_values = np.concatenate([[portfolio_value_start], portfolio_value_start * apv_ratios])

running_returns = apv_ratios - 1

running_max = np.maximum.accumulate(portfolio_values)
running_drawdown = (running_max - portfolio_values) / running_max
running_max_drawdown = np.maximum.accumulate(running_drawdown)

# Add to dataframe (all arrays are length n_test_periods + 1, matching df_test_results)
df_test_results['returns'] = np.concatenate([[np.nan], step_returns])
df_test_results['running_log_returns'] = np.concatenate([[np.nan], running_log_returns])
df_test_results['apv_ratios'] = np.concatenate([[np.nan], apv_ratios])
df_test_results['portfolio_values'] = portfolio_values
df_test_results['running_returns'] = np.concatenate([[np.nan], running_returns])
df_test_results['running_max'] = running_max
df_test_results['running_drawdown'] = running_drawdown
df_test_results['running_max_drawdown'] = running_max_drawdown

# Calculate agent summary statistics
risk_free_return = 0
periodic_sharpe_numerator = np.mean(step_returns - risk_free_return)
periodic_sharpe_denominator = np.sqrt(np.var(step_returns - risk_free_return, ddof=1))
periodic_sharpe_ratio = periodic_sharpe_numerator / periodic_sharpe_denominator
annualized_sharpe_ratio = np.sqrt(365 * 24 * 60 / RESOLUTION_MINUTES) * periodic_sharpe_ratio

# Calculate BTC benchmark metrics
btc_log_returns = np.log(df_test_results['close_btc'] / df_test_results['close_btc'].shift(1)).dropna()
btc_returns = np.exp(btc_log_returns) - 1
btc_final_apv = df_test_results['close_btc'].iloc[-1] / df_test_results['close_btc'].iloc[0]
btc_sharpe = np.sqrt(365 * 24 * 60 / RESOLUTION_MINUTES) * np.mean(btc_returns) / np.sqrt(np.var(btc_returns, ddof=1))

# Calculate ETH benchmark metrics
eth_log_returns = np.log(df_test_results['close_eth'] / df_test_results['close_eth'].shift(1)).dropna()
eth_returns = np.exp(eth_log_returns) - 1
eth_final_apv = df_test_results['close_eth'].iloc[-1] / df_test_results['close_eth'].iloc[0]
eth_sharpe = np.sqrt(365 * 24 * 60 / RESOLUTION_MINUTES) * np.mean(eth_returns) / np.sqrt(np.var(eth_returns, ddof=1))

# Print summary statistics
print(f"Agent: fAPV={apv_ratios[-1]:.4f}, SR={annualized_sharpe_ratio:.4f}, MDD={running_max_drawdown[-1]:.4f}")
print(f"BTC:   fAPV={btc_final_apv:.4f}, SR={btc_sharpe:.4f}")
print(f"ETH:   fAPV={eth_final_apv:.4f}, SR={eth_sharpe:.4f}")

# %%

# Calculate value multipliers for benchmarks
btc_value_multipliers = df_test_results['close_btc'] / df_test_results['close_btc'].iloc[0]
eth_value_multipliers = df_test_results['close_eth'] / df_test_results['close_eth'].iloc[0]

# Extract weights data (excluding first and last rows which have None)
weights_btc = df_test_results['weight_btc'].dropna().values
weights_eth = df_test_results['weight_eth'].dropna().values
weights_btc_before = df_test_results['weight_before_btc'].dropna().values
weights_eth_before = df_test_results['weight_before_eth'].dropna().values
weights_datetimes = df_test_results['datetimes'].iloc[1:-1].values

# Prepare data for plotting
plot_datetimes = df_test_results['datetimes'].values
apv_plot = df_test_results['apv_ratios'].dropna().values
apv_datetimes = df_test_results['datetimes'].iloc[1:].values

fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06, subplot_titles=('Value Multipliers', 'Weights', 'Drawdown'))
fig.add_trace(go.Scatter(x=plot_datetimes, y=btc_value_multipliers, mode='lines', name='BTC', line=dict(color='orange')), row=1, col=1)
fig.add_trace(go.Scatter(x=plot_datetimes, y=eth_value_multipliers, mode='lines', name='ETH', line=dict(color='blue')), row=1, col=1)
fig.add_trace(go.Scatter(x=apv_datetimes, y=apv_plot, mode='lines', name='Agent', line=dict(color='black')), row=1, col=1)
fig.add_trace(go.Scatter(x=weights_datetimes, y=weights_btc, mode='lines', name='BTC Target', line=dict(color='orange', dash='solid')), row=2, col=1)
fig.add_trace(go.Scatter(x=weights_datetimes, y=weights_eth, mode='lines', name='ETH Target', line=dict(color='blue', dash='solid')), row=2, col=1)
fig.add_trace(go.Scatter(x=weights_datetimes, y=weights_btc_before, mode='lines', name='BTC Before', line=dict(color='orange', dash='dash')), row=2, col=1)
fig.add_trace(go.Scatter(x=weights_datetimes, y=weights_eth_before, mode='lines', name='ETH Before', line=dict(color='blue', dash='dash')), row=2, col=1)
fig.add_trace(go.Scatter(x=plot_datetimes, y=df_test_results['running_drawdown'], mode='lines', name='Running Drawdown', line=dict(color='orange')), row=3, col=1)
fig.add_trace(go.Scatter(x=plot_datetimes, y=df_test_results['running_max_drawdown'], mode='lines', name='Running Max Drawdown', line=dict(color='red')), row=3, col=1)
fig.update_layout(height=1000, showlegend=True, hovermode='x unified')
fig.update_xaxes(title_text='Date', row=3, col=1)
fig.update_yaxes(title_text='Value Multiplier', row=1, col=1)
fig.update_yaxes(title_text='Weight', row=2, col=1)
fig.update_yaxes(title_text='Drawdown', row=3, col=1)
fig.show()

# %%
# Test CNN Policy using new function
policy.eval()

print("\nTesting CNN Policy with new function...")
df_cnn_new = run_walk_forward_test(
    policy=policy,
    initial_portfolio_weights=np.array([0, 0.5, 0.5]),
    train_prices=train_prices,
    test_prices=test_prices,
    all_datetimes=all_datetimes,
    assets=assets,
    n_recent_periods=n_recent_periods,
    commission_rate=commission_rate,
    device=device
)
cnn_metrics = calculate_performance_metrics(df_cnn_new, RESOLUTION_MINUTES)
print(f"CNN Policy: fAPV={cnn_metrics['fAPV']:.4f}, SR={cnn_metrics['SR']:.4f}, MDD={cnn_metrics['MDD']:.4f}")

# %%
# Test Equal Weight Policy
print("\nTesting Equal Weight Policy...")
equal_weight_policy = EqualWeightPolicy(n_non_cash_assets=2)
df_equal_weight = run_walk_forward_test(
    policy=equal_weight_policy,
    initial_portfolio_weights=np.array([0, 0.5, 0.5]),
    train_prices=train_prices,
    test_prices=test_prices,
    all_datetimes=all_datetimes,
    assets=assets,
    n_recent_periods=n_recent_periods,
    commission_rate=commission_rate,
    device=device
)
equal_weight_metrics = calculate_performance_metrics(df_equal_weight, RESOLUTION_MINUTES)
print(f"Equal Weight: fAPV={equal_weight_metrics['fAPV']:.4f}, SR={equal_weight_metrics['SR']:.4f}, MDD={equal_weight_metrics['MDD']:.4f}")

# %%
# Test Buy-and-Hold BTC Policy
print("\nTesting Buy-and-Hold BTC Policy...")
buy_hold_btc_policy = BuyAndHoldPolicy()
df_buy_hold_btc = run_walk_forward_test(
    policy=buy_hold_btc_policy,
    initial_portfolio_weights=np.array([0, 0.5, 0.5]),
    train_prices=train_prices,
    test_prices=test_prices,
    all_datetimes=all_datetimes,
    assets=assets,
    n_recent_periods=n_recent_periods,
    commission_rate=commission_rate,
    device=device
)
buy_hold_btc_metrics = calculate_performance_metrics(df_buy_hold_btc, RESOLUTION_MINUTES)
print(f"Buy-Hold BTC: fAPV={buy_hold_btc_metrics['fAPV']:.4f}, SR={buy_hold_btc_metrics['SR']:.4f}, MDD={buy_hold_btc_metrics['MDD']:.4f}")

# %%
# Comparison Summary Table
print("\n" + "="*80)
print("STRATEGY COMPARISON SUMMARY")
print("="*80)

comparison_data = {
    'Strategy': ['CNN Policy', 'Equal Weight', 'Buy-Hold BTC'],
    'fAPV': [cnn_metrics['fAPV'], equal_weight_metrics['fAPV'], buy_hold_btc_metrics['fAPV']],
    'SR': [cnn_metrics['SR'], equal_weight_metrics['SR'], buy_hold_btc_metrics['SR']],
    'MDD': [cnn_metrics['MDD'], equal_weight_metrics['MDD'], buy_hold_btc_metrics['MDD']]
}
df_comparison = pd.DataFrame(comparison_data)
print(df_comparison.to_string(index=False))
print("="*80)

# %%
# Comparative Visualization - 3 panels
apv_datetimes = df_cnn_new['datetimes'].iloc[1:].values
weights_datetimes = df_cnn_new['datetimes'].iloc[1:-1].values

fig_comparison = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06, subplot_titles=('Value Multipliers', 'Weights', 'Drawdown'))

# Panel 1: Value Multipliers for all strategies
fig_comparison.add_trace(go.Scatter(x=apv_datetimes, y=cnn_metrics['apv_ratios'], mode='lines', name='CNN Policy', line=dict(color='black', width=2)), row=1, col=1)
fig_comparison.add_trace(go.Scatter(x=apv_datetimes, y=equal_weight_metrics['apv_ratios'], mode='lines', name='Equal Weight', line=dict(color='green', width=2)), row=1, col=1)
fig_comparison.add_trace(go.Scatter(x=apv_datetimes, y=buy_hold_btc_metrics['apv_ratios'], mode='lines', name='Buy-Hold BTC', line=dict(color='orange', width=2)), row=1, col=1)

# Panel 2: Weights for all strategies
fig_comparison.add_trace(go.Scatter(x=weights_datetimes, y=df_cnn_new['weight_btc'].dropna().values, mode='lines', name='CNN BTC', line=dict(color='black', dash='solid')), row=2, col=1)
fig_comparison.add_trace(go.Scatter(x=weights_datetimes, y=df_cnn_new['weight_eth'].dropna().values, mode='lines', name='CNN ETH', line=dict(color='gray', dash='solid')), row=2, col=1)
fig_comparison.add_trace(go.Scatter(x=weights_datetimes, y=df_equal_weight['weight_btc'].dropna().values, mode='lines', name='EW BTC', line=dict(color='green', dash='dash')), row=2, col=1)
fig_comparison.add_trace(go.Scatter(x=weights_datetimes, y=df_equal_weight['weight_eth'].dropna().values, mode='lines', name='EW ETH', line=dict(color='lightgreen', dash='dash')), row=2, col=1)
fig_comparison.add_trace(go.Scatter(x=weights_datetimes, y=df_buy_hold_btc['weight_btc'].dropna().values, mode='lines', name='BH BTC', line=dict(color='orange', dash='dot')), row=2, col=1)
fig_comparison.add_trace(go.Scatter(x=weights_datetimes, y=df_buy_hold_btc['weight_eth'].dropna().values, mode='lines', name='BH ETH', line=dict(color='lightsalmon', dash='dot')), row=2, col=1)

# Panel 3: Drawdown for all strategies
plot_datetimes = df_cnn_new['datetimes'].values
fig_comparison.add_trace(go.Scatter(x=plot_datetimes, y=cnn_metrics['running_drawdown'], mode='lines', name='CNN DD', line=dict(color='black')), row=3, col=1)
fig_comparison.add_trace(go.Scatter(x=plot_datetimes, y=equal_weight_metrics['running_drawdown'], mode='lines', name='EW DD', line=dict(color='green')), row=3, col=1)
fig_comparison.add_trace(go.Scatter(x=plot_datetimes, y=buy_hold_btc_metrics['running_drawdown'], mode='lines', name='BH DD', line=dict(color='orange')), row=3, col=1)

fig_comparison.update_layout(height=1000, showlegend=True, hovermode='x unified')
fig_comparison.update_xaxes(title_text='Date', row=3, col=1)
fig_comparison.update_yaxes(title_text='Value Multiplier', row=1, col=1)
fig_comparison.update_yaxes(title_text='Weight', row=2, col=1)
fig_comparison.update_yaxes(title_text='Drawdown', row=3, col=1)
fig_comparison.show()

# %%


# online_epoch = 0
# print(f"Online epoch {online_epoch+1} / {n_online_epochs}")
# valid_online_batch_data_start_indices = sample_valid_online_batch_data_start_indices(current_train_prices)
# online_epoch_avg_log_return = run_one_epoch(valid_online_batch_data_start_indices, current_portfolio_vector_memory, current_train_prices, policy, optimizer)
# print(f"\tEpoch avg log-return: {online_epoch_avg_log_return:.9f}")
# writer.add_scalar('TrainOnline/AvgLogReturn', online_epoch_avg_log_return, online_epoch+1)

