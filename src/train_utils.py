import numpy as np
import torch
import torch.nn.functional as F
from src.portfolio import approximate_mu


def prepare_batch_gpu(prices_tensor, portfolio_vector_memory, batch_start_idx, batch_size, n_recent_periods):
    """
    GPU-optimized batch preparation using vectorized torch ops instead of Python loop.
    """
    # Extract sliding windows using unfold - avoids Python loop over batch items
    start = batch_start_idx - n_recent_periods + 1
    all_prices = prices_tensor[:, :, start:batch_start_idx + batch_size]
    windows = all_prices.unfold(dimension=2, size=n_recent_periods, step=1)
    batch_price_histories = windows.permute(2, 0, 1, 3)  # (batch_size, F, M, n_recent_periods)

    # Normalize by latest close price
    latest_close_prices = batch_price_histories[:, -1:, :, -1:]
    batch_normalized_price_histories = batch_price_histories / latest_close_prices

    batch_previous_weights = portfolio_vector_memory[batch_start_idx-1:batch_start_idx-1+batch_size]

    # Price relatives from close prices (last feature)
    batch_previous_close = prices_tensor[-1, :, batch_start_idx-1:batch_start_idx-1+batch_size]
    batch_current_close = prices_tensor[-1, :, batch_start_idx:batch_start_idx+batch_size]
    batch_next_close = prices_tensor[-1, :, batch_start_idx+1:batch_start_idx+1+batch_size]
    batch_current_price_relatives = (batch_current_close / batch_previous_close).T
    batch_next_price_relatives = (batch_next_close / batch_current_close).T

    return {
        "batch_start_idx": batch_start_idx,
        "normalized_price_histories": batch_normalized_price_histories,
        "current_price_relatives": batch_current_price_relatives,
        "previous_weights": batch_previous_weights,
        "next_price_relatives": batch_next_price_relatives,
    }


def geometrically_sample_batch_start_indices(n_samples, n_available_periods, batch_size, geometric_parameter, n_recent_periods):
    max_batch_start_index = n_available_periods - batch_size - 1 # must have enough data for batch_size consecutive actions plus one final reward
    min_batch_start_index = n_recent_periods - 1 # must have enough data for the price history
    chosen_indices = []
    while len(chosen_indices) < n_samples:
        geometric_sample = np.random.geometric(p=geometric_parameter, size=n_samples)
        potential_indices = max_batch_start_index - (geometric_sample - 1)
        valid_indices = potential_indices[potential_indices >= min_batch_start_index]
        chosen_indices.extend(valid_indices)
    chosen_indices = np.array(chosen_indices[:n_samples])
    assert chosen_indices.min() >= min_batch_start_index, "Sampled batch start index is out of bounds"
    assert chosen_indices.max() <= max_batch_start_index, "Sampled batch start index is out of bounds"
    return chosen_indices

def prepare_batch_of_consecutive_periods(prices_array, portfolio_vector_memory, batch_start_idx, batch_size, n_recent_periods):
    '''
    Prepares a mini-batch of training data, consisting of a consecutive sequence of price histories, previous portfolio vectors and price relatives.
    In other words, for an action at time t, we have X_t and w_{t-1} and y_{t+1}
    '''
    assert prices_array.shape[1] == portfolio_vector_memory.shape[-1] - 1, "Number of assets in prices must match portfolio (excluding cash)"

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

def perform_one_minibatch_update(batch, policy, optimizer, device, batch_size, commission_rate):
    '''
    At time t, the agent chooses w_t based on X_t and w_{t-1}, which must be rewarded based on the price-relative vector y_{t+1}.

    The paper suggests using the log-return r_{t+1} as a reward, which depends on p_{t+1}. This is the portfolio value after rebalancing
    once more at time t+1. In other words, in order to reward action w_t, we discard mu_t and instead need to take another action w_{t+1}.
    As such, the signal the agent receives about rebalancing costs are not directly linked to w_t. This is very inefficient.

    We instead use the reward r'_{t+1} = ln(p'_{t+1}/p'_t). This depends on the portfolio values BEFORE rebalancing. This way, the reward
    for action w_t actually depends on mu_t, linking w_t directly to both the quality of investments (through y_{t+1}) as well as the
    effect on transaction costs through mu_t. This way it learns to balance expected returns and costs.  
    '''

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

    policy.eval()

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