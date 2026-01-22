# %%

import time
import numpy as np
import pandas as pd
import torch
from dataclasses import dataclass
from typing import List
from src.portfolio import approximate_mu
from src.train_utils import geometrically_sample_batch_start_indices, run_one_epoch


@dataclass
class WalkForwardConfig:
    """Fixed parameters for walk-forward evaluation (don't change per call)."""
    n_recent_periods: int
    commission_rate: float
    device: torch.device
    assets: List[str]
    n_osbl_update_steps: int
    osbl_batch_size: int
    geometric_parameter: float


def run_walk_forward(policy, initial_weights, seen_prices, unseen_prices, all_datetimes, use_osbl, config, availability_mask, optimizer=None, verbose=False):
    """
    Run walk-forward evaluation on a trading policy.

    Args:
        policy: Trading policy (CNNPolicy, EqualWeightPolicy, etc.)
        initial_weights: Initial portfolio weights (1D array, shape [n_assets + 1] including cash)
        seen_prices: Historical price data used as context (shape [n_features, n_assets, n_periods])
        unseen_prices: Future price data to walk through (shape [n_features, n_assets, n_periods])
        all_datetimes: Array of datetime objects for all periods
        use_osbl: Whether to use Online Stochastic Batch Learning
        config: WalkForwardConfig with fixed parameters
        availability_mask: shape (n_periods, n_non_cash_assets), True where asset data exists.
            For validation/test data, pass an all-True mask.
        optimizer: PyTorch optimizer (required if use_osbl=True)
        verbose: Whether to print progress updates (default False)

    Returns:
        DataFrame with walk-forward results including weights, returns, and metrics.
    """
    if use_osbl and optimizer is None:
        raise ValueError("optimizer is required when use_osbl=True")

    if hasattr(policy, 'eval'):
        policy.eval()

    seen_prices = seen_prices.copy()
    unseen_prices = unseen_prices.copy()

    n_initial_periods = seen_prices.shape[-1]
    current_idx = n_initial_periods - 1

    n_non_cash_assets = seen_prices.shape[1]
    current_portfolio_vector_memory = np.ones((n_initial_periods - 1, n_non_cash_assets + 1)) / (n_non_cash_assets + 1)
    current_portfolio_vector_memory[-1] = initial_weights

    test_results = {
        'datetimes': [],
        'log_returns': [],
        'indices': [],
        'relative_turnover': [],
        'transaction_remainder_factor': [],
    }
    for i, asset in enumerate(config.assets):
        test_results[f'weight_before_{asset}'] = []
        test_results[f'weight_{asset}'] = []
        test_results[f'close_{asset}'] = []
    if use_osbl:
        test_results['osbl_avg_log_return'] = []

    previous_transaction_remainder_factor = 1.0

    total_steps = unseen_prices.shape[-1]
    steps_processed = 0
    last_reported_percent = -1
    start_time = time.time()

    while unseen_prices.shape[-1] > 0:

        # At time t, at the start of Period t+1, we observe the price history up to and including time t.
        price_history = seen_prices[:, :, -config.n_recent_periods:]
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
            normalized_price_tensor = torch.tensor(normalized_price_history, dtype=torch.float32, device=config.device)
            previous_weights_tensor = torch.tensor(previous_weights, dtype=torch.float32, device=config.device)
            current_mask_tensor = torch.tensor(availability_mask[current_idx], dtype=torch.bool, device=config.device)
            new_weights = policy(normalized_price_tensor, previous_weights_tensor, current_mask_tensor)
            new_weights = new_weights.squeeze(0).cpu().numpy()

        # The new weights at time t allow use to compute the transaction remainder factor
        transaction_remainder_factor, weights_before_rebalancing = approximate_mu(
            torch.from_numpy(previous_weights).unsqueeze(0),
            torch.from_numpy(price_relative).unsqueeze(0),
            torch.from_numpy(new_weights).unsqueeze(0),
            config.commission_rate,
            train_mode=False,
            return_w_prime=True
        )
        transaction_remainder_factor = transaction_remainder_factor.item()
        weights_before_rebalancing = weights_before_rebalancing.squeeze(0).numpy()
        relative_turnover = (1 - transaction_remainder_factor) / config.commission_rate

        test_results['relative_turnover'].append(relative_turnover)
        test_results['transaction_remainder_factor'].append(transaction_remainder_factor)
        for i, asset in enumerate(config.assets):
            test_results[f'weight_before_{asset}'].append(weights_before_rebalancing[i+1])
            test_results[f'weight_{asset}'].append(new_weights[i+1])
            test_results[f'close_{asset}'].append(latest_close_prices.squeeze()[i])

        previous_transaction_remainder_factor = transaction_remainder_factor
        current_portfolio_vector_memory = np.concatenate([current_portfolio_vector_memory, new_weights[np.newaxis, :]], axis=0)

        if use_osbl:
            osbl_batch_start_indices = geometrically_sample_batch_start_indices(
                n_samples=config.n_osbl_update_steps,
                n_available_periods=seen_prices.shape[-1],
                batch_size=config.osbl_batch_size,
                geometric_parameter=config.geometric_parameter,
                n_recent_periods=config.n_recent_periods
            )

            policy.train()
            osbl_update_avg_log_return = run_one_epoch(
                prices_array=seen_prices,
                batch_start_indices=osbl_batch_start_indices,
                portfolio_vector_memory=current_portfolio_vector_memory,
                policy=policy,
                optimizer=optimizer,
                n_recent_periods=config.n_recent_periods,
                batch_size=config.osbl_batch_size,
                device=config.device,
                commission_rate=config.commission_rate,
                availability_mask=availability_mask,
            )
            policy.eval()

            test_results['osbl_avg_log_return'].append(osbl_update_avg_log_return)

        # take a step and observe new prices
        current_idx += 1
        seen_prices = np.concatenate([seen_prices, unseen_prices[:, :, :1]], axis=-1)
        unseen_prices = unseen_prices[:, :, 1:]

        steps_processed += 1
        if verbose:
            current_percent = (steps_processed * 10) // total_steps * 10
            if current_percent > last_reported_percent:
                elapsed = time.time() - start_time
                eta = (elapsed / steps_processed) * (total_steps - steps_processed) if steps_processed > 0 else 0
                print(f"Walk-forward: {steps_processed}/{total_steps} steps ({current_percent}%) | Elapsed: {elapsed/60:.1f}m | ETA: {eta/60:.1f}m")
                last_reported_percent = current_percent

    df_results = pd.DataFrame(test_results)

    # Compute cash weights as 1 - sum of all non-cash asset weights
    weight_cols = [f'weight_{asset}' for asset in config.assets]
    df_results['weight_cash'] = 1 - df_results[weight_cols].sum(axis=1)

    weight_before_cols = [f'weight_before_{asset}' for asset in config.assets]
    df_results['weight_before_cash'] = 1 - df_results[weight_before_cols].sum(axis=1)
    df_results['start_of_period'] = df_results.index.values + 1

    all_weight_cols = ['weight_cash'] + weight_cols
    all_weights = df_results[all_weight_cols]

    all_weights_clipped = np.clip(all_weights, 0, 1)
    epsilon = 1e-9
    entropy = (-1) * np.sum(all_weights_clipped * np.log(all_weights_clipped + epsilon), axis=1)
    max_entropy = np.log(len(all_weights.columns))
    df_results['normalized_entropy'] = entropy / max_entropy

    return df_results


def calculate_performance_metrics(df_results, resolution_minutes=30, commission_rate=0.0005):
    """
    Calculate performance metrics from walk-forward test results.

    The turnover is the traded volume (buying AND selling) and as such has a value between and 2.
    The "(absolute) turnover" is in terms of dollars (i.e. the initial portfolio value),
    while the "relative turnover" is in terms of the portfolio value before rebalancing, which is useful if the portfolio value changes significantly over time.
    """
    initial_portfolio_value = 1
    risk_free_return = 0

    # Note that all metrics are measured BEFORE rebalancing occurs
    step_log_returns = df_results['log_returns'].values
    avg_log_return = np.mean(step_log_returns)
    step_returns = np.exp(step_log_returns) - 1
    apv_ratios = np.exp(np.cumsum(step_log_returns)) # more numerically stable than np.cumprod(np.exp(step_log_returns))

    transaction_remainder_factors = df_results['transaction_remainder_factor'].values
    portfolio_values_before_rebalancing = initial_portfolio_value * apv_ratios
    transaction_costs = portfolio_values_before_rebalancing * (1 - transaction_remainder_factors) # by definition of the transaction remainder factor

    relative_step_turnovers = (1 - transaction_remainder_factors) / commission_rate

    running_transaction_costs = np.cumsum(transaction_costs)
    running_turnover = running_transaction_costs / commission_rate

    running_max = np.maximum.accumulate(portfolio_values_before_rebalancing)
    running_drawdown = (running_max - portfolio_values_before_rebalancing) / running_max
    running_max_drawdown = np.maximum.accumulate(running_drawdown)

    periodic_sharpe_ratio = np.mean(step_returns - risk_free_return) / (np.sqrt(np.var(step_returns - risk_free_return, ddof=1)) + 1e-12)
    annualized_sharpe_ratio = np.sqrt(365 * 24 * 60 / resolution_minutes) * periodic_sharpe_ratio

    avg_normalized_entropy = df_results['normalized_entropy'].mean()

    return {
        'fAPV': portfolio_values_before_rebalancing[-1],
        'SR': annualized_sharpe_ratio,
        'MDD': running_max_drawdown[-1],
        'avg_normalized_entropy': avg_normalized_entropy,
        'avg_transaction_cost': transaction_costs.mean(),
        'avg_relative_turnover': relative_step_turnovers.mean(),
        'avg_log_return': avg_log_return,
        'avg_cash_weight': df_results['weight_cash'].mean(),
        'total_transaction_costs': running_transaction_costs[-1],
        'total_turnover': running_turnover[-1],
        'apv_ratios': apv_ratios,
        'running_drawdown': running_drawdown,
        'running_max_drawdown': running_max_drawdown,
        'step_log_returns': step_log_returns,
        'step_returns': step_returns,
        'running_transaction_costs': running_transaction_costs,
        'running_turnover': running_turnover,
    }


# %%
