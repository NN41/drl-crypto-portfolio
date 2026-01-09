# %%

import numpy as np
import pandas as pd
import torch
from src.portfolio import approximate_mu
from src.train_utils import geometrically_sample_batch_start_indices, run_one_epoch

def run_walk_forward_test(policy, initial_portfolio_weights, initial_prices, forward_prices, all_datetimes, assets, n_recent_periods, commission_rate, device, use_osbl, n_osbl_update_steps, optimizer):

    try:
        policy.eval()
    except:
        pass

    seen_prices = initial_prices.copy()
    unseen_prices = forward_prices.copy()

    n_initial_periods = seen_prices.shape[-1]
    current_idx = n_initial_periods - 1

    n_non_cash_assets = initial_prices.shape[1]
    current_portfolio_vector_memory = np.ones((n_initial_periods - 1, n_non_cash_assets + 1)) / (n_non_cash_assets + 1)
    current_portfolio_vector_memory[-1] = initial_portfolio_weights

    test_results = {
        'datetimes': [],
        'log_returns': [],
        'indices': [],
        'relative_turnover': [],
        'transaction_remainder_factor': [],
    }
    for i, asset in enumerate(assets):
        test_results[f'weight_before_{asset}'] = []
        test_results[f'weight_{asset}'] = []
        test_results[f'close_{asset}'] = []
    if use_osbl:
        test_results['osbl_avg_log_return'] = []

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

        # The new weights at time t allow use to compute the transaction remainder factor
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

        if use_osbl:
            batch_size = int(50 * 5.5)
            geometric_parameter = 5e-5
            osbl_batch_start_indices = geometrically_sample_batch_start_indices(
                n_samples=n_osbl_update_steps,
                n_available_periods=seen_prices.shape[-1],
                batch_size=batch_size,
                geometric_parameter=geometric_parameter,
                n_recent_periods=n_recent_periods
            )

            policy.train()
            osbl_update_avg_log_return = run_one_epoch(
                prices_array=seen_prices,
                batch_start_indices=osbl_batch_start_indices,
                portfolio_vector_memory=current_portfolio_vector_memory,
                policy=policy,
                optimizer=optimizer,
                n_recent_periods=n_recent_periods,
                batch_size=batch_size,
                device=device,
                commission_rate=commission_rate
            )
            policy.eval()

            test_results['osbl_avg_log_return'].append(osbl_update_avg_log_return)

        # take a step and observe new prices
        current_idx += 1
        seen_prices = np.concatenate([seen_prices, unseen_prices[:, :, :1]], axis=-1)
        unseen_prices = unseen_prices[:, :, 1:]

    df_results = pd.DataFrame(test_results)

    # Compute cash weights as 1 - sum of all non-cash asset weights
    weight_cols = [f'weight_{asset}' for asset in assets]
    df_results['weight_cash'] = 1 - df_results[weight_cols].sum(axis=1)

    weight_before_cols = [f'weight_before_{asset}' for asset in assets]
    df_results['weight_before_cash'] = 1 - df_results[weight_before_cols].sum(axis=1)
    df_results['start_of_period'] = df_results.index.values + 1

    return df_results


def calculate_performance_metrics(df_results, resolution_minutes=30, commission_rate=0.0005):
    """
    Calculate performance metrics from walk-forward test results.

    Args:
        df_results: DataFrame with 'log_returns' and 'transaction_remainder_factor' columns
        resolution_minutes: Time resolution in minutes (default 30)
        commission_rate: Commission rate for turnover calculation (default 0.0005)
    """
    initial_portfolio_value = 1
    risk_free_return = 0

    # Note that all metrics are measured BEFORE rebalancing occurs
    step_log_returns = df_results['log_returns'].values
    avg_log_return = np.mean(step_log_returns)
    step_returns = np.exp(step_log_returns) - 1
    step_portfolio_value_multipliers = np.exp(step_log_returns)
    apv_ratios = np.cumprod(step_portfolio_value_multipliers)
    assert np.sum(np.abs(apv_ratios - np.exp(np.cumsum(step_log_returns)))) < 1e-9, "Large deviation between equivalent calculations of apv ratios"

    transaction_remainder_factors = df_results['transaction_remainder_factor'].values
    portfolio_values_before_rebalancing = initial_portfolio_value * apv_ratios
    transaction_costs = portfolio_values_before_rebalancing * (1 - transaction_remainder_factors) # in terms of the initial portfolio value
    turnovers = transaction_costs / commission_rate

    running_transaction_costs = np.cumsum(transaction_costs)
    running_turnover = running_transaction_costs / commission_rate

    running_max = np.maximum.accumulate(portfolio_values_before_rebalancing)
    running_drawdown = (running_max - portfolio_values_before_rebalancing) / running_max
    running_max_drawdown = np.maximum.accumulate(running_drawdown)


    periodic_sharpe_ratio = np.mean(step_returns - risk_free_return) / (np.sqrt(np.var(step_returns - risk_free_return, ddof=1)) + 1e-12)
    annualized_sharpe_ratio = np.sqrt(365 * 24 * 60 / resolution_minutes) * periodic_sharpe_ratio

    return {
        'fAPV': portfolio_values_before_rebalancing[-1],
        'SR': annualized_sharpe_ratio,
        'MDD': running_max_drawdown[-1],
        'avg_log_return': avg_log_return,
        'total_transaction_costs': running_transaction_costs[-1],
        'total_turnover': running_turnover[-1],
        'apv_ratios': apv_ratios,
        'running_drawdown': running_drawdown,
        'running_max_drawdown': running_max_drawdown,
        'step_log_returns': step_log_returns,
        'step_returns': step_returns,
        'running_transaction_costs': running_transaction_costs,
        'running_turnover': running_turnover
    }
