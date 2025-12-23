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
        'log_returns': [],
        'datetimes': [],
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
    return df_results


def calculate_performance_metrics(df_results, resolution_minutes=30):
    step_log_returns = df_results['log_returns'].values
    step_returns = np.exp(step_log_returns) - 1
    running_log_returns = np.cumsum(step_log_returns)

    apv_ratios = np.exp(running_log_returns)
    portfolio_values = apv_ratios

    running_max = np.maximum.accumulate(portfolio_values)
    running_drawdown = (running_max - portfolio_values) / running_max
    running_max_drawdown = np.maximum.accumulate(running_drawdown)

    risk_free_return = 0
    periodic_sharpe_ratio = np.mean(step_returns - risk_free_return) / (np.sqrt(np.var(step_returns - risk_free_return, ddof=1)) + 1e-12)
    annualized_sharpe_ratio = np.sqrt(365 * 24 * 60 / resolution_minutes) * periodic_sharpe_ratio

    return {
        'fAPV': apv_ratios[-1],
        'SR': annualized_sharpe_ratio,
        'MDD': running_max_drawdown[-1],
        'apv_ratios': apv_ratios,
        'portfolio_values': portfolio_values,
        'running_drawdown': running_drawdown,
        'running_max_drawdown': running_max_drawdown
    }

if __name__ == '__main__':
    from datetime import datetime, timezone
    from src.policies import EqualWeightPolicy

    np.random.seed(42)
    n_features, n_assets, n_periods = 3, 2, 200
    prices = np.random.rand(n_features, n_assets, n_periods) * 100 + 100

    initial_prices = prices[:, :, :100]
    forward_prices = prices[:, :, 100:]

    start_time = datetime(2024, 1, 1, tzinfo=timezone.utc)
    datetimes = pd.date_range(start=start_time, periods=n_periods, freq='30min').values

    policy = EqualWeightPolicy(n_non_cash_assets=n_assets)
    device = torch.device('cpu')

    print("Running baseline walk-forward test...")
    results = run_walk_forward_test(
        policy=policy,
        initial_portfolio_weights=np.array([0, 0.5, 0.5]),
        initial_prices=initial_prices,
        forward_prices=forward_prices,
        all_datetimes=datetimes,
        assets=['asset0', 'asset1'],
        n_recent_periods=10,
        commission_rate=0.0005,
        device=device,
        use_osbl=False,
        n_osbl_update_steps=None,
        optimizer=None
    )
    metrics = calculate_performance_metrics(results, resolution_minutes=30)

    print(f"Shape: {results.shape}")
    print(f"Periods: {len(results)}")
    print(f"fAPV: {metrics['fAPV']:.6f}")
    print(f"SR: {metrics['SR']:.6f}")
    print(f"MDD: {metrics['MDD']:.6f}")
    print(f"Mean turnover: {results['relative_turnover'].mean():.6f}")

# %%
