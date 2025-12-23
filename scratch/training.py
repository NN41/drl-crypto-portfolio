# %%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta, datetime, timezone

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.policies import CNNPolicy, BuyAndHoldPolicy, EqualWeightPolicy
from src.portfolio import approximate_mu
from src.training import geometrically_sample_batch_start_indices, prepare_batch_of_consecutive_periods, perform_one_minibatch_update, run_one_epoch
from src.model_io import save_model, load_model

commission_rate = 0.0005 # 0.0005 = 5 bips
n_recent_periods = 50 # number of periods passed to the policy to choose a portfolio
batch_size = int(50 * 5.5) # x5.5 to match the number of training data points used per update; number of actions in a single batch
n_online_batches = 30
geometric_parameter = 5e-5
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

df_btc = pd.read_csv('./data/raw/ohlcv/BTC-PERPETUAL_resolution_30.csv').sort_values('timestamp')
df_eth = pd.read_csv('./data/raw/ohlcv/ETH-PERPETUAL_resolution_30.csv').sort_values('timestamp')

df_btc['datetime'] = pd.to_datetime(df_btc['datetime'], utc=True) + timedelta(minutes=RESOLUTION_MINUTES)
df_eth['datetime'] = pd.to_datetime(df_eth['datetime'], utc=True) + timedelta(minutes=RESOLUTION_MINUTES)

assets = ['btc', 'eth']
features = ['high', 'low', 'close'] # follow the standard order of the OHLC acronym O-H-L-C
n_train_periods = 32504
# n_validation_periods = 2456
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

seed_everything(seed=42)

n_features, n_non_cash_assets, n_train_periods = train_prices.shape
learning_rate = 3e-5 #1e-4 # as opposed to the paper's 3e-5
weight_decay = 1e-8
n_epochs = 1000 # to match the total number of meaningful data points seen during training
n_epochs_per_validation = 10
n_batches_per_epoch = 2000
n_total_updates = n_epochs * n_batches_per_epoch
geometric_parameter = 5e-5

n_available_periods = train_prices.shape[-1]
prices_array = train_prices

portfolio_vector_memory = np.ones((n_available_periods, n_non_cash_assets + 1)) / (n_non_cash_assets + 1)
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

    if epoch % n_epochs_per_validation == 0:
        print(f"\tRunning validation...")
        initial_portfolio = np.array([1] * (n_non_cash_assets + 1)) / (n_non_cash_assets + 1)
        validation_results = run_walk_forward_test(
            policy=policy,
            initial_portfolio_weights=initial_portfolio,
            train_prices=train_prices,
            test_prices=test_prices,
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

writer.close()

# %%
save_model(policy, optimizer, save_dir='./models', n_epochs=n_epochs, commission_rate=commission_rate, learning_rate=learning_rate, weight_decay=weight_decay, n_features=n_features, n_recent_periods=n_recent_periods)



# %%

# def run_walk_forward_test(policy, initial_portfolio_weights, train_prices, test_prices, all_datetimes, assets, n_recent_periods, commission_rate, device):

initial_portfolio_weights = np.array([1.,0.,0.])
policy.eval()

def run_walk_forward_test(policy, initial_portfolio_weights, train_prices, test_prices, all_datetimes, assets, n_recent_periods, commission_rate, device, use_osbl, n_osbl_update_steps, optimizer):

    try:
        policy.eval()
    except:
        pass

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


def calculate_performance_metrics(df_results, resolution_minutes=RESOLUTION_MINUTES):
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
    annualized_sharpe_ratio = np.sqrt(365 * 24 * 60 / RESOLUTION_MINUTES) * periodic_sharpe_ratio

    return {
        'fAPV': apv_ratios[-1],
        'SR': annualized_sharpe_ratio,
        'MDD': running_max_drawdown[-1],
        'apv_ratios': apv_ratios,
        'portfolio_values': portfolio_values,
        'running_drawdown': running_drawdown,
        'running_max_drawdown': running_max_drawdown
    }


# %%

policy, optimizer, checkpoint = load_model('./models/pretrained/cnn_policy_20251030_094203.pt', CNNPolicy, device, learning_rate, weight_decay)

# %%

results_cnn_policy_no_osbl = run_walk_forward_test(
    policy=policy,
    initial_portfolio_weights=np.array([0, 0.5, 0.5]),
    train_prices=train_prices,
    test_prices=test_prices,
    all_datetimes=all_datetimes,
    assets=assets,
    n_recent_periods=n_recent_periods,
    commission_rate=commission_rate,
    device=device,
    use_osbl=False,
    n_osbl_update_steps=None,
    optimizer=None,
)
metrics_cnn_policy_no_osbl = calculate_performance_metrics(results_cnn_policy_no_osbl, RESOLUTION_MINUTES)
print(f"CNN Policy: fAPV={metrics_cnn_policy_no_osbl['fAPV']:.4f}, SR={metrics_cnn_policy_no_osbl['SR']:.4f}, MDD={metrics_cnn_policy_no_osbl['MDD']:.4f}")

# %%

results_cnn_policy_with_osbl = run_walk_forward_test(
    policy=policy,
    initial_portfolio_weights=np.array([0, 0.5, 0.5]),
    train_prices=train_prices,
    test_prices=test_prices,
    all_datetimes=all_datetimes,
    assets=assets,
    n_recent_periods=n_recent_periods,
    commission_rate=commission_rate,
    device=device,
    use_osbl=True,
    n_osbl_update_steps=n_osbl_update_steps,
    optimizer=optimizer,
)
metrics_cnn_policy_with_osbl = calculate_performance_metrics(results_cnn_policy_with_osbl, RESOLUTION_MINUTES)
print(f"CNN Policy: fAPV={metrics_cnn_policy_with_osbl['fAPV']:.4f}, SR={metrics_cnn_policy_with_osbl['SR']:.4f}, MDD={metrics_cnn_policy_with_osbl['MDD']:.4f}")

# %%

print("\nTesting Equal Weight Policy...")
equal_weight_policy = EqualWeightPolicy(n_non_cash_assets=2)
df_equal_weight = run_walk_forward_test(
    policy=equal_weight_policy,
    initial_portfolio_weights=np.array([0., 0.5, 0.5]),
    train_prices=train_prices,
    test_prices=test_prices,
    all_datetimes=all_datetimes,
    assets=assets,
    n_recent_periods=n_recent_periods,
    commission_rate=commission_rate,
    device=device,
    use_osbl=False,
    n_osbl_update_steps=None,
    optimizer=None,
)
equal_weight_metrics = calculate_performance_metrics(df_equal_weight, RESOLUTION_MINUTES)
print(f"Equal Weight: fAPV={equal_weight_metrics['fAPV']:.4f}, SR={equal_weight_metrics['SR']:.4f}, MDD={equal_weight_metrics['MDD']:.4f}")


# %%



# %%
# initial_portfolio_weights = np.array([0, 0.5, 0.5])
initial_portfolio_weights = np.array([0, 1, 0])

# Test Buy-and-Hold BTC Policy
print("\nTesting Buy-and-Hold BTC Policy...")
buy_hold_btc_policy = BuyAndHoldPolicy()
df_buy_hold_btc = run_walk_forward_test(
    policy=buy_hold_btc_policy,
    initial_portfolio_weights=initial_portfolio_weights,
    train_prices=train_prices,
    test_prices=test_prices,
    all_datetimes=all_datetimes,
    assets=assets,
    n_recent_periods=n_recent_periods,
    commission_rate=commission_rate,
    device=device,
    use_osbl=False,
    n_osbl_update_steps=None,
    optimizer=None,
)
buy_hold_btc_metrics = calculate_performance_metrics(df_buy_hold_btc, RESOLUTION_MINUTES)
print(f"Buy-Hold BTC: fAPV={buy_hold_btc_metrics['fAPV']:.4f}, SR={buy_hold_btc_metrics['SR']:.4f}, MDD={buy_hold_btc_metrics['MDD']:.4f}")

# %%


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

