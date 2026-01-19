# %%

import json
import numpy as np
import pandas as pd
from datetime import datetime, timezone
import torch
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.policies import CNNPolicy, EqualWeightPolicy
from src.data_loading import load_and_split_data
from src.evaluation import run_walk_forward, calculate_performance_metrics, WalkForwardConfig
from src.model_io import load_checkpoint

# %%

run_dir = './runs_completed/runs_batch/260113_batch_size_500_baseline_v1'
config_path = f'{run_dir}/run_config.json'

with open(config_path, 'r') as f:
    config = json.load(f)

commission_rate = config['commission_rate']
n_recent_periods = config['n_recent_periods']
learning_rate = config['learning_rate']
weight_decay = config['weight_decay']
n_features = config['n_features']
n_non_cash_assets = config['n_non_cash_assets']
RESOLUTION_MINUTES = config['RESOLUTION_MINUTES']
instrument_names = config['instrument_names']
features = config['features']

START_DATE_TRAIN = datetime.fromisoformat(config['START_DATE_TRAIN']).replace(tzinfo=timezone.utc)
START_DATE_VALIDATION = datetime.fromisoformat(config['START_DATE_VALIDATION']).replace(tzinfo=timezone.utc)
START_DATE_TEST = datetime.fromisoformat(config['START_DATE_TEST']).replace(tzinfo=timezone.utc)
END_DATE_TEST = datetime.fromisoformat(config['END_DATE_TEST']).replace(tzinfo=timezone.utc)

assets = [s.split('_')[0].split('-')[0].lower() for s in instrument_names]

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

wf_config = WalkForwardConfig(
    n_recent_periods=n_recent_periods,
    commission_rate=commission_rate,
    device=device,
    assets=assets,
    n_osbl_update_steps=config['n_osbl_update_steps'],
    osbl_batch_size=config['batch_size'],
    geometric_parameter=config['geometric_parameter'],
)

print(f"Using device {device}")
print(f"Loaded config from {config_path}")

# %%

train_prices, validation_prices, test_prices, n_train_periods, n_validation_periods, n_test_periods, all_datetimes = load_and_split_data(
    instrument_names, features, START_DATE_TRAIN, START_DATE_VALIDATION, START_DATE_TEST, END_DATE_TEST, RESOLUTION_MINUTES
)

print(f"(n_train_periods, n_validation_periods, n_test_periods) = {n_train_periods, n_validation_periods, n_test_periods}")

# %%

policy = CNNPolicy(n_features=n_features, n_recent_periods=n_recent_periods).to(device)
optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate, weight_decay=weight_decay)

checkpoint_path = f'{run_dir}/checkpoints/checkpoint_epoch_4400.pt'
epoch = load_checkpoint(checkpoint_path, policy, optimizer, device)

# %%



print("Running walk-forward test on validation set (no OSBL)...")
# initial_portfolio = np.ones(n_non_cash_assets + 1) / (n_non_cash_assets + 1)
initial_portfolio = np.insert(np.zeros(n_non_cash_assets), 0, 1)
validation_results = run_walk_forward(
    policy=policy,
    initial_weights=initial_portfolio,
    seen_prices=train_prices,
    unseen_prices=validation_prices,
    all_datetimes=all_datetimes,
    use_osbl=False,
    config=wf_config,
)
validation_metrics = calculate_performance_metrics(validation_results, RESOLUTION_MINUTES, commission_rate)
print(f"VALIDATION RESULTS: fAPV={validation_metrics['fAPV']:.4f}, SR={validation_metrics['SR']:.4f}, MDD={validation_metrics['MDD']:.4f}")


# %%

# Compute individual asset metrics and cumulative fAPV over time
risk_free_return = 0
annualization_factor = np.sqrt(365 * 24 * 60 / RESOLUTION_MINUTES)

metrics_data = []
asset_apv_over_time = {}  # Store cumulative fAPV for plotting

for asset in assets:
    close_prices = validation_results[f'close_{asset}'].values
    apv_ratios = close_prices / close_prices[0]  # Cumulative fAPV over time
    asset_apv_over_time[asset] = apv_ratios

    price_ratios = close_prices[1:] / close_prices[:-1]
    step_log_returns = np.log(price_ratios)
    step_returns = np.exp(step_log_returns) - 1

    fAPV = apv_ratios[-1]
    periodic_sr = np.mean(step_returns - risk_free_return) / (np.sqrt(np.var(step_returns - risk_free_return, ddof=1)) + 1e-12)
    SR = annualization_factor * periodic_sr

    running_max = np.maximum.accumulate(apv_ratios)
    running_drawdown = (running_max - apv_ratios) / running_max
    MDD = np.max(running_drawdown)

    metrics_data.append({'Asset': asset.upper(), 'fAPV': fAPV, 'SR': SR, 'MDD': MDD})

# Add portfolio metrics
portfolio_apv_over_time = np.concatenate([[1.0], validation_metrics['apv_ratios']])
metrics_data.append({'Asset': 'PORTFOLIO', 'fAPV': validation_metrics['fAPV'], 'SR': validation_metrics['SR'], 'MDD': validation_metrics['MDD']})

# Individual asset metrics computed above (displayed in combined table later)

# %%

# UBAH: Uniform Buy and Hold - equal weight in all crypto assets, no rebalancing
# Portfolio value = (1/n) * sum(price_t / price_0) = average of individual asset fAPV
ubah_apv_over_time = np.mean([asset_apv_over_time[asset] for asset in assets], axis=0)
ubah_fAPV = ubah_apv_over_time[-1]

# Compute UBAH step returns for Sharpe ratio
ubah_step_log_returns = np.log(ubah_apv_over_time[1:] / ubah_apv_over_time[:-1])
ubah_step_returns = np.exp(ubah_step_log_returns) - 1
ubah_periodic_sr = np.mean(ubah_step_returns) / (np.sqrt(np.var(ubah_step_returns, ddof=1)) + 1e-12)
ubah_SR = annualization_factor * ubah_periodic_sr

# Compute UBAH MDD
ubah_running_max = np.maximum.accumulate(ubah_apv_over_time)
ubah_running_drawdown = (ubah_running_max - ubah_apv_over_time) / ubah_running_max
ubah_MDD = np.max(ubah_running_drawdown)

print(f"\nUBAH: fAPV={ubah_fAPV:.4f}, SR={ubah_SR:.4f}, MDD={ubah_MDD:.4f}")

# %%

# UCRP: Uniform Constant Rebalanced Portfolios - rebalance to equal weights each period
print("\nRunning UCRP walk-forward...")
ucrp_policy = EqualWeightPolicy(n_non_cash_assets)
ucrp_initial = np.array([0.0] + [1.0 / n_non_cash_assets] * n_non_cash_assets)
ucrp_results = run_walk_forward(
    policy=ucrp_policy, initial_weights=ucrp_initial, seen_prices=train_prices,
    unseen_prices=validation_prices, all_datetimes=all_datetimes, use_osbl=False, config=wf_config,
)
ucrp_metrics = calculate_performance_metrics(ucrp_results, RESOLUTION_MINUTES, commission_rate)
ucrp_apv_over_time = np.concatenate([[1.0], ucrp_metrics['apv_ratios']])
print(f"UCRP: fAPV={ucrp_metrics['fAPV']:.4f}, SR={ucrp_metrics['SR']:.4f}, MDD={ucrp_metrics['MDD']:.4f}")

# %%

# Best Stock: select the best performing individual asset
best_asset_idx = np.argmax([m['fAPV'] for m in metrics_data if m['Asset'] not in ['PORTFOLIO']])
best_stock_metrics = metrics_data[best_asset_idx]
best_stock_name = best_stock_metrics['Asset']
best_stock_apv_over_time = asset_apv_over_time[best_stock_name.lower()]
print(f"\nBest Stock: {best_stock_name} with fAPV={best_stock_metrics['fAPV']:.4f}")

# %%

# Combined metrics table with all strategies and individual assets
main_strategies = [
    {'Strategy': 'CNN (No OSBL)', 'fAPV': validation_metrics['fAPV'], 'SR': validation_metrics['SR'], 'MDD': validation_metrics['MDD']},
    {'Strategy': f'BEST ASSET ({best_stock_name})', 'fAPV': best_stock_metrics['fAPV'], 'SR': best_stock_metrics['SR'], 'MDD': best_stock_metrics['MDD']},
    {'Strategy': 'UBAH', 'fAPV': ubah_fAPV, 'SR': ubah_SR, 'MDD': ubah_MDD},
    {'Strategy': 'UCRP', 'fAPV': ucrp_metrics['fAPV'], 'SR': ucrp_metrics['SR'], 'MDD': ucrp_metrics['MDD']},
]
individual_assets = []
for asset in assets:
    asset_metrics = next(m for m in metrics_data if m['Asset'] == asset.upper())
    individual_assets.append({'Strategy': asset.upper(), 'fAPV': asset_metrics['fAPV'], 'SR': asset_metrics['SR'], 'MDD': asset_metrics['MDD']})

combined_df = pd.DataFrame(main_strategies + individual_assets)
table_lines = combined_df.to_string(index=False, float_format=lambda x: f'{x:.4f}').split('\n')

print("\n=== Performance Comparison: All Strategies ===")
print(table_lines[0])
for i in range(1, 5):
    print(table_lines[i])
print("-" * len(table_lines[0]))
for i in range(5, len(table_lines)):
    print(table_lines[i])

# %%

weights_datetimes = validation_results['datetimes'].values
entropy_datetimes = validation_results['datetimes'].values

# Plot fAPV over time with benchmarks
fig2 = make_subplots(rows=3, cols=1, subplot_titles=('fAPV Over Time', 'Portfolio Weights', 'Normalized Entropy'), vertical_spacing=0.08, row_heights=[0.4, 0.35, 0.25], shared_xaxes=True)

# fAPV subplot
fig2.add_trace(go.Scatter(x=weights_datetimes, y=portfolio_apv_over_time, mode='lines', name='Portfolio', line=dict(width=2)), row=1, col=1)
fig2.add_trace(go.Scatter(x=weights_datetimes, y=ubah_apv_over_time, mode='lines', name='UBAH', line=dict(dash='dash')), row=1, col=1)
fig2.add_trace(go.Scatter(x=weights_datetimes, y=ucrp_apv_over_time, mode='lines', name='UCRP', line=dict(dash='dot')), row=1, col=1)
fig2.add_trace(go.Scatter(x=weights_datetimes, y=best_stock_apv_over_time, mode='lines', name=f'Best ({best_stock_name})', line=dict(dash='dashdot')), row=1, col=1)
for asset in assets:
    fig2.add_trace(go.Scatter(x=weights_datetimes, y=asset_apv_over_time[asset], mode='lines', name=asset.upper(), opacity=0.4, legendgroup=asset.upper()), row=1, col=1)

# Weights subplot
fig2.add_trace(go.Scatter(x=weights_datetimes, y=validation_results['weight_cash'].values, mode='lines', name='Cash', legendgroup='Cash'), row=2, col=1)
for asset in assets:
    fig2.add_trace(go.Scatter(x=weights_datetimes, y=validation_results[f'weight_{asset}'].values, mode='lines', name=asset.upper(), legendgroup=asset.upper(), showlegend=False), row=2, col=1)

# Entropy subplot
fig2.add_trace(go.Scatter(x=weights_datetimes, y=validation_results['normalized_entropy'].values, mode='lines', name='Entropy', line=dict(color='purple', width=2), showlegend=False), row=3, col=1)

fig2.update_layout(height=1100, showlegend=True, hovermode='x unified')
fig2.update_yaxes(title_text='fAPV', row=1, col=1)
fig2.update_yaxes(title_text='Weight', row=2, col=1)
fig2.update_yaxes(title_text='Entropy', range=[0, 1], row=3, col=1)
fig2.update_xaxes(title_text='Date', row=3, col=1)
fig2.show()

# %%
