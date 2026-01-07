# %%

import os
import numpy as np
import pandas as pd
from datetime import timedelta, datetime, timezone
from pathlib import Path
import torch
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.policies import CNNPolicy, BuyAndHoldPolicy, EqualWeightPolicy
from src.evaluation import run_walk_forward_test, calculate_performance_metrics
from src.model_io import save_model, load_model

commission_rate = 0.0005 # 0.0005 = 5 bips
n_recent_periods = 50 # number of periods passed to the policy to choose a portfolio||
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


seed_everything(seed=42)

# %%

RESOLUTION_MINUTES = 30
START_DATE = datetime(2023, 10, 17, 17, 0, 0, tzinfo=timezone.utc)
END_DATE = datetime(2025, 10, 15, 0, 30, 0, tzinfo=timezone.utc)
START_TEST_DATE = datetime(2025, 8, 24, 21, 0, 0, tzinfo=timezone.utc)

# 11-asset list from inspect_data.py REDUCED_LIST
asset_symbols = [
    'ADA_USDC-PERPETUAL',
    'AVAX_USDC-PERPETUAL',
    'BTC-PERPETUAL',
    'BNB_USDC-PERPETUAL',
    'DOGE_USDC-PERPETUAL',
    'ETH-PERPETUAL',
    'LINK_USDC-PERPETUAL',
    'PAXG_USDC-PERPETUAL',
    'SOL_USDC-PERPETUAL',
    'TRUMP_USDC-PERPETUAL',
    'XRP_USDC-PERPETUAL',
]
assets = [s.split('_')[0].split('-')[0].lower() for s in asset_symbols]

features = ['high', 'low', 'close']

# Load all asset data with datetime processing
data_dir = Path('./data/raw/ohlcv')
dfs = {}
for symbol in asset_symbols:
    csv_file = data_dir / f'{symbol}_resolution_30.csv'
    df = pd.read_csv(csv_file).sort_values('timestamp')
    df['datetime'] = pd.to_datetime(df['datetime'], utc=True) + timedelta(minutes=RESOLUTION_MINUTES)
    dfs[symbol] = df

# Forward-fill missing early data for each asset
for symbol in asset_symbols:
    df = dfs[symbol]
    first_valid_close = df['close'].dropna().iloc[0]

    # Create continuous datetime index from START_DATE to END_DATE
    date_range = pd.date_range(START_DATE, END_DATE, freq='30min', tz='UTC')
    df_full = df.set_index('datetime').reindex(date_range)

    # Forward-fill OHLC with first available close for missing early data
    missing_mask = df_full[features].isna().any(axis=1)
    for col in features:
        df_full.loc[missing_mask, col] = first_valid_close
    df_full = df_full.reset_index().rename(columns={'index': 'datetime'})
    dfs[symbol] = df_full

# Verify all dataframes have same datetime index
common_dates = dfs[asset_symbols[0]]['datetime'].values
for symbol in asset_symbols[1:]:
    assert np.array_equal(dfs[symbol]['datetime'].values, common_dates), f"Date mismatch for {symbol}"

# Stack all assets into single array: shape (n_features, 11, n_periods) as in paper
all_prices = np.stack([dfs[symbol][features].values for symbol in asset_symbols]).transpose(2, 0, 1)
all_datetimes = common_dates

# Calculate train/test split based on START_TEST_DATE
split_idx = np.searchsorted(all_datetimes, pd.Timestamp(START_TEST_DATE).to_datetime64())
n_train_periods = split_idx
n_test_periods = len(all_datetimes) - n_train_periods

train_prices = all_prices[:, :, :n_train_periods]
test_prices = all_prices[:, :, n_train_periods:]


# %%

learning_rate = 3e-5
weight_decay = 1e-8
n_features = train_prices.shape[0]
n_non_cash_assets = train_prices.shape[1]

policy, optimizer, checkpoint = load_model('./models/pretrained/cnn_policy_20251030_094203.pt', CNNPolicy, device, learning_rate, weight_decay)
print(f"Loaded model trained for {checkpoint['n_epochs']} epochs")

# %%

print("Evaluating CNN Policy without OSBL on test set...")
initial_portfolio = np.ones(n_non_cash_assets + 1) / (n_non_cash_assets + 1)
results_cnn_no_osbl = run_walk_forward_test(
    policy=policy,
    initial_portfolio_weights=initial_portfolio,
    initial_prices=train_prices,
    forward_prices=test_prices,
    all_datetimes=all_datetimes,
    assets=assets,
    n_recent_periods=n_recent_periods,
    commission_rate=commission_rate,
    device=device,
    use_osbl=False,
    n_osbl_update_steps=None,
    optimizer=None,
)
metrics_cnn_no_osbl = calculate_performance_metrics(results_cnn_no_osbl, RESOLUTION_MINUTES)

# %%

# Extract policy name from model path
model_path = './models/pretrained/cnn_policy_20251030_094203.pt'
policy_name = model_path.split('/')[-1].replace('.pt', '')
results_dir = './data/walk_forward_results'
os.makedirs(results_dir, exist_ok=True)
results_path = f'{results_dir}/osbl_{policy_name}.csv'

# Try to load existing OSBL results, otherwise run and save
try:
    results_cnn_with_osbl = pd.read_csv(results_path)
    results_cnn_with_osbl['datetimes'] = pd.to_datetime(results_cnn_with_osbl['datetimes'], utc=True)
    print(f"Loaded OSBL results from {results_path}")
except FileNotFoundError:
    print("Evaluating CNN Policy with OSBL on test set...")
    results_cnn_with_osbl = run_walk_forward_test(
        policy=policy,
        initial_portfolio_weights=initial_portfolio,
        initial_prices=train_prices,
        forward_prices=test_prices,
        all_datetimes=all_datetimes,
        assets=assets,
        n_recent_periods=n_recent_periods,
        commission_rate=commission_rate,
        device=device,
        use_osbl=True,
        n_osbl_update_steps=n_osbl_update_steps,
        optimizer=optimizer,
    )
    results_cnn_with_osbl.to_csv(results_path, index=False)
    print(f"Saved OSBL results to {results_path}")

metrics_cnn_with_osbl = calculate_performance_metrics(results_cnn_with_osbl, RESOLUTION_MINUTES)

# %%

print("Evaluating Equal Weight Policy on test set...")
equal_weight_policy = EqualWeightPolicy(n_non_cash_assets=n_non_cash_assets)
equal_weight_portfolio = np.zeros(n_non_cash_assets + 1)
equal_weight_portfolio[1:] = 1.0 / n_non_cash_assets
results_equal_weight = run_walk_forward_test(
    policy=equal_weight_policy,
    initial_portfolio_weights=equal_weight_portfolio,
    initial_prices=train_prices,
    forward_prices=test_prices,
    all_datetimes=all_datetimes,
    assets=assets,
    n_recent_periods=n_recent_periods,
    commission_rate=commission_rate,
    device=device,
    use_osbl=False,
    n_osbl_update_steps=None,
    optimizer=None,
)
metrics_equal_weight = calculate_performance_metrics(results_equal_weight, RESOLUTION_MINUTES)
metrics_equal_weight

# %%

print("Evaluating Buy-and-Hold Policy on test set...")
buy_hold_btc_policy = BuyAndHoldPolicy()
btc_idx = asset_symbols.index('BNB_USDC-PERPETUAL')
btc_portfolio = np.zeros(n_non_cash_assets + 1)
btc_portfolio[btc_idx + 1] = 1.0
results_buy_hold_btc = run_walk_forward_test(
    policy=buy_hold_btc_policy,
    initial_portfolio_weights=btc_portfolio,
    initial_prices=train_prices,
    forward_prices=test_prices,
    all_datetimes=all_datetimes,
    assets=assets,
    n_recent_periods=n_recent_periods,
    commission_rate=commission_rate,
    device=device,
    use_osbl=False,
    n_osbl_update_steps=None,
    optimizer=None,
)
metrics_buy_hold_btc = calculate_performance_metrics(results_buy_hold_btc, RESOLUTION_MINUTES)
metrics_buy_hold_btc

# %%

print("Evaluating Buy-and-Hold ETH Policy on test set...")
buy_hold_eth_policy = BuyAndHoldPolicy()
eth_idx = asset_symbols.index('ETH-PERPETUAL')
eth_portfolio = np.zeros(n_non_cash_assets + 1)
eth_portfolio[eth_idx + 1] = 1.0
results_buy_hold_eth = run_walk_forward_test(
    policy=buy_hold_eth_policy,
    initial_portfolio_weights=eth_portfolio,
    initial_prices=train_prices,
    forward_prices=test_prices,
    all_datetimes=all_datetimes,
    assets=assets,
    n_recent_periods=n_recent_periods,
    commission_rate=commission_rate,
    device=device,
    use_osbl=False,
    n_osbl_update_steps=None,
    optimizer=None,
)
metrics_buy_hold_eth = calculate_performance_metrics(results_buy_hold_eth, RESOLUTION_MINUTES)

# %%

train_start = pd.Timestamp(all_datetimes[0])
train_end = pd.Timestamp(all_datetimes[n_train_periods - 1])
test_start = pd.Timestamp(all_datetimes[n_train_periods])
test_end = pd.Timestamp(all_datetimes[-1])

print("\n" + "="*80)
print("STRATEGY COMPARISON SUMMARY - TEST SET")
print("="*80)
print(f"Training period: {train_start} to {train_end} ({n_train_periods} periods)")
print(f"Test period:     {test_start} to {test_end} ({n_test_periods} periods)")
print("="*80)

comparison_data = {
    'Strategy': ['CNN (no OSBL)', 'CNN (with OSBL)', 'Equal Weight (rebalancing)', 'Buy-Hold BTC', 'Buy-Hold ETH'],
    'fAPV': [metrics_cnn_no_osbl['fAPV'], metrics_cnn_with_osbl['fAPV'], metrics_equal_weight['fAPV'], metrics_buy_hold_btc['fAPV'], metrics_buy_hold_eth['fAPV']],
    'SR': [metrics_cnn_no_osbl['SR'], metrics_cnn_with_osbl['SR'], metrics_equal_weight['SR'], metrics_buy_hold_btc['SR'], metrics_buy_hold_eth['SR']],
    'MDD': [metrics_cnn_no_osbl['MDD'], metrics_cnn_with_osbl['MDD'], metrics_equal_weight['MDD'], metrics_buy_hold_btc['MDD'], metrics_buy_hold_eth['MDD']]
}
df_comparison = pd.DataFrame(comparison_data)
print(df_comparison.to_string(index=False))
print("="*80)

# %%

apv_datetimes = results_cnn_no_osbl['datetimes'].iloc[1:].values
weights_datetimes = results_cnn_no_osbl['datetimes'].iloc[1:-1].values
plot_datetimes = results_cnn_no_osbl['datetimes'].values

fig_comparison = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06, subplot_titles=('Value Multipliers', 'Portfolio Weights', 'Drawdowns'))

# Subplot 1: BTC, ETH, CNN policies, and Equal Weight
fig_comparison.add_trace(go.Scatter(x=apv_datetimes, y=metrics_buy_hold_btc['apv_ratios'], mode='lines', name='BTC', legendgroup='BTC', line=dict(color='orange', width=2), showlegend=True), row=1, col=1)
fig_comparison.add_trace(go.Scatter(x=apv_datetimes, y=metrics_buy_hold_eth['apv_ratios'], mode='lines', name='ETH', legendgroup='ETH', line=dict(color='blue', width=2), showlegend=True), row=1, col=1)
fig_comparison.add_trace(go.Scatter(x=apv_datetimes, y=metrics_cnn_no_osbl['apv_ratios'], mode='lines', name='CNN (no OSBL)', legendgroup='CNN_no_OSBL', line=dict(color='red', width=2), showlegend=True), row=1, col=1)
fig_comparison.add_trace(go.Scatter(x=apv_datetimes, y=metrics_cnn_with_osbl['apv_ratios'], mode='lines', name='CNN (with OSBL)', legendgroup='CNN_with_OSBL', line=dict(color='purple', width=2), showlegend=True), row=1, col=1)
fig_comparison.add_trace(go.Scatter(x=apv_datetimes, y=metrics_equal_weight['apv_ratios'], mode='lines', name='Equal Weight', legendgroup='EW', line=dict(color='green', width=2), showlegend=True), row=1, col=1)

# Subplot 2: Portfolio weights
fig_comparison.add_trace(go.Scatter(x=weights_datetimes, y=results_cnn_no_osbl['weight_cash'].dropna().values, mode='lines', name='CNN (no OSBL) Cash', legendgroup='CNN_no_OSBL', line=dict(color='red', dash='dot'), showlegend=False, hovertemplate='Cash: %{y:.6f}<extra></extra>'), row=2, col=1)
fig_comparison.add_trace(go.Scatter(x=weights_datetimes, y=results_cnn_no_osbl['weight_btc'].dropna().values, mode='lines', name='CNN (no OSBL) BTC', legendgroup='CNN_no_OSBL', line=dict(color='red', dash='solid'), showlegend=False, hovertemplate='BTC: %{y:.6f}<extra></extra>'), row=2, col=1)
fig_comparison.add_trace(go.Scatter(x=weights_datetimes, y=results_cnn_no_osbl['weight_eth'].dropna().values, mode='lines', name='CNN (no OSBL) ETH', legendgroup='CNN_no_OSBL', line=dict(color='red', dash='dash'), showlegend=False, hovertemplate='ETH: %{y:.6f}<extra></extra>'), row=2, col=1)
fig_comparison.add_trace(go.Scatter(x=weights_datetimes, y=results_cnn_with_osbl['weight_cash'].dropna().values, mode='lines', name='CNN (with OSBL) Cash', legendgroup='CNN_with_OSBL', line=dict(color='purple', dash='dot'), showlegend=False, hovertemplate='Cash: %{y:.6f}<extra></extra>'), row=2, col=1)
fig_comparison.add_trace(go.Scatter(x=weights_datetimes, y=results_cnn_with_osbl['weight_btc'].dropna().values, mode='lines', name='CNN (with OSBL) BTC', legendgroup='CNN_with_OSBL', line=dict(color='purple', dash='solid'), showlegend=False, hovertemplate='BTC: %{y:.6f}<extra></extra>'), row=2, col=1)
fig_comparison.add_trace(go.Scatter(x=weights_datetimes, y=results_cnn_with_osbl['weight_eth'].dropna().values, mode='lines', name='CNN (with OSBL) ETH', legendgroup='CNN_with_OSBL', line=dict(color='purple', dash='dash'), showlegend=False, hovertemplate='ETH: %{y:.6f}<extra></extra>'), row=2, col=1)

# Subplot 3: Drawdowns for BTC, ETH, CNN policies, and Equal Weight
fig_comparison.add_trace(go.Scatter(x=plot_datetimes, y=metrics_buy_hold_btc['running_drawdown'], mode='lines', name='BTC', legendgroup='BTC', line=dict(color='orange'), showlegend=False), row=3, col=1)
fig_comparison.add_trace(go.Scatter(x=plot_datetimes, y=metrics_buy_hold_eth['running_drawdown'], mode='lines', name='ETH', legendgroup='ETH', line=dict(color='blue'), showlegend=False), row=3, col=1)
fig_comparison.add_trace(go.Scatter(x=plot_datetimes, y=metrics_cnn_no_osbl['running_drawdown'], mode='lines', name='CNN (no OSBL)', legendgroup='CNN_no_OSBL', line=dict(color='red'), showlegend=False), row=3, col=1)
fig_comparison.add_trace(go.Scatter(x=plot_datetimes, y=metrics_cnn_with_osbl['running_drawdown'], mode='lines', name='CNN (with OSBL)', legendgroup='CNN_with_OSBL', line=dict(color='purple'), showlegend=False), row=3, col=1)
fig_comparison.add_trace(go.Scatter(x=plot_datetimes, y=metrics_equal_weight['running_drawdown'], mode='lines', name='Equal Weight', legendgroup='EW', line=dict(color='green'), showlegend=False), row=3, col=1)

fig_comparison.update_layout(height=1200, showlegend=True, hovermode='x unified')
fig_comparison.update_xaxes(title_text='Date', row=3, col=1)
fig_comparison.update_yaxes(title_text='Value Multiplier', row=1, col=1)
fig_comparison.update_yaxes(title_text='Weight', row=2, col=1)
fig_comparison.update_yaxes(title_text='Drawdown', row=3, col=1)
fig_comparison.show()

# %%

# Separate plot for training data
train_datetimes = all_datetimes[:n_train_periods]
train_btc_close = train_prices[2, 0, :]
train_eth_close = train_prices[2, 1, :]

train_btc_normalized = train_btc_close / train_btc_close[0]
train_eth_normalized = train_eth_close / train_eth_close[0]

fig_training = go.Figure()
fig_training.add_trace(go.Scatter(x=train_datetimes, y=train_btc_normalized, mode='lines', name='BTC', line=dict(color='orange', width=2)))
fig_training.add_trace(go.Scatter(x=train_datetimes, y=train_eth_normalized, mode='lines', name='ETH', line=dict(color='blue', width=2)))
fig_training.update_layout(title='Training Data - Normalized Prices', xaxis_title='Date', yaxis_title='Value Multiplier', height=500, showlegend=True, hovermode='x unified')
fig_training.show()

# %%
