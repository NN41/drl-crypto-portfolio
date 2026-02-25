# %%
# Post-training analysis script.
# Loads CSVs from post_training.py and creates visualizations + summary tables for train/validation/test sets.

import glob
import json
import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from ipywidgets import IntText, Button, VBox, HBox, Output
from IPython.display import display, clear_output

from src.evaluation import calculate_performance_metrics

# %%
# Color scheme and styling constants

# High-contrast palette for individual assets (distinct on white background)
DEFAULT_COLORS = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52', '#1F77B4', '#2CA02C', '#D62728', '#9467BD']

def get_asset_color(asset):
    """Get color for asset based on its position in the assets list."""
    asset_lower = asset.lower()
    if asset_lower == 'cash':
        return '#888888'
    if asset_lower in assets:
        return DEFAULT_COLORS[assets.index(asset_lower) % len(DEFAULT_COLORS)]
    return '#888888'

POLICY_STYLES = {
    'pretrained': {'color': '#1f77b4', 'dash': 'solid', 'width': 2.5},
    'osbl': {'color': '#2ca02c', 'dash': 'solid', 'width': 2.5},
    'ucrp': {'color': '#ff7f0e', 'dash': 'dot', 'width': 2},
    'ubah': {'color': '#9467bd', 'dash': 'dash', 'width': 2},
    'best': {'color': '#d62728', 'dash': 'dashdot', 'width': 2},
}

# %%
# Discovery: List available post-trained models

post_training_dirs = glob.glob('./runs_completed/**/post_training/epoch_*', recursive=True)
print("Available post-trained models:")
for path in sorted(post_training_dirs):
    parts = path.replace('\\', '/').split('/')
    epoch_folder = parts[-1]
    run_name = parts[-4] if len(parts) >= 4 else 'unknown'
    print(f"  {run_name} / {epoch_folder}")

# %%
# Configuration

run_dir = './runs_completed/runs_batch/260113_batch_size_500_baseline_v1'
epoch = 5000

config_path = f'{run_dir}/run_config.json'
with open(config_path, 'r') as f:
    config = json.load(f)

commission_rate = config['commission_rate']
RESOLUTION_MINUTES = config['RESOLUTION_MINUTES']
instrument_names = config['instrument_names']
assets = [s.split('_')[0].split('-')[0].lower() for s in instrument_names]

print(f"Loaded config from {config_path}")
print(f"Epoch: {epoch}, Assets: {assets}")

# %%
# Load CSVs for all data sets

csv_dir = f'{run_dir}/post_training/epoch_{epoch}'
data_sets = ['train', 'validation', 'test']

results = {}
for data_set in data_sets:
    results[data_set] = {}

    # Pretrained (always available)
    pretrained_path = f'{csv_dir}/{data_set}_pretrained_results.csv'
    if os.path.exists(pretrained_path):
        results[data_set]['pretrained'] = pd.read_csv(pretrained_path)

    # OSBL (validation/test only)
    osbl_path = f'{csv_dir}/{data_set}_osbl_results.csv'
    if os.path.exists(osbl_path):
        results[data_set]['osbl'] = pd.read_csv(osbl_path)

    # UCRP (always available)
    ucrp_path = f'{csv_dir}/{data_set}_ucrp_results.csv'
    if os.path.exists(ucrp_path):
        results[data_set]['ucrp'] = pd.read_csv(ucrp_path)

print(f"\nLoaded data sets: {[ds for ds in data_sets if results[ds]]}")
for ds in data_sets:
    if results[ds]:
        print(f"  {ds}: {list(results[ds].keys())}")

# %%
# Compute metrics for all strategies and benchmarks (per data set)

all_metrics = {}
annualization_factor = np.sqrt(365 * 24 * 60 / RESOLUTION_MINUTES)

for data_set in data_sets:
    if not results[data_set]:
        continue

    all_metrics[data_set] = {}
    ref_df = results[data_set].get('pretrained') if 'pretrained' in results[data_set] else results[data_set]['ucrp']

    # Pretrained, OSBL, UCRP metrics
    for strategy in ['pretrained', 'osbl', 'ucrp']:
        if strategy in results[data_set]:
            metrics = calculate_performance_metrics(results[data_set][strategy], RESOLUTION_MINUTES, commission_rate)
            all_metrics[data_set][strategy] = metrics

    # Individual asset metrics and APV over time (no transaction costs)
    # Note: Some assets have missing early data filled with first available price - trim leading constant prices for metrics
    asset_apv = {}
    for asset in assets:
        close_prices = ref_df[f'close_{asset}'].values
        apv = close_prices / close_prices[0]
        asset_apv[asset] = apv  # Full APV for plotting

        # Trim leading constant prices for metrics
        diff = np.diff(close_prices)
        first_change_idx = np.argmax(diff != 0)
        trimmed = close_prices[first_change_idx:] if diff[first_change_idx] != 0 else close_prices
        trimmed_apv = trimmed / trimmed[0]

        step_returns = trimmed_apv[1:] / trimmed_apv[:-1] - 1
        periodic_sr = np.mean(step_returns) / (np.std(step_returns, ddof=1) + 1e-12)
        running_max = np.maximum.accumulate(trimmed_apv)
        all_metrics[data_set][asset] = {'fAPV': trimmed_apv[-1], 'SR': annualization_factor * periodic_sr, 'MDD': ((running_max - trimmed_apv) / running_max).max()}

    # UBAH: mean of all asset APVs
    ubah_apv = np.mean([asset_apv[a] for a in assets], axis=0)
    step_returns = ubah_apv[1:] / ubah_apv[:-1] - 1
    periodic_sr = np.mean(step_returns) / (np.std(step_returns, ddof=1) + 1e-12)
    running_max = np.maximum.accumulate(ubah_apv)
    all_metrics[data_set]['ubah'] = {'fAPV': ubah_apv[-1], 'SR': annualization_factor * periodic_sr, 'MDD': ((running_max - ubah_apv) / running_max).max(), 'apv_over_time': ubah_apv}

    # Store asset APVs for plotting
    all_metrics[data_set]['_asset_apv'] = asset_apv

    # Best stock
    best_asset = max(assets, key=lambda a: all_metrics[data_set][a]['fAPV'])
    all_metrics[data_set]['_best_asset'] = best_asset

print("Metrics computed for all data sets.")

# %%
# Summary tables (one per data set)

for data_set in data_sets:
    if data_set not in all_metrics:
        continue

    m = all_metrics[data_set]
    best = m['_best_asset']

    def make_row(name, metrics, is_policy=False):
        row = {'Strategy': name, 'fAPV': metrics['fAPV'], 'SR': metrics['SR'], 'MDD': metrics['MDD']}
        if is_policy:
            row['AvgEntropy'] = metrics.get('avg_normalized_entropy', np.nan)
            row['AvgTurnover'] = metrics.get('avg_relative_turnover', np.nan)
            row['AvgTxCost'] = metrics.get('avg_transaction_cost', np.nan)
            row['AvgCash'] = metrics.get('avg_cash_weight', np.nan)
        else:
            row['AvgEntropy'] = np.nan
            row['AvgTurnover'] = np.nan
            row['AvgTxCost'] = np.nan
            row['AvgCash'] = np.nan
        return row

    rows = []
    if 'pretrained' in m:
        rows.append(make_row('Pretrained', m['pretrained'], is_policy=True))
    if 'osbl' in m:
        rows.append(make_row('OSBL', m['osbl'], is_policy=True))
    if 'ucrp' in m:
        rows.append(make_row('UCRP', m['ucrp'], is_policy=True))
    rows.append(make_row('UBAH', m['ubah'], is_policy=False))
    rows.append(make_row(f'Best ({best.upper()})', m[best], is_policy=False))

    for asset in assets:
        rows.append(make_row(asset.upper(), m[asset], is_policy=False))

    df = pd.DataFrame(rows)
    cols = ['Strategy', 'fAPV', 'SR', 'MDD', 'AvgEntropy', 'AvgTurnover', 'AvgTxCost', 'AvgCash']
    df = df[cols]

    print(f"\n=== {data_set.upper()} Performance ===")
    n_policies = sum(1 for s in ['pretrained', 'osbl', 'ucrp'] if s in m)
    n_main = n_policies + 2  # policies + UBAH + Best

    def fmt(x):
        if pd.isna(x): return '    -'
        return f'{x:.4f}'

    table_lines = df.to_string(index=False, formatters={c: fmt for c in cols[1:]}).split('\n')
    print(table_lines[0])
    for i in range(1, n_main + 1):
        print(table_lines[i])
    print("-" * len(table_lines[0]))
    for i in range(n_main + 1, len(table_lines)):
        print(table_lines[i])

# %%
# Visualization function

def plot_dataset(data_set):
    """Plot log returns, weights, entropy, turnover, and drawdown for a given data set."""
    if data_set not in all_metrics or 'pretrained' not in results[data_set]:
        print(f"No data available for {data_set}")
        return

    m = all_metrics[data_set]
    df = results[data_set]['pretrained']
    datetimes = df['datetimes'].values
    ps = POLICY_STYLES

    # Cumulative log returns
    pretrained_cum_log_ret = np.cumsum(m['pretrained']['step_log_returns'])
    ucrp_cum_log_ret = np.cumsum(m['ucrp']['step_log_returns']) if 'ucrp' in m else None
    osbl_cum_log_ret = np.cumsum(m['osbl']['step_log_returns']) if 'osbl' in m else None
    ubah_cum_log_ret = np.log(m['ubah']['apv_over_time'])
    asset_apv = m['_asset_apv']

    # Period turnover and drawdown
    pretrained_turnover = (1 - df['transaction_remainder_factor'].values) / commission_rate
    pretrained_drawdown = m['pretrained']['running_drawdown']
    osbl_turnover = (1 - results[data_set]['osbl']['transaction_remainder_factor'].values) / commission_rate if 'osbl' in results[data_set] else None
    osbl_drawdown = m['osbl']['running_drawdown'] if 'osbl' in m else None

    fig = make_subplots(rows=5, cols=1, subplot_titles=(f'{data_set.upper()} SET: Cumulative Log Returns', 'Portfolio Weights', 'Normalized Entropy', 'Period Turnover', 'Running Drawdown'), vertical_spacing=0.05, row_heights=[0.6, 0.2, 0.2, 0.2, 0.2], shared_xaxes=True)

    # Cumulative log returns subplot - policies
    fig.add_trace(go.Scatter(x=datetimes, y=pretrained_cum_log_ret, mode='lines', name='Pretrained', line=dict(color=ps['pretrained']['color'], dash=ps['pretrained']['dash'], width=ps['pretrained']['width']), legendgroup='Pretrained'), row=1, col=1)
    if osbl_cum_log_ret is not None:
        fig.add_trace(go.Scatter(x=datetimes, y=osbl_cum_log_ret, mode='lines', name='OSBL', line=dict(color=ps['osbl']['color'], dash=ps['osbl']['dash'], width=ps['osbl']['width']), legendgroup='OSBL'), row=1, col=1)
    if ucrp_cum_log_ret is not None:
        fig.add_trace(go.Scatter(x=datetimes, y=ucrp_cum_log_ret, mode='lines', name='UCRP', line=dict(color=ps['ucrp']['color'], dash=ps['ucrp']['dash'], width=ps['ucrp']['width']), legendgroup='UCRP'), row=1, col=1)
    fig.add_trace(go.Scatter(x=datetimes, y=ubah_cum_log_ret, mode='lines', name='UBAH', line=dict(color=ps['ubah']['color'], dash=ps['ubah']['dash'], width=ps['ubah']['width']), legendgroup='UBAH'), row=1, col=1)
    # Cumulative log returns subplot - individual assets
    for asset in assets:
        asset_cum_log_ret = np.log(asset_apv[asset])
        fig.add_trace(go.Scatter(x=datetimes, y=asset_cum_log_ret, mode='lines', name=asset.upper(), line=dict(color=get_asset_color(asset), width=1), legendgroup=asset.upper()), row=1, col=1)

    # Weights subplot
    fig.add_trace(go.Scatter(x=datetimes, y=df['weight_cash'].values, mode='lines', name='Cash', line=dict(color=get_asset_color('cash')), legendgroup='Cash'), row=2, col=1)
    for asset in assets:
        fig.add_trace(go.Scatter(x=datetimes, y=df[f'weight_{asset}'].values, mode='lines', name=asset.upper(), line=dict(color=get_asset_color(asset)), legendgroup=asset.upper(), showlegend=False), row=2, col=1)

    # Entropy subplot - both policies
    fig.add_trace(go.Scatter(x=datetimes, y=df['normalized_entropy'].values, mode='lines', name='Pretrained', line=dict(color=ps['pretrained']['color'], width=1.5), legendgroup='Pretrained', showlegend=False), row=3, col=1)
    if 'osbl' in results[data_set]:
        osbl_entropy = results[data_set]['osbl']['normalized_entropy'].values
        fig.add_trace(go.Scatter(x=datetimes, y=osbl_entropy, mode='lines', name='OSBL', line=dict(color=ps['osbl']['color'], width=1.5), legendgroup='OSBL', showlegend=False), row=3, col=1)

    # Period turnover subplot
    fig.add_trace(go.Scatter(x=datetimes, y=pretrained_turnover, mode='lines', name='Pretrained', line=dict(color=ps['pretrained']['color'], width=1.5), legendgroup='Pretrained', showlegend=False), row=4, col=1)
    if osbl_turnover is not None:
        fig.add_trace(go.Scatter(x=datetimes, y=osbl_turnover, mode='lines', name='OSBL', line=dict(color=ps['osbl']['color'], width=1.5), legendgroup='OSBL', showlegend=False), row=4, col=1)

    # Running drawdown subplot
    fig.add_trace(go.Scatter(x=datetimes, y=pretrained_drawdown, mode='lines', name='Pretrained', line=dict(color=ps['pretrained']['color'], width=1.5), legendgroup='Pretrained', showlegend=False), row=5, col=1)
    if osbl_drawdown is not None:
        fig.add_trace(go.Scatter(x=datetimes, y=osbl_drawdown, mode='lines', name='OSBL', line=dict(color=ps['osbl']['color'], width=1.5), legendgroup='OSBL', showlegend=False), row=5, col=1)

    fig.update_layout(height=1400, showlegend=True, hovermode='x unified')
    fig.update_yaxes(title_text='Cum. Log Return', row=1, col=1)
    fig.update_yaxes(title_text='Weight', row=2, col=1)
    fig.update_yaxes(title_text='Entropy', range=[0, 1], row=3, col=1)
    fig.update_yaxes(title_text='Turnover', row=4, col=1)
    fig.update_yaxes(title_text='Drawdown', row=5, col=1)
    for row in range(1, 6):
        fig.update_xaxes(title_text='Date', showticklabels=True, row=row, col=1)
    fig.show()

# %%
# plot_dataset('train')

# %%
# plot_dataset('validation')

# %%
plot_dataset('test')

# %%
# Interactive N-day aggregation for weights and entropy

def plot_aggregated(data_set, n_days):
    """Plot weights and entropy aggregated into N-day buckets."""
    if data_set not in all_metrics or 'pretrained' not in results[data_set]:
        print(f"No data available for {data_set}")
        return

    df = results[data_set]['pretrained']
    datetimes = pd.to_datetime(df['datetimes'])
    periods_per_day = int(24 * 60 / RESOLUTION_MINUTES)
    bucket_size = n_days * periods_per_day

    n_periods = len(df)
    n_buckets = n_periods // bucket_size
    if n_buckets == 0:
        print(f"N={n_days} days is too large for this dataset ({n_periods} periods)")
        return

    # Aggregate weights and entropy into buckets
    weight_cols = ['weight_cash'] + [f'weight_{a}' for a in assets]
    bucket_datetimes, bucket_weights, bucket_entropy = [], {col: [] for col in weight_cols}, []

    for i in range(n_buckets):
        start_idx, end_idx = i * bucket_size, (i + 1) * bucket_size
        bucket_datetimes.append(datetimes.iloc[start_idx])
        for col in weight_cols:
            bucket_weights[col].append(df[col].iloc[start_idx:end_idx].mean())
        bucket_entropy.append(df['normalized_entropy'].iloc[start_idx:end_idx].mean())

    fig = make_subplots(rows=2, cols=1, subplot_titles=(f'{data_set.upper()}: Avg Weights ({n_days}-day buckets)', f'Avg Entropy ({n_days}-day buckets)'), vertical_spacing=0.1, row_heights=[0.6, 0.4], shared_xaxes=True)

    fig.add_trace(go.Scatter(x=bucket_datetimes, y=bucket_weights['weight_cash'], mode='lines', name='Cash', line=dict(color=get_asset_color('cash')), legendgroup='Cash'), row=1, col=1)
    for asset in assets:
        col = f'weight_{asset}'
        fig.add_trace(go.Scatter(x=bucket_datetimes, y=bucket_weights[col], mode='lines', name=asset.upper(), line=dict(color=get_asset_color(asset)), legendgroup=asset.upper()), row=1, col=1)

    fig.add_trace(go.Scatter(x=bucket_datetimes, y=bucket_entropy, mode='lines', name='Entropy', line=dict(color='purple', width=2), showlegend=False), row=2, col=1)

    fig.update_layout(height=800, showlegend=True, hovermode='x unified')
    fig.update_yaxes(title_text='Weight', row=1, col=1)
    fig.update_yaxes(title_text='Entropy', range=[0, 1], row=2, col=1)
    fig.update_xaxes(title_text='Date', row=1, col=1)
    fig.update_xaxes(title_text='Date', row=2, col=1)
    fig.show()

# %%
# Interactive widget for N-day aggregation

output = Output()

def on_button_click(b):
    with output:
        clear_output(wait=True)
        plot_aggregated(dataset_selector.value, n_days_input.value)

from ipywidgets import Dropdown
dataset_selector = Dropdown(options=['train', 'validation', 'test'], value='validation', description='Dataset:')
n_days_input = IntText(value=7, description='N days:', min=1)
refresh_button = Button(description='Refresh Plot')
refresh_button.on_click(on_button_click)

display(HBox([dataset_selector, n_days_input, refresh_button]))
display(output)
on_button_click(None)

# %%


