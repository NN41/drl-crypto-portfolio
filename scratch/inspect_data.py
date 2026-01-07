# %%

import pandas as pd
from datetime import timedelta, datetime, timezone
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

RESOLUTION_MINUTES = 30

# Date ranges for train, validation, and test sets
START_DATE_TRAIN = datetime(2021, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
START_DATE_VALIDATION = datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
START_DATE_TEST = datetime(2025, 7, 1, 0, 0, 0, 0, tzinfo=timezone.utc)
END_DATE_TEST = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)

# Report statistics for each set
for name, start, end in [('Training', START_DATE_TRAIN, START_DATE_VALIDATION), ('Validation', START_DATE_VALIDATION, START_DATE_TEST), ('Test', START_DATE_TEST, END_DATE_TEST)]:
    delta = end - start
    periods = int(delta.total_seconds() / (RESOLUTION_MINUTES * 60))
    print(f"{name}: {periods:,} periods, {delta.days / 365:.1f} years, {delta.days / 365 * 12:.1f} months")

# %%



REDUCED_LIST = None
# REDUCED_LIST = [
#     'ADA_USDC-PERPETUAL',
#     'AVAX_USDC-PERPETUAL',
#     'BTC-PERPETUAL',
#     'BNB_USDC-PERPETUAL',
#     'DOGE_USDC-PERPETUAL',
#     'ETH-PERPETUAL',
#     'LINK_USDC-PERPETUAL',
#     'PAXG_USDC-PERPETUAL',
#     'SOL_USDC-PERPETUAL',
#     'TRUMP_USDC-PERPETUAL',
#     'XRP_USDC-PERPETUAL'
# ]


# %%

data_dir = Path('./data/raw/ohlcv')
csv_files = sorted(data_dir.glob('*_resolution_30.csv'))

instruments = {}
for csv_file in csv_files:
    instrument_name = csv_file.stem.replace('_resolution_30', '')
    df = pd.read_csv(csv_file).sort_values('timestamp')
    df['datetime'] = pd.to_datetime(df['datetime'], utc=True) + timedelta(minutes=RESOLUTION_MINUTES)
    df = df.set_index('datetime')
    df_resampled = pd.DataFrame({
        'close': df['close'].resample('1D').last(),
        'cost': df['cost'].resample('1D').sum()
    }).reset_index()
    df_resampled['volume_30d'] = df_resampled['cost'].rolling(window=30, min_periods=1).sum() / 1e6
    instruments[instrument_name] = df_resampled

# %%

# Metric type: 'cumulative_return', 'cumulative_log_return'
METRIC_TYPE = 'cumulative_return'

def calculate_metric(prices):
    if METRIC_TYPE == 'cumulative_return':
        return prices / prices.iloc[0]
    elif METRIC_TYPE == 'cumulative_log_return':
        return np.log(prices / prices.iloc[0])

metric_labels = {'cumulative_return': 'Total Return', 'cumulative_log_return': 'Total Log Return'}
metric_titles = {'cumulative_return': 'Total Return', 'cumulative_log_return': 'Total Log Return'}

import plotly.express as px
colors = px.colors.qualitative.Plotly
color_map = {name: colors[i % len(colors)] for i, name in enumerate(instruments.keys())}

fig = make_subplots(rows=3, cols=1, shared_xaxes=False, vertical_spacing=0.05, subplot_titles=(f'Training Set - {metric_titles[METRIC_TYPE]}', f'Validation Set - {metric_titles[METRIC_TYPE]}', f'Test Set - {metric_titles[METRIC_TYPE]}'))

for instrument_name, df in instruments.items():
    if REDUCED_LIST is not None and instrument_name not in REDUCED_LIST:
        continue

    # Training set
    df_train = df[(df['datetime'] >= START_DATE_TRAIN) & (df['datetime'] < START_DATE_VALIDATION)].copy()
    if len(df_train) > 0:
        metric_train = calculate_metric(df_train['close'])
        color = color_map[instrument_name]
        fig.add_trace(go.Scatter(x=df_train['datetime'], y=metric_train, mode='lines', name=instrument_name, line=dict(width=2, color=color), legendgroup=instrument_name, showlegend=True), row=1, col=1)

    # Validation set
    df_val = df[(df['datetime'] >= START_DATE_VALIDATION) & (df['datetime'] < START_DATE_TEST)].copy()
    if len(df_val) > 0:
        metric_val = calculate_metric(df_val['close'])
        fig.add_trace(go.Scatter(x=df_val['datetime'], y=metric_val, mode='lines', name=instrument_name, line=dict(width=2, color=color), legendgroup=instrument_name, showlegend=False), row=2, col=1)

    # Test set
    df_test = df[(df['datetime'] >= START_DATE_TEST) & (df['datetime'] <= END_DATE_TEST)].copy()
    if len(df_test) > 0:
        metric_test = calculate_metric(df_test['close'])
        fig.add_trace(go.Scatter(x=df_test['datetime'], y=metric_test, mode='lines', name=instrument_name, line=dict(width=2, color=color), legendgroup=instrument_name, showlegend=False), row=3, col=1)

fig.update_layout(height=1200, showlegend=True, hovermode='x unified')
fig.update_xaxes(title_text='Date', row=1, col=1)
fig.update_xaxes(title_text='Date', row=2, col=1)
fig.update_xaxes(title_text='Date', row=3, col=1)
fig.update_yaxes(title_text=metric_labels[METRIC_TYPE], row=1, col=1)
fig.update_yaxes(title_text=metric_labels[METRIC_TYPE], row=2, col=1)
fig.update_yaxes(title_text=metric_labels[METRIC_TYPE], row=3, col=1)
fig.show()

# %%

stats = []
for instrument_name, df in instruments.items():
    if REDUCED_LIST is not None and instrument_name not in REDUCED_LIST:
        continue
    df_filtered = df.copy()
    if START_DATE_TRAIN is not None:
        df_filtered = df_filtered[df_filtered['datetime'] >= START_DATE_TRAIN]
    if START_DATE_TEST is not None:
        df_filtered = df_filtered[df_filtered['datetime'] < START_DATE_TEST]

    if len(df_filtered) > 0:
        normalized = df_filtered['close'] / df_filtered['close'].iloc[0]
        daily_returns = df_filtered['close'].pct_change().dropna()
        annualized_volatility = daily_returns.std() * np.sqrt(365)

        stats.append({
            'Instrument': instrument_name,
            'Volume at END (M USD)': round(df_filtered['volume_30d'].iloc[-1], 2),
            'Min Normalized Price': round(normalized.min(), 2),
            'Max Normalized Price': round(normalized.max(), 2),
            'Annualized Volatility': round(annualized_volatility, 2),
            'Final Normalized Price': round(normalized.iloc[-1], 2)
        })

df_stats = pd.DataFrame(stats).sort_values('Volume at END (M USD)', ascending=False)
print(df_stats.to_string(index=False))

# %%
