# %%

import pandas as pd
from datetime import timedelta, datetime, timezone
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

RESOLUTION_MINUTES = 30

# Date ranges and split ratio
START_DATE = datetime(2022, 4, 28, 0, 0, 0, tzinfo=timezone.utc)
END_DATE = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
SPLIT_RATIO = (14, 1, 1)  # Train, Validation, Test

# Calculate split dates
total_duration = END_DATE - START_DATE
total_parts = sum(SPLIT_RATIO)
resolution_seconds = RESOLUTION_MINUTES * 60

START_DATE_TRAIN = START_DATE
val_timestamp = (START_DATE + total_duration * SPLIT_RATIO[0] / total_parts).timestamp()
START_DATE_VALIDATION = datetime.fromtimestamp(round(val_timestamp / resolution_seconds) * resolution_seconds, tz=timezone.utc)
test_timestamp = (START_DATE_VALIDATION + total_duration * SPLIT_RATIO[1] / total_parts).timestamp()
START_DATE_TEST = datetime.fromtimestamp(round(test_timestamp / resolution_seconds) * resolution_seconds, tz=timezone.utc)
END_DATE_TEST = END_DATE

# Report statistics for each set
for name, start, end in [('Training Set', START_DATE_TRAIN, START_DATE_VALIDATION), ('Validation Set', START_DATE_VALIDATION, START_DATE_TEST), ('Test Set', START_DATE_TEST, END_DATE_TEST)]:
    delta = end - start
    periods = int(delta.total_seconds() / (RESOLUTION_MINUTES * 60))
    print(f"{name}:\t{start} to {end}")
    print(f"\t\t{periods:,} periods, or {delta.days / 7:.1f} weeks, or {delta.days / 365 * 12:.1f} months")

# %%

REDUCED_LIST = None
# REDUCED_LIST = [
#     'BTC-PERPETUAL',
#     'ETH-PERPETUAL',
#     'SOL_USDC-PERPETUAL',
#     'XRP_USDC-PERPETUAL',
#     'DOGE_USDC-PERPETUAL',
#     'PAXG_USDC-PERPETUAL',
#     'ADA_USDC-PERPETUAL',
#     'AVAX_USDC-PERPETUAL',
#     'DOT_USDC-PERPETUAL',
#     'BNB_USDC-PERPETUAL',
#     'UNI_USDC-PERPETUAL',
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

# Sort instruments by 30-day volume at end of training set
instrument_volumes = []
for name, df in instruments.items():
    df_train = df[(df['datetime'] >= START_DATE_TRAIN) & (df['datetime'] < START_DATE_VALIDATION)]
    if len(df_train) > 0:
        instrument_volumes.append((name, df_train['volume_30d'].iloc[-1]))
instrument_volumes.sort(key=lambda x: x[1], reverse=True)
sorted_instruments = [name for name, _ in instrument_volumes]

color_map = {name: colors[i % len(colors)] for i, name in enumerate(sorted_instruments)}

fig = make_subplots(rows=3, cols=1, shared_xaxes=False, vertical_spacing=0.05, subplot_titles=(f'Training Set - {metric_titles[METRIC_TYPE]}', f'Validation Set - {metric_titles[METRIC_TYPE]}', f'Test Set - {metric_titles[METRIC_TYPE]}'))

for instrument_name in sorted_instruments:
    df = instruments[instrument_name]
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

annualization_factor = (365 * 24 * 60 / RESOLUTION_MINUTES)

stats = []
for instrument_name, df in instruments.items():
    if REDUCED_LIST is not None and instrument_name not in REDUCED_LIST:
        continue

    df_train = df[(df['datetime'] >= START_DATE_TRAIN) & (df['datetime'] < START_DATE_VALIDATION)].copy()

    if len(df_train) > 0:
        time_delta = df_train['datetime'].iloc[-1] - df_train['datetime'].iloc[0]
        years_in_data = time_delta.total_seconds() / (365 * 24 * 60 * 60)

        prices = df_train['close'].values

        period_log_returns = np.log(prices[1:] / prices[:-1])
        avg_period_log_return = np.mean(period_log_returns)
        total_log_return = np.log(prices[-1] / prices[0])
        final_portfolio_value_multiplier = np.exp(total_log_return)

        period_returns = prices[1:] / prices[:-1] - 1
        risk_free_return = 0
        periodic_sharpe_ratio = np.mean(period_returns - risk_free_return) / (np.sqrt(np.var(period_returns - risk_free_return, ddof=1)) + 1e-12)
        annualized_sharpe_ratio = np.sqrt(annualization_factor) * periodic_sharpe_ratio

        portfolio_values = np.exp(np.cumsum(period_log_returns))
        running_max = np.maximum.accumulate(portfolio_values)
        running_drawdown = (running_max - portfolio_values) / running_max
        running_max_drawdown = np.maximum.accumulate(running_drawdown)
        max_drawdown = running_max_drawdown[-1]

        stats.append({
            'Instrument': instrument_name,
            'Years loaded': round(years_in_data, 2),
            'Final 30d Volume (M USD)': round(df_train['volume_30d'].iloc[-1], 2),
            'Avg Period Log Return': round(avg_period_log_return, 6),
            'Total Log Return': round(total_log_return, 2),
            'Final APV': round(final_portfolio_value_multiplier, 2),
            'Sharpe Ratio': round(annualized_sharpe_ratio, 1),
            'Max Drawdown': round(max_drawdown, 2),
        })

df_stats = pd.DataFrame(stats).sort_values('Final 30d Volume (M USD)', ascending=False)
print(df_stats.to_string(index=False))

# %%
