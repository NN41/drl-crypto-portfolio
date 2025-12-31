# %%

import pandas as pd
from datetime import timedelta, datetime, timezone
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots

RESOLUTION_MINUTES = 30
START_DATE = datetime(2023, 10, 17, 17, 0, 0, tzinfo=timezone.utc)
# START_DATE = datetime(2025, 8, 24, 21, 0, 0, tzinfo=timezone.utc)
END_DATE = datetime(2025, 10, 15, 0, 30, 0, tzinfo=timezone.utc)

# START_TEST_DATE = None  # Set to None to hide the train/test split line
START_TEST_DATE = datetime(2025, 8, 24, 21, 0, 0, tzinfo=timezone.utc)

# REDUCED_LIST = None
REDUCED_LIST = [
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
    'XRP_USDC-PERPETUAL'
]


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

import plotly.express as px
colors = px.colors.qualitative.Plotly
color_map = {name: colors[i % len(colors)] for i, name in enumerate(instruments.keys())}

fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, subplot_titles=('Normalized Close Prices', '30-Day Rolling Volume (Millions USD)'))

for instrument_name, df in instruments.items():
    if REDUCED_LIST is not None and instrument_name not in REDUCED_LIST:
        continue
    df_filtered = df.copy()
    if START_DATE is not None:
        df_filtered = df_filtered[df_filtered['datetime'] >= START_DATE]
    if END_DATE is not None:
        df_filtered = df_filtered[df_filtered['datetime'] <= END_DATE]

    if len(df_filtered) > 0:
        normalized = df_filtered['close'] / df_filtered['close'].iloc[0]
        color = color_map[instrument_name]
        fig.add_trace(go.Scatter(x=df_filtered['datetime'], y=normalized, mode='lines', name=instrument_name, line=dict(width=2, color=color), legendgroup=instrument_name, showlegend=True), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_filtered['datetime'], y=df_filtered['volume_30d'], mode='lines', name=instrument_name, line=dict(width=2, color=color), legendgroup=instrument_name, showlegend=False), row=2, col=1)

if START_TEST_DATE is not None:
    fig.add_vline(x=START_TEST_DATE, line_dash='dash', line_color='black', row='all')
fig.update_layout(height=900, showlegend=True, hovermode='x unified')
fig.update_xaxes(title_text='Date', row=2, col=1)
fig.update_yaxes(title_text='Value Multiplier', row=1, col=1)
fig.update_yaxes(title_text='Volume (Millions USD)', row=2, col=1)
fig.show()

# %%

import numpy as np

stats = []
for instrument_name, df in instruments.items():
    if REDUCED_LIST is not None and instrument_name not in REDUCED_LIST:
        continue
    df_filtered = df.copy()
    if START_DATE is not None:
        df_filtered = df_filtered[df_filtered['datetime'] >= START_DATE]
    if START_TEST_DATE is not None:
        df_filtered = df_filtered[df_filtered['datetime'] < START_TEST_DATE]

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
