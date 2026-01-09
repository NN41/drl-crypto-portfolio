# %%

import json
import numpy as np
from datetime import datetime, timezone
import torch
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.policies import CNNPolicy
from src.data_loading import load_and_split_data
from src.evaluation import run_walk_forward_test, calculate_performance_metrics
from src.model_io import load_checkpoint

# %%

run_dir = './runs/260108_174432'
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

checkpoint_path = f'{run_dir}/checkpoints/checkpoint_epoch_0.pt'
epoch = load_checkpoint(checkpoint_path, policy, optimizer, device)

# %%



print("Running walk-forward test on validation set (no OSBL)...")
initial_portfolio = np.ones(n_non_cash_assets + 1) / (n_non_cash_assets + 1)
# initial_portfolio = np.insert(np.zeros(n_non_cash_assets), 0, 1)
validation_results = run_walk_forward_test(
    policy=policy,
    initial_portfolio_weights=initial_portfolio,
    initial_prices=train_prices,
    forward_prices=validation_prices,
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
print(f"VALIDATION RESULTS: fAPV={validation_metrics['fAPV']:.4f}, SR={validation_metrics['SR']:.4f}, MDD={validation_metrics['MDD']:.4f}")

# %%

weights_datetimes = validation_results['datetimes'].iloc[1:-1].values

fig = make_subplots(rows=1, cols=1, subplot_titles=('Portfolio Weights',))

fig.add_trace(go.Scatter(x=weights_datetimes, y=validation_results['weight_cash'].dropna().values, mode='lines', name='Cash', hovertemplate='Cash: %{y:.6f}<extra></extra>'), row=1, col=1)

for asset in assets:
    weight_col = f'weight_{asset}'
    fig.add_trace(go.Scatter(x=weights_datetimes, y=validation_results[weight_col].dropna().values, mode='lines', name=asset.upper(), hovertemplate=f'{asset.upper()}: %{{y:.6f}}<extra></extra>'), row=1, col=1)

fig.update_layout(height=600, showlegend=True, hovermode='x unified')
fig.update_xaxes(title_text='Date', row=1, col=1)
fig.update_yaxes(title_text='Weight', row=1, col=1)
fig.show()

# %%