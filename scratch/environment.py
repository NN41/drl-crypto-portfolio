# %%

import gymnasium as gym
import numpy as np
import pandas as pd
from typing import Optional
from scipy.special import softmax
import matplotlib.pyplot as plt
from datetime import timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

RESOLUTION_MINUTES = 30

df_btc = pd.read_csv('./data/raw/ohlcv/BTC-PERPETUAL_resolution_30.csv')
df_eth = pd.read_csv('./data/raw/ohlcv/ETH-PERPETUAL_resolution_30.csv')

df_btc['datetime'] = pd.to_datetime(df_btc['datetime'], utc=True) + timedelta(minutes=RESOLUTION_MINUTES)
df_eth['datetime'] = pd.to_datetime(df_eth['datetime'], utc=True) + timedelta(minutes=RESOLUTION_MINUTES)

cols = ['datetime','high','low','close']
df_btc = df_btc[cols].copy()
df_eth = df_eth[cols].copy()

df_merged = pd.merge(df_btc, df_eth, on='datetime', how='outer', suffixes=('_btc','_eth')).sort_values('datetime')

assets = ['btc', 'eth']
cols_features = ['high','low','close']
n_periods = len(df_merged)
n_features = len(cols_features)
total_price_history = np.concatenate([
    np.ones((1, n_periods, n_features)),
    np.stack([
        df_merged[[f'{c}_btc' for c in cols_features]].values,
        df_merged[[f'{c}_eth' for c in cols_features]].values
    ])
], axis=0)

total_history_close_datetimes = df_merged['datetime']

# Array indexing constants (derived from data structure)
ASSET_CASH = 0
ASSET_BTC = assets.index('btc') + 1
ASSET_ETH = assets.index('eth') + 1
N_ASSETS = len(assets) + 1

FEAT_HIGH = cols_features.index('high')
FEAT_LOW = cols_features.index('low')
FEAT_CLOSE = cols_features.index('close')
N_FEATURES = n_features

# %%

class CandlestickEnv(gym.Env):

    def __init__(self, total_price_history: np.array, total_history_close_datetimes: pd.Series, n_recent_periods: int = 50):

        n_assets, n_periods, n_features = total_price_history.shape

        self._all_close_datetimes = total_history_close_datetimes
        self._n_recent_periods = n_recent_periods
        self._n_periods = n_periods
        self._n_assets = n_assets # includes cash
        self._n_features = n_features

        self._current_idx = None # index i-1 corresponds to Period i and time interval [i-1, i) with close price at time i
        self._recent_price_history = None
        self._portfolio_vector_period_end = np.array([1] + [0] * (self._n_assets - 1))
        self._portfolio_vector_period_start = None
        self._episode_period = None

        self.observation_space = gym.spaces.Dict({
            'recent_price_history': gym.spaces.Box(low=0, high=np.inf, shape=(n_assets, n_recent_periods, n_features), dtype=np.float32),
            'previous_portfolio_vector': gym.spaces.Box(low=0, high=1, shape=(n_assets,), dtype=np.float32)
        })
        self.action_space = gym.spaces.Box(low=0, high=1, shape=(n_assets,), dtype=np.float32)

    def _get_obs(self):
        return {
            "recent_price_history": self._recent_price_history,
            "previous_portfolio_vector": self._portfolio_vector_period_start
        }

    def _get_info(self):
        info = {
            'data_index': self._current_idx,
            # 'data_period': self._current_idx + 1,
            'datetime': self._all_close_datetimes[self._current_idx],
            'btc_close': self._btc_close,
            'episode_period': self._episode_period,
            'portfolio_vector_period_end': self._portfolio_vector_period_end,
            'portfolio_vector_period_start': self._portfolio_vector_period_start,
        }
        return info

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        # Check if options specifies starting at minimum index
        start_from_beginning = options.get('start_from_beginning', False) if options else False

        if start_from_beginning:
            self._current_idx = self._n_recent_periods - 1
        else:
            self._current_idx = np.random.randint(self._n_recent_periods, self._n_periods)

        end_idx = self._current_idx + 1
        start_idx = end_idx - self._n_recent_periods
        self._recent_price_history = total_price_history[:, start_idx:end_idx]
        self._btc_close = self._recent_price_history[ASSET_BTC, -1, FEAT_CLOSE]

        observation = self._get_obs()
        info = self._get_info()

        self._episode_period = 0

        return observation, info

    def step(self, action):

        # At the start of Period t+1 (at time t), set the portfolio weights. The first element corresponds to the cash
        assert np.abs(np.sum(action) - 1) < 1e-6, f"action doesn't sum to 1 ({np.sum(action)})"
        self._portfolio_vector_period_start = action

        # Progress the period by executing one timestep from time t to time t+1 (that is, run Period t+1)
        self._episode_period += 1
        self._current_idx += 1

        # Take a step forward from time t to t+1 and observe the price at t+1 (the close price of Period t+1)
        end_idx = self._current_idx + 1
        start_idx = end_idx - self._n_recent_periods
        self._recent_price_history = total_price_history[:, start_idx:end_idx]

        self._btc_close = self._recent_price_history[ASSET_BTC, -1, FEAT_CLOSE]

        current_price_vector = self._recent_price_history[:, -1, FEAT_CLOSE] # price at time t+1 (close price of Period t+1)
        previous_price_vector = self._recent_price_history[:, -2, FEAT_CLOSE] # price at time t (close price of Period t)
        self._price_relative_vector = current_price_vector / previous_price_vector # relative price changes over Period t+1

        numerator = self._price_relative_vector * self._portfolio_vector_period_start
        denominator = self._price_relative_vector @ self._portfolio_vector_period_start
        self._portfolio_vector_period_end = numerator / denominator

        transaction_remainder_factor = 1 # dummy data (fees are zero)
        log_return = np.log(transaction_remainder_factor * self._price_relative_vector @ self._portfolio_vector_period_start) # logarithmic rate of return over [t, t+1)
        reward = log_return # reward r_{t+1} = R(s_t, a_t, s_{t+1}) 

        terminated = False
        truncated = self._current_idx >= self._n_periods - 1 # check if we reached the end of the entire price history

        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info



# %%

env = CandlestickEnv(total_price_history, total_history_close_datetimes)

options = {
    'start_from_beginning': False
}

episode_over = False
all_rewards = []
all_weights_start = []
all_weights_end = []
ep_datetimes = []

max_steps = 4
step = 0
obs, info = env.reset(options=options)
start_dt = info['datetime']
while not episode_over:

    unnormalized_action = env.action_space.sample()
    # action = unnormalized_action / np.sum(unnormalized_action)
    # # action = np.array([1, 0, 0]) # cash only
    # # action = np.array([0, 1, 0]) # btc only
    action = np.array([0, 0.5, 0.5]) # equal-weighted
    obs, reward, terminated, truncated, info = env.step(action)
    assert (action == obs['previous_portfolio_vector']).all()
    episode_over = terminated or truncated

    all_rewards.append(reward)
    all_weights_start.append(info['portfolio_vector_period_start'])
    all_weights_end.append(info['portfolio_vector_period_end'])
    ep_datetimes.append(info['datetime'])

    print(info)

    step += 1
    if step >= max_steps:
        break
end_dt = info['datetime']
env.close()

# %%
step_log_returns = np.array(all_rewards)
step_returns = np.exp(step_log_returns) - 1

running_log_returns = np.cumsum(step_log_returns)

portfolio_value_start = 1
apv_ratios = np.exp(running_log_returns)
portfolio_values = np.insert(portfolio_value_start * apv_ratios, 0, portfolio_value_start)

running_returns = apv_ratios - 1

risk_free_return = 0
periodic_sharpe_numerator = np.mean(step_returns - risk_free_return)
periodic_sharpe_denominator = np.sqrt(np.var(step_returns - risk_free_return, ddof=1))
periodic_sharpe_ratio = periodic_sharpe_numerator / periodic_sharpe_denominator
annualized_sharpe_ratio = np.sqrt(365 * 24 * 60 / RESOLUTION_MINUTES) * periodic_sharpe_ratio
annualized_sharpe_ratio

running_max = np.maximum.accumulate(portfolio_values)
running_drawdown = (running_max - portfolio_values) / running_max
running_max_drawdown = np.maximum.accumulate(running_drawdown)


df_episode = df_merged[(df_merged['datetime'] >= start_dt) & (df_merged['datetime'] <= end_dt)]
btc_value_multipliers = df_episode['close_btc'] / df_episode['close_btc'].iloc[0]
eth_value_multipliers = df_episode['close_eth'] / df_episode['close_eth'].iloc[0]

# BTC metrics
btc_log_returns = np.log(df_episode['close_btc'] / df_episode['close_btc'].shift(1)).dropna()
btc_returns = np.exp(btc_log_returns) - 1
btc_final_apv = btc_value_multipliers.iloc[-1]
btc_sharpe = np.sqrt(365 * 24 * 60 / RESOLUTION_MINUTES) * np.mean(btc_returns) / np.sqrt(np.var(btc_returns, ddof=1))

# ETH metrics
eth_log_returns = np.log(df_episode['close_eth'] / df_episode['close_eth'].shift(1)).dropna()
eth_returns = np.exp(eth_log_returns) - 1
eth_final_apv = eth_value_multipliers.iloc[-1]
eth_sharpe = np.sqrt(365 * 24 * 60 / RESOLUTION_MINUTES) * np.mean(eth_returns) / np.sqrt(np.var(eth_returns, ddof=1))

print(f"Agent: fAPV={apv_ratios[-1]:.4f}, SR={annualized_sharpe_ratio:.4f}, MDD={running_max_drawdown[-1]:.4f}")
print(f"BTC:   fAPV={btc_final_apv:.4f}, SR={btc_sharpe:.4f}")
print(f"ETH:   fAPV={eth_final_apv:.4f}, SR={eth_sharpe:.4f}")


# %%

weights_start_array = np.array(all_weights_start)
weights_end_array = np.array(all_weights_end)

weights_btc_start = weights_start_array[:, ASSET_BTC]
weights_eth_start = weights_start_array[:, ASSET_ETH]
weights_btc_end = weights_end_array[:, ASSET_BTC]
weights_eth_end = weights_end_array[:, ASSET_ETH]

fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06, subplot_titles=('Value Multipliers', 'Weights', 'Drawdown'))
fig.add_trace(go.Scatter(x=df_episode['datetime'], y=btc_value_multipliers, mode='lines', name='BTC', line=dict(color='orange')), row=1, col=1)
fig.add_trace(go.Scatter(x=df_episode['datetime'], y=eth_value_multipliers, mode='lines', name='ETH', line=dict(color='blue')), row=1, col=1)
fig.add_trace(go.Scatter(x=ep_datetimes, y=apv_ratios, mode='lines', name='Agent', line=dict(color='black')), row=1, col=1)
fig.add_trace(go.Scatter(x=ep_datetimes, y=weights_btc_start, mode='lines', name='BTC Start', line=dict(color='orange', dash='solid')), row=2, col=1)
fig.add_trace(go.Scatter(x=ep_datetimes, y=weights_eth_start, mode='lines', name='ETH Start', line=dict(color='blue', dash='solid')), row=2, col=1)
fig.add_trace(go.Scatter(x=ep_datetimes, y=weights_btc_end, mode='lines', name='BTC End', line=dict(color='orange', dash='dash')), row=2, col=1)
fig.add_trace(go.Scatter(x=ep_datetimes, y=weights_eth_end, mode='lines', name='ETH End', line=dict(color='blue', dash='dash')), row=2, col=1)
fig.add_trace(go.Scatter(x=ep_datetimes, y=running_drawdown[1:], mode='lines', name='Running Drawdown', line=dict(color='orange')), row=3, col=1)
fig.add_trace(go.Scatter(x=ep_datetimes, y=running_max_drawdown[1:], mode='lines', name='Running Max Drawdown', line=dict(color='red')), row=3, col=1)
fig.update_layout(height=1000, showlegend=True, hovermode='x unified')
fig.update_xaxes(title_text='Date', row=3, col=1)
fig.update_yaxes(title_text='Value Multiplier', row=1, col=1)
fig.update_yaxes(title_text='Weight', row=2, col=1)
fig.update_yaxes(title_text='Drawdown', row=3, col=1)
fig.show()

# %%




# %%

import torch
import torch.nn as nn
import torch.nn.functional as F

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'

recent_prices = obs['recent_price_history'][1:, :, :] # removes the cash asset (shape: n_non_cash_assets, n_periods, n_features)
latest_close_prices = recent_prices[:, 0:1, FEAT_CLOSE:FEAT_CLOSE+1] # latest close prices for all non-cash assets (shape allows broadcasting)
normalized_recent_prices = recent_prices / latest_close_prices # normalize all prices by latest close. Order: H-L-C
portfolio_vector = obs['previous_portfolio_vector']

price_tensor = torch.Tensor(normalized_recent_prices.transpose(2,0,1)).to(device)
weight_tensor = torch.Tensor(portfolio_vector).to(device)

price_batch = torch.stack((price_tensor, price_tensor, price_tensor, price_tensor, price_tensor))
weight_batch = torch.stack((weight_tensor, weight_tensor, weight_tensor, weight_tensor, weight_tensor))

# %%

class CNNPolicy(nn.Module):
    def __init__(self, n_features: int, n_recent_periods: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=n_features, out_channels=2, kernel_size=(1, 3))
        self.conv2 = nn.Conv2d(in_channels=2, out_channels=20, kernel_size=(1, n_recent_periods-2))
        self.conv3 = nn.Conv2d(in_channels=21, out_channels=1, kernel_size=1) # one extra in-channel for the previous portfolio vector
        self.cash_bias = nn.Parameter(torch.zeros(1)) # add a learnable cash score, initialized at 0

    def forward(self, normalized_historical_prices: torch.Tensor, previous_portfolio_weights: torch.Tensor) -> torch.Tensor:
        # normalized_historical_prices must have shape (B, F, M, N) or (F, M, N)
        # previous_portfolio_weights must have shape (B, M+1) or (M+1,)
        # for B = batch size, F = number of features (=3), M = number of non-cash assets, N = number of periods (=50)
        x = normalized_historical_prices
        w_prev = previous_portfolio_weights

        if x.dim() == 3: # add batch dimension if necessary
            x = x.unsqueeze(0)
        elif x.dim() != 4:
            raise ValueError(f"normalized_historical_prices must be 3d or 4d, got {x.dim()}")
        
        if w_prev.dim() == 1: # add batch dimension if necessary
            w_prev = w_prev.unsqueeze(0)
        elif w_prev.dim() != 2:
            raise ValueError(f"previous_portfolio_weights must be 1d or 2d, got {x.dim()}")
        
        batch_size = x.shape[0]

        h1 = F.relu(self.conv1(x)) # (B, 2, M, N-2)
        h2 = F.relu(self.conv2(h1)) # (B, 20, M, 1)

        # inject previous non-cash portfolio weights as a feature map
        w_map = w_prev[:, 1:].unsqueeze(1).unsqueeze(-1) # (B, M+1) -> (B, 1, M, 1)
        h2 = torch.cat([w_map, h2], dim=1)

        scores = self.conv3(h2).squeeze(-1).squeeze(1) # (B, M) non-cash asset logits
        cash_score = self.cash_bias.expand(batch_size, 1) # broadcast (1,) to (B, 1) without copying
        logits = torch.cat([cash_score, scores], dim=1)
        weights = F.softmax(logits, dim=1).squeeze() # (B, M+1) or (M+1,)

        return weights
            
policy = CNNPolicy(n_features=3, n_recent_periods=50)

single_action = policy(price_tensor, weight_tensor)
batch_action = policy(price_batch, weight_batch)

single_action.squeeze(0)
batch_action.squeeze(0)
