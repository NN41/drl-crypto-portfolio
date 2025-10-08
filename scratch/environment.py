# %%

import gymnasium as gym
import numpy as np
import pandas as pd
from typing import Optional
from scipy.special import softmax
import matplotlib.pyplot as plt
from datetime import timedelta

df_btc = pd.read_csv('./data/raw/ohlcv/BTC-PERPETUAL_resolution_30.csv')
df_eth = pd.read_csv('./data/raw/ohlcv/ETH-PERPETUAL_resolution_30.csv')

df_btc['datetime'] = pd.to_datetime(df_btc['datetime'], utc=True)
df_eth['datetime'] = pd.to_datetime(df_eth['datetime'], utc=True)

cols = ['datetime','open','high','low','close']
df_btc = df_btc[cols].copy()
df_eth = df_eth[cols].copy()

df_merged = pd.merge(df_btc, df_eth, on='datetime', how='outer', suffixes=('_btc','_eth')).sort_values('datetime')
cols_features = ['low','high','close']

n_periods = len(df_merged)
n_features = len(cols_features)
cash_history = np.ones((1, n_periods, n_features))

total_price_history = np.concatenate([
    cash_history,
    np.stack([
        df_merged[[f'{c}_btc' for c in cols_features]].values,
        df_merged[[f'{c}_eth' for c in cols_features]].values
    ])
], axis=0)

total_price_history_datetimes = df_merged['datetime']


# %%

class CandlestickEnv(gym.Env):

    def __init__(self, total_price_history: np.array, total_price_history_datetimes: pd.Series, n_periods_price_tensor: int = 50):

        n_assets, n_periods, n_features = total_price_history.shape

        self._all_datetimes = total_price_history_datetimes
        self._n_recent_periods = n_periods_price_tensor
        self._n_periods = n_periods
        self._n_assets = n_assets
        self._n_features = n_features

        # index i-1 corresponds to period i and time interval [i-1, i) with close price at time i
        self._period_idx = None # index i corresponds to Period i+1 and time interval [i, i+1)
        self._recent_price_history = None
        self._portfolio_vector = np.array([1] + [0] * (self._n_assets - 1))

        self.observation_space = gym.spaces.Box(low=0, high=np.inf, shape=(n_assets, n_periods_price_tensor, n_features))
        self.action_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(n_assets,)) # the actions are logits, we then use softmax to get the weights

    def _get_obs(self):
        return self._recent_price_history # of shape (n_assets, n_periods_price_tensor, n_features)

    def _get_info(self):
        info = {
            'period_idx': self._period_idx,
            'period': self._period_idx + 1,
            'datetime': self._all_datetimes[self._period_idx],
            'portfolio_vector': self._portfolio_vector
        }
        return info

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        self._period_idx = np.random.randint(self._n_recent_periods, self._n_periods) # start at a random point in the price history
        end_idx = self._period_idx
        start_idx = end_idx - self._n_recent_periods
        self._recent_price_history = total_price_history[:, start_idx:end_idx]

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action):
        # Execute one timestep from time t to time t+1

        # At the start of Period t+1 (at time t), fix the portfolio weights
        # by taking the softmax of a vector of logit actions. The first element corresponds to the cash
        self._portfolio_vector = softmax(action)

        # Take a step forward from time t to t+1 and observe the price at t+1 (the close price of Period t+1)
        self._period_idx += 1
        end_idx = self._period_idx
        start_idx = end_idx - self._n_recent_periods
        self._recent_price_history = total_price_history[:, start_idx:end_idx]

        next_price_vector = self._recent_price_history[:,-1,-1] # price at time t (close price of Period t)
        price_vector = self._recent_price_history[:,-2,-1] # price at time t+1 (close price of Period t+1)
        price_relative_vector = next_price_vector / price_vector # relative price changes

        transaction_remainder_factor = 1 # dummy data (fees are zero)

        log_return = np.log(transaction_remainder_factor * price_relative_vector @ self._portfolio_vector) # logarithmic rate of return over [t, t+1)
        reward = log_return # reward r_t for taking action a_t

        terminated = False
        truncated = self._period_idx >= self._n_periods - 1 # check if we reached the end of the entire price history

        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info

# %%

env = CandlestickEnv(total_price_history, total_price_history_datetimes)

episode_over = False
all_rewards = []
all_weights = []
all_datetimes = []

max_steps = 1000
step = 0
obs, info = env.reset()
while not episode_over:

    # action = env.action_space.sample()
    # action = np.array([1e62, -1e62, -1e62]) # cash only
    action = np.array([-1e62, 1e62, -1e62]) # btc only
    obs, reward, terminated, truncated, info = env.step(action)

    episode_over = terminated or truncated

    all_rewards.append(reward)
    all_weights.append(info['portfolio_vector'])
    all_datetimes.append(info['datetime'])

    step += 1
    if step >= max_steps:
        break
env.close()

total_reward = np.sum(all_rewards)

# %%
start_dt = all_datetimes[1]
end_dt = np.max(all_datetimes)

df_episode = df_merged[(df_merged['datetime'] >= start_dt) & (df_merged['datetime'] <= end_dt)]

btc_ep_log_return = np.log(df_episode['close_btc'].iloc[-1] / df_episode['close_btc'].iloc[0])
eth_ep_log_return = np.log(df_episode['close_eth'].iloc[-1] / df_episode['close_eth'].iloc[0])

print(f"Total episode log return: agent = {total_reward:.4f}, BTC = {btc_ep_log_return:.4f}, ETH = {eth_ep_log_return:.4f}")

# %%





log_returns_btc = np.log(df_episode['close_btc'] / df_episode['close_btc'].shift(1))
log_returns_eth = np.log(df_episode['close_eth'] / df_episode['close_eth'].shift(1))

# %%
fig, ax1 = plt.subplots(figsize=(12, 6))
ax1.plot(df_episode['datetime'], log_returns_btc, label='BTC', color='orange', alpha=0.7)
ax1.plot(df_episode['datetime'], log_returns_eth, label='ETH', color='blue', alpha=0.7)
ax1.set_xlabel('Date')
ax1.set_ylabel('Log Returns')
ax1.legend(loc='upper left')
ax1.grid(alpha=0.3)

ax2 = ax1.twinx()
ax2.plot(df_episode['datetime'], np.cumsum(all_rewards), label='Cumulative Reward', color='green', linewidth=2)
ax2.set_ylabel('Cumulative Reward')
ax2.legend(loc='upper right')

plt.tight_layout()
plt.show()

# %%

pd.DataFrame(all_weights, columns=['cash','btc','eth']).plot()

# %%

plt.plot(np.cumsum(all_rewards))

# %%

