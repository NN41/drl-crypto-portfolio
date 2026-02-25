# Deep Reinforcement Learning for Cryptocurrency Portfolio Management

This repository contains a PyTorch implementation of the deep reinforcement learning framework for crypto portfolio management proposed by [Jiang et al. (2017)](https://arxiv.org/abs/1706.10059).

The agent is trained on 3.5 years of historical 30-minute OHLC data for 11 Deribit perpetuals. On the 3-month test set, the agent easily outperforms benchmarks, achieving 59% ROI / 2.15 Sharpe, vs. -39% / -2.11 (uniform buy-and-hold) and 7.3% / 1.28 (best individual asset).

We also identify key behavior patterns, such as high asset concentration, zero cash allocation and high turnover, and suggest and test a number of improvements.

## Key Features & Implementations
- **CNN-based EIIE Policy:** A CNN implementation of the Ensemble of Identical Independent Evaluators (EIIE), in which each asset is judged independently by a CNN-based sub-network with shared parameters. Final asset scores are combined via softmax (with a learnable cash bias) to produce portfolio weights.
- **Transaction Cost Model:** Iterative approximation of the transaction remainder factor $\mu_t$ following the paper's formulation, allowing the agent to take into account the effect of transaction costs.
- **Portfolio-Vector Memory (PVM):** Stores and retrieves previous portfolio weights during training (similar to experience replay memory), significantly speeding up training by allowing parallelization.
- **Online Stochastic Batch Learning (OSBL):** Geometrically-weighted sampling of mini-batches for online adaptation during walk-forward evaluation, allowing the agent to continuously learn from new market data.
- **Walk-Forward Evaluation:** An out-of-sample testing framework that steps through unseen data one period at a time, optionally updating the policy via OSBL.
<!-- - **Asset Availability Masking:** Handles missing data for assets that don't exist throughout the full history by masking them out of the softmax allocation. -->

## Demo
Below you can see the performance of the trained agent ("OSBL") on the 3-month test set, compared to standard benchmarks*. The OSBL agent achieves 59% ROI with a 2.15 Sharpe ratio, outperforming all benchmarks (including the best individual asset, PAXG) despite a broadly declining market.


![Test Set Performance Summary](./assets/demo_test_summary.png)

![Test Set Performance Plot](./assets/demo_test_set_performance.png)

*\*Benchmarks: UCRP = uniform constant rebalanced portfolio, UBAH = uniform buy-and-hold. Metrics: fAPV = final accumulated portfolio value, SR = annualized Sharpe ratio, MDD = maximum drawdown.*


## Background & Implementation

This section provides a brief overview of the framework and its implementation. The project follows the methodology from the [Jiang et al. (2017) paper](https://arxiv.org/abs/1706.10059).

### Problem Setting

<!-- TODO: Brief description of the portfolio management problem:
- m+1 assets (m crypto + cash), portfolio weight vector w_t, price relative vector y_t
- Objective: maximize cumulative log-return (Eq. 21 in paper)
- 30-minute trading periods on Deribit exchange
-->

### Price Tensor & Normalization

<!-- TODO: Describe the input tensor X_t with shape (f, n, m):
- f=3 features (high, low, close), n=50 lookback periods, m=11 assets
- All prices normalized by latest closing price (Eq. 18 in paper)
-->

### EIIE Network Topology

<!-- TODO: Describe the CNN architecture:
- Conv1: (3, M, 50) -> (2, M, 48), kernel (1,3), ReLU
- Conv2: (2, M, 48) -> (20, M, 1), kernel (1,48), ReLU
- Concatenate previous portfolio weights -> (21, M, 1)
- Conv3: (21, M, 1) -> (1, M, 1), kernel (1,1)
- Add learnable cash bias, softmax -> portfolio weights
- Key insight: all convolution kernels have height 1, so assets are independent until softmax
-->

### Transaction Cost

<!-- TODO: Describe the transaction remainder factor mu_t:
- Iterative approximation (Theorem 1 in paper)
- Initial guess from Eq. 16
- Fixed iterations during training, convergence-based during backtest
-->

### Reward Function

<!-- TODO: Describe the reward function:
- Maximize average log return R = (1/t_f) * sum(r_t)
- r_t = ln(mu_t * y_t . w_{t-1})
- Full exploitation approach: no exploration needed
- Deterministic policy gradient
-->

### Online Stochastic Batch Learning

<!-- TODO: Describe OSBL:
- Geometrically distributed batch sampling (Eq. 26)
- Recent market events weighted more heavily
- Enables continuous online learning during trading
-->

## Data

<!-- TODO: Describe the dataset:
- Source: Deribit exchange
- 11 crypto assets: BTC, ETH perpetual futures + SOL, XRP, DOGE, ADA, AVAX, DOT, BNB, UNI, PAXG spot perpetuals
- Resolution: 30-minute OHLCV candles
- Train/validation/test split dates and sizes
- Commission rate used
-->

## Experiments & Results
Diving into the behavior of the trained agent, we discover three key behaviors:
1. Zero cash allocation. Trained agents never allocate any money to cash asset under any circumstances. Attempts to mitigate this effect through hyperparameter tweaking (such as weight decay) always fail.
2. Low portfolio diversification.
3. High turnover. 

<!-- TODO: For each experiment group, add a subsection with:
- What was varied and why
- Key findings (1-2 paragraphs)
- Figures showing results
-->

### Experiment 1: Batch Size

<!-- TODO: Describe batch size experiments from runs_completed/runs_batch/ -->

### Experiment 2: Weight Decay

### Experiment 3: Asset Availability Masking

<!-- TODO: Describe masking experiments from runs_completed/runs_masking/ -->

<!-- TODO: Add a performance comparison table (fAPV, SR, MDD) for best model vs benchmarks -->

## Key Learnings & Obstacles

<!-- TODO: Write personal reflections, e.g.:
- Challenges encountered during implementation
- Differences from the paper's setup (Deribit vs Poloniex, different time period, etc.)
- Insights about the framework's strengths and weaknesses
-->

## Future Work

- [ ] TODO
- [ ] TODO
- [ ] TODO
