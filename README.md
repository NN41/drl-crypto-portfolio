# Deep Reinforcement Learning for Cryptocurrency Portfolio Management

This repository contains a PyTorch implementation of the deep reinforcement learning framework for portfolio management proposed by [Jiang et al. (2017)](https://arxiv.org/abs/1706.10059). The framework uses a CNN-based Ensemble of Identical Independent Evaluators (EIIE) to learn optimal portfolio allocations across cryptocurrency assets traded on Deribit, using 30-minute trading periods. The agent is trained to maximize the log-return of a portfolio while accounting for transaction costs.

<!-- TODO: Add a one-liner on key results, e.g. "Achieves X-fold returns over Y days in backtests." -->

## Key Features & Implementations
- **CNN EIIE Policy:** A fully convolutional network where each asset is evaluated by an identical, independent sub-network (IIE) with shared parameters. Asset scores are combined via softmax (with a learnable cash bias) to produce portfolio weights.
- **Transaction Cost Model:** Iterative approximation of the transaction remainder factor $\mu_t$ following the paper's formulation (Eq. 14), supporting separate buy/sell commission rates.
- **Portfolio-Vector Memory (PVM):** Stores and retrieves previous portfolio weights during training, enabling parallel mini-batch training while accounting for transaction costs.
- **Online Stochastic Batch Learning (OSBL):** Geometrically-weighted sampling of mini-batches for online adaptation during walk-forward evaluation, allowing the agent to continuously learn from new market data.
- **Walk-Forward Evaluation:** A rigorous out-of-sample testing framework that steps through unseen data one period at a time, optionally updating the policy via OSBL.
- **Asset Availability Masking:** Handles missing data for assets that don't exist throughout the full history by masking them out of the softmax allocation.

## Demo

<!-- TODO: Add an equity curve figure here, e.g. ![Equity Curve](./assets/equity_curve.png) -->
<!-- TODO: Show a comparison of CNN EIIE vs benchmarks (UCRP, Buy-and-Hold) -->

## Setup & Usage

### Installation

1. **Clone the repository:**
    ```bash
    git clone https://github.com/NN41/drl-crypto-portfolio.git
    cd drl-crypto-portfolio
    ```

2. **Create and activate the Conda environment:**
    ```bash
    conda env create -f environment.yml
    conda activate drl-crypto-portfolio
    ```

<!-- TODO: Populate environment.yml / requirements.txt with actual dependencies (torch, pandas, numpy, plotly, etc.) -->

### Running Training & Evaluation

Training and evaluation are run interactively through VSCode's Python Interactive window using `# %%` cell markers.

- **Training:** Open `scratch/training.py` and run cells sequentially. Training metrics and checkpoints are saved to the `runs/` directory.
- **Post-Training Evaluation:** Open `scratch/post_training.py` to run walk-forward evaluation on train/validation/test splits and export results to CSV.
- **Analysis & Visualization:** Open `scratch/analysis_v2.py` to load post-training CSVs and generate performance tables and plots.

<!-- TODO: Add a brief note on how to configure hyperparameters (where to change them, key ones to tweak) -->

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

<!-- TODO: For each experiment group, add a subsection with:
- What was varied and why
- Key findings (1-2 paragraphs)
- Figures showing results
-->

### Experiment 1: Batch Size

<!-- TODO: Describe batch size experiments from runs_completed/runs_batch/ -->

### Experiment 2: Commission Rate Sensitivity

<!-- TODO: Describe commission rate experiments from runs_completed/runs_fees/ -->

### Experiment 3: Learning Rate Decay

<!-- TODO: Describe decay experiments from runs_completed/runs_decay/ -->

### Experiment 4: Asset Availability Masking

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
