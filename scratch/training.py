# %%

import numpy as np
from datetime import datetime, timezone
import time

import torch
from torch.utils.tensorboard import SummaryWriter

from src.policies import CNNPolicy
from src.train_utils import geometrically_sample_batch_start_indices, run_one_epoch
from src.evaluation import run_walk_forward_test, calculate_performance_metrics
from src.model_io import save_model, save_checkpoint, save_run_config
from src.data_loading import load_and_split_data

commission_rate = 0.0005 # 0.0005 = 5 bips
n_recent_periods = 50 # number of periods passed to the policy to choose a portfolio
batch_size = 50 # with 2 assets, do x5.5 to match the number of training data points used per update; number of actions in a single batch
n_online_batches = 30
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

# %%

RESOLUTION_MINUTES = 30

START_DATE_TRAIN = datetime(2022, 4, 28, 0, 0, 0, tzinfo=timezone.utc)
START_DATE_VALIDATION = datetime(2025, 7, 17, 0, 0, 0, tzinfo=timezone.utc)
START_DATE_TEST = datetime(2025, 10, 9, 0, 0, 0, tzinfo=timezone.utc)
END_DATE_TEST = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)

instrument_names = [
    'BTC-PERPETUAL',
    'ETH-PERPETUAL',
    'SOL_USDC-PERPETUAL',
    'XRP_USDC-PERPETUAL',
    'DOGE_USDC-PERPETUAL',
    'PAXG_USDC-PERPETUAL',
    'ADA_USDC-PERPETUAL',
    'AVAX_USDC-PERPETUAL',
    'DOT_USDC-PERPETUAL',
    'BNB_USDC-PERPETUAL',
    'UNI_USDC-PERPETUAL',
]
assets = [s.split('_')[0].split('-')[0].lower() for s in instrument_names]
features = ['high', 'low', 'close'] # follow the standard order of the OHLC acronym O-H-L-C

train_prices, validation_prices, test_prices, n_train_periods, n_validation_periods, n_test_periods, all_datetimes = load_and_split_data(
    instrument_names, features, START_DATE_TRAIN, START_DATE_VALIDATION, START_DATE_TEST, END_DATE_TEST, RESOLUTION_MINUTES
)
print(f"(n_train_periods, n_validation_periods, n_test_periods) = {n_train_periods, n_validation_periods, n_test_periods}")

# %%
seed_everything(seed=42)

n_features, n_non_cash_assets, n_train_periods = train_prices.shape
learning_rate = 3e-5 #* np.sqrt(10)
weight_decay = 1e-8
n_epochs = 100
n_epochs_per_validation = 100
n_batches_per_epoch = 1000
geometric_parameter = 5e-5

n_available_periods = train_prices.shape[-1]
prices_array = train_prices

portfolio_vector_memory = np.ones((n_available_periods, n_non_cash_assets + 1)) / (n_non_cash_assets + 1)
policy = CNNPolicy(n_features=n_features, n_recent_periods=n_recent_periods).to(device)
optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate, weight_decay=weight_decay)

# %%

run_timestamp = datetime.now(tz=timezone.utc).strftime("%y%m%d_%H%M%S")
run_dir = f'./runs/{run_timestamp}'
checkpoint_dir = f'{run_dir}/checkpoints'

# Save run configuration
run_config = {
    'commission_rate': commission_rate,
    'n_recent_periods': n_recent_periods,
    'batch_size': batch_size,
    'n_online_batches': n_online_batches,
    'n_osbl_update_steps': n_osbl_update_steps,
    'learning_rate': learning_rate,
    'weight_decay': weight_decay,
    'n_epochs': n_epochs,
    'n_epochs_per_validation': n_epochs_per_validation,
    'n_batches_per_epoch': n_batches_per_epoch,
    'geometric_parameter': geometric_parameter,
    'n_features': n_features,
    'n_non_cash_assets': n_non_cash_assets,
    'n_train_periods': n_train_periods,
    'n_validation_periods': n_validation_periods,
    'n_test_periods': n_test_periods,
    'RESOLUTION_MINUTES': RESOLUTION_MINUTES,
    'START_DATE_TRAIN': START_DATE_TRAIN,
    'START_DATE_VALIDATION': START_DATE_VALIDATION,
    'START_DATE_TEST': START_DATE_TEST,
    'END_DATE_TEST': END_DATE_TEST,
    'instrument_names': instrument_names,
    'features': features,
}
save_run_config(run_config, run_dir)

save_checkpoint(policy, optimizer, 0, checkpoint_dir)
print("Initial checkpoint saved (epoch 0, untrained model)")

writer = SummaryWriter(log_dir=run_dir)
training_start_time = time.time()
for epoch in range(n_epochs):
    print(f"Epoch {epoch+1} / {n_epochs}")

    batch_start_indices = geometrically_sample_batch_start_indices(
        n_samples=n_batches_per_epoch,
        n_available_periods=n_available_periods,
        batch_size=batch_size,
        geometric_parameter=geometric_parameter,
        n_recent_periods=n_recent_periods
    )

    pvm_before_update = portfolio_vector_memory.copy()

    epoch_avg_log_return = run_one_epoch(
        prices_array=prices_array,
        batch_start_indices=batch_start_indices,
        portfolio_vector_memory=portfolio_vector_memory,
        policy=policy,
        optimizer=optimizer,
        n_recent_periods=n_recent_periods,
        batch_size=batch_size,
        device=device,
        commission_rate=commission_rate
    )

    n_pvm_updates = batch_size * n_batches_per_epoch
    avg_abs_pvm_change = np.sum(np.abs(portfolio_vector_memory - pvm_before_update)) / n_pvm_updates

    print(f"\tEpoch avg log-return: {epoch_avg_log_return:.9f}")
    writer.add_scalar('Train/AvgLogReturn', epoch_avg_log_return, epoch+1)
    writer.add_scalar('Train/Avg_Abs_PVM_Change', avg_abs_pvm_change, epoch+1)
    writer.add_scalar('Model_CashBias/Value', policy.cash_bias.item(), epoch+1)
    writer.add_scalar('Model_CashBias/Gradient', policy.cash_bias.grad, epoch+1)
    if epoch % int(n_epochs * 0.1) == 0:
        for name, param in policy.named_parameters():
            writer.add_histogram(f'Model_Parameters/{name}', param, epoch+1)
            writer.add_histogram(f'Model_Gradients/{name}', param.grad, epoch+1)

    if (epoch + 1) % n_epochs_per_validation == 0:
        print(f"\tRunning validation...")
        initial_portfolio = np.array([1] * (n_non_cash_assets + 1)) / (n_non_cash_assets + 1)
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
        writer.add_scalar('Validation/Final_Portfolio_Value_Multiplier', validation_metrics['fAPV'], epoch+1)
        writer.add_scalar('Validation/Annualized_Sharpe_Ratio', validation_metrics['SR'], epoch+1)
        writer.add_scalar('Validation/Maximum_Drawdown', validation_metrics['MDD'], epoch+1)
        print(f"\tVALIDATION RESULTS: fAPV={validation_metrics['fAPV']:.4f}, SR={validation_metrics['SR']:.4f}, MDD={validation_metrics['MDD']:.4f}")

        save_checkpoint(policy, optimizer, epoch+1, checkpoint_dir)
        print(f"\tCheckpoint saved at epoch {epoch+1}")

training_elapsed = time.time() - training_start_time
print(f"\nTraining completed in {training_elapsed/3600:.2f} hours ({training_elapsed/60:.1f} minutes)")
writer.close()

save_model(policy, optimizer, save_dir=run_dir, filename='final_model.pt', n_epochs=n_epochs, commission_rate=commission_rate, learning_rate=learning_rate, weight_decay=weight_decay, n_features=n_features, n_recent_periods=n_recent_periods)


# %%


initial_portfolio = np.array([1] * (n_non_cash_assets + 1)) / (n_non_cash_assets + 1)
initial_portfolio = np.insert(np.zeros(n_non_cash_assets), 0, 1)
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

# %%

df_results = validation_results

df_results
resolution_minutes = 30

# %%

initial_portfolio_value = 1
risk_free_return = 0

# Note that all metrics are measured BEFORE rebalancing occurs
step_log_returns = df_results['log_returns'].values
avg_log_return = np.mean(step_log_returns)
step_returns = np.exp(step_log_returns) - 1
step_portfolio_value_multipliers = np.exp(step_log_returns)
apv_ratios = np.cumprod(step_portfolio_value_multipliers)
assert np.sum(np.abs(apv_ratios - np.exp(np.cumsum(step_log_returns)))) < 1e-9, "Large deviation between equivalent calculations of apv ratios"

transaction_remainder_factors = df_results['transaction_remainder_factor'].values
portfolio_values_before_rebalancing = initial_portfolio_value * apv_ratios
transaction_costs = portfolio_values_before_rebalancing * (1 - transaction_remainder_factors) # in terms of the initial portfolio value
turnovers = transaction_costs / commission_rate

running_transaction_costs = np.cumsum(transaction_costs)
running_turnover = running_transaction_costs / commission_rate

running_max = np.maximum.accumulate(portfolio_values_before_rebalancing)
running_drawdown = (running_max - portfolio_values_before_rebalancing) / running_max
running_max_drawdown = np.maximum.accumulate(running_drawdown)


periodic_sharpe_ratio = np.mean(step_returns - risk_free_return) / (np.sqrt(np.var(step_returns - risk_free_return, ddof=1)) + 1e-12)
annualized_sharpe_ratio = np.sqrt(365 * 24 * 60 / resolution_minutes) * periodic_sharpe_ratio

{
    'fAPV': portfolio_values_before_rebalancing[-1],
    'SR': annualized_sharpe_ratio,
    'MDD': running_max_drawdown[-1],
    'avg_log_return': avg_log_return,
    'total_transaction_costs': running_transaction_costs[-1],
    'total_turnover': running_turnover[-1],
    'apv_ratios': apv_ratios,
    'running_drawdown': running_drawdown,
    'running_max_drawdown': running_max_drawdown,
    'step_log_returns': step_log_returns,
    'step_returns': step_returns,
    'running_transaction_costs': running_transaction_costs,
    'running_turnover': running_turnover
}