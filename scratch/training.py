# %%

import numpy as np
from datetime import datetime, timezone
import time

import torch
from torch.utils.tensorboard import SummaryWriter

from src.policies import CNNPolicy
from src.train_utils import geometrically_sample_batch_start_indices, run_one_epoch, seed_everything
from src.evaluation import run_walk_forward, calculate_performance_metrics, WalkForwardConfig
from src.model_io import save_model, save_checkpoint, save_run_config
from src.data_loading import load_and_split_data


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f"Using device {device}")

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

commission_rate = 0.0005 # 0.0005 = 5 bips
batch_size = 500

geometric_parameter = 5e-5
n_online_batches = 30
n_osbl_update_steps = 30
n_recent_periods = 50 # number of periods passed to the policy to choose a portfolio

# periods_per_epoch = 1e4 * 5
# periods_per_validation = periods_per_epoch * 100 / 2
# paper_total_steps = 2e6 * 10
periods_per_epoch = int(1e4)
periods_per_validation = int(periods_per_epoch * 100)
paper_total_steps = int(2e6)

paper_batch_size = 50
total_periods = int(paper_total_steps * paper_batch_size)
n_epochs = int(total_periods / periods_per_epoch)
n_validations = int(total_periods / periods_per_validation)
n_epochs_per_validation = int(n_epochs / n_validations)
n_batches_per_epoch = int(periods_per_epoch / batch_size)

paper_learing_rate = 3e-5
learning_rate_scaling_factor = np.sqrt(batch_size / paper_batch_size)
learning_rate = paper_learing_rate * learning_rate_scaling_factor
weight_decay = 1e-8

print(f"paper_total_steps={paper_total_steps:.0e}, paper_batch_size={paper_batch_size}, total_periods={total_periods:.0e}")
print(f"periods_per_epoch={periods_per_epoch:.0e}, periods_per_validation={periods_per_validation:.0e}")
print(f"learning_rate={learning_rate:.1e} (learning_rate_scaling_factor={learning_rate_scaling_factor:.2f})")
print(f"n_epochs={n_epochs}, n_validations={n_validations}, n_epochs_per_validation={n_epochs_per_validation}, n_batches_per_epoch={n_batches_per_epoch}")

# %%

n_available_periods = train_prices.shape[-1]

# Convert to GPU tensors for vectorized_tensor_mode training
prices_array = torch.tensor(train_prices, device=device, dtype=torch.float32)
portfolio_vector_memory = torch.ones((n_available_periods, n_non_cash_assets + 1), device=device, dtype=torch.float32) / (n_non_cash_assets + 1)
policy = CNNPolicy(n_features=n_features, n_recent_periods=n_recent_periods).to(device)
optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate, weight_decay=weight_decay)

wf_config = WalkForwardConfig(
    n_recent_periods=n_recent_periods,
    commission_rate=commission_rate,
    device=device,
    assets=assets,
    n_osbl_update_steps=n_osbl_update_steps,
    osbl_batch_size=batch_size,
    geometric_parameter=geometric_parameter,
)

run_timestamp = datetime.now(tz=timezone.utc).strftime("%y%m%d_%H%M%S")
run_dir = f'./runs/{run_timestamp}'
checkpoint_dir = f'{run_dir}/checkpoints'

# Save run configuration
run_config = {
    'n_epochs': n_epochs,
    'n_batches_per_epoch': n_batches_per_epoch,
    'batch_size': batch_size,
    'learning_rate': learning_rate,
    'weight_decay': weight_decay,
    'n_recent_periods': n_recent_periods,
    'geometric_parameter': geometric_parameter,
    'n_online_batches': n_online_batches,
    'n_osbl_update_steps': n_osbl_update_steps,
    'RESOLUTION_MINUTES': RESOLUTION_MINUTES,
    'START_DATE_TRAIN': START_DATE_TRAIN,
    'START_DATE_VALIDATION': START_DATE_VALIDATION,
    'START_DATE_TEST': START_DATE_TEST,
    'END_DATE_TEST': END_DATE_TEST,
    'n_train_periods': n_train_periods,
    'n_validation_periods': n_validation_periods,
    'n_test_periods': n_test_periods,
    'instrument_names': instrument_names,
    'n_non_cash_assets': n_non_cash_assets,
    'features': features,
    'n_features': n_features,
    'commission_rate': commission_rate,
    'n_epochs_per_validation': n_epochs_per_validation,
    'paper_total_steps': paper_total_steps,
    'paper_batch_size': paper_batch_size,
    'total_periods': total_periods,
    'periods_per_epoch': periods_per_epoch,
    'periods_per_validation': periods_per_validation,
    'paper_learing_rate': paper_learing_rate,
    'learning_rate_scaling_factor': learning_rate_scaling_factor,
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

    pvm_before_update = portfolio_vector_memory.clone()

    epoch_avg_log_return = run_one_epoch(
        prices_array=prices_array,
        batch_start_indices=batch_start_indices,
        portfolio_vector_memory=portfolio_vector_memory,
        policy=policy,
        optimizer=optimizer,
        n_recent_periods=n_recent_periods,
        batch_size=batch_size,
        device=device,
        commission_rate=commission_rate,
        vectorized_tensor_mode=True
    )

    n_pvm_updates = batch_size * n_batches_per_epoch
    avg_abs_pvm_change = torch.sum(torch.abs(portfolio_vector_memory - pvm_before_update)).item() / n_pvm_updates

    print(f"\tEpoch avg log-return: {epoch_avg_log_return:.9f}")
    writer.add_scalar('Train/AvgLogReturn', epoch_avg_log_return, epoch+1)
    writer.add_scalar('Train/Avg_Abs_PVM_Change', avg_abs_pvm_change, epoch+1)
    writer.add_scalar('Model_CashBias/Value', policy.cash_bias.item(), epoch+1)
    writer.add_scalar('Model_CashBias/Gradient', policy.cash_bias.grad, epoch+1)

    # Compute L2 norms for each parameter and gradient individually
    total_param_l2 = sum(torch.norm(p).item()**2 for p in policy.parameters())**0.5
    total_grad_l2 = sum(torch.norm(p.grad).item()**2 for p in policy.parameters() if p.grad is not None)**0.5
    writer.add_scalar('Model_Param_Norms/_Total_L2', total_param_l2, epoch+1)
    writer.add_scalar('Model_Grad_Norms/_Total_L2', total_grad_l2, epoch+1)
    for name, param in policy.named_parameters():
        writer.add_scalar(f'Model_Param_Norms/{name}_L2', torch.norm(param).item(), epoch+1)
        if param.grad is not None:
            writer.add_scalar(f'Model_Grad_Norms/{name}_L2', torch.norm(param.grad).item(), epoch+1)

    if (epoch + 1) % n_epochs_per_validation == 0:
        print(f"\tRunning validation...")
        initial_portfolio = np.array([1] * (n_non_cash_assets + 1)) / (n_non_cash_assets + 1)
        validation_results = run_walk_forward(
            policy=policy,
            initial_weights=initial_portfolio,
            seen_prices=train_prices,
            unseen_prices=validation_prices,
            all_datetimes=all_datetimes,
            use_osbl=False,
            config=wf_config,
        )
        validation_metrics = calculate_performance_metrics(validation_results, RESOLUTION_MINUTES, commission_rate)
        writer.add_scalar('Validation/Final_Portfolio_Value_Multiplier', validation_metrics['fAPV'], epoch+1)
        writer.add_scalar('Validation/Annualized_Sharpe_Ratio', validation_metrics['SR'], epoch+1)
        writer.add_scalar('Validation/Maximum_Drawdown', validation_metrics['MDD'], epoch+1)
        writer.add_scalar('Validation/Total_Transaction_Costs', validation_metrics['total_transaction_costs'], epoch+1)
        writer.add_scalar('Validation/Total_Turnover', validation_metrics['total_turnover'], epoch+1)
        writer.add_scalar('Validation_Detailed/Avg_Log_Return', validation_metrics['avg_log_return'], epoch+1)
        writer.add_scalar('Validation_Detailed/Avg_Normalized_Entropy', validation_metrics['avg_normalized_entropy'], epoch+1)
        writer.add_scalar('Validation_Detailed/Avg_Transaction_Cost', validation_metrics['avg_transaction_cost'], epoch+1)
        writer.add_scalar('Validation_Detailed/Avg_Relative_Turnover', validation_metrics['avg_relative_turnover'], epoch+1)
        writer.add_scalar('Validation_Detailed/Avg_Cash_Weight', validation_metrics['avg_cash_weight'], epoch+1)
        # for name, param in policy.named_parameters():
        #     writer.add_histogram(f'Model_Parameters/{name}', param, epoch+1)
        #     writer.add_histogram(f'Model_Gradients/{name}', param.grad, epoch+1)

        for asset in assets:
            mean_weight = validation_results[f'weight_{asset}'].mean()
            writer.add_scalar(f'Validation_Avg_Portfolio_Weights/{asset}', mean_weight, epoch+1)
        writer.add_scalar('Validation_Avg_Portfolio_Weights/_cash', validation_results['weight_cash'].mean(), epoch+1)

        save_checkpoint(policy, optimizer, epoch+1, checkpoint_dir)
        print(f"\tCheckpoint saved at epoch {epoch+1}")

training_elapsed = time.time() - training_start_time
print(f"\nTraining completed in {training_elapsed/3600:.2f} hours ({training_elapsed/60:.1f} minutes)")
writer.close()

save_model(policy, optimizer, save_dir=run_dir, filename='final_model.pt', n_epochs=n_epochs, commission_rate=commission_rate, learning_rate=learning_rate, weight_decay=weight_decay, n_features=n_features, n_recent_periods=n_recent_periods)


# %%
