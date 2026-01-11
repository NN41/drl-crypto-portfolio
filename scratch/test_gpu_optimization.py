# %%

import sys
sys.path.insert(0, '.')

import numpy as np
import torch
import time
from datetime import datetime, timezone

from src.policies import CNNPolicy
from src.train_utils import geometrically_sample_batch_start_indices, run_one_epoch, prepare_batch_of_consecutive_periods, prepare_batch_gpu
from src.evaluation import run_walk_forward_test, calculate_performance_metrics
from src.data_loading import load_and_split_data

device = torch.device('cuda')
commission_rate = 0.0005
n_recent_periods = 50
batch_size = 500
n_batches_per_epoch = 200
geometric_parameter = 5e-5

RESOLUTION_MINUTES = 30
START_DATE_TRAIN = datetime(2022, 4, 28, 0, 0, 0, tzinfo=timezone.utc)
START_DATE_VALIDATION = datetime(2025, 7, 17, 0, 0, 0, tzinfo=timezone.utc)
START_DATE_TEST = datetime(2025, 10, 9, 0, 0, 0, tzinfo=timezone.utc)
END_DATE_TEST = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
instrument_names = ['BTC-PERPETUAL', 'ETH-PERPETUAL', 'SOL_USDC-PERPETUAL']
assets = [s.split('_')[0].split('-')[0].lower() for s in instrument_names]
features = ['high', 'low', 'close']

print("Loading data...")
train_prices, validation_prices, test_prices, n_train, n_val, n_test, all_datetimes = load_and_split_data(
    instrument_names, features, START_DATE_TRAIN, START_DATE_VALIDATION, START_DATE_TEST, END_DATE_TEST, RESOLUTION_MINUTES
)
n_features, n_non_cash_assets, n_available_periods = train_prices.shape
print(f"Data loaded: {n_features} features, {n_non_cash_assets} assets, {n_available_periods} periods")

# TEST 1: prepare_batch_gpu produces same outputs as original
print("\n=== TEST 1: prepare_batch_gpu equivalence ===")
batch_start_idx = 1000

pvm_np = np.ones((n_available_periods, n_non_cash_assets + 1)) / (n_non_cash_assets + 1)
batch_cpu = prepare_batch_of_consecutive_periods(train_prices, pvm_np, batch_start_idx, batch_size, n_recent_periods)

prices_gpu = torch.tensor(train_prices, device=device, dtype=torch.float32)
pvm_gpu = torch.ones((n_available_periods, n_non_cash_assets + 1), device=device, dtype=torch.float32) / (n_non_cash_assets + 1)
batch_gpu = prepare_batch_gpu(prices_gpu, pvm_gpu, batch_start_idx, batch_size, n_recent_periods)

for key in ['normalized_price_histories', 'previous_weights', 'next_price_relatives', 'current_price_relatives']:
    cpu_val = batch_cpu[key]
    gpu_val = batch_gpu[key].cpu().numpy()
    max_diff = np.abs(cpu_val - gpu_val).max()
    print(f"  {key}: max_diff = {max_diff:.2e}")
    assert max_diff < 1e-5, f"Mismatch in {key}"
print("TEST 1 PASSED")

# TEST 2: run_one_epoch with gpu_mode=True produces similar loss
print("\n=== TEST 2: run_one_epoch gpu_mode equivalence ===")
np.random.seed(42)
torch.manual_seed(42)

batch_start_indices = geometrically_sample_batch_start_indices(n_batches_per_epoch, n_available_periods, batch_size, geometric_parameter, n_recent_periods)

policy_cpu = CNNPolicy(n_features=n_features, n_recent_periods=n_recent_periods).to(device)
optimizer_cpu = torch.optim.Adam(policy_cpu.parameters(), lr=3e-5)
pvm_cpu = np.ones((n_available_periods, n_non_cash_assets + 1)) / (n_non_cash_assets + 1)

loss_cpu = run_one_epoch(
    prices_array=train_prices, batch_start_indices=batch_start_indices, portfolio_vector_memory=pvm_cpu,
    policy=policy_cpu, optimizer=optimizer_cpu, n_recent_periods=n_recent_periods, batch_size=batch_size,
    device=device, commission_rate=commission_rate, gpu_mode=False
)

np.random.seed(42)
torch.manual_seed(42)

policy_gpu = CNNPolicy(n_features=n_features, n_recent_periods=n_recent_periods).to(device)
optimizer_gpu = torch.optim.Adam(policy_gpu.parameters(), lr=3e-5)
prices_tensor = torch.tensor(train_prices, device=device, dtype=torch.float32)
pvm_gpu = torch.ones((n_available_periods, n_non_cash_assets + 1), device=device, dtype=torch.float32) / (n_non_cash_assets + 1)

loss_gpu = run_one_epoch(
    prices_array=prices_tensor, batch_start_indices=batch_start_indices, portfolio_vector_memory=pvm_gpu,
    policy=policy_gpu, optimizer=optimizer_gpu, n_recent_periods=n_recent_periods, batch_size=batch_size,
    device=device, commission_rate=commission_rate, gpu_mode=True
)

print(f"  CPU mode loss: {loss_cpu:.9f}")
print(f"  GPU mode loss: {loss_gpu:.9f}")
print(f"  Difference: {abs(loss_cpu - loss_gpu):.2e}")
assert abs(loss_cpu - loss_gpu) < 1e-4, "Loss values differ too much"
print("TEST 2 PASSED")

# TEST 3: walk_forward_test without OSBL still works
print("\n=== TEST 3: walk_forward_test without OSBL ===")
policy = CNNPolicy(n_features=n_features, n_recent_periods=n_recent_periods).to(device)
initial_portfolio = np.array([1] * (n_non_cash_assets + 1)) / (n_non_cash_assets + 1)
small_validation = validation_prices[:, :, :100]

results = run_walk_forward_test(
    policy=policy, initial_portfolio_weights=initial_portfolio, initial_prices=train_prices,
    forward_prices=small_validation, all_datetimes=all_datetimes, assets=assets,
    n_recent_periods=n_recent_periods, commission_rate=commission_rate, device=device,
    use_osbl=False, n_osbl_update_steps=None, optimizer=None
)
metrics = calculate_performance_metrics(results, RESOLUTION_MINUTES, commission_rate)
print(f"  fAPV: {metrics['fAPV']:.4f}, SR: {metrics['SR']:.4f}")
assert len(results) == 100
print("TEST 3 PASSED")

# TEST 4: walk_forward_test with OSBL still works
print("\n=== TEST 4: walk_forward_test with OSBL ===")
policy = CNNPolicy(n_features=n_features, n_recent_periods=n_recent_periods).to(device)
optimizer = torch.optim.Adam(policy.parameters(), lr=3e-5)
initial_portfolio = np.array([1] * (n_non_cash_assets + 1)) / (n_non_cash_assets + 1)
small_validation = validation_prices[:, :, :10]

results = run_walk_forward_test(
    policy=policy, initial_portfolio_weights=initial_portfolio, initial_prices=train_prices,
    forward_prices=small_validation, all_datetimes=all_datetimes, assets=assets,
    n_recent_periods=n_recent_periods, commission_rate=commission_rate, device=device,
    use_osbl=True, n_osbl_update_steps=5, optimizer=optimizer
)
assert 'osbl_avg_log_return' in results.columns
print(f"  OSBL ran {len(results)} steps")
print("TEST 4 PASSED")
# %%
# TEST 5: Timing comparison
print("\n=== TEST 5: Timing comparison ===")
n_epochs_test = 100

np.random.seed(42)
torch.manual_seed(42)
policy = CNNPolicy(n_features=n_features, n_recent_periods=n_recent_periods).to(device)
optimizer = torch.optim.Adam(policy.parameters(), lr=3e-5)
pvm = np.ones((n_available_periods, n_non_cash_assets + 1)) / (n_non_cash_assets + 1)

torch.cuda.synchronize()
t0 = time.time()
for _ in range(n_epochs_test):
    print(_)
    batch_indices = geometrically_sample_batch_start_indices(n_batches_per_epoch, n_available_periods, batch_size, geometric_parameter, n_recent_periods)
    run_one_epoch(train_prices, batch_indices, pvm, policy, optimizer, n_recent_periods, batch_size, device, commission_rate, gpu_mode=False)
torch.cuda.synchronize()
cpu_time = time.time() - t0

np.random.seed(42)
torch.manual_seed(42)
policy = CNNPolicy(n_features=n_features, n_recent_periods=n_recent_periods).to(device)
optimizer = torch.optim.Adam(policy.parameters(), lr=3e-5)
prices_tensor = torch.tensor(train_prices, device=device, dtype=torch.float32)
pvm_gpu = torch.ones((n_available_periods, n_non_cash_assets + 1), device=device, dtype=torch.float32) / (n_non_cash_assets + 1)

torch.cuda.synchronize()
t0 = time.time()
for _ in range(n_epochs_test):
    print(_)
    batch_indices = geometrically_sample_batch_start_indices(n_batches_per_epoch, n_available_periods, batch_size, geometric_parameter, n_recent_periods)
    run_one_epoch(prices_tensor, batch_indices, pvm_gpu, policy, optimizer, n_recent_periods, batch_size, device, commission_rate, gpu_mode=True)
torch.cuda.synchronize()
gpu_time = time.time() - t0

speedup = cpu_time / gpu_time
print(f"  CPU mode: {cpu_time:.2f}s for {n_epochs_test} epochs")
print(f"  GPU mode: {gpu_time:.2f}s for {n_epochs_test} epochs")
print(f"  Speedup: {speedup:.2f}x")
print("TEST 5 PASSED")

print("\n=== ALL TESTS PASSED ===")

# %%
