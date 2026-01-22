# %%
# Post-training walk-forward evaluation script.
#
# Run order to generate full dataset for analysis_v2.py:
#   1. data_set='train'      - Generates pretrained and ucrp CSVs on training data (investigates what model learned from training)
#   2. data_set='validation' - Generates pretrained, osbl, ucrp CSVs + saves OSBL model for test set
#   3. data_set='test'       - Generates pretrained, osbl, ucrp CSVs using validation OSBL model
#
# This workflow provides a complete 360-degree view of policy performance across train/validation/test sets,
# which can be loaded in analysis_v2.py for comprehensive comparison to UCRP and other benchmarks.
#
# Output: CSV files in {run_dir}/post_training/epoch_{epoch}/

import json
import os
import numpy as np
from datetime import datetime, timezone
import torch

from src.policies import CNNPolicy, EqualWeightPolicy
from src.data_loading import load_and_split_data
from src.evaluation import run_walk_forward, calculate_performance_metrics, WalkForwardConfig
from src.model_io import load_checkpoint, save_model
from src.train_utils import seed_everything


# %%

run_dir = './runs_completed/runs_fees/260111_fees_5_bips_baseline_v0'
checkpoint_filename = 'checkpoint_epoch_10000.pt'
data_set = 'test'  # 'train', 'validation', or 'test'

# %%

seed_everything(seed=42)

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

train_prices, validation_prices, test_prices, n_train_periods, n_validation_periods, n_test_periods, all_datetimes, train_availability_mask, validation_availability_mask, test_availability_mask = load_and_split_data(
    instrument_names, features, START_DATE_TRAIN, START_DATE_VALIDATION, START_DATE_TEST, END_DATE_TEST, RESOLUTION_MINUTES
)
print(f"(n_train_periods, n_validation_periods, n_test_periods) = {n_train_periods, n_validation_periods, n_test_periods}")

policy = CNNPolicy(n_features=n_features, n_recent_periods=n_recent_periods).to(device)
optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate, weight_decay=weight_decay)

checkpoint_path = f'{run_dir}/checkpoints/{checkpoint_filename}'
epoch = load_checkpoint(checkpoint_path, policy, optimizer, device)
print(f"Loaded checkpoint from epoch {epoch}")

if data_set == 'test':
    seen_prices = np.concatenate([train_prices, validation_prices], axis=-1)
    unseen_prices = test_prices
    availability_mask = np.concatenate([train_availability_mask, validation_availability_mask, test_availability_mask], axis=0)
    validation_osbl_model_path = f'{run_dir}/post_training/epoch_{epoch}/validation_osbl_model.pt'
    if not os.path.exists(validation_osbl_model_path):
        raise FileNotFoundError(
            f"Validation OSBL model not found. "
            f"Run with data_set='validation' first to generate it."
        )
elif data_set == 'validation':
    seen_prices = train_prices
    unseen_prices = validation_prices
    availability_mask = np.concatenate([train_availability_mask, validation_availability_mask], axis=0)
elif data_set == 'train':
    seen_prices = train_prices[:, :, :n_recent_periods]
    unseen_prices = train_prices[:, :, n_recent_periods:]
    availability_mask = train_availability_mask
else:
    raise ValueError(f"Invalid data_set: {data_set}. Must be 'train', 'validation', or 'test'.")

# Models trained before 2026-01-20 didn't use availability masking - use all-True mask for backward compatibility
run_dir_name = os.path.basename(run_dir.rstrip('/'))
run_timestamp_str = run_dir_name[:6]  # yymmdd format
run_date = datetime.strptime(run_timestamp_str, '%y%m%d').replace(tzinfo=timezone.utc)
masking_cutoff_date = datetime(2026, 1, 20, tzinfo=timezone.utc)
if run_date < masking_cutoff_date:
    print(f"Model trained on {run_date.date()} (before {masking_cutoff_date.date()}) - using all-True availability mask")
    availability_mask = np.ones_like(availability_mask, dtype=bool)

print(f"Dataset: {data_set}, seen_prices: {seen_prices.shape}, unseen_prices: {unseen_prices.shape}")

paper_rolling_steps = 30
rolling_steps = max(1, int(paper_rolling_steps * config['paper_batch_size'] / config['batch_size']))  # to match the number of periods sampled per time step

wf_config = WalkForwardConfig(
    n_recent_periods=n_recent_periods,
    commission_rate=commission_rate,
    device=device,
    assets=assets,
    n_osbl_update_steps=rolling_steps,
    osbl_batch_size=config['batch_size'],
    geometric_parameter=config['geometric_parameter'],
)
initial_portfolio_weights = np.array([1.] + [0.] * n_non_cash_assets)


print("\nRunning walk-forward without OSBL...")
pretrained_results = run_walk_forward(
    policy=policy,
    initial_weights=initial_portfolio_weights,
    seen_prices=seen_prices.copy(),
    unseen_prices=unseen_prices.copy(),
    all_datetimes=all_datetimes,
    use_osbl=False,
    config=wf_config,
    availability_mask=availability_mask,
    verbose=True,
)
pretrained_metrics = calculate_performance_metrics(pretrained_results, RESOLUTION_MINUTES, commission_rate)
print(f"No OSBL: fAPV={pretrained_metrics['fAPV']:.4f}, SR={pretrained_metrics['SR']:.4f}, MDD={pretrained_metrics['MDD']:.4f}")


# OSBL run (skip for train set - only need pretrained and ucrp results)
if data_set != 'train':
    policy_osbl = CNNPolicy(n_features=n_features, n_recent_periods=n_recent_periods).to(device)
    optimizer_osbl = torch.optim.Adam(policy_osbl.parameters(), lr=learning_rate, weight_decay=weight_decay)

    if data_set == 'test':
        load_checkpoint(validation_osbl_model_path, policy_osbl, optimizer_osbl, device)
        print(f"Loaded validation OSBL model from {validation_osbl_model_path}")
    else:
        load_checkpoint(checkpoint_path, policy_osbl, optimizer_osbl, device)
        print(f"Loaded checkpoint from {checkpoint_path}")

    print("\nRunning walk-forward with OSBL...")
    osbl_results = run_walk_forward(
        policy=policy_osbl,
        initial_weights=initial_portfolio_weights,
        seen_prices=seen_prices.copy(),
        unseen_prices=unseen_prices.copy(),
        all_datetimes=all_datetimes,
        use_osbl=True,
        config=wf_config,
        availability_mask=availability_mask,
        optimizer=optimizer_osbl,
        verbose=True,
    )
    osbl_metrics = calculate_performance_metrics(osbl_results, RESOLUTION_MINUTES, commission_rate)
    print(f"OSBL: fAPV={osbl_metrics['fAPV']:.4f}, SR={osbl_metrics['SR']:.4f}, MDD={osbl_metrics['MDD']:.4f}")


print("\nRunning UCRP walk-forward...")
ucrp_policy = EqualWeightPolicy(n_non_cash_assets)
ucrp_results = run_walk_forward(
    policy=ucrp_policy,
    initial_weights=initial_portfolio_weights,
    seen_prices=seen_prices,
    unseen_prices=unseen_prices,
    all_datetimes=all_datetimes,
    use_osbl=False,
    config=wf_config,
    availability_mask=availability_mask,
    verbose=True,
)
ucrp_metrics = calculate_performance_metrics(ucrp_results, RESOLUTION_MINUTES, commission_rate)
print(f"UCRP: fAPV={ucrp_metrics['fAPV']:.4f}, SR={ucrp_metrics['SR']:.4f}, MDD={ucrp_metrics['MDD']:.4f}")


print("\n=== Results Summary ===")
print(f"Data set: {data_set}")
print(f"{'Strategy':<15} {'fAPV':>10} {'SR':>10} {'MDD':>10}")
print("-" * 47)
print(f"{'No OSBL':<15} {pretrained_metrics['fAPV']:>10.4f} {pretrained_metrics['SR']:>10.4f} {pretrained_metrics['MDD']:>10.4f}")
if data_set != 'train':
    print(f"{'OSBL':<15} {osbl_metrics['fAPV']:>10.4f} {osbl_metrics['SR']:>10.4f} {osbl_metrics['MDD']:>10.4f}")
print(f"{'UCRP':<15} {ucrp_metrics['fAPV']:>10.4f} {ucrp_metrics['SR']:>10.4f} {ucrp_metrics['MDD']:>10.4f}")

output_dir = f'{run_dir}/post_training/epoch_{epoch}'
os.makedirs(output_dir, exist_ok=True)

pretrained_results.to_csv(f'{output_dir}/{data_set}_pretrained_results.csv', index=False)
ucrp_results.to_csv(f'{output_dir}/{data_set}_ucrp_results.csv', index=False)

if data_set != 'train':
    osbl_results.to_csv(f'{output_dir}/{data_set}_osbl_results.csv', index=False)
    save_model(
        policy_osbl,
        optimizer_osbl,
        save_dir=output_dir,
        filename=f'{data_set}_osbl_model.pt',
        epoch=epoch,
        commission_rate=commission_rate,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        n_features=n_features,
        n_recent_periods=n_recent_periods
    )

print(f"\nResults saved to {output_dir}/")
print(f"  - {data_set}_pretrained_results.csv")
if data_set != 'train':
    print(f"  - {data_set}_osbl_results.csv")
    print(f"  - {data_set}_osbl_model.pt")
print(f"  - {data_set}_ucrp_results.csv")

# %%
