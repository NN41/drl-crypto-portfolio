import os
import json
import numpy as np
import torch
from datetime import datetime, timezone

def save_model(policy, optimizer, save_dir='./models', filename=None, **metadata):
    """Save model and optimizer state with training metadata."""
    os.makedirs(save_dir, exist_ok=True)
    if filename is None:
        run_timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename = f'cnn_policy_{run_timestamp}.pt'
    model_path = os.path.join(save_dir, filename)
    checkpoint = {'model_state_dict': policy.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}
    checkpoint.update(metadata)
    torch.save(checkpoint, model_path)
    print(f"Model saved to {model_path}")
    return model_path

def load_model(model_path, policy_class, device, default_learning_rate=1e-4, default_weight_decay=1e-8):
    """Load model and optimizer from checkpoint."""
    checkpoint = torch.load(model_path, map_location=device)
    policy = policy_class(n_features=checkpoint['n_features'], n_recent_periods=checkpoint['n_recent_periods']).to(device)
    policy.load_state_dict(checkpoint['model_state_dict'])
    learning_rate = checkpoint.get('learning_rate', default_learning_rate)
    weight_decay = checkpoint.get('weight_decay', default_weight_decay)
    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate, weight_decay=weight_decay)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print(f"Loaded model from {model_path}")
    print(f"Model trained for {checkpoint.get('n_epochs', 'unknown')} epochs with commission_rate={checkpoint.get('commission_rate', 'unknown')}")
    return policy, optimizer, checkpoint

def save_checkpoint(policy, optimizer, epoch, checkpoint_dir):
    """Save minimal checkpoint for resuming training."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')
    checkpoint = {
        'model_state_dict': policy.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'rng_state_numpy': np.random.get_state(),
        'rng_state_torch': torch.get_rng_state(),
        'rng_state_torch_cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")
    return checkpoint_path

def load_checkpoint(checkpoint_path, policy, optimizer, device):
    """Load checkpoint and restore training state (model, optimizer, RNG states)."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    policy.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    np.random.set_state(checkpoint['rng_state_numpy'])
    torch.set_rng_state(checkpoint['rng_state_torch'])
    if checkpoint['rng_state_torch_cuda'] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(checkpoint['rng_state_torch_cuda'])
    epoch = checkpoint['epoch']
    print(f"Loaded checkpoint from {checkpoint_path} (epoch {epoch})")
    return epoch

def save_run_config(config_dict, run_dir):
    """Save all hyperparameters to JSON for reference."""
    os.makedirs(run_dir, exist_ok=True)
    config_path = os.path.join(run_dir, 'run_config.json')
    config_serializable = {}
    for key, value in config_dict.items():
        if isinstance(value, (list, tuple)):
            config_serializable[key] = list(value)
        elif isinstance(value, datetime):
            config_serializable[key] = value.isoformat()
        elif isinstance(value, np.integer):
            config_serializable[key] = int(value)
        elif isinstance(value, np.floating):
            config_serializable[key] = float(value)
        else:
            config_serializable[key] = value
    with open(config_path, 'w') as f:
        json.dump(config_serializable, f, indent=2)
    print(f"Run config saved to {config_path}")