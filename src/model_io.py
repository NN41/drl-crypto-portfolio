import os
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