"""
Example training script for the two-stage latent ODE model.
"""

import os
import sys
import yaml
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models import TwoStageLatentODE
from train import Trainer, LatentODELoss


def load_config(config_path='config.yaml'):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def set_seed(seed):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def create_synthetic_data(num_samples=1000, obs_len=12, pred_len=36):
    """
    Create synthetic time series data for demonstration.
    
    In practice, replace this with real postprandial glucose data.
    
    Args:
        num_samples: Number of samples
        obs_len: Length of observation sequence
        pred_len: Length of prediction sequence
        
    Returns:
        dict: Dictionary containing data tensors
    """
    # Create synthetic glucose-like curves (postprandial response)
    # Peak glucose followed by decay
    
    time_obs = torch.linspace(0, 1, obs_len).repeat(num_samples, 1)
    time_pred = torch.linspace(1, 4, pred_len).repeat(num_samples, 1)
    
    # Parameters for synthetic curves
    baseline = 100 + torch.randn(num_samples, 1) * 10
    peak = 50 + torch.randn(num_samples, 1) * 15
    decay_rate = 0.5 + torch.randn(num_samples, 1) * 0.2
    
    # Generate observations (first part of curve)
    x_obs = baseline + peak * time_obs
    
    # Generate targets (continuation of curve with decay)
    t_combined = torch.cat([time_obs, time_pred], dim=1)
    targets_full = baseline + peak * torch.exp(-decay_rate * t_combined[:, obs_len:])
    
    # Add noise
    x_obs = x_obs + torch.randn_like(x_obs) * 2
    targets_full = targets_full + torch.randn_like(targets_full) * 2
    
    # Reshape for model input
    x_obs = x_obs.unsqueeze(-1)  # (num_samples, obs_len, 1)
    targets = targets_full.unsqueeze(-1)  # (num_samples, pred_len, 1)
    
    return {
        'x_obs': x_obs,
        't_obs': time_obs,
        't_pred': time_pred,
        'targets': targets
    }


def create_dataloaders(config):
    """
    Create train and validation dataloaders.
    
    In practice, load real data here.
    """
    # Create synthetic data
    num_train = 800
    num_val = 200
    
    obs_len = config['data']['obs_window'] // config['data']['sampling_freq']
    pred_len = config['data']['pred_window'] // config['data']['sampling_freq']
    
    train_data = create_synthetic_data(num_train, obs_len, pred_len)
    val_data = create_synthetic_data(num_val, obs_len, pred_len)
    
    # Create datasets
    train_dataset = TensorDataset(
        train_data['x_obs'],
        train_data['t_obs'],
        train_data['t_pred'],
        train_data['targets']
    )
    
    val_dataset = TensorDataset(
        val_data['x_obs'],
        val_data['t_obs'],
        val_data['t_pred'],
        val_data['targets']
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=0
    )
    
    # Wrap in dictionaries for trainer
    def collate_fn(batch):
        return {
            'x_obs': batch[0],
            't_obs': batch[1],
            't_pred': batch[2],
            'targets': batch[3]
        }
    
    train_loader.collate_fn = collate_fn
    val_loader.collate_fn = collate_fn
    
    # Manually collate
    train_loader_wrapped = []
    for batch in train_loader:
        train_loader_wrapped.append({
            'x_obs': batch[0],
            't_obs': batch[1],
            't_pred': batch[2],
            'targets': batch[3]
        })
    
    val_loader_wrapped = []
    for batch in val_loader:
        val_loader_wrapped.append({
            'x_obs': batch[0],
            't_obs': batch[1],
            't_pred': batch[2],
            'targets': batch[3]
        })
    
    return train_loader_wrapped, val_loader_wrapped


def main():
    """Main training function."""
    # Load configuration
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config.yaml')
    config = load_config(config_path)
    
    # Set random seed
    set_seed(config['seed'])
    
    # Set device
    device = config['device'] if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create model
    model = TwoStageLatentODE(
        input_dim=config['model']['input_dim'],
        latent_dim=config['model']['latent_dim'],
        output_dim=config['model']['output_dim'],
        encoder_hidden_dim=config['model']['encoder']['hidden_dim'],
        ode_hidden_dim=config['model']['ode']['hidden_dim'],
        decoder_hidden_dim=config['model']['decoder']['hidden_dim'],
        num_encoder_layers=config['model']['encoder']['num_layers'],
        num_ode_layers=config['model']['ode']['num_layers'],
        num_decoder_layers=config['model']['decoder']['num_layers'],
        rnn_type=config['model']['encoder']['rnn_type'],
        ode_method=config['model']['ode']['method'],
        ode_rtol=config['model']['ode']['rtol'],
        ode_atol=config['model']['ode']['atol']
    )
    
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create loss function
    loss_fn = LatentODELoss(
        kl_weight=config['training']['loss']['kl_weight'],
        recon_loss_type=config['training']['loss']['recon_loss_type']
    )
    
    # Create optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Create dataloaders
    print("Creating dataloaders...")
    train_loader, val_loader = create_dataloaders(config)
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=device,
        checkpoint_dir=config['paths']['checkpoint_dir']
    )
    
    # Train model
    print("\nStarting training...")
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config['training']['num_epochs']
    )
    
    print("\nTraining complete!")
    print(f"Best validation loss: {min(history['val_loss']):.4f}")


if __name__ == '__main__':
    main()
