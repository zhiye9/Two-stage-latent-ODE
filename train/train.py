"""
Trainer class for the two-stage latent ODE model.
"""

import torch
import torch.optim as optim
from tqdm import tqdm
import os


class Trainer:
    """
    Trainer for the two-stage latent ODE model.
    
    Args:
        model (nn.Module): The model to train
        loss_fn: Loss function
        optimizer: PyTorch optimizer
        device (str): Device to train on ('cuda' or 'cpu')
        checkpoint_dir (str): Directory to save checkpoints
    """
    
    def __init__(
        self,
        model,
        loss_fn,
        optimizer=None,
        device='cpu',
        checkpoint_dir='checkpoints'
    ):
        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        
        # Default optimizer if not provided
        if optimizer is None:
            self.optimizer = optim.Adam(model.parameters(), lr=1e-3)
        else:
            self.optimizer = optimizer
        
        # Create checkpoint directory
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'recon_loss': [],
            'kl_loss': []
        }
    
    def train_epoch(self, train_loader):
        """
        Train for one epoch.
        
        Args:
            train_loader: DataLoader for training data
            
        Returns:
            dict: Average losses for the epoch
        """
        self.model.train()
        
        epoch_loss = 0.0
        epoch_recon = 0.0
        epoch_kl = 0.0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc='Training')
        for batch in pbar:
            # Move data to device
            x_obs = batch['x_obs'].to(self.device)
            t_obs = batch['t_obs'].to(self.device)
            t_pred = batch['t_pred'].to(self.device)
            targets = batch['targets'].to(self.device)
            
            # Optional mask
            mask = batch.get('mask', None)
            if mask is not None:
                mask = mask.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(x_obs, t_obs, t_pred)
            
            # Compute loss
            loss_dict = self.loss_fn(
                outputs['predictions'],
                targets,
                outputs['latent_mean'],
                outputs['latent_logvar'],
                mask
            )
            
            # Backward pass
            loss_dict['loss'].backward()
            self.optimizer.step()
            
            # Accumulate losses
            epoch_loss += loss_dict['loss'].item()
            epoch_recon += loss_dict['recon_loss'].item()
            epoch_kl += loss_dict['kl_loss'].item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss_dict['loss'].item():.4f}",
                'recon': f"{loss_dict['recon_loss'].item():.4f}",
                'kl': f"{loss_dict['kl_loss'].item():.4f}"
            })
        
        return {
            'train_loss': epoch_loss / num_batches,
            'recon_loss': epoch_recon / num_batches,
            'kl_loss': epoch_kl / num_batches
        }
    
    def validate(self, val_loader):
        """
        Validate the model.
        
        Args:
            val_loader: DataLoader for validation data
            
        Returns:
            dict: Average validation losses
        """
        self.model.eval()
        
        epoch_loss = 0.0
        epoch_recon = 0.0
        epoch_kl = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Validation'):
                # Move data to device
                x_obs = batch['x_obs'].to(self.device)
                t_obs = batch['t_obs'].to(self.device)
                t_pred = batch['t_pred'].to(self.device)
                targets = batch['targets'].to(self.device)
                
                mask = batch.get('mask', None)
                if mask is not None:
                    mask = mask.to(self.device)
                
                # Forward pass
                outputs = self.model(x_obs, t_obs, t_pred)
                
                # Compute loss
                loss_dict = self.loss_fn(
                    outputs['predictions'],
                    targets,
                    outputs['latent_mean'],
                    outputs['latent_logvar'],
                    mask
                )
                
                epoch_loss += loss_dict['loss'].item()
                epoch_recon += loss_dict['recon_loss'].item()
                epoch_kl += loss_dict['kl_loss'].item()
                num_batches += 1
        
        return {
            'val_loss': epoch_loss / num_batches,
            'recon_loss': epoch_recon / num_batches,
            'kl_loss': epoch_kl / num_batches
        }
    
    def train(self, train_loader, val_loader=None, num_epochs=100):
        """
        Train the model for multiple epochs.
        
        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data (optional)
            num_epochs (int): Number of epochs to train
            
        Returns:
            dict: Training history
        """
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            
            # Train
            train_metrics = self.train_epoch(train_loader)
            
            # Log training metrics
            self.history['train_loss'].append(train_metrics['train_loss'])
            self.history['recon_loss'].append(train_metrics['recon_loss'])
            self.history['kl_loss'].append(train_metrics['kl_loss'])
            
            print(f"Train Loss: {train_metrics['train_loss']:.4f}, "
                  f"Recon: {train_metrics['recon_loss']:.4f}, "
                  f"KL: {train_metrics['kl_loss']:.4f}")
            
            # Validate
            if val_loader is not None:
                val_metrics = self.validate(val_loader)
                self.history['val_loss'].append(val_metrics['val_loss'])
                
                print(f"Val Loss: {val_metrics['val_loss']:.4f}")
                
                # Save best model
                if val_metrics['val_loss'] < best_val_loss:
                    best_val_loss = val_metrics['val_loss']
                    self.save_checkpoint('best_model.pth')
                    print("Saved best model!")
            
            # Save periodic checkpoint
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch + 1}.pth')
        
        return self.history
    
    def save_checkpoint(self, filename):
        """Save model checkpoint."""
        filepath = os.path.join(self.checkpoint_dir, filename)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history
        }, filepath)
    
    def load_checkpoint(self, filename):
        """Load model checkpoint."""
        filepath = os.path.join(self.checkpoint_dir, filename)
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint.get('history', self.history)
