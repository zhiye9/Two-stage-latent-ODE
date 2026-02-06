"""
Loss functions for training the two-stage latent ODE model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def reconstruction_loss(predictions, targets, mask=None, loss_type='mse'):
    """
    Compute reconstruction loss between predictions and targets.
    
    Args:
        predictions (torch.Tensor): Model predictions
        targets (torch.Tensor): Ground truth values
        mask (torch.Tensor, optional): Mask for missing/irregular observations
        loss_type (str): Type of loss ('mse' or 'mae')
        
    Returns:
        torch.Tensor: Scalar loss value
    """
    if loss_type == 'mse':
        loss = F.mse_loss(predictions, targets, reduction='none')
    elif loss_type == 'mae':
        loss = F.l1_loss(predictions, targets, reduction='none')
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
    
    if mask is not None:
        loss = loss * mask
        return loss.sum() / (mask.sum() + 1e-8)
    else:
        return loss.mean()


def kl_divergence(mean, logvar):
    """
    Compute KL divergence between latent distribution and standard normal.
    KL(q(z|x) || p(z)) where p(z) = N(0, I)
    
    Args:
        mean (torch.Tensor): Mean of latent distribution, shape (batch_size, latent_dim)
        logvar (torch.Tensor): Log variance of latent distribution
        
    Returns:
        torch.Tensor: Scalar KL divergence
    """
    # KL(N(μ, σ²) || N(0, 1)) = -0.5 * Σ(1 + log(σ²) - μ² - σ²)
    kl = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
    # Normalize by batch size and latent dimension
    kl = kl / (mean.size(0) * mean.size(1))
    return kl


class LatentODELoss(nn.Module):
    """
    Combined loss for training the two-stage latent ODE model.
    Combines reconstruction loss and KL divergence for VAE-style training.
    
    Args:
        kl_weight (float): Weight for KL divergence term
        recon_loss_type (str): Type of reconstruction loss ('mse' or 'mae')
    """
    
    def __init__(self, kl_weight=1.0, recon_loss_type='mse'):
        super(LatentODELoss, self).__init__()
        self.kl_weight = kl_weight
        self.recon_loss_type = recon_loss_type
    
    def forward(self, predictions, targets, latent_mean, latent_logvar, mask=None):
        """
        Compute total loss.
        
        Args:
            predictions (torch.Tensor): Model predictions
            targets (torch.Tensor): Ground truth values
            latent_mean (torch.Tensor): Mean of latent distribution
            latent_logvar (torch.Tensor): Log variance of latent distribution
            mask (torch.Tensor, optional): Mask for observations
            
        Returns:
            dict: Dictionary containing total loss and individual components
        """
        # Reconstruction loss
        recon_loss = reconstruction_loss(
            predictions, targets, mask, self.recon_loss_type
        )
        
        # KL divergence
        kl_loss = kl_divergence(latent_mean, latent_logvar)
        
        # Total loss
        total_loss = recon_loss + self.kl_weight * kl_loss
        
        return {
            'loss': total_loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss
        }
