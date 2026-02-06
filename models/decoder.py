"""
Decoder module for the two-stage latent ODE model.
Decodes latent representations back to observation space.
"""

import torch
import torch.nn as nn


class Decoder(nn.Module):
    """
    Decoder that reconstructs observations from latent representations.
    
    Args:
        latent_dim (int): Dimension of latent representation
        hidden_dim (int): Dimension of hidden layers
        output_dim (int): Dimension of output features (e.g., glucose predictions)
        num_layers (int): Number of hidden layers
    """
    
    def __init__(
        self,
        latent_dim,
        hidden_dim=64,
        output_dim=1,
        num_layers=2
    ):
        super(Decoder, self).__init__()
        
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Build decoder network
        layers = []
        layers.append(nn.Linear(latent_dim, hidden_dim))
        layers.append(nn.ReLU())
        
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, z):
        """
        Decode latent representation to observation space.
        
        Args:
            z (torch.Tensor): Latent representation
                             Shape: (batch_size, seq_len, latent_dim) or (batch_size, latent_dim)
            
        Returns:
            torch.Tensor: Reconstructed observations
                         Shape: (batch_size, seq_len, output_dim) or (batch_size, output_dim)
        """
        return self.net(z)
