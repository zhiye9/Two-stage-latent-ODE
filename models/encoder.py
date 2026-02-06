"""
Encoder module for the two-stage latent ODE model.
Encodes irregular time series observations into latent representations.
"""

import torch
import torch.nn as nn


class Encoder(nn.Module):
    """
    RNN-based encoder that processes time series data and outputs latent representations.
    
    Args:
        input_dim (int): Dimension of input features (e.g., glucose, insulin measurements)
        hidden_dim (int): Dimension of hidden state in RNN
        latent_dim (int): Dimension of latent representation
        num_layers (int): Number of RNN layers
        rnn_type (str): Type of RNN ('gru' or 'lstm')
        bidirectional (bool): Whether to use bidirectional RNN
    """
    
    def __init__(
        self,
        input_dim,
        hidden_dim=64,
        latent_dim=32,
        num_layers=1,
        rnn_type='gru',
        bidirectional=False
    ):
        super(Encoder, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # Select RNN type
        if rnn_type.lower() == 'gru':
            self.rnn = nn.GRU(
                input_dim,
                hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=bidirectional
            )
        elif rnn_type.lower() == 'lstm':
            self.rnn = nn.LSTM(
                input_dim,
                hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=bidirectional
            )
        else:
            raise ValueError(f"Unknown RNN type: {rnn_type}")
        
        # Project RNN output to latent space
        rnn_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc_mean = nn.Linear(rnn_output_dim, latent_dim)
        self.fc_logvar = nn.Linear(rnn_output_dim, latent_dim)
        
    def forward(self, x, mask=None):
        """
        Forward pass through encoder.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_dim)
            mask (torch.Tensor, optional): Mask for irregular observations
            
        Returns:
            tuple: (mean, logvar) of latent distribution, each of shape (batch_size, latent_dim)
        """
        # Process through RNN
        output, hidden = self.rnn(x)
        
        # Take the last hidden state (for GRU) or last hidden (for LSTM)
        if isinstance(hidden, tuple):  # LSTM case
            hidden = hidden[0]
        
        # If bidirectional, concatenate forward and backward
        if self.bidirectional:
            # hidden shape: (num_layers * 2, batch_size, hidden_dim)
            hidden_forward = hidden[-2, :, :]
            hidden_backward = hidden[-1, :, :]
            hidden_final = torch.cat([hidden_forward, hidden_backward], dim=1)
        else:
            # hidden shape: (num_layers, batch_size, hidden_dim)
            hidden_final = hidden[-1, :, :]
        
        # Project to latent space
        mean = self.fc_mean(hidden_final)
        logvar = self.fc_logvar(hidden_final)
        
        return mean, logvar
    
    def reparameterize(self, mean, logvar):
        """
        Reparameterization trick for VAE.
        
        Args:
            mean (torch.Tensor): Mean of latent distribution
            logvar (torch.Tensor): Log variance of latent distribution
            
        Returns:
            torch.Tensor: Sampled latent vector
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std
