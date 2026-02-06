"""
Two-Stage Latent ODE Model for Personalized Metabolic Modeling.

This model combines:
1. Stage 1 (Encoder): Encodes observed time series into latent representations
2. Stage 2 (Latent ODE): Models continuous-time dynamics in latent space
3. Decoder: Reconstructs predictions from latent trajectories
"""

import torch
import torch.nn as nn
from .encoder import Encoder
from .decoder import Decoder
from .ode_solver import ODEFunc, ODEBlock


class TwoStageLatentODE(nn.Module):
    """
    Two-Stage Latent ODE Model for modeling postprandial metabolic dynamics.
    
    Args:
        input_dim (int): Dimension of input observations
        latent_dim (int): Dimension of latent space
        output_dim (int): Dimension of output predictions
        encoder_hidden_dim (int): Hidden dimension for encoder RNN
        ode_hidden_dim (int): Hidden dimension for ODE function network
        decoder_hidden_dim (int): Hidden dimension for decoder network
        num_encoder_layers (int): Number of RNN layers in encoder
        num_ode_layers (int): Number of layers in ODE function network
        num_decoder_layers (int): Number of layers in decoder
        rnn_type (str): Type of RNN ('gru' or 'lstm')
        ode_method (str): ODE solver method
        ode_rtol (float): Relative tolerance for ODE solver
        ode_atol (float): Absolute tolerance for ODE solver
    """
    
    def __init__(
        self,
        input_dim,
        latent_dim=32,
        output_dim=1,
        encoder_hidden_dim=64,
        ode_hidden_dim=64,
        decoder_hidden_dim=64,
        num_encoder_layers=1,
        num_ode_layers=2,
        num_decoder_layers=2,
        rnn_type='gru',
        ode_method='dopri5',
        ode_rtol=1e-3,
        ode_atol=1e-4
    ):
        super(TwoStageLatentODE, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        
        # Stage 1: Encoder
        self.encoder = Encoder(
            input_dim=input_dim,
            hidden_dim=encoder_hidden_dim,
            latent_dim=latent_dim,
            num_layers=num_encoder_layers,
            rnn_type=rnn_type
        )
        
        # Stage 2: Latent ODE
        self.ode_func = ODEFunc(
            latent_dim=latent_dim,
            hidden_dim=ode_hidden_dim,
            num_layers=num_ode_layers
        )
        
        self.ode_block = ODEBlock(
            ode_func=self.ode_func,
            method=ode_method,
            rtol=ode_rtol,
            atol=ode_atol
        )
        
        # Decoder
        self.decoder = Decoder(
            latent_dim=latent_dim,
            hidden_dim=decoder_hidden_dim,
            output_dim=output_dim,
            num_layers=num_decoder_layers
        )
    
    def forward(self, x_obs, t_obs, t_pred):
        """
        Forward pass through the two-stage latent ODE model.
        
        Args:
            x_obs (torch.Tensor): Observed time series data
                                 Shape: (batch_size, obs_len, input_dim)
            t_obs (torch.Tensor): Observation time points
                                 Shape: (batch_size, obs_len) or (obs_len,)
            t_pred (torch.Tensor): Time points for prediction
                                  Shape: (batch_size, pred_len) or (pred_len,)
        
        Returns:
            dict: Dictionary containing:
                - 'predictions': Predicted values at t_pred
                - 'latent_mean': Mean of latent distribution
                - 'latent_logvar': Log variance of latent distribution
                - 'latent_trajectory': ODE trajectory in latent space
        """
        # Stage 1: Encode observations to latent space
        latent_mean, latent_logvar = self.encoder(x_obs)
        
        # Sample from latent distribution using reparameterization trick
        z0 = self.encoder.reparameterize(latent_mean, latent_logvar)
        
        # Stage 2: Solve ODE in latent space
        # Ensure t_pred is 1D tensor
        if t_pred.dim() > 1:
            t_pred = t_pred[0]  # Assume same time points for all batches
        
        # Solve ODE from z0 over time span t_pred
        latent_trajectory = self.ode_block(z0, t_pred)
        
        # Decode latent trajectory to observation space
        predictions = self.decoder(latent_trajectory)
        
        return {
            'predictions': predictions,
            'latent_mean': latent_mean,
            'latent_logvar': latent_logvar,
            'latent_trajectory': latent_trajectory,
            'z0': z0
        }
    
    def encode(self, x_obs):
        """
        Encode observations to latent space.
        
        Args:
            x_obs (torch.Tensor): Observed time series data
            
        Returns:
            tuple: (latent_mean, latent_logvar)
        """
        return self.encoder(x_obs)
    
    def predict(self, z0, t_pred):
        """
        Predict from initial latent state z0.
        
        Args:
            z0 (torch.Tensor): Initial latent state
            t_pred (torch.Tensor): Time points for prediction
            
        Returns:
            torch.Tensor: Predictions at t_pred
        """
        # Ensure t_pred is 1D
        if t_pred.dim() > 1:
            t_pred = t_pred[0]
        
        # Solve ODE
        latent_trajectory = self.ode_block(z0, t_pred)
        
        # Decode
        predictions = self.decoder(latent_trajectory)
        
        return predictions
    
    def get_nfe(self):
        """Get number of function evaluations in last ODE solve."""
        return self.ode_block.get_nfe()
