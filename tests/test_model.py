"""
Unit tests for the two-stage latent ODE model.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import pytest
from models import Encoder, Decoder, ODEFunc, ODEBlock, TwoStageLatentODE
from train import LatentODELoss


class TestEncoder:
    """Test cases for Encoder module."""
    
    def test_encoder_forward(self):
        """Test encoder forward pass."""
        encoder = Encoder(input_dim=1, hidden_dim=32, latent_dim=16)
        x = torch.randn(8, 10, 1)  # (batch, seq_len, input_dim)
        
        mean, logvar = encoder(x)
        
        assert mean.shape == (8, 16), "Mean shape incorrect"
        assert logvar.shape == (8, 16), "Logvar shape incorrect"
    
    def test_encoder_reparameterize(self):
        """Test reparameterization trick."""
        encoder = Encoder(input_dim=1, latent_dim=16)
        mean = torch.randn(8, 16)
        logvar = torch.randn(8, 16)
        
        z = encoder.reparameterize(mean, logvar)
        
        assert z.shape == (8, 16), "Latent vector shape incorrect"


class TestODESolver:
    """Test cases for ODE solver components."""
    
    def test_ode_func(self):
        """Test ODE function evaluation."""
        ode_func = ODEFunc(latent_dim=16, hidden_dim=32)
        x = torch.randn(8, 16)
        t = torch.tensor(0.5)
        
        dx_dt = ode_func(t, x)
        
        assert dx_dt.shape == (8, 16), "Derivative shape incorrect"
    
    def test_ode_block(self):
        """Test ODE block integration."""
        ode_func = ODEFunc(latent_dim=16, hidden_dim=32)
        ode_block = ODEBlock(ode_func)
        
        x0 = torch.randn(8, 16)
        t_span = torch.linspace(0, 1, 10)
        
        trajectory = ode_block(x0, t_span)
        
        assert trajectory.shape == (8, 10, 16), "Trajectory shape incorrect"


class TestDecoder:
    """Test cases for Decoder module."""
    
    def test_decoder_forward(self):
        """Test decoder forward pass."""
        decoder = Decoder(latent_dim=16, hidden_dim=32, output_dim=1)
        z = torch.randn(8, 10, 16)  # (batch, seq_len, latent_dim)
        
        output = decoder(z)
        
        assert output.shape == (8, 10, 1), "Output shape incorrect"


class TestTwoStageLatentODE:
    """Test cases for complete model."""
    
    def test_model_forward(self):
        """Test full model forward pass."""
        model = TwoStageLatentODE(
            input_dim=1,
            latent_dim=16,
            output_dim=1,
            encoder_hidden_dim=32,
            ode_hidden_dim=32,
            decoder_hidden_dim=32
        )
        
        x_obs = torch.randn(8, 10, 1)
        t_obs = torch.linspace(0, 1, 10)
        t_pred = torch.linspace(1, 2, 20)
        
        outputs = model(x_obs, t_obs, t_pred)
        
        assert 'predictions' in outputs, "Missing predictions"
        assert 'latent_mean' in outputs, "Missing latent_mean"
        assert 'latent_logvar' in outputs, "Missing latent_logvar"
        assert outputs['predictions'].shape == (8, 20, 1), "Predictions shape incorrect"
    
    def test_model_encode(self):
        """Test model encoding."""
        model = TwoStageLatentODE(input_dim=1, latent_dim=16)
        x_obs = torch.randn(8, 10, 1)
        
        mean, logvar = model.encode(x_obs)
        
        assert mean.shape == (8, 16), "Mean shape incorrect"
        assert logvar.shape == (8, 16), "Logvar shape incorrect"


class TestLoss:
    """Test cases for loss functions."""
    
    def test_latent_ode_loss(self):
        """Test combined loss function."""
        loss_fn = LatentODELoss(kl_weight=0.1)
        
        predictions = torch.randn(8, 20, 1)
        targets = torch.randn(8, 20, 1)
        latent_mean = torch.randn(8, 16)
        latent_logvar = torch.randn(8, 16)
        
        loss_dict = loss_fn(predictions, targets, latent_mean, latent_logvar)
        
        assert 'loss' in loss_dict, "Missing total loss"
        assert 'recon_loss' in loss_dict, "Missing reconstruction loss"
        assert 'kl_loss' in loss_dict, "Missing KL loss"
        assert loss_dict['loss'].item() > 0, "Loss should be positive"


if __name__ == '__main__':
    # Run tests
    pytest.main([__file__, '-v'])
