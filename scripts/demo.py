"""
Simple demo script showing basic usage of the Two-Stage Latent ODE model.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
import matplotlib.pyplot as plt
from models import TwoStageLatentODE
from visualization import plot_predictions


def create_demo_data():
    """
    Create simple demo data simulating postprandial glucose response.
    """
    # Time points
    t_obs = torch.linspace(0, 60, 12)  # 0-60 minutes, 12 observations
    t_pred = torch.linspace(60, 240, 36)  # 60-240 minutes, 36 predictions
    
    # Simulate glucose curve: baseline + peak response + decay
    baseline = 100.0
    peak = 80.0
    decay_rate = 0.02
    
    # Generate observations (first hour after meal)
    time_all = torch.cat([t_obs, t_pred])
    glucose_curve = baseline + peak * torch.exp(-decay_rate * time_all)
    
    # Split into observation and prediction
    x_obs = glucose_curve[:12].unsqueeze(0).unsqueeze(-1)  # (1, 12, 1)
    targets = glucose_curve[12:].unsqueeze(0).unsqueeze(-1)  # (1, 36, 1)
    
    # Add some noise to observations
    x_obs = x_obs + torch.randn_like(x_obs) * 3
    
    return x_obs, t_obs, t_pred, targets


def main():
    """Run simple demo."""
    print("Two-Stage Latent ODE Model - Demo")
    print("=" * 50)
    
    # Create demo data
    print("\n1. Creating demo data...")
    x_obs, t_obs, t_pred, targets = create_demo_data()
    print(f"   Observations shape: {x_obs.shape}")
    print(f"   Prediction times: {len(t_pred)} points")
    
    # Create model
    print("\n2. Creating model...")
    model = TwoStageLatentODE(
        input_dim=1,
        latent_dim=16,
        output_dim=1,
        encoder_hidden_dim=32,
        ode_hidden_dim=32,
        decoder_hidden_dim=32,
        num_encoder_layers=1,
        num_ode_layers=2,
        num_decoder_layers=2,
        rnn_type='gru',
        ode_method='dopri5'
    )
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"   Model created with {num_params:,} parameters")
    
    # Set to evaluation mode
    model.eval()
    
    # Forward pass
    print("\n3. Running inference...")
    with torch.no_grad():
        outputs = model(x_obs, t_obs, t_pred)
        predictions = outputs['predictions']
        latent_mean = outputs['latent_mean']
        nfe = model.get_nfe()
    
    print(f"   Predictions shape: {predictions.shape}")
    print(f"   Latent dimension: {latent_mean.shape[1]}")
    print(f"   ODE function evaluations: {nfe}")
    
    # Calculate simple error (before training, will be high)
    with torch.no_grad():
        mae = torch.mean(torch.abs(predictions - targets)).item()
        rmse = torch.sqrt(torch.mean((predictions - targets) ** 2)).item()
    
    print(f"\n4. Metrics (untrained model):")
    print(f"   MAE: {mae:.2f} mg/dL")
    print(f"   RMSE: {rmse:.2f} mg/dL")
    
    # Visualize
    print("\n5. Creating visualization...")
    plot_predictions(
        t_obs=t_obs,
        x_obs=x_obs[0],
        t_pred=t_pred,
        predictions=predictions[0],
        targets=targets[0],
        title='Two-Stage Latent ODE Demo (Untrained)',
        xlabel='Time (minutes)',
        ylabel='Glucose (mg/dL)',
        save_path='demo_prediction.png'
    )
    print("   Plot saved to: demo_prediction.png")
    
    print("\n" + "=" * 50)
    print("Demo complete!")
    print("\nNote: This model is untrained. To see good predictions,")
    print("train the model using: python scripts/train_model.py")


if __name__ == '__main__':
    main()
