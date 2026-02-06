# Two-Stage Latent ODE Model

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**A Biologically Informed Deep Dynamical Model for Personalized Postprandial Metabolic Modeling**

This repository contains the implementation of a two-stage latent ordinary differential equation (ODE) model for modeling personalized postprandial (post-meal) metabolic dynamics, particularly glucose response patterns.

## Overview

The two-stage latent ODE model combines:
1. **Stage 1 (Encoder)**: RNN/GRU-based encoder that processes irregular time series observations and encodes them into a latent representation
2. **Stage 2 (Latent ODE)**: Neural ODE that models continuous-time dynamics in the latent space, learning differential equations governing metabolic processes
3. **Decoder**: Neural network that reconstructs predictions from latent trajectories

This architecture enables:
- Learning patient-specific metabolic dynamics from sparse glucose/insulin measurements
- Capturing nonlinear postprandial metabolic responses
- Personalization through learned latent representations
- Continuous-time predictions with Neural ODEs

## Features

- ✅ **Flexible Architecture**: Configurable encoder, ODE solver, and decoder
- ✅ **VAE-style Training**: Supports variational inference with KL divergence regularization
- ✅ **Multiple ODE Solvers**: Dopri5, RK4, Euler, and more via `torchdiffeq`
- ✅ **Clinical Metrics**: Time-in-range, glucose variability, RMSE, MAE
- ✅ **Irregular Time Series**: Handles sparse and irregular observations
- ✅ **Personalized Modeling**: Extracts patient-specific latent representations

## Installation

### Requirements

- Python 3.8+
- PyTorch 2.0+
- torchdiffeq 0.2.3+

### Setup

1. Clone the repository:
```bash
git clone https://github.com/zhiye9/Two-stage-latent-ODE.git
cd Two-stage-latent-ODE
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Repository Structure

```
Two-stage-latent-ODE/
├── models/                          # Model components
│   ├── encoder.py                   # RNN/GRU encoder
│   ├── ode_solver.py               # Neural ODE solver
│   ├── decoder.py                  # Decoder network
│   └── two_stage_latent_ode.py    # Main model
├── train/                          # Training utilities
│   ├── loss.py                     # Loss functions
│   └── train.py                    # Trainer class
├── evaluation/                     # Evaluation metrics
│   └── metrics.py                  # Clinical metrics (RMSE, MAE, TIR, etc.)
├── scripts/                        # Example scripts
│   └── train_model.py             # Training script
├── data/                           # Data directory
│   ├── raw/                        # Raw data
│   ├── processed/                  # Preprocessed data
│   └── synthetic/                  # Synthetic data
├── config.yaml                     # Configuration file
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## Quick Start

### Training a Model

1. Configure your model in `config.yaml`:
```yaml
model:
  input_dim: 1
  latent_dim: 32
  output_dim: 1
  encoder:
    hidden_dim: 64
    rnn_type: 'gru'
  ode:
    method: 'dopri5'
```

2. Run training:
```bash
python scripts/train_model.py
```

### Using the Model

```python
import torch
from models import TwoStageLatentODE

# Create model
model = TwoStageLatentODE(
    input_dim=1,
    latent_dim=32,
    output_dim=1,
    encoder_hidden_dim=64,
    ode_hidden_dim=64,
    decoder_hidden_dim=64
)

# Prepare data
x_obs = torch.randn(32, 12, 1)  # (batch, obs_length, input_dim)
t_obs = torch.linspace(0, 1, 12)  # Observation times
t_pred = torch.linspace(1, 4, 36)  # Prediction times

# Forward pass
outputs = model(x_obs, t_obs, t_pred)
predictions = outputs['predictions']  # (batch, pred_length, output_dim)
```

## Model Architecture

### Encoder
The encoder processes irregular time series observations using an RNN (GRU or LSTM) and outputs parameters of a latent distribution:

```
x_obs → RNN → h → [μ, log σ²]
z ~ N(μ, σ²)  (via reparameterization trick)
```

### Latent ODE
The latent ODE models continuous dynamics in latent space:

```
dz/dt = f_θ(z, t)
z(t) = ODESolve(z₀, t)
```

where `f_θ` is a neural network parameterizing the derivative function.

### Decoder
The decoder reconstructs predictions from latent trajectories:

```
ŷ = g_φ(z(t))
```

## Configuration

Key configuration options in `config.yaml`:

- **Model Architecture**: Dimensions, number of layers, RNN type
- **ODE Solver**: Method (dopri5, rk4, euler), tolerances
- **Training**: Batch size, learning rate, number of epochs
- **Loss**: KL divergence weight, reconstruction loss type
- **Data**: Observation/prediction windows, sampling frequency

## Evaluation Metrics

The model includes several clinical metrics for glucose monitoring:

- **RMSE/MAE**: Standard prediction errors
- **MAPE**: Mean absolute percentage error
- **Time-in-Range (TIR)**: Percentage of time in target glucose range (70-180 mg/dL)
- **Glucose Variability**: Standard deviation and coefficient of variation
- **Time Below/Above Range**: Clinical safety metrics

## Training

The training process uses a VAE-style objective combining:
1. **Reconstruction Loss**: MSE or MAE between predictions and targets
2. **KL Divergence**: Regularization of latent distribution

```python
Loss = Reconstruction_Loss + β * KL_Divergence
```

## Data Format

Expected input format:
- `x_obs`: Observed values, shape `(batch_size, obs_length, input_dim)`
- `t_obs`: Observation times, shape `(batch_size, obs_length)` or `(obs_length,)`
- `t_pred`: Prediction times, shape `(batch_size, pred_length)` or `(pred_length,)`
- `targets`: Target values, shape `(batch_size, pred_length, output_dim)`

For glucose monitoring:
- Input: Glucose measurements (mg/dL) and optionally insulin, meals
- Times: Minutes or hours from start of observation
- Output: Predicted glucose values

## Citation

If you use this code in your research, please cite:

```bibtex
@article{two_stage_latent_ode,
  title={A Biologically Informed Deep Dynamical Model Reveals Personalized Postprandial Metabolic},
  author={Ye, Zhi},
  year={2026}
}
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built with [PyTorch](https://pytorch.org/)
- Neural ODE implementation using [torchdiffeq](https://github.com/rtqichen/torchdiffeq)
- Inspired by research in continuous-time deep learning and metabolic modeling

## References

1. Chen, R. T., Rubanova, Y., Bettencourt, J., & Duvenaud, D. K. (2018). Neural ordinary differential equations. NeurIPS.
2. Rubanova, Y., Chen, R. T., & Duvenaud, D. K. (2019). Latent ordinary differential equations for irregularly-sampled time series. NeurIPS.
3. Bergman, R. N., et al. (1981). Quantitative estimation of insulin sensitivity. American Journal of Physiology.

## Contact

For questions and feedback, please open an issue on GitHub or contact the author.
