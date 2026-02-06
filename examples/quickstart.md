# Quick Start Example for Two-Stage Latent ODE Model

This example demonstrates how to use the Two-Stage Latent ODE model for postprandial glucose prediction.

## Basic Usage

```python
import torch
from models import TwoStageLatentODE
from visualization import plot_predictions

# 1. Create model
model = TwoStageLatentODE(
    input_dim=1,           # Glucose only
    latent_dim=32,         # Latent space dimension
    output_dim=1,          # Predict glucose
    encoder_hidden_dim=64,
    ode_hidden_dim=64,
    decoder_hidden_dim=64,
    rnn_type='gru',        # or 'lstm'
    ode_method='dopri5'    # ODE solver method
)

# 2. Prepare your data
# Observations: glucose measurements over first hour
x_obs = torch.randn(1, 12, 1)  # (batch, time_steps, features)
t_obs = torch.linspace(0, 60, 12)  # Time in minutes

# Prediction: future time points
t_pred = torch.linspace(60, 240, 36)  # Predict next 3 hours

# 3. Make predictions
model.eval()
with torch.no_grad():
    outputs = model(x_obs, t_obs, t_pred)
    predictions = outputs['predictions']
    latent_mean = outputs['latent_mean']  # Patient-specific representation

# 4. Visualize results
plot_predictions(
    t_obs=t_obs,
    x_obs=x_obs[0],
    t_pred=t_pred,
    predictions=predictions[0],
    title='Postprandial Glucose Prediction',
    save_path='prediction.png'
)
```

## Training a Model

```python
import torch
from torch.utils.data import DataLoader, TensorDataset
from models import TwoStageLatentODE
from train import Trainer, LatentODELoss

# 1. Prepare your dataset
# Replace this with your actual data loading
x_obs_train = torch.randn(100, 12, 1)
t_obs_train = torch.linspace(0, 60, 12).repeat(100, 1)
t_pred_train = torch.linspace(60, 240, 36).repeat(100, 1)
targets_train = torch.randn(100, 36, 1)

# Create dataset and dataloader
train_data = []
for i in range(len(x_obs_train)):
    train_data.append({
        'x_obs': x_obs_train[i],
        't_obs': t_obs_train[i],
        't_pred': t_pred_train[i],
        'targets': targets_train[i]
    })

# 2. Create model and loss
model = TwoStageLatentODE(input_dim=1, latent_dim=32, output_dim=1)
loss_fn = LatentODELoss(kl_weight=0.1)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# 3. Create trainer
trainer = Trainer(
    model=model,
    loss_fn=loss_fn,
    optimizer=optimizer,
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

# 4. Train
history = trainer.train(
    train_loader=train_data,
    val_loader=None,  # Add validation data if available
    num_epochs=50
)
```

## Extracting Patient-Specific Features

```python
# Encode observations to latent space
model.eval()
with torch.no_grad():
    latent_mean, latent_logvar = model.encode(x_obs)
    
# latent_mean is the patient-specific representation
# Use it for clustering, classification, or personalization
print(f"Patient latent vector shape: {latent_mean.shape}")
```

## Clinical Evaluation

```python
from evaluation import calculate_all_metrics

# Calculate clinical metrics
metrics = calculate_all_metrics(
    predictions=predictions,
    targets=targets
)

print(f"RMSE: {metrics['rmse']:.2f} mg/dL")
print(f"MAE: {metrics['mae']:.2f} mg/dL")
print(f"Time in Range: {metrics['tir']:.1f}%")
print(f"Glucose CV: {metrics['cv']:.1f}%")
```

## Loading and Saving Models

```python
# Save model
torch.save(model.state_dict(), 'model.pth')

# Load model
model = TwoStageLatentODE(input_dim=1, latent_dim=32, output_dim=1)
model.load_state_dict(torch.load('model.pth'))
model.eval()
```

## Advanced: Multi-modal Input

```python
# For glucose + insulin + meals
model = TwoStageLatentODE(
    input_dim=3,  # glucose, insulin, carbs
    latent_dim=64,
    output_dim=1  # predict glucose only
)

# Input data shape: (batch, time_steps, 3)
x_obs = torch.randn(1, 12, 3)
```

## Next Steps

- See `scripts/train_model.py` for complete training example
- See `scripts/demo.py` for interactive demo
- Read `README.md` for full documentation
- Check `config.yaml` for hyperparameter options
