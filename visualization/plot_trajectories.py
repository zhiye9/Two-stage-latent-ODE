"""
Plotting utilities for trajectories and predictions.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_predictions(
    t_obs,
    x_obs,
    t_pred,
    predictions,
    targets=None,
    title='Glucose Predictions',
    xlabel='Time (minutes)',
    ylabel='Glucose (mg/dL)',
    save_path=None,
    figsize=(10, 6)
):
    """
    Plot observed data and model predictions.
    
    Args:
        t_obs: Observation times
        x_obs: Observed values
        t_pred: Prediction times
        predictions: Model predictions
        targets: Ground truth values (optional)
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        save_path: Path to save figure (optional)
        figsize: Figure size
    """
    # Convert tensors to numpy
    if torch.is_tensor(t_obs):
        t_obs = t_obs.detach().cpu().numpy()
    if torch.is_tensor(x_obs):
        x_obs = x_obs.detach().cpu().numpy()
    if torch.is_tensor(t_pred):
        t_pred = t_pred.detach().cpu().numpy()
    if torch.is_tensor(predictions):
        predictions = predictions.detach().cpu().numpy()
    if targets is not None and torch.is_tensor(targets):
        targets = targets.detach().cpu().numpy()
    
    # Handle batch dimension
    if len(x_obs.shape) > 1:
        x_obs = x_obs.squeeze()
    if len(predictions.shape) > 1:
        predictions = predictions.squeeze()
    if targets is not None and len(targets.shape) > 1:
        targets = targets.squeeze()
    
    plt.figure(figsize=figsize)
    
    # Plot observations
    plt.scatter(t_obs, x_obs, label='Observations', color='blue', s=50, zorder=3)
    
    # Plot predictions
    plt.plot(t_pred, predictions, label='Predictions', color='red', linewidth=2)
    
    # Plot ground truth if available
    if targets is not None:
        plt.plot(t_pred, targets, label='Ground Truth', color='green', 
                linewidth=2, linestyle='--', alpha=0.7)
    
    # Add clinical thresholds for glucose
    if 'glucose' in ylabel.lower():
        plt.axhline(y=70, color='orange', linestyle=':', alpha=0.5, label='Low (70 mg/dL)')
        plt.axhline(y=180, color='orange', linestyle=':', alpha=0.5, label='High (180 mg/dL)')
    
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_glucose_curve(
    time_points,
    glucose_values,
    meal_times=None,
    title='Postprandial Glucose Response',
    save_path=None,
    figsize=(12, 6)
):
    """
    Plot glucose curve with clinical context.
    
    Args:
        time_points: Time points (minutes)
        glucose_values: Glucose measurements (mg/dL)
        meal_times: Times of meals (optional)
        title: Plot title
        save_path: Path to save figure (optional)
        figsize: Figure size
    """
    # Convert tensors to numpy
    if torch.is_tensor(time_points):
        time_points = time_points.detach().cpu().numpy()
    if torch.is_tensor(glucose_values):
        glucose_values = glucose_values.detach().cpu().numpy()
    
    # Handle batch dimension
    if len(glucose_values.shape) > 1:
        glucose_values = glucose_values.squeeze()
    
    plt.figure(figsize=figsize)
    
    # Plot glucose curve
    plt.plot(time_points, glucose_values, linewidth=2.5, color='darkblue')
    plt.fill_between(time_points, glucose_values, alpha=0.3, color='lightblue')
    
    # Add clinical zones
    plt.axhspan(70, 180, alpha=0.1, color='green', label='Target Range (70-180 mg/dL)')
    plt.axhspan(0, 70, alpha=0.1, color='red', label='Hypoglycemia (<70 mg/dL)')
    plt.axhspan(180, 400, alpha=0.1, color='orange', label='Hyperglycemia (>180 mg/dL)')
    
    # Mark meal times
    if meal_times is not None:
        for meal_time in meal_times:
            plt.axvline(x=meal_time, color='brown', linestyle='--', 
                       linewidth=2, alpha=0.7, label='Meal')
    
    plt.xlabel('Time (minutes)', fontsize=12)
    plt.ylabel('Glucose (mg/dL)', fontsize=12)
    plt.title(title, fontsize=14)
    plt.ylim([0, max(400, glucose_values.max() * 1.1)])
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_multiple_trajectories(
    t_obs_list,
    x_obs_list,
    t_pred_list,
    predictions_list,
    patient_ids=None,
    title='Multiple Patient Trajectories',
    save_path=None,
    figsize=(12, 8)
):
    """
    Plot trajectories for multiple patients.
    
    Args:
        t_obs_list: List of observation times
        x_obs_list: List of observed values
        t_pred_list: List of prediction times
        predictions_list: List of predictions
        patient_ids: List of patient identifiers (optional)
        title: Plot title
        save_path: Path to save figure (optional)
        figsize: Figure size
    """
    num_patients = len(x_obs_list)
    
    plt.figure(figsize=figsize)
    
    colors = plt.cm.tab10(np.linspace(0, 1, num_patients))
    
    for i in range(num_patients):
        # Convert tensors to numpy
        t_obs = t_obs_list[i]
        x_obs = x_obs_list[i]
        t_pred = t_pred_list[i]
        predictions = predictions_list[i]
        
        if torch.is_tensor(t_obs):
            t_obs = t_obs.detach().cpu().numpy()
        if torch.is_tensor(x_obs):
            x_obs = x_obs.detach().cpu().numpy()
        if torch.is_tensor(t_pred):
            t_pred = t_pred.detach().cpu().numpy()
        if torch.is_tensor(predictions):
            predictions = predictions.detach().cpu().numpy()
        
        # Handle batch dimension
        if len(x_obs.shape) > 1:
            x_obs = x_obs.squeeze()
        if len(predictions.shape) > 1:
            predictions = predictions.squeeze()
        
        label = f"Patient {patient_ids[i]}" if patient_ids else f"Patient {i+1}"
        
        # Plot observations and predictions
        plt.scatter(t_obs, x_obs, color=colors[i], s=30, alpha=0.7)
        plt.plot(t_pred, predictions, color=colors[i], linewidth=2, label=label)
    
    plt.xlabel('Time (minutes)', fontsize=12)
    plt.ylabel('Glucose (mg/dL)', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(loc='best', ncol=2)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    else:
        plt.show()
    
    plt.close()
