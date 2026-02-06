"""
Clinical metrics for evaluating postprandial metabolic predictions.
"""

import numpy as np
import torch


def rmse(predictions, targets, mask=None):
    """
    Root Mean Squared Error.
    
    Args:
        predictions: Predicted values (numpy array or torch tensor)
        targets: Ground truth values
        mask: Optional mask for valid observations
        
    Returns:
        float: RMSE value
    """
    # Convert to numpy if needed
    if torch.is_tensor(predictions):
        predictions = predictions.detach().cpu().numpy()
    if torch.is_tensor(targets):
        targets = targets.detach().cpu().numpy()
    if mask is not None and torch.is_tensor(mask):
        mask = mask.detach().cpu().numpy()
    
    # Flatten arrays
    predictions = predictions.flatten()
    targets = targets.flatten()
    
    if mask is not None:
        mask = mask.flatten()
        predictions = predictions[mask > 0]
        targets = targets[mask > 0]
    
    return np.sqrt(np.mean((predictions - targets) ** 2))


def mae(predictions, targets, mask=None):
    """
    Mean Absolute Error.
    
    Args:
        predictions: Predicted values (numpy array or torch tensor)
        targets: Ground truth values
        mask: Optional mask for valid observations
        
    Returns:
        float: MAE value
    """
    # Convert to numpy if needed
    if torch.is_tensor(predictions):
        predictions = predictions.detach().cpu().numpy()
    if torch.is_tensor(targets):
        targets = targets.detach().cpu().numpy()
    if mask is not None and torch.is_tensor(mask):
        mask = mask.detach().cpu().numpy()
    
    # Flatten arrays
    predictions = predictions.flatten()
    targets = targets.flatten()
    
    if mask is not None:
        mask = mask.flatten()
        predictions = predictions[mask > 0]
        targets = targets[mask > 0]
    
    return np.mean(np.abs(predictions - targets))


def mape(predictions, targets, mask=None, epsilon=1e-8):
    """
    Mean Absolute Percentage Error.
    
    Args:
        predictions: Predicted values (numpy array or torch tensor)
        targets: Ground truth values
        mask: Optional mask for valid observations
        epsilon: Small value to avoid division by zero
        
    Returns:
        float: MAPE value as percentage
    """
    # Convert to numpy if needed
    if torch.is_tensor(predictions):
        predictions = predictions.detach().cpu().numpy()
    if torch.is_tensor(targets):
        targets = targets.detach().cpu().numpy()
    if mask is not None and torch.is_tensor(mask):
        mask = mask.detach().cpu().numpy()
    
    # Flatten arrays
    predictions = predictions.flatten()
    targets = targets.flatten()
    
    if mask is not None:
        mask = mask.flatten()
        predictions = predictions[mask > 0]
        targets = targets[mask > 0]
    
    return np.mean(np.abs((targets - predictions) / (targets + epsilon))) * 100


def time_in_range(glucose_values, low_threshold=70, high_threshold=180):
    """
    Calculate Time In Range (TIR) - percentage of time glucose is in target range.
    Standard range: 70-180 mg/dL
    
    Args:
        glucose_values: Array of glucose values (mg/dL)
        low_threshold: Lower bound of target range (default: 70 mg/dL)
        high_threshold: Upper bound of target range (default: 180 mg/dL)
        
    Returns:
        dict: Dictionary with TIR, time below range, time above range
    """
    if torch.is_tensor(glucose_values):
        glucose_values = glucose_values.detach().cpu().numpy()
    
    glucose_values = glucose_values.flatten()
    
    in_range = np.logical_and(
        glucose_values >= low_threshold,
        glucose_values <= high_threshold
    )
    below_range = glucose_values < low_threshold
    above_range = glucose_values > high_threshold
    
    total = len(glucose_values)
    
    return {
        'tir': np.sum(in_range) / total * 100,
        'tbr': np.sum(below_range) / total * 100,
        'tar': np.sum(above_range) / total * 100
    }


def glucose_variability(glucose_values):
    """
    Calculate glucose variability metrics.
    
    Args:
        glucose_values: Array of glucose values
        
    Returns:
        dict: Dictionary with standard deviation and coefficient of variation
    """
    if torch.is_tensor(glucose_values):
        glucose_values = glucose_values.detach().cpu().numpy()
    
    glucose_values = glucose_values.flatten()
    
    mean_glucose = np.mean(glucose_values)
    std_glucose = np.std(glucose_values)
    cv = (std_glucose / mean_glucose) * 100 if mean_glucose > 0 else 0
    
    return {
        'mean': mean_glucose,
        'std': std_glucose,
        'cv': cv
    }


def calculate_all_metrics(predictions, targets, mask=None):
    """
    Calculate all evaluation metrics.
    
    Args:
        predictions: Predicted values
        targets: Ground truth values
        mask: Optional mask for valid observations
        
    Returns:
        dict: Dictionary containing all metrics
    """
    metrics = {
        'rmse': rmse(predictions, targets, mask),
        'mae': mae(predictions, targets, mask),
        'mape': mape(predictions, targets, mask)
    }
    
    # If values are in glucose range (assumed mg/dL), calculate clinical metrics
    # Typical glucose range is 40-400 mg/dL
    if torch.is_tensor(predictions):
        pred_np = predictions.detach().cpu().numpy()
    else:
        pred_np = predictions
    
    if np.mean(pred_np) > 10:  # Likely in mg/dL scale
        tir_metrics = time_in_range(predictions)
        var_metrics = glucose_variability(predictions)
        
        metrics.update(tir_metrics)
        metrics.update(var_metrics)
    
    return metrics
