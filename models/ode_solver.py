"""
Neural ODE solver components for modeling continuous-time dynamics in latent space.
"""

import torch
import torch.nn as nn
from torchdiffeq import odeint


class ODEFunc(nn.Module):
    """
    Neural network that parameterizes the derivative function dx/dt = f(x, t).
    This represents the continuous dynamics in the latent space.
    
    Args:
        latent_dim (int): Dimension of latent space
        hidden_dim (int): Dimension of hidden layers
        num_layers (int): Number of hidden layers
    """
    
    def __init__(self, latent_dim, hidden_dim=64, num_layers=2):
        super(ODEFunc, self).__init__()
        
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        
        # Build neural network layers
        layers = []
        layers.append(nn.Linear(latent_dim, hidden_dim))
        layers.append(nn.Tanh())
        
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())
        
        layers.append(nn.Linear(hidden_dim, latent_dim))
        
        self.net = nn.Sequential(*layers)
        
        # Track number of function evaluations for monitoring
        self.nfe = 0
    
    def forward(self, t, x):
        """
        Compute the derivative dx/dt at time t.
        
        Args:
            t (torch.Tensor): Current time (scalar or tensor)
            x (torch.Tensor): Current state, shape (batch_size, latent_dim)
            
        Returns:
            torch.Tensor: Time derivative dx/dt, shape (batch_size, latent_dim)
        """
        self.nfe += 1
        return self.net(x)


class ODEBlock(nn.Module):
    """
    ODE solver block that integrates latent dynamics over time.
    
    Args:
        ode_func (ODEFunc): Function defining the ODE dynamics
        method (str): ODE solver method ('dopri5', 'rk4', 'euler', etc.)
        rtol (float): Relative tolerance for adaptive solvers
        atol (float): Absolute tolerance for adaptive solvers
    """
    
    def __init__(
        self,
        ode_func,
        method='dopri5',
        rtol=1e-3,
        atol=1e-4
    ):
        super(ODEBlock, self).__init__()
        
        self.ode_func = ode_func
        self.method = method
        self.rtol = rtol
        self.atol = atol
    
    def forward(self, x0, t_span):
        """
        Solve ODE from initial state x0 over time span t_span.
        
        Args:
            x0 (torch.Tensor): Initial state, shape (batch_size, latent_dim)
            t_span (torch.Tensor): Time points to evaluate, shape (num_timepoints,)
            
        Returns:
            torch.Tensor: Trajectory of latent states, shape (batch_size, num_timepoints, latent_dim)
        """
        # Reset function evaluation counter
        self.ode_func.nfe = 0
        
        # Solve ODE
        # odeint expects shape (batch_size, latent_dim) for initial state
        # and returns (num_timepoints, batch_size, latent_dim)
        trajectory = odeint(
            self.ode_func,
            x0,
            t_span,
            method=self.method,
            rtol=self.rtol,
            atol=self.atol
        )
        
        # Transpose to (batch_size, num_timepoints, latent_dim)
        trajectory = trajectory.permute(1, 0, 2)
        
        return trajectory
    
    def get_nfe(self):
        """Get number of function evaluations used in last forward pass."""
        return self.ode_func.nfe
