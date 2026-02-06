"""
Visualization of latent space embeddings.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def plot_latent_embeddings(
    latent_vectors,
    labels=None,
    method='pca',
    title='Latent Space Embeddings',
    save_path=None,
    figsize=(10, 8)
):
    """
    Visualize latent space embeddings using dimensionality reduction.
    
    Args:
        latent_vectors: Latent representations, shape (num_samples, latent_dim)
        labels: Optional labels for coloring points
        method: Dimensionality reduction method ('pca' or 'tsne')
        title: Plot title
        save_path: Path to save figure (optional)
        figsize: Figure size
    """
    # Convert to numpy
    if torch.is_tensor(latent_vectors):
        latent_vectors = latent_vectors.detach().cpu().numpy()
    
    # Ensure 2D array
    if len(latent_vectors.shape) == 1:
        latent_vectors = latent_vectors.reshape(1, -1)
    
    # Apply dimensionality reduction
    if method.lower() == 'pca':
        reducer = PCA(n_components=2)
        embeddings_2d = reducer.fit_transform(latent_vectors)
        explained_var = reducer.explained_variance_ratio_
        subtitle = f"PCA (Explained variance: {explained_var[0]:.2%}, {explained_var[1]:.2%})"
    elif method.lower() == 'tsne':
        reducer = TSNE(n_components=2, random_state=42)
        embeddings_2d = reducer.fit_transform(latent_vectors)
        subtitle = "t-SNE"
    else:
        raise ValueError(f"Unknown method: {method}. Use 'pca' or 'tsne'.")
    
    # Create plot
    plt.figure(figsize=figsize)
    
    if labels is not None:
        # Convert labels to numpy
        if torch.is_tensor(labels):
            labels = labels.detach().cpu().numpy()
        
        # Plot with colors
        unique_labels = np.unique(labels)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            mask = labels == label
            plt.scatter(
                embeddings_2d[mask, 0],
                embeddings_2d[mask, 1],
                c=[colors[i]],
                label=f"Class {label}",
                s=50,
                alpha=0.6
            )
        plt.legend(loc='best')
    else:
        # Plot without labels
        plt.scatter(
            embeddings_2d[:, 0],
            embeddings_2d[:, 1],
            c=np.arange(len(embeddings_2d)),
            cmap='viridis',
            s=50,
            alpha=0.6
        )
        plt.colorbar(label='Sample Index')
    
    plt.xlabel(f'{method.upper()} Component 1', fontsize=12)
    plt.ylabel(f'{method.upper()} Component 2', fontsize=12)
    plt.title(f'{title}\n{subtitle}', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_latent_trajectories(
    latent_trajectories,
    time_points,
    latent_dims=None,
    title='Latent Space Trajectories',
    save_path=None,
    figsize=(12, 8)
):
    """
    Plot trajectories in latent space over time.
    
    Args:
        latent_trajectories: Latent trajectories, shape (num_samples, time_steps, latent_dim)
        time_points: Time points
        latent_dims: Which latent dimensions to plot (default: first 4)
        title: Plot title
        save_path: Path to save figure (optional)
        figsize: Figure size
    """
    # Convert to numpy
    if torch.is_tensor(latent_trajectories):
        latent_trajectories = latent_trajectories.detach().cpu().numpy()
    if torch.is_tensor(time_points):
        time_points = time_points.detach().cpu().numpy()
    
    # Get latent dimension
    if len(latent_trajectories.shape) == 2:
        latent_trajectories = latent_trajectories.unsqueeze(0)
    
    num_samples, time_steps, latent_dim = latent_trajectories.shape
    
    # Select latent dimensions to plot
    if latent_dims is None:
        latent_dims = min(4, latent_dim)
    if isinstance(latent_dims, int):
        latent_dims = list(range(latent_dims))
    
    num_dims = len(latent_dims)
    
    # Create subplots
    fig, axes = plt.subplots(num_dims, 1, figsize=figsize, sharex=True)
    if num_dims == 1:
        axes = [axes]
    
    colors = plt.cm.tab10(np.linspace(0, 1, num_samples))
    
    for i, dim in enumerate(latent_dims):
        ax = axes[i]
        
        for j in range(num_samples):
            ax.plot(
                time_points,
                latent_trajectories[j, :, dim],
                color=colors[j],
                linewidth=2,
                alpha=0.7,
                label=f'Sample {j+1}' if i == 0 else None
            )
        
        ax.set_ylabel(f'Latent Dim {dim}', fontsize=11)
        ax.grid(True, alpha=0.3)
        
        if i == 0 and num_samples <= 10:
            ax.legend(loc='best', ncol=2, fontsize=9)
    
    axes[-1].set_xlabel('Time', fontsize=12)
    fig.suptitle(title, fontsize=14, y=0.995)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_latent_distribution(
    latent_mean,
    latent_logvar,
    title='Latent Distribution',
    save_path=None,
    figsize=(12, 5)
):
    """
    Plot the mean and variance of latent distributions.
    
    Args:
        latent_mean: Mean of latent distribution, shape (num_samples, latent_dim)
        latent_logvar: Log variance of latent distribution
        title: Plot title
        save_path: Path to save figure (optional)
        figsize: Figure size
    """
    # Convert to numpy
    if torch.is_tensor(latent_mean):
        latent_mean = latent_mean.detach().cpu().numpy()
    if torch.is_tensor(latent_logvar):
        latent_logvar = latent_logvar.detach().cpu().numpy()
    
    latent_std = np.exp(0.5 * latent_logvar)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Plot mean
    im1 = ax1.imshow(latent_mean.T, aspect='auto', cmap='viridis')
    ax1.set_xlabel('Sample Index', fontsize=11)
    ax1.set_ylabel('Latent Dimension', fontsize=11)
    ax1.set_title('Latent Mean', fontsize=12)
    plt.colorbar(im1, ax=ax1)
    
    # Plot standard deviation
    im2 = ax2.imshow(latent_std.T, aspect='auto', cmap='plasma')
    ax2.set_xlabel('Sample Index', fontsize=11)
    ax2.set_ylabel('Latent Dimension', fontsize=11)
    ax2.set_title('Latent Standard Deviation', fontsize=12)
    plt.colorbar(im2, ax=ax2)
    
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    else:
        plt.show()
    
    plt.close()
