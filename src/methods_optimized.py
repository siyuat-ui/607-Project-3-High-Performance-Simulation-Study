"""Optimized methods for neural network-based engression estimation.

This module provides optimized neural network architectures and loss functions
for engression tasks.

OPTIMIZATIONS:
- Vectorized epsilon generation (single forward pass)
- torch.cdist() for efficient pairwise distances
- Reduced memory allocations
- Removed unnecessary mask creation
"""

import numpy as np
import torch
import torch.nn as nn


def get_device():
    """Determine the appropriate device for computation.
    
    Returns
    -------
    torch.device
        The device to use for computations (mps, cuda, or cpu)
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


class EngressionNet(nn.Module):
    """Neural network for engression estimation.
    
    Parameters
    ----------
    input_dim : int, default=2
        Dimension of input features
    output_dim : int, default=2
        Dimension of output
    hidden_dim : int, default=64
        Dimension of hidden layers
    num_layers : int, default=3
        Number of hidden layers
    dropout : float, default=0.0
        Dropout probability
    use_batchnorm : bool, default=False
        Whether to use batch normalization
    """
    
    def __init__(self, input_dim=2, output_dim=2, hidden_dim=64, 
                 num_layers=3, dropout=0.0, use_batchnorm=False):
        super(EngressionNet, self).__init__()
        layers = []
        in_dim = input_dim

        for i in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim

        layers.append(nn.Linear(hidden_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """Forward pass through the network.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor
            
        Returns
        -------
        torch.Tensor
            Output from the network
        """
        return self.network(x)


def engression_loss(g, X_batch, m=5, device=None):
    """Compute the engression loss (optimized version).
    
    The loss consists of two terms:
    - term1: Average distance from X_i to g(epsilon) samples
    - term2: Average pairwise distances between g(epsilon) samples
    
    OPTIMIZATIONS:
    - Single forward pass for all epsilons (batch_size * m instead of batch_size, m)
    - torch.cdist() for efficient pairwise distances (GPU-optimized)
    - Diagonal extraction instead of mask creation (saves memory)
    - Reduced intermediate tensor allocations
    
    Parameters
    ----------
    g : EngressionNet
        The engression network
    X_batch : torch.Tensor
        Batch of observed data, shape (batch_size, output_dim)
    m : int, default=5
        Number of epsilon samples per batch element
    device : torch.device, optional
        Device for computation. If None, uses get_device()
        
    Returns
    -------
    tuple of torch.Tensor
        (loss, term1_mean, term2_mean) where:
        - loss: scalar loss value
        - term1_mean: mean of term1 over batch
        - term2_mean: mean of term2 over batch
    """
    if device is None:
        device = get_device()
    
    batch_size, output_dim = X_batch.shape
    input_dim = g.network[0].in_features

    # OPTIMIZATION 1: Generate all epsilons at once (batch_size * m, input_dim)
    # Single forward pass instead of batch_size forward passes
    epsilons = torch.randn(batch_size * m, input_dim, device=device)
    g_eps = g(epsilons).view(batch_size, m, output_dim)

    # OPTIMIZATION 2: Use torch.cdist for term1 (efficient L2 distances)
    # Avoids explicit expansion and subtraction
    term1 = torch.cdist(X_batch.unsqueeze(1), g_eps, p=2).squeeze(1).mean(dim=1)

    # OPTIMIZATION 3: Use torch.cdist for term2 pairwise distances
    # GPU-optimized, vectorized computation
    pairwise_dist = torch.cdist(g_eps, g_eps, p=2)  # (batch_size, m, m)
    
    # OPTIMIZATION 4: Exclude diagonal without creating mask
    # Extract diagonal and subtract from sum (more memory efficient)
    term2 = (pairwise_dist.sum(dim=(1, 2)) - pairwise_dist.diagonal(dim1=1, dim2=2).sum(dim=1)) / (m * (m - 1))

    # loss: mean over batch
    loss = (term1 - 0.5 * term2).mean()
    
    return loss, term1.mean(), term2.mean()
