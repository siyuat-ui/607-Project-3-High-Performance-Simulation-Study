"""Training and inference for engression networks (OPTIMIZED VERSION).

This module uses the optimized engression_loss function for better performance.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from methods_optimized import EngressionNet, engression_loss, get_device

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False


class EngressionTrainer:
    """Trainer for EngressionNet models with early stopping."""
    
    def __init__(self, batch_size=128, learning_rate=1e-4, num_epochs=200, 
                 m=50, patience=20, hidden_dim=64, num_layers=3, 
                 dropout=0.0, use_batchnorm=False, input_dim=128, device=None):
        """Initialize the trainer."""
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.m = m
        self.patience = patience
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_batchnorm = use_batchnorm
        self.input_dim = input_dim
        self.device = device if device is not None else get_device()
        
        self.training_history = {
            'loss': [],
            'term1': [],
            'term2': []
        }
    
    def train(self, X, model=None, verbose=True):
        """Train an engression network."""
        # Convert to tensor if needed
        if not isinstance(X, torch.Tensor):
            X = torch.from_numpy(X).float()
        else:
            X = X.float()
        
        # Move to device
        X = X.to(self.device)
        
        # Initialize model if not provided
        if model is None:
            output_dim = X.shape[1]
            model = EngressionNet(
                input_dim=self.input_dim,
                output_dim=output_dim,
                hidden_dim=self.hidden_dim,
                num_layers=self.num_layers,
                dropout=self.dropout,
                use_batchnorm=self.use_batchnorm
            ).to(self.device)
        
        # Store input_dim for later reference
        self.model_input_dim = self.input_dim
        
        # Initialize optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        
        # Prepare DataLoader
        dataset = TensorDataset(X)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Early stopping variables
        best_loss = float('inf')
        epochs_no_improve = 0
        best_model_state = None
        
        # Training loop with early stopping
        for epoch in range(1, self.num_epochs + 1):
            epoch_loss = 0.0
            epoch_term1 = 0.0
            epoch_term2 = 0.0
            
            for batch in dataloader:
                X_batch = batch[0]
                
                optimizer.zero_grad()
                loss, t1, t2 = engression_loss(model, X_batch, m=self.m, device=self.device)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item() * X_batch.size(0)
                epoch_term1 += t1.item() * X_batch.size(0)
                epoch_term2 += t2.item() * X_batch.size(0)
            
            # Average over dataset
            epoch_loss /= len(dataset)
            epoch_term1 /= len(dataset)
            epoch_term2 /= len(dataset)
            
            # Store history
            self.training_history['loss'].append(epoch_loss)
            self.training_history['term1'].append(epoch_term1)
            self.training_history['term2'].append(epoch_term2)
            
            # Print progress
            if verbose and (epoch % 10 == 0 or epoch == 1):
                print(f"Epoch {epoch:03d} | Loss: {epoch_loss:.4f} | "
                      f"Term1: {epoch_term1:.4f} | Term2: {epoch_term2:.4f}")
            
            # Early stopping check
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                epochs_no_improve = 0
                best_model_state = model.state_dict().copy()
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= self.patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch} | Best Loss: {best_loss:.4f}")
                    model.load_state_dict(best_model_state)
                    break
        
        return model, self.training_history


def generate_samples(model, num_samples=1000, input_dim=None, device=None):
    """Generate samples from a trained engression network."""
    if device is None:
        device = next(model.parameters()).device
    
    if input_dim is None:
        input_dim = model.network[0].in_features
    
    model.eval()
    eps = torch.randn(num_samples, input_dim, device=device)
    
    with torch.no_grad():
        samples = model(eps)
    
    return samples


def train_and_generate(X, num_samples=1000, batch_size=128, learning_rate=1e-4,
                       num_epochs=200, m=50, patience=20, input_dim=128,
                       hidden_dim=64, num_layers=3, dropout=0.0, 
                       use_batchnorm=False, verbose=True, device=None):
    """Convenience function to train a model and generate samples."""
    if device is None:
        device = get_device()
    
    # Initialize trainer
    trainer = EngressionTrainer(
        batch_size=batch_size,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        m=m,
        patience=patience,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        use_batchnorm=use_batchnorm,
        input_dim=input_dim,
        device=device
    )
    
    # Train model
    model, history = trainer.train(X, verbose=verbose)
    
    # Generate samples
    samples = generate_samples(model, num_samples=num_samples, 
                              input_dim=input_dim, device=device)
    
    return model, history, samples
