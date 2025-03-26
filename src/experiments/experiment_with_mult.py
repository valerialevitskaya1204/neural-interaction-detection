from src.neural_detection.multilayer_perceptron import MLP, train, get_weights, get_interactions
from src.utils import preprocess_data, get_anyorder_R_precision, get_pairwise_auc, sanitize_tensor
from src.plots.draw_smth import plot_metrics_mult
import numpy as np
import torch
import os
import matplotlib.pyplot as plt

use_main_effect_nets = True
num_samples = 30000
num_features = 10
np.random.seed(52)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def multicollinearity_analysis(n_clones_list=[1, 5, 10, 20, 50], exact_clones=True, noise_level=0.1):
    """Study interaction detection with cloned/correlated features"""
    metrics_results = {}
    
    for n_clones in n_clones_list:
        # Generate synthetic data with multicollinearity
        X, Y, ground_truth = generate_multicollinear_data(
            num_samples, 
            num_features,
            n_clones,
            exact_clones,
            noise_level
        )
        
        # Preprocess data using your existing pipeline
        X = torch.Tensor(X)

        Y = sanitize_tensor(torch.Tensor(Y))
        data_loaders = preprocess_data(
            X, Y, 
            valid_size=10000, 
            test_size=10000, 
            std_scale=False, 
            get_torch_loaders=True
        )
        
        # Initialize and train model
        model = MLP(
            num_features, 
            [140, 100, 60, 20], 
            use_main_effect_nets=use_main_effect_nets
        ).to(device)
        
        model, _ = train(
            model, 
            data_loaders, 
            device=device, 
            learning_rate=1e-2, 
            l1_const=5e-5, 
            verbose=False
        )
        
        # Get interaction metrics
        weights = get_weights(model)
        anyorder_interactions = get_interactions(weights, one_indexed=True)
        pairwise_interactions = get_interactions(weights, pairwise=True, one_indexed=True)
        
        # Calculate performance metrics
        metrics = {
            'auc': get_pairwise_auc(pairwise_interactions, ground_truth),
            'r_precision': get_anyorder_R_precision(anyorder_interactions, ground_truth),
            'n_detected': len(anyorder_interactions)
        }
        
        metrics_results[n_clones] = metrics
        print(f"n_clones={n_clones}: AUC={metrics['auc']:.2f}, R-Prec={metrics['r_precision']:.2f}")
    
    return metrics_results

def generate_multicollinear_data(n_samples, n_features, n_clones, exact_clones=True, noise=0.1):
    """Generate data with multicollinear features and ground truth interactions"""
    # Base features
    X1 = np.random.normal(size=(n_samples, 1))
    X2 = np.random.normal(size=(n_samples, 1))
    
    # Create clones/correlated features
    if exact_clones:
        clones = np.repeat(X1, n_clones, axis=1)
    else:  # Create correlated features
        clones = np.zeros((n_samples, n_clones))
        for i in range(n_clones):
            clones[:, i] = X1.flatten() + np.random.normal(scale=noise, size=n_samples)
    
    # Add noise features to reach total feature count
    remaining = n_features - n_clones - 1
    noise_features = np.random.normal(size=(n_samples, remaining))
    
    # Combine all features
    X = np.hstack([clones, X2, noise_features])
    
    # Create target with interaction term
    Y = X1 * X2 + np.random.normal(scale=0.1, size=n_samples)
    
    # Ground truth interactions (X1 clones with X2)
    ground_truth = [(i, n_clones) for i in range(n_clones)]  # X2 is at index n_clones
    
    return X, Y.reshape(-1, 1), ground_truth

def run_analysis():
    # Run for both exact and correlated clones
    exact_metrics = multicollinearity_analysis(exact_clones=True)
    corr_metrics = multicollinearity_analysis(exact_clones=False, noise_level=0.1)
    
    # Plot results using your existing visualization
    plot_metrics_mult({
        'Exact Clones': exact_metrics,
        'Correlated Clones': corr_metrics
    })
    
    return exact_metrics, corr_metrics