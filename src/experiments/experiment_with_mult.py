from src.neural_detection.multilayer_perceptron import MLP, train, get_weights, get_interactions
from src.utils import preprocess_data, get_anyorder_R_precision, get_pairwise_auc, sanitize_tensor, get_strength
from src.plots.draw_smth import plot_metrics_mult, plot_str_against_n
import numpy as np
import torch
import matplotlib.pyplot as plt

use_main_effect_nets = True
num_samples = 30000
num_features = 61
np.random.seed(52)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def multicollinearity_analysis(n_clones_list=list(range(5, 30, 5)), exact_clones=True, noise_level=0.1):
    """Study interaction detection with cloned/correlated features"""
    metrics_results = {}

    p_strengths = []
    a_strengths = []
    
    for n_clones in n_clones_list:
        X, Y, ground_truth = generate_multicollinear_data(
            num_samples, 
            num_features,
            n_clones,
            exact_clones,
            noise_level
        )
        X = torch.Tensor(X)
        Y = sanitize_tensor(torch.Tensor(Y))
        data_loaders = preprocess_data(
            X, Y, 
            valid_size=10000, 
            test_size=10000, 
            std_scale=False, 
            get_torch_loaders=True
        )
        
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
            verbose=True
        )
        
        weights = get_weights(model)
        anyorder_interactions = get_interactions(weights, one_indexed=True)
        pairwise_interactions = get_interactions(weights, pairwise=True, one_indexed=True)

        print(anyorder_interactions, "ANYORDER")

        p_inter, p_strength, a_strength, a_inter = get_strength(pairwise_interactions, anyorder_interactions)

        p_strengths.append((p_inter, p_strength))
        a_strengths.append((a_strength, a_inter))

        auc = get_pairwise_auc(pairwise_interactions, ground_truth)
        metrics = {
            'auc': auc if auc is not None else 0.5,
            'r_precision': get_anyorder_R_precision(anyorder_interactions, ground_truth),
            'n_detected': len(anyorder_interactions),
            'p': p_strength,
            'a': a_strengths,
        }
        
        metrics_results[n_clones] = metrics
        print(f"n_clones={n_clones}: AUC={metrics['auc']:.2f}, R-Prec={metrics['r_precision']:.2f}")
    # if exact_clones:
    #     plot_str_against_n(n_clones_list, p_strengths, a_strengths, task="clons")
    # else:
    #     plot_str_against_n(n_clones_list, p_strengths, a_strengths)
    return metrics_results


def generate_multicollinear_data(n_samples, n_features, n_clones, exact_clones=True, noise=0.1):
    """Generate data with cloned features for both X1 and X2, and ground truth interactions.
    
    The features:
      - X1: base feature (to be cloned to X11, X12, ..., X1n)
      - X2: base feature (to be cloned to X21, X22, ..., X2n)
      - noise_features: additional features if needed to reach n_features
    
    The target is:
        Y = X11*X21 + X12*X22 + ... + X1n*X2n
    """
    if n_features < 2 * n_clones:
        raise ValueError("n_features must be at least 2*n_clones")
    
    X1 = np.random.normal(size=(n_samples, 1))
    X2 = np.random.normal(size=(n_samples, 1))
    
    if exact_clones:
        clones_x1 = np.repeat(X1, n_clones, axis=1)  # X11, X12, ..., X1n
        clones_x2 = np.repeat(X2, n_clones, axis=1)  # X21, X22, ..., X2n
    else:
        clones_x1 = np.hstack([
            X1 + np.random.normal(scale=noise, size=(n_samples, 1))
            for _ in range(n_clones)
        ])
        clones_x2 = np.hstack([
            X2 + np.random.normal(scale=noise, size=(n_samples, 1))
            for _ in range(n_clones)
        ])

    Y = np.sum(clones_x1 * clones_x2, axis=1) + np.random.normal(scale=noise, size=n_samples)
    
    remaining = n_features - 2 * n_clones
    if remaining > 0:
        noise_features = np.random.normal(size=(n_samples, remaining))
        X = np.hstack([clones_x1, clones_x2, noise_features])
    else:
        X = np.hstack([clones_x1, clones_x2])

    ground_truth_1 = [{i, i + n_clones} for i in range(1, n_clones)]
    
    return X, Y, ground_truth_1


def run_analysis():
    # Run for both exact and correlated clones
    exact_metrics = multicollinearity_analysis(exact_clones=True)
    corr_metrics = multicollinearity_analysis(exact_clones=False, noise_level=0.1)

    plot_metrics_mult({
        'Exact Clones': exact_metrics,
        'Correlated Clones': corr_metrics
                }, "src/plots/mult_corr.png")
    return exact_metrics, corr_metrics




