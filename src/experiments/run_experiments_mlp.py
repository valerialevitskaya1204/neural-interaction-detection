from src.datasets.synthetic_datasets import F1, F2, F3, F4, F5, F6, F7, F8, F9, F10, F11, F12
from src.neural_detection.multilayer_perceptron import MLP, train, get_weights, get_interactions
from src.utils import preprocess_data, get_anyorder_R_precision, get_pairwise_auc, print_rankings, sanitize_tensor
from src.plots.draw_smth import draw_heatmap, plot_metrics
import numpy as np
import torch
import os
import matplotlib.pyplot as plt

use_main_effect_nets = False
num_samples = 30000
num_features = 10
np.random.seed(52)
functions = [F12, F1, F11, F2, F3, F4, F5, F6, F7, F8, F9, F10, F11, F1] 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def repeat_result_from_paper():
    """Generate model results for all functions in a single run"""
    results = {}
    
    for func in functions:
        func_name = func.__name__
        results[func_name] = {}
        if func_name not in ["F11", "F12"]:
            X = np.random.uniform(low=-1, high=1, size=(num_samples, num_features))
        else:
            X = np.random.normal(loc=0, scale=1, size=(num_samples, num_features))
        X = torch.Tensor(X)
        Y, ground_truth = func(X)
        Y = sanitize_tensor(Y)
        data_loaders = preprocess_data(
            X, Y, 
            valid_size=10000, 
            test_size=10000, 
            std_scale=True, 
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
        
        results[func_name]['ground_truth'] = ground_truth
            
        results[func_name]['weights'] = get_weights(model)
    return results

def get_results():
    """Calculate interaction metrics for all results"""
    results = repeat_result_from_paper()
    metrics = {}
    
    for func_name in results:
        model_weights = results[func_name]['weights']
        ground_truth = results[func_name]['ground_truth']
        
        anyorder_interactions = get_interactions(model_weights, one_indexed=True)
        pairwise_interactions = get_interactions(model_weights, pairwise=True, one_indexed=True)
        draw_heatmap(pairwise_interactions, func_name)


        if func_name not in ["F11", "F12"]:
            auc = get_pairwise_auc(pairwise_interactions, ground_truth)
            r_prec = get_anyorder_R_precision(anyorder_interactions, ground_truth)

            metrics[func_name] = {
                'pairwise_interactions': pairwise_interactions,
                'anyorder_interactions': anyorder_interactions,
                'auc': auc,
                'r_precision': r_prec
            }
            plot_metrics(metrics)
            print("Pairwise AUC", auc, ", Any-order R-Precision", r_prec)
        else:
            metrics[func_name] = {
                'pairwise_interactions': pairwise_interactions,
                'anyorder_interactions': anyorder_interactions,
            }

        
        print(
            print_rankings(pairwise_interactions, anyorder_interactions, top_k=10, spacing=14)
            )
            
    return metrics, results



