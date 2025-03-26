from src.datasets.synthetic_datasets import F1, F2, F3, F4, F5, F6, F7, F8, F9, F10
from src.datasets.realworld_datasets import load_real_dataset
from src.neural_detection.multilayer_perceptron import MLP, train, get_weights, get_interactions
from src.utils import preprocess_data, get_anyorder_R_precision, get_pairwise_auc, print_rankings
from src.plots.draw_smth import draw_heatmap, plot_metrics, draw_heatmap_real_data
from pathlib import Path

import numpy as np
import torch
import pickle
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

use_main_effect_nets = True
num_samples = 30000
num_features = 10
np.random.seed(52)
synth_functions = [F1, F2, F3, F4, F5, F6, F7, F8, F9, F10] 
real_functions = ["cal_housing", "letter", "parkinsons", "images", "robots", "seoul_bikes", "higgs_boson"] 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def sanitize_tensor(tensor, nan_val=0.0, posinf=1e6, neginf=-1e6):
        return torch.nan_to_num(tensor, nan=nan_val, posinf=posinf, neginf=neginf)
    


def repeat_result_from_paper(task='synth', save_dir="model"):
    """Generate model results for all functions in a single run"""
    
    results = {}
    
    if task == "synth":
        path = Path(f"{save_dir}/synth")
        path.mkdir(parents=True, exist_ok=True)
        functions = synth_functions
        
        for func in tqdm(functions):
            func_name = func.__name__
            results[func_name] = {}
            if func_name not in ["F11", "F12"]:
                X = np.random.uniform(low=-1, high=1, size=(num_samples, num_features))
                X = torch.Tensor(X)
                # print(func(X))
                Y, ground_truth = func(X)
                Y = sanitize_tensor(Y)
            else:
                X = np.random.normal(loc=0, scale=1, size=(num_samples, num_features))
                X = torch.Tensor(X)
                Y, ground_truth  = func(X)
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
            
            torch.save(model, f"{save_dir}{func}.pth")
            
            # results[func_name]['weights'] = get_weights(model)
            with open('model_weights.pkl', 'wb') as f:
                pickle.dump(get_weights(model), f)
            results[func_name] = {'ground_truth': ground_truth, "weights":get_weights(model)}
            
            
    elif task == "real":
        functions = real_functions
        path = Path(f"{save_dir}/real")
        path.mkdir(parents=True, exist_ok=True)
        
        for func in functions[2:3]:
            func_name = func
            results[func_name] = {}
            
            X, Y = load_real_dataset(func)
            num_features = X.shape[-1]
            
            data_loaders = preprocess_data(
                X, Y, 
                valid_size=500, 
                test_size=500, 
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
            
            
            torch.save(model, Path(f"{save_dir}/real/{func}.pth"))
            
            with open(f"{save_dir}/real/{func}.pkl", 'wb') as f:
                pickle.dump(get_weights(model), f)
            # results[func_name]['weights'] = get_weights(model)
            results[func_name] = {"weights":get_weights(model)}
        
        
    return results

def get_results(task="synth"):
    """Calculate interaction metrics for all results"""
    
    results = repeat_result_from_paper(task=task)
    # metrics = {}
    # for i in results:
    # with open(f'/workspace/kate/neural-interaction-detection/model/real/cal_housing.pkl', 'rb') as f:
    #     model_weights = pickle.load(f)
    #     print(model_weights)
        
    # anyorder_interactions = get_interactions(model_weights, one_indexed=True)
    # pairwise_interactions = get_interactions(model_weights, pairwise=True, one_indexed=True)
    # print(pairwise_interactions)

    # if task == "synth":
    #     draw_heatmap(pairwise_interactions, func_name)
    # else:
    #     draw_heatmap_real_data(pairwise_interactions, 'cal_housing')
            
        # auc = get_pairwise_auc(pairwise_interactions, ground_truth)
        # r_prec = get_anyorder_R_precision(anyorder_interactions, ground_truth)
        
        # metrics[func_name] = {
        #     'pairwise_interactions': pairwise_interactions,
        #     'anyorder_interactions': anyorder_interactions,
        #     'auc': auc,
        #     'r_precision': r_prec
        # }

        # plot_metrics(metrics, task=task)

        # print("Pairwise AUC", auc, ", Any-order R-Precision", r_prec)
        # print(
        #     print_rankings(pairwise_interactions, anyorder_interactions, top_k=10, spacing=14)
        #     )
            

    



