"""
Neural Interaction Detection Reproduction Script
"""

# Import synthetic dataset functions
from src.datasets.synthetic_datasets import (
    generate_synthetic_data,
    F1, F2, F3, F4, F5, F6, F7, F8, F9, F10
)

# Import MLP-related classes/functions
from src.neural_detection.multilayer_perceptron import (
    MLP, MLP_Cutoff, MLP_M, train, get_weights
)

# Import real-world dataset loader
from src.datasets.realworld_datasets import load_real_dataset

# Import interaction detection utilities
from src.neural_detection.neural_interaction_detection import (
    get_interactions, prune_redundant_interactions
)

# Import general utilities
from src.utils import (
    preprocess_data, get_anyorder_R_precision, convert_to_torch_loaders
)

from typing import Dict
import logging
import numpy as np
import torch
import os
import time
import matplotlib.pyplot as plt


# --------------------------
# Configuration
# --------------------------
CONFIG = {
    "seed": 42,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "synthetic": {
        "functions": [F1, F2, F3, F4, F5, F6, F7, F8, F9, F10],
        "ground_truth" : {
            F1: [{0,1,2}, {3}, {4,5}, {6,7,8,9}, {1,6}],
            F2: [{0,1,2}, {3}, {2,4}, {6,7,8,9}, {1,6}],
            F3: [{0,1}, {1,2}, {3,4,6,7}, {3}, {9}],
            F4: [{0,1}, {1,2}, {3,4,6,7}, {3}, {9}, {0,3}],
            F5: [{0,1,2}, {3,4}, {5,6}, {7,8,9}],
            F6: [{0,1}, {2,3}, {4,5,7}, {7,8,9}],
            F7: [{0,1}, {2,3,5}, {3,4,5,6,7}, {6,8}],
            F8: [{0,1}, {2,4,5}, {2,3,4,6}, {6,7,8}, {9}],
            F9: [{0,1,2,3,4}, {4,5}, {5,6,7}, {8,9}],
            F10: [{0,1}, {2,4,6}, {3,4}, {6,8}]
        },
        "trials": 10,
        "l1_range": [5e-6, 1e-5, 5e-5, 1e-4, 5e-4],
        "data_points": 30000,
        "batch_size": 100
    },
    "real_world": {
        "datasets": ["cal_housing", "bike_sharing", "higgs_boson", "letter"],
        "batch_size": 256,
        "l1_const": 5e-5
    },
    "paths": {
        "logs": "./logs",
        "results": "./results",
        "plots": "./plots"
    },
    "training": {
        "epochs": 100,
        "patience": 5,
        "learning_rate": 0.001
    }
}

# --------------------------
# Setup
# --------------------------
def setup_environment(config: Dict):
    """Initialize reproduction environment"""
    # Create directories
    for path in config["paths"].values():
        os.makedirs(path, exist_ok=True)
    
    # Set seeds
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config["seed"])

# --------------------------
# Synthetic Experiments
# --------------------------

def run_synthetic_trial(func, config: Dict) -> Dict:
    """Run trial with all three models"""
    try:
        # Data preparation
        X, y = generate_synthetic_data(func, config["synthetic"]["data_points"])
        Xd, Yd = preprocess_data(X, y, valid_size=10000, test_size=10000, std_scale=True)
        data_loaders = convert_to_torch_loaders(Xd, Yd, config["synthetic"]["batch_size"])

        results = {}
        
        # =======================================================================
        # 1. Train MLP and MLP-M with L1 regularization
        # =======================================================================
        for model_class in [MLP, MLP_M]:
            model_name = model_class.__name__
            model_results = []
            
            # Hyperparameter tuning
            best_l1, best_val_loss = None, float('inf')
            for l1 in config["synthetic"]["l1_range"]:
                model = model_class(
                    num_features=10,
                    hidden_units=[140, 100, 60, 20],
                    main_effect_net_units=[10, 10, 10] if model_class == MLP_M else None
                ).to(config["device"])
                
                _, val_loss = train(
                    model, data_loaders,
                    l1_const=l1,
                    nepochs=config["training"]["epochs"],
                    device=config["device"]
                )
                if val_loss < best_val_loss:
                    best_l1, best_val_loss = l1, val_loss

            # Final training
            model = model_class(
                num_features=10,
                hidden_units=[140, 100, 60, 20],
                main_effect_net_units=[10, 10, 10] if model_class == MLP_M else None
            ).to(config["device"])
            
            trained_model, _ = train(
                model, data_loaders,
                l1_const=best_l1,
                nepochs=config["training"]["epochs"],
                device=config["device"]
            )

            # Detect interactions
            weights = get_weights(trained_model)
            interactions = get_interactions(weights)
            
            # Store results
            model_results.append({
                "l1": best_l1,
                "interactions": interactions,
                "model_state": trained_model.state_dict()
            })
            
            results[model_name] = model_results[-1]

        # =======================================================================
        # 2. Train MLP-Cutoff with L2 regularization
        # =======================================================================
        # Get top interactions from MLP-M as candidates
        top_interactions = [inter[0] for inter in results["MLP_M"]["interactions"][:20]]
        
        cutoff_model = MLP_Cutoff(
            num_features=10,
            interaction_list=top_interactions,
            hidden_units=[140, 100, 60, 20]
        ).to(config["device"])
        
        trained_cutoff, _ = train(
            cutoff_model, data_loaders,
            l2_const=1e-4,  # Fixed L2 as per paper
            nepochs=config["training"]["epochs"],
            device=config["device"]
        )
        
        # Evaluate cutoff performance
        cutoff_interactions = prune_redundant_interactions(results["MLP_M"]["interactions"])
        results["MLP_Cutoff"] = {
            "interactions": cutoff_interactions,
            "model_state": trained_cutoff.state_dict()
        }

        # =======================================================================
        # 3. Evaluation against ground truth
        # =======================================================================
        gt_interactions = [set(inter) for inter in config["synthetic"]["ground_truth"][func]]
        
        for model_name in ["MLP", "MLP_M", "MLP_Cutoff"]:
            # Count correct interactions before first false positive
            correct_count = 0
            for inter, _ in results[model_name]["interactions"]:
                inter_set = set(inter)
                if any(inter_set == gt for gt in gt_interactions):
                    correct_count += 1
                else:
                    break  # Stop at first false positive
                    
            results[model_name]["correct_top"] = correct_count
            results[model_name]["r_precision"] = get_anyorder_R_precision(
                results[model_name]["interactions"], gt_interactions
            )

        return results

    except Exception as e:
        return {"status": "failed", "error": str(e)}

def run_realworld_experiment(dataset: str, config: Dict) -> Dict:
    """Real-world experiment with full model suite"""
    try:
        X, y = load_real_dataset(dataset)
        Xd, Yd = preprocess_data(X, y, valid_size=0.1, test_size=0.1, std_scale=True)
        data_loaders = convert_to_torch_loaders(Xd, Yd, config["real_world"]["batch_size"])
        
        results = {}
        num_features = X.shape[1]

        # 1. Train MLP-M
        mlp_m = MLP_M(
            num_features,
            [140, 100, 60, 20],
            main_effect_net_units=[10, 10, 10]
        ).to(config["device"])
        
        trained_mlp_m, _ = train(
            mlp_m, data_loaders,
            l1_const=config["real_world"]["l1_const"],
            nepochs=config["training"]["epochs"],
            device=config["device"]
        )
        weights = get_weights(trained_mlp_m)
        interactions = get_interactions(weights)
        results["MLP_M"] = {"interactions": interactions}

        # 2. Train MLP-Cutoff
        top_interactions = [inter[0] for inter in interactions[:20]]
        cutoff_model = MLP_Cutoff(
            num_features,
            interaction_list=top_interactions,
            hidden_units=[140, 100, 60, 20]
        ).to(config["device"])
        
        trained_cutoff, _ = train(
            cutoff_model, data_loaders,
            l2_const=1e-4,
            nepochs=config["training"]["epochs"],
            device=config["device"]
        )
        
        # Get interactions FROM CUTOFF MODEL
        cutoff_weights = get_weights(trained_cutoff)
        cutoff_interactions = get_interactions(cutoff_weights)
        
        results["MLP_Cutoff"] = {
            "interactions": prune_redundant_interactions(cutoff_interactions),  
            "model_state": trained_cutoff.state_dict()
        }

    except Exception as e:
        return {"status": "failed", "error": str(e)}

# ... [Keep visualization and main execution sections] ...

# --------------------------
# Visualization
# --------------------------
def plot_results(results: Dict, config: Dict):
    """Generate result visualizations"""
    # Synthetic Results
    plt.figure(figsize=(12, 6))
    r_precision = [
        np.mean([t["r_precision"] for t in results["synthetic"][f.__name__] if t["status"] == "success"])
        for f in config["synthetic"]["functions"]
    ]
    plt.bar([f.__name__ for f in config["synthetic"]["functions"]], r_precision)
    plt.title("R-Precision Across Synthetic Functions")
    plt.ylabel("R-Precision")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(config["paths"]["plots"], "synthetic_results.png"))
    
    # Real-World Interactions
    plt.figure(figsize=(12, 8))
    for i, dataset in enumerate(config["real_world"]["datasets"]):
        plt.subplot(2, 2, i+1)
        strengths = [s for _, s in results["real_world"][dataset]["interactions"][:10]]
        plt.barh(range(10), strengths[::-1])
        plt.title(f"Top Interactions - {dataset}")
        plt.yticks(range(10), [str(t[0]) for t in results["real_world"][dataset]["interactions"][:10]][::-1])
    plt.tight_layout()
    plt.savefig(os.path.join(config["paths"]["plots"], "realworld_interactions.png"))

# --------------------------
# Main Execution
# --------------------------
if __name__ == "__main__":
    setup_environment(CONFIG)
    results = {"synthetic": {}, "real_world": {}}
    
    # Synthetic Experiments
    print("Starting synthetic experiments")
    for func in CONFIG["synthetic"]["functions"]:
        func_results = []
        for trial in range(CONFIG["synthetic"]["trials"]):
            try:
                print(f"Running {func.__name__} trial {trial+1}/{CONFIG['synthetic']['trials']}")
                result = run_synthetic_trial(func, CONFIG)
                func_results.append(result)
            except Exception as e:
                print(f"Trial failed for {func.__name__}: {str(e)}")
            continue

        results["synthetic"][func.__name__] = func_results
    
    # # Real-World Experiments
    print("\nStarting real-world experiments")
    for dataset in CONFIG["real_world"]["datasets"]:
        print(f"Processing {dataset}")
        results["real_world"][dataset] = run_realworld_experiment(dataset, CONFIG)

    torch.save(results, os.path.join(CONFIG["paths"]["results"], "full_results.pt"))
    plot_results(results, CONFIG)