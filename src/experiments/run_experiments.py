from src.datasets.synthetic_datasets import generate_synthetic_data, F1
from src.neural_detection.multilayer_perceptron import MLP, train, get_weights
from src.datasets.realworld_datasets import load_real_dataset
from src.neural_detection.neural_interaction_detection import get_interactions, prune_redundant_interactions
from src.utils import preprocess_data, get_anyorder_R_precision, convert_to_torch_loaders, get_pairwise_auc
from typing import Dict
import numpy as np
import torch
import os

import matplotlib.pyplot as plt


CONFIG = {
    "seed": 42,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "synthetic": {
        "functions": [F1],
        "ground_truth": {F1: [{0,1,2}, {3}, {4,5}, {6,7,8,9}, {1,6}]},
        "trials": 2,
        "l1_range": [5e-6],
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
        "epochs": 1,
        "patience": 5,
        "learning_rate": 0.001
    }
}

def setup_environment(config: Dict):
    for path in config["paths"].values():
        os.makedirs(path, exist_ok=True)
    np.random.seed(config["seed"])


def run_synthetic_trial(func, config: Dict) -> Dict:
    try:
        os.makedirs("saved_models_new", exist_ok=True)
        X, y = generate_synthetic_data(func, config["synthetic"]["data_points"])
        Xd, Yd = preprocess_data(X, y, valid_size=10000, test_size=10000, std_scale=True)
        data_loaders = convert_to_torch_loaders(Xd, Yd, config["synthetic"]["batch_size"])

        results = {}
        
        model_configs = [
            {"name": "MLP", "use_me": False},
            {"name": "MLP_M", "use_me": True, "me_units": [10, 10, 10]}
        ]
        
        for model_config in model_configs:
            model_name = model_config["name"]
            use_me = model_config["use_me"]
            me_units = model_config.get("me_units", None)
            model_results = []
            
            best_l1, best_val_loss = None, float('inf')
            for l1 in config["synthetic"]["l1_range"]:
                model = MLP(
                    num_features=10,
                    hidden_units=[140, 100, 60, 20],
                    use_main_effect_nets=use_me,
                    main_effect_net_units=me_units
                ).to(config["device"])
                
                _, val_loss = train(
                    model, data_loaders,
                    l1_const=l1,
                    nepochs=config["training"]["epochs"],
                    device=config["device"])
                if val_loss < best_val_loss:
                    best_l1, best_val_loss = l1, val_loss

            model = MLP(
                num_features=10,
                hidden_units=[140, 100, 60, 20],
                use_main_effect_nets=use_me,
                main_effect_net_units=me_units
            ).to(config["device"])
            
            trained_model, _ = train(
                model, data_loaders,
                l1_const=best_l1,
                nepochs=config["training"]["epochs"],
                device=config["device"])
            
            mlp_path = os.path.join("saved_models", f"{func.__name__}_{model_name}.pth")
            torch.save({'model_state_dict': trained_model.state_dict()}, mlp_path)

            weights = get_weights(trained_model)
            anyorder_interactions = get_interactions(weights, one_indexed=True)
            print("interactions", anyorder_interactions)
            pairwise_interactions = get_interactions(weights, pairwise=True, one_indexed=True)
            print("interactions_pairwise", pairwise_interactions)
            model_results.append({
                "l1": best_l1,
                "interactions_anyorder": anyorder_interactions,
                "interactions_pairwise": pairwise_interactions
            })
            
            results[model_name] = model_results[-1]

        top_interactions = [inter[0] for inter in results["MLP_M"]["interactions_anyorder"][:20]]
        
        cutoff_model = MLP_Cutoff(
            num_features=10,
            interaction_list=top_interactions,
            hidden_units=[140, 100, 60, 20]
        ).to(config["device"])
        
        trained_cutoff, _ = train(
            cutoff_model, data_loaders,
            l2_const=1e-4,
            nepochs=config["training"]["epochs"],
            device=config["device"])
        mlp_cutoff_path = os.path.join("saved_models", f"{func.__name__}_mlp_cutoff.pth")
        torch.save({'model_state_dict': trained_cutoff.state_dict()}, mlp_cutoff_path)
        
        cutoff_interactions = prune_redundant_interactions(results["MLP_M"]["interactions_anyorder"])
        results["MLP_Cutoff"] = {"interactions": cutoff_interactions, "model_state": trained_cutoff.state_dict()}
        gt_interactions = [set(inter) for inter in config["synthetic"]["ground_truth"][func]]
        
        for model_name in ["MLP", "MLP_M", "MLP_Cutoff"]:        
            results[model_name]["correct_top"] = get_pairwise_auc(results[model_name]["interactions_pairwise"], gt_interactions)
            results[model_name]["r_precision"] = get_anyorder_R_precision(results[model_name]["interactions_anyorder"], gt_interactions)

        return results

    except Exception as e:
        return {"status": "failed", "error": str(e)}

def run_realworld_experiment(dataset: str, config: Dict) -> Dict:
    try:
        os.makedirs("saved_models_new", exist_ok=True)
        X, y = load_real_dataset(dataset)
        Xd, Yd = preprocess_data(X, y, valid_size=0.1, test_size=0.1, std_scale=True)
        data_loaders = convert_to_torch_loaders(Xd, Yd, config["real_world"]["batch_size"])
        
        results = {}
        num_features = X.shape[1]

        mlp_m = MLP(
            num_features,
            [140, 100, 60, 20],
            use_main_effect_nets=True,
            main_effect_net_units=[10, 10, 10]
        ).to(config["device"])
        
        trained_mlp_m, _ = train(
            mlp_m, data_loaders,
            l1_const=config["real_world"]["l1_const"],
            nepochs=config["training"]["epochs"],
            device=config["device"])
        mlp_m_path = os.path.join("saved_models", f"{dataset}_mlp_m.pth")
        torch.save({'model_state_dict': trained_mlp_m.state_dict()}, mlp_m_path)

        weights = get_weights(trained_mlp_m)
        interactions = get_interactions(weights, one_indexed=True)
        pairwise_interactions = get_interactions(weights, pairwise=True, one_indexed=True)
        results["MLP_M"] = {"interactions_anyorder": interactions, "interactions_pairwise": pairwise_interactions}

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
            device=config["device"])
        mlp_cutoff_path = os.path.join("saved_models", f"{dataset}_mlp_cutoff.pth")
        torch.save({'model_state_dict': trained_cutoff.state_dict()}, mlp_cutoff_path)
        cutoff_weights = get_weights(trained_cutoff)
        cutoff_interactions = get_interactions(cutoff_weights)
        
        results["MLP_Cutoff"] = {
            "interactions": prune_redundant_interactions(cutoff_interactions),  
            "model_state": trained_cutoff.state_dict()
        }

    except Exception as e:
        return {"status": "failed", "error": str(e)}

def plot_results(results: Dict, config: Dict):
    plt.figure(figsize=(15, 8))
    models = ["MLP", "MLP_M", "MLP_Cutoff"]
    function_keys = [f.__name__ for f in config["synthetic"]["functions"]]

    data = {model: [] for model in models}
    for f_key in function_keys:
        for model in models:
            r_precs = [run[model]["r_precision"] for run in results["synthetic"][f_key]]
            data[model].append(np.mean(r_precs))
    
    x = np.arange(len(function_keys))
    width = 0.25
    for i, model in enumerate(models):
        plt.bar(x + i*width, data[model], width, label=model)
    
    plt.title("R-Precision Across Synthetic Functions by Model")
    plt.ylabel("R-Precision")
    plt.xlabel("Functions")
    plt.xticks(x + width, function_keys, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(config["paths"]["plots"], "synthetic_results.png"))
    plt.close()

    datasets = config["real_world"]["datasets"]
    interaction_types = ["interactions_pairwise", "interactions_anyorder"]
    
    plt.figure(figsize=(14, 6 * len(datasets)))
    for d_idx, dataset in enumerate(datasets):
        for i_idx, i_type in enumerate(interaction_types):
            plt.subplot(len(datasets), 2, d_idx*2 + i_idx + 1)
            interactions = results["real_world"][dataset][i_type][:10]
            strengths = [s for _, s in interactions]
            
            plt.barh(range(10), strengths[::-1], color='skyblue')
            plt.title(f"{dataset} - {i_type.split('_')[-1].title()}")
            plt.yticks(range(10), [str(t[0]) for t in interactions][::-1])
            plt.gca().invert_yaxis() 

    plt.tight_layout()
    plt.savefig(os.path.join(config["paths"]["plots"], "realworld_interactions.png"))
    plt.close()