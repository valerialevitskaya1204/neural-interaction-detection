import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from pathlib import Path
from matplotlib.colors import LogNorm

def draw_heatmap(pairwise_interactions, func_name,  num_features=10, save_dir="../src/plots/synthetic_data_heatmap"):
    """Visualize pairwise interactions as a heatmap."""
    path = Path(save_dir)
    path.mkdir(parents=True, exist_ok=True)
    matrix = np.zeros((num_features, num_features))
    
    for (i, j), value in pairwise_interactions:
        i_idx = int(i) - 1
        j_idx = int(j) - 1
        matrix[i_idx, j_idx] = value
        matrix[j_idx, i_idx] = value 

    labels = [f'X{i+1}' for i in range(num_features)]

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".2f",
        cmap='coolwarm',
        xticklabels=labels,
        yticklabels=labels,
        vmin=0,
        vmax=1
    )
    
    plt.title(f'Pairwise Interactions for {func_name}')
    plt.xlabel('Features')
    plt.ylabel('Features')
    plt.tight_layout()
    
    plt.savefig(os.path.join(f'{save_dir}/{func_name}_heatmap.png'))
    plt.close()



def plot_metrics(metrics_dict, task="synth"):
    if task == "synth":
        save_path= "src/plots/synthetic_data_metrics"
    else:
        save_path= "src/plots/real_data_metrics"
        
    path = Path(save_path)
    path.mkdir(parents=True, exist_ok=True)
    """Plot AUC and R-precision metrics for all functions"""
    plt.figure(figsize=(10, 6))
    
    func_names = list(metrics_dict.keys())
    auc_scores = [metrics_dict[name]['auc'] for name in func_names]
    r_prec_scores = [metrics_dict[name]['r_precision'] for name in func_names]
    
    x = np.arange(len(func_names))
    width = 0.35

    auc_bars = plt.bar(x - width/2, auc_scores, width, label='AUC', color='royalblue')
    r_prec_bars = plt.bar(x + width/2, r_prec_scores, width, label='R-Precision', color='lightcoral')
    
    plt.ylabel('Scores')
    plt.title('Model Performance Metrics by Function')
    plt.xticks(x, func_names)
    plt.ylim(0, 1.1)
    plt.legend()
    
    
    add_labels(auc_bars)
    add_labels(r_prec_bars)
    
    plt.tight_layout()
    plt.savefig(path)
    plt.show()
    plt.close()


def add_labels(bars):
    for bar in bars:
        height = bar.get_height()
        plt.annotate(f'{height:.2f}',
                         xy=(bar.get_x() + bar.get_width() / 2, height),
                         xytext=(0, 3),  # 3 points vertical offset
                         textcoords="offset points",
                         ha='center', va='bottom')
        
        
def min_max_scale(data):
    min_vals = data.min(axis=0)  # Minimum of each column
    max_vals = data.max(axis=0)  # Maximum of each column
    range_vals = max_vals - min_vals  # Range of each column
    
    # Handle case where range is zero (same values across the column)
    range_vals[range_vals == 0] = 1  # Avoid division by zero by setting the range to 1 where max = min
    
    scaled_data = (data - min_vals) / range_vals
    return scaled_data
        

def draw_heatmap_real_data(pairwise_interactions, dataset_name, num_feat, feature_names=None, save_dir="/workspace/kate/neural-interaction-detection/src/plots/real_data_heatmap"):
    """Visualize pairwise interactions for real datasets without ground truth"""
    path = Path(save_dir)
    path.mkdir(parents=True, exist_ok=True)
    
    num_features = num_feat
    matrix = np.zeros((num_features, num_features))
    mask = np.triu(np.ones_like(matrix, dtype=bool), k=0)
    
    for (i, j), value in pairwise_interactions:
        i_idx = int(i) - 1  
        j_idx = int(j) - 1
        matrix[i_idx, j_idx] = value
        matrix[j_idx, i_idx] = value 
        
    # matrix = min_max_scale(matrix)

    labels = feature_names if feature_names else [f'{i+1}' for i in range(num_features)]
    print()

    if dataset_name == "seoul_bikes":

        plt.figure(figsize=(12, 10))
        ax = sns.heatmap(
            matrix,
            mask=mask,
            vmin=matrix.min(),
            vmax=matrix.max(),
            annot=False,
            fmt=".2f",
            square=True,
            norm=LogNorm(),
            cmap='coolwarm',
            xticklabels=labels,
            yticklabels=labels,
            cbar=False,
        )
    else:
        plt.figure(figsize=(12, 10))
        ax = sns.heatmap(
            matrix,
            mask=mask,
            vmin=matrix.min(),
            vmax=matrix.max(),
            annot=False,
            fmt=".2f",
            square=True,
            # norm=LogNorm(),
            cmap='coolwarm',
            xticklabels=labels,
            yticklabels=labels,
            cbar=False,
        )

    # Increase font sizes
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=18)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=18)

    plt.tight_layout()
    
    path_save = Path(f'{save_dir}/{dataset_name}_heatmap.png')
    
    plt.savefig(path_save)
    plt.close()