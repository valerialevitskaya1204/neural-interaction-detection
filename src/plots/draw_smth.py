import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

def draw_heatmap(pairwise_interactions, func_name,  num_features=10, save_dir="src/plots"):
    """Visualize pairwise interactions as a heatmap."""
    os.makedirs(save_dir, exist_ok=True)
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
    
    plt.savefig(os.path.join(f'src/plots/{func_name}_heatmap.png'))
    plt.close()



def plot_metrics(metrics_dict, save_path="src/plots/metrics_plot.png"):
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
    plt.savefig(save_path)
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
        

def draw_heatmap_real_data(pairwise_interactions, dataset_name, feature_names=None):
    """Visualize pairwise interactions for real datasets without ground truth"""
    os.makedirs("real_data_heatmaps", exist_ok=True)
    
    num_features = len(pairwise_interactions)
    matrix = np.zeros((num_features, num_features))
    
    for (i, j), value in pairwise_interactions:
        i_idx = int(i) - 1  
        j_idx = int(j) - 1
        matrix[i_idx, j_idx] = value
        matrix[j_idx, i_idx] = value 

    labels = feature_names if feature_names else [f'Feature {i+1}' for i in range(num_features)]

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
    
    plt.title(f'Learned Pairwise Interactions for {dataset_name}')
    plt.xlabel('Features')
    plt.ylabel('Features')
    plt.tight_layout()
    
    plt.savefig(f'src/plots/real_data_heatmaps/{dataset_name}_heatmap.png')
    plt.close()