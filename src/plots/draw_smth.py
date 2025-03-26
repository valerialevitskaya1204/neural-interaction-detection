import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

def min_max_scale(data):
    min_vals = data.min(axis=0)  # Minimum of each column
    max_vals = data.max(axis=0)  # Maximum of each column
    range_vals = max_vals - min_vals  # Range of each column
    
    # Handle case where range is zero (same values across the column)
    range_vals[range_vals == 0] = 1  # Avoid division by zero by setting the range to 1 where max = min
    
    scaled_data = (data - min_vals) / range_vals
    return scaled_data


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

    matrix = min_max_scale(matrix)

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
    
    # Filter out F11 and F12 if they exist
    func_names = [name for name in metrics_dict.keys() if name not in ["F11", "F12"]]
    
    # Get scores, filtering out None values but keeping track of which functions have valid scores
    auc_scores = []
    r_prec_scores = []
    valid_func_names = []
    
    for name in func_names:
        auc = metrics_dict[name].get('auc')
        r_prec = metrics_dict[name].get('r_precision')
        if auc is not None or r_prec is not None:
            valid_func_names.append(name)
            auc_scores.append(auc)
            r_prec_scores.append(r_prec)
    
    x = np.arange(len(valid_func_names))
    width = 0.35
    
    # Plot AUC scores if available
    if any(score is not None for score in auc_scores):
        auc_bars = plt.bar(x - width/2, [s if s is not None else 0 for s in auc_scores], 
                          width, label='AUC', color='royalblue')
    else:
        auc_bars = None
    
    # Plot R-precision scores if available
    if any(score is not None for score in r_prec_scores):
        r_prec_bars = plt.bar(x + width/2, [s if s is not None else 0 for s in r_prec_scores], 
                             width, label='R-Precision', color='lightcoral')
    else:
        r_prec_bars = None
    
    plt.ylabel('Scores')
    plt.title('Model Performance Metrics by Function')
    plt.xticks(x, valid_func_names)
    plt.ylim(0, 1.1)
    plt.legend()
    
    if auc_bars:
        add_labels(auc_bars)
    if r_prec_bars:
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



def plot_metrics_mult(metrics_dict, save_path=None):
    """
    Plot AUC and R-Precision metrics for multicollinearity analysis
    
    Args:
        metrics_dict: Dictionary containing results for different conditions
            Format: {'Exact Clones': {n_clones: metrics}, 'Correlated Clones': {...}}
        save_path: Optional path to save the figure
    """
    plt.figure(figsize=(14, 6))
    
    # Create subplots
    ax1 = plt.subplot(121)
    ax2 = plt.subplot(122)
    
    # Plot settings
    colors = {'Exact Clones': 'blue', 'Correlated Clones': 'red'}
    markers = {'Exact Clones': 'o', 'Correlated Clones': 's'}
    
    # Plot each metric type
    for condition, metrics in metrics_dict.items():
        n_clones = sorted(metrics.keys())
        aucs = [metrics[n]['auc'] for n in n_clones]
        r_precs = [metrics[n]['r_precision'] for n in n_clones]
        
        ax1.plot(n_clones, aucs, 
                label=condition, 
                color=colors[condition],
                marker=markers[condition],
                linestyle='--')
        
        ax2.plot(n_clones, r_precs,
                label=condition,
                color=colors[condition],
                marker=markers[condition],
                linestyle='--')
    
    # Configure plots
    ax1.set_title('Pairwise Interaction Detection (AUC)')
    ax1.set_xlabel('Number of Clones (n)')
    ax1.set_ylabel('AUC Score')
    ax1.grid(True)
    ax1.legend()
    
    ax2.set_title('Any-Order Interaction Detection (R-Precision)')
    ax2.set_xlabel('Number of Clones (n)')
    ax2.set_ylabel('R-Precision Score')
    ax2.grid(True)
    ax2.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()