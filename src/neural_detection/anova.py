from src.datasets.synthetic_datasets import F1, F2, F3, F4, F5, F6, F7, F8, F9, F10
from src.utils import preprocess_data, get_anyorder_R_precision, get_pairwise_auc

import pandas as pd
import itertools
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import ols
import seaborn as sns
import matplotlib.pyplot as plt

import torch
import statsmodels.api as sm
from statsmodels.formula.api import ols

# -------------------------
# Step 1: Define F1 function (your code)
# -------------------------
# def F1(X):
#     X1, X2, X3, X4, X5, X6, X7, X8, X9, X10 = X.transpose(0, 1)

#     part1 = torch.pow(torch.pi, X1 * X2) * torch.sqrt(torch.clamp(2 * X3, min=1e-6))
#     part2 = torch.arcsin(torch.clamp(X4, min=-1 + 1e-6, max=1 - 1e-6))
#     part3 = torch.log(torch.clamp(X3 + X5, min=1e-6))
#     part4 = -(X9 / torch.clamp(X10, min=1e-6)) * torch.sqrt(torch.clamp(X7 / torch.clamp(X8, min=1e-6), min=1e-6))
#     part5 = -X2 * X7

#     result = part1 + part2 + part3 + part4 + part5

#     ground_truth = [{1, 2, 3}, {3, 5}, {7, 8, 9, 10}, {2, 7}]
#     return result, ground_truth

# -------------------------
# Step 2: Generate data and compute target
# -------------------------
torch.manual_seed(42)
num_samples = 30000
num_features = 10

X = torch.rand(num_samples, num_features) * 2 - 1  # Uniform in [-1, 1]
y, ground_truth = F10(X)

# Convert to pandas DataFrame
df = pd.DataFrame(X.numpy(), columns=[f'f{i}' for i in range(1, 11)])
df['y'] = y.numpy()

# -------------------------
# Step 3: Compute ANOVA F-value matrix
# -------------------------
features = [f'f{i}' for i in range(1, 11)]
f_matrix = pd.DataFrame(np.nan, index=features, columns=features)

for f1, f2 in itertools.combinations(features, 2):
    try:
        formula = f"y ~ {f1} + {f2} + {f1}:{f2}"
        model = ols(formula, data=df).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)
        f_val = anova_table.loc[f'{f1}:{f2}', 'F']
        f_matrix.loc[f1, f2] = f_val
        f_matrix.loc[f2, f1] = f_val
    except Exception as e:
        print(f"Skipped ({f1}, {f2}):", e)

np.fill_diagonal(f_matrix.values, 0)

# -------------------------
# Step 4: Done — show result
# -------------------------
print("ANOVA Interaction F-value Matrix:")
print(f_matrix.round(2))

# plt.figure(figsize=(10, 8))
# sns.heatmap(f_matrix, annot=True, fmt=".1f", cmap="viridis", square=True, linewidths=0.5, cbar_kws={'label': 'F-value'})
# plt.title("Pairwise Feature Interaction (ANOVA F-values)")
# plt.tight_layout()
# plt.show()
interaction_list = []

for f1, f2 in itertools.combinations(features, 2):
    f_val = f_matrix.loc[f1, f2]
    if pd.notna(f_val):
        i1 = int(f1[1:])
        i2 = int(f2[1:])
        interaction_list.append(((i1, i2), f_val))

# Sort by F-value (descending)
interaction_list.sort(key=lambda x: x[1], reverse=True)

auc = get_pairwise_auc(interaction_list, ground_truth)
print(auc)
