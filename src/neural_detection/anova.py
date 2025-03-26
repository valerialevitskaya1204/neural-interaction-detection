from src.datasets.synthetic_datasets import F1, F2, F3, F4, F5, F6, F7, F8, F9, F10, F11, F12
from src.utils import preprocess_data, get_anyorder_R_precision, get_pairwise_auc

import pandas as pd
import itertools
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import ols
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt

import torch
import statsmodels.api as sm
from statsmodels.formula.api import ols

torch.manual_seed(42)
num_samples = 10000
num_features = 10

functions = [F1, F2, F3, F4, F5, F6, F7, F8, F9, F10]
aucs = {}

for func in tqdm(functions):

    X = torch.rand(num_samples, num_features) * 2 - 1  # Uniform in [-1, 1]
    y, ground_truth = func(X)

    # Convert to pandas DataFrame
    df = pd.DataFrame(X.numpy(), columns=[f'f{i}' for i in range(1, 11)])
    df['y'] = y.numpy()

    # -------------------------
    # Step 3: Compute ANOVA F-value matrix
    # -------------------------
    features = [f'f{i}' for i in range(1, 11)]
    f_matrix = pd.DataFrame(np.nan, index=features, columns=features)

    
    nonlinear_terms = []
    for f1, f2 in itertools.combinations([f'f{i}' for i in range(1, 11)], 2):
        # 1. f1^2 * f2
        squared_col = f'{f1}_squared_{f2}'
        df[squared_col] = df[f1]**2 * df[f2]
        nonlinear_terms.append((f1, f2, squared_col))

        # 2. log(|f1 * f2| + epsilon)
        log_col = f'log_{f1}_{f2}'
        df[log_col] = np.log(np.abs(df[f1] * df[f2]) + 1e-6)
        nonlinear_terms.append((f1, f2, log_col)) 

    for f1, f2 in itertools.combinations(features, 2):
        f_vals = []

        # Get all nonlinear terms for this pair
        terms = [term for a, b, term in nonlinear_terms if {a, b} == {f1, f2}]
        if not terms:
            continue

        try:
            formula = f"y ~ {f1} + {f2} + {' + '.join(terms)}"
            model = ols(formula, data=df).fit()
            anova_table = sm.stats.anova_lm(model, typ=2)

            for term in terms:
                if term in anova_table.index:
                    f_vals.append(anova_table.loc[term, 'F'])

            if f_vals:
                f_matrix.loc[f1, f2] = np.mean(f_vals)
                f_matrix.loc[f2, f1] = np.mean(f_vals)

        except Exception as e:
            print(f"Skipped ({f1}, {f2}):", e)


    np.fill_diagonal(f_matrix.values, 0)

    # -------------------------
    # Step 4: Done — show result
    # -------------------------
    # print("ANOVA Interaction F-value Matrix:")
    # print(f_matrix.round(2))
# 
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
    

    aucs[func.__name__] = auc
    print(aucs)


print(f"MEAN AUC: {np.mean(list(aucs.values()))}")
