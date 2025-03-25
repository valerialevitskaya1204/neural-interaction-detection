import torch
import numpy as np

from src.datasets.synthetic_datasets import (
    F1,
    F2,
    F3,
    F4,
    F5,
    F6,
    F7,
    F8,
    F9,
    F10,
    F11,
    F12,
)

from src.experiments.run_experiments import CONFIG
from src.utils import preprocess_data, set_seed, print_rankings
from src.neural_detection.multilayer_perceptron import MLP, train, get_weights
from src.neural_detection.neural_interaction_detection import (
    get_interactions,
    get_pairwise_auc,
    get_anyorder_R_precision,
)


if __name__ == "__main__":

    use_main_effect_nets = True  # toggle this to use "main effect" nets
    num_samples = 30000
    num_features = 10

    # X, y = generate_synthetic_data(F11, num_samples)
    # # ground = CONFIG["synthetic"]["ground_truth"][F1]
    # # print(X)

    # def synth_func(X):
    #     X1, X2, X3, X4, X5, X6, X7, X8, X9, X10 = X.transpose()

    #     interaction1 = np.exp(np.abs(X1 - X2))
    #     interaction2 = np.abs(X2 * X3)
    #     interaction3 = -1 * (X3**2) ** np.abs(X4)
    #     interaction4 = (X1 * X4) ** 2
    #     interaction5 = np.log(X4**2 + X5**2 + X7**2 + X8**2)
    #     main_effects = X9 + 1 / (1 + X10**2)

    #     Y = (
    #         interaction1
    #         + interaction2
    #         + interaction3 * interaction4
    #         + interaction5
    #         + main_effects
    #     )
    #     ground_truth = [{1, 2}, {2, 3}, {3, 4}, {1, 4}, {4, 5, 7, 8}]

    #     return Y, ground_truth

    X = np.random.uniform(low=-1, high=1, size=(num_samples, num_features))
    # X = np.random.normal(0, 1, (num_samples, 10))
    X = torch.Tensor(X)
    Y = F3(X)
    print(Y)

    def sanitize_tensor(tensor, nan_val=0.0, posinf=1e6, neginf=-1e6):
        """
        Replaces NaNs and infinities in a tensor with safe values.
        """
        return torch.nan_to_num(tensor, nan=nan_val, posinf=posinf, neginf=neginf)

    nan_mask = torch.isnan(Y)
    Y = sanitize_tensor(Y)
    if nan_mask.any():
        print("NaNs found at positions:", nan_mask.nonzero(as_tuple=True))
    # print(torch.isnan(Y).any())

    data_loaders = preprocess_data(
        X, Y, valid_size=10000, test_size=10000, std_scale=True, get_torch_loaders=True
    )

    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP(
        num_features, [140, 100, 60, 20], use_main_effect_nets=use_main_effect_nets
    ).to(device)

    model, mlp_loss = train(
        model,
        data_loaders,
        device=device,
        learning_rate=1e-2,
        l1_const=5e-5,
        verbose=True,
    )

    model_weights = get_weights(model)
    torch.save(model_weights, "F1.pth")

    anyorder_interactions = get_interactions(model_weights, one_indexed=True)
    pairwise_interactions = get_interactions(
        model_weights, pairwise=True, one_indexed=True
    )

    print(
        print_rankings(
            pairwise_interactions, anyorder_interactions, top_k=10, spacing=14
        )
    )

    auc = get_pairwise_auc(pairwise_interactions, ground_truth)
    r_prec = get_anyorder_R_precision(anyorder_interactions, ground_truth)

    print("Pairwise AUC", auc, ", Any-order R-Precision", r_prec)
