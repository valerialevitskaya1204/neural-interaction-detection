import torch
from torch.utils import data
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score



def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def force_float(X):
    """
    Convert input (NumPy array or PyTorch tensor) to a PyTorch float32 tensor.
    
    Args:
        X: Input array (NumPy array or PyTorch tensor).
    
    Returns:
        torch.Tensor: A float32 tensor.
    """
    if isinstance(X, np.ndarray):
        # Case 1: X is a NumPy array -> convert to torch.float32
        return torch.from_numpy(X.astype(np.float32))
    elif isinstance(X, torch.Tensor):
        # Case 2: X is already a tensor -> ensure dtype is float32
        return X.to(dtype=torch.float32)
    else:
        raise TypeError("Input must be a NumPy array or PyTorch tensor.")


# def convert_to_torch_loaders(Xd, Yd, batch_size):
#     if type(Xd) != dict and type(Yd) != dict:
#         Xd = {"train": Xd}
#         Yd = {"train": Yd}

#     data_loaders = {}
#     for k in Xd:
#         if k == "scaler":
#             continue
#         feats = force_float(Xd[k])
#         targets = force_float(Yd[k])
#         dataset = data.TensorDataset(feats, targets)
#         data_loaders[k] = data.DataLoader(dataset, batch_size, shuffle=(k == "train"))

#     return data_loaders


# def preprocess_data(
#     X,
#     Y,
#     valid_size=500,
#     test_size=500,
#     std_scale=False,
#     get_torch_loaders=False,
#     batch_size=100,
# ):

#     n, p = X.shape
#     ## Make dataset splits
#     ntrain, nval, ntest = n - valid_size - test_size, valid_size, test_size

#     Xd = {
#         "train": X[:ntrain],
#         "val": X[ntrain : ntrain + nval],
#         "test": X[ntrain + nval : ntrain + nval + ntest],
#     }
#     Yd = {
#         "train": np.expand_dims(Y[:ntrain], axis=1),
#         "val": np.expand_dims(Y[ntrain : ntrain + nval], axis=1),
#         "test": np.expand_dims(Y[ntrain + nval : ntrain + nval + ntest], axis=1),
#     }

#     for k in Xd:
#         if len(Xd[k]) == 0:
#             assert k != "train"
#             del Xd[k]
#             del Yd[k]

#     if std_scale:
#         scaler_x = StandardScaler()
#         scaler_y = StandardScaler()

#         scaler_x.fit(Xd["train"])
#         scaler_y.fit(Yd["train"])

#         for k in Xd:
#             Xd[k] = scaler_x.transform(Xd[k])
#             Yd[k] = scaler_y.transform(Yd[k])

#         Xd["scaler"] = scaler_x
#         Yd["scaler"] = scaler_y

#     if get_torch_loaders:
#         return convert_to_torch_loaders(Xd, Yd, batch_size)

#     else:
#         return Xd, Yd

def convert_to_torch_loaders(Xd, Yd, batch_size):
    data_loaders = {}
    
    for k in Xd:
        if k == "scaler":
            continue
            
        feats = force_float(Xd[k])
        targets = force_float(Yd[k]).float()  # Ensure float type
        
        # Ensure targets are 2D [batch_size, 1] if needed
        if len(targets.shape) == 1:
            targets = targets.unsqueeze(-1)
        
        dataset = data.TensorDataset(feats, targets)
        data_loaders[k] = data.DataLoader(dataset, batch_size, shuffle=(k == "train"))
        
    return data_loaders


def preprocess_data(
    X,
    Y,
    valid_size=500,
    test_size=500,
    std_scale=False,
    get_torch_loaders=False,
    batch_size=100,
):
    
    n, p = X.shape
    ntrain, nval, ntest = n - valid_size - test_size, valid_size, test_size

    # Convert Y to proper shape (handle both torch and numpy)
    if isinstance(Y, torch.Tensor):
        Y_np = Y.numpy()
    else:
        Y_np = np.array(Y)
    
    # Ensure Y is 1D before splitting (squeeze all extra dimensions)
    Y_np = Y_np.squeeze()
    
    # Create splits
    Xd = {
        "train": X[:ntrain],
        "val": X[ntrain : ntrain + nval],
        "test": X[ntrain + nval : ntrain + nval + ntest],
    }
    
    # Create targets without adding extra dimensions
    Yd = {
        "train": Y_np[:ntrain],
        "val": Y_np[ntrain : ntrain + nval],
        "test": Y_np[ntrain + nval : ntrain + nval + ntest],
    }
    # Clean empty splits
    for k in list(Xd.keys()):
        if len(Xd[k]) == 0:
            print(f"Removing empty split: {k}")
            del Xd[k]
            del Yd[k]

    if std_scale:
        print("\nApplying standard scaling")
        scaler_x = StandardScaler()
        scaler_y = StandardScaler()

        scaler_x.fit(Xd["train"])
        scaler_y.fit(Yd["train"].reshape(-1, 1))  # Reshape for scaler

        for k in Xd:
            if k != "scaler":
                Xd[k] = scaler_x.transform(Xd[k])
                Yd[k] = scaler_y.transform(Yd[k].reshape(-1, 1)).flatten()
                print(f"Scaled {k} split - X mean: {Xd[k].mean():.2f}, Y mean: {Yd[k].mean():.2f}")

        Xd["scaler"] = scaler_x
        Yd["scaler"] = scaler_y

    if get_torch_loaders:
        print("\nConverting to Torch DataLoaders")
        return convert_to_torch_loaders(Xd, Yd, batch_size)
    else:
        print("\nReturning numpy arrays")
        return Xd, Yd

def get_pairwise_auc(interactions, ground_truth):
    strengths = []
    gt_binary_list = []
    for inter, strength in interactions:
        inter_set = set(inter)  # assume 1-indexed
        strengths.append(strength)
        if any(inter_set <= gt for gt in ground_truth):
            gt_binary_list.append(1)
        else:
            gt_binary_list.append(0)
    if interactions:
        auc = roc_auc_score(gt_binary_list, strengths)
    else:
        auc = 0.5
    return auc


def get_anyorder_R_precision(interactions, ground_truth):

    R = len(ground_truth)
    recovered_gt = []
    counter = 0

    for inter, strength in interactions:
        if counter == R:
            break

        inter_set = set(inter)  # assume 1-indexed

        if any(inter_set < gt for gt in ground_truth):
            continue
        counter += 1
        if inter_set in ground_truth:
            recovered_gt.append(inter_set)

    R_precision = len(recovered_gt) / R

    return R_precision


def print_rankings(pairwise_interactions, anyorder_interactions, top_k=10, spacing=14):
    print(
        justify(["Pairwise interactions", "", "Arbitrary-order interactions"], spacing)
    )
    if isinstance(pairwise_interactions, list) and len(pairwise_interactions) > 0 and isinstance(anyorder_interactions, list) and len(anyorder_interactions) > 0:
        actual_top_k = min(
            top_k, 
            len(pairwise_interactions), 
            len(anyorder_interactions)
        )
        for i in range(actual_top_k):
            p_inter, p_strength = pairwise_interactions[i]
            a_inter, a_strength = anyorder_interactions[i]


            p_inter = tuple(x.item() for x in p_inter)
            a_inter = tuple(x.item() for x in a_inter)

            print(
                justify(
                    [
                        p_inter,
                        "{0:.4f}".format(p_strength),
                        "",
                        a_inter,
                        "{0:.4f}".format(a_strength),
                    ],
                    spacing,
                )
            )
    else:
        p_inter, p_strength = pairwise_interactions
        a_inter, a_strength = anyorder_interactions


        p_inter = tuple(x.item() for x in p_inter)
        a_inter = tuple(x.item() for x in a_inter)

        print(
            justify(
                    [
                        p_inter,
                        "{0:.4f}".format(p_strength),
                        "",
                        a_inter,
                        "{0:.4f}".format(a_strength),
                    ],
                    spacing,
                )
            )



def justify(row, spacing=14):
    return "".join(str(item).ljust(spacing) for item in row)

def sanitize_tensor(tensor, nan_val=0.0, posinf=1e6, neginf=-1e6):
        """
        Replaces NaNs and infinities in a tensor with safe values.
        """
        tensor = torch.nan_to_num(tensor, nan=nan_val, posinf=posinf, neginf=neginf)
        return tensor.view(-1, 1) if tensor.dim() == 1 else tensor


def get_strength(pairwise_interactions, anyorder_interactions, top_k=10):
    if isinstance(pairwise_interactions, list) and len(pairwise_interactions) > 0 and isinstance(anyorder_interactions, list) and len(anyorder_interactions) > 0:
        actual_top_k = min(
            top_k, 
            len(pairwise_interactions), 
            len(anyorder_interactions)
        )
        for i in range(actual_top_k):
            p_inter, p_strength = pairwise_interactions[i]
            a_inter, a_strength = anyorder_interactions[i]


            p_inter = tuple(x.item() for x in p_inter)
            a_inter = tuple(x.item() for x in a_inter)

    else:
        p_inter, p_strength = pairwise_interactions
        a_inter, a_strength = anyorder_interactions


        p_inter = tuple(x.item() for x in p_inter)
        a_inter = tuple(x.item() for x in a_inter)
    return p_inter, p_strength, a_strength, a_inter

