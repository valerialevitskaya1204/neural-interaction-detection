import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import copy
from bisect import bisect, insort
import operator
import numpy as np
import torch

def get_weights(model):
    weights = []
    for name, param in model.named_parameters():
        if "interaction_mlp" in name and "weight" in name:
            weights.append(param.cpu().detach().numpy())
    return weights


class MLP(nn.Module):
    def __init__(
        self,
        num_features,
        hidden_units,
        use_main_effect_nets=False,
        main_effect_net_units=[10, 10, 10],
    ):
        super(MLP, self).__init__()

        self.hidden_units = hidden_units
        self.use_main_effect_nets = use_main_effect_nets
        self.interaction_mlp = create_mlp([num_features] + hidden_units + [1])

        if main_effect_net_units == [1]:
            use_linear = True
        else:
            use_linear = False
        self.use_linear = use_linear

        if self.use_main_effect_nets:

            if use_linear:
                self.linear = nn.Linear(num_features, 1, bias=False)
            else:
                self.univariate_mlps = self.create_main_effect_nets(
                    num_features, main_effect_net_units, False, "uni"
                )

    def forward(self, x):
        y = self.interaction_mlp(x)

        if self.use_main_effect_nets:
            if self.use_linear:
                y += self.linear(x)
            else:
                y += self.forward_main_effect_nets(x, self.univariate_mlps)
        return y

    def create_main_effect_nets(self, num_features, hidden_units, out_bias, name):
        mlp_list = [
            create_mlp([1] + hidden_units + [1], out_bias=out_bias)
            for _ in range(num_features)
        ]
        for i in range(num_features):
            setattr(self, name + "_" + str(i), mlp_list[i])
        return mlp_list

    def forward_main_effect_nets(self, x, mlps):
        forwarded_mlps = []
        for i, mlp in enumerate(mlps):
            forwarded_mlps.append(mlp(x[:, [i]]))
        forwarded_mlp = sum(forwarded_mlps)
        return forwarded_mlp


def create_mlp(layer_sizes, out_bias=True):
    ls = list(layer_sizes)
    layers = nn.ModuleList()
    for i in range(1, len(ls) - 1):
        layers.append(nn.Linear(int(ls[i - 1]), int(ls[i])))
        layers.append(nn.ReLU())
    layers.append(nn.Linear(int(ls[-2]), int(ls[-1]), bias=out_bias))
    return nn.Sequential(*layers)


def train(
    net,
    data_loaders,
    criterion=nn.MSELoss(reduction="mean"),
    nepochs=100,
    verbose=False,
    early_stopping=True,
    patience=5,
    l1_const=1e-4,
    l2_const=0,
    learning_rate=0.01,
    opt_func=optim.Adam,
    device=torch.device("cpu"),
):
    optimizer = opt_func(net.parameters(), lr=learning_rate, weight_decay=l2_const)

    def evaluate(net, data_loader, criterion, device):
        losses = []
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            loss = criterion(net(inputs), labels).cpu().data
            losses.append(loss)
        return torch.stack(losses).mean()

    best_loss = float("inf")
    best_net = None

    if "val" not in data_loaders:
        early_stopping = False

    patience_counter = 0

    if verbose:
        print("starting to train")
        if early_stopping:
            print("early stopping enabled")

    for epoch in range(nepochs):
        running_loss = 0.0
        run_count = 0
        for i, data in enumerate(data_loaders["train"], 0):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels).mean()

            reg_loss = 0
            for name, param in net.named_parameters():
                if "interaction_mlp" in name and "weight" in name:
                    reg_loss += torch.sum(torch.abs(param))
            (loss + reg_loss * l1_const).backward()
            optimizer.step()
            running_loss += loss.item()
            run_count += 1

        if epoch % 1 == 0:
            key = "val" if "val" in data_loaders else "train"
            val_loss = evaluate(net, data_loaders[key], criterion, device)

            if epoch % 2 == 0:
                if verbose:
                    print(
                        "[epoch %d, total %d] train loss: %.4f, val loss: %.4f"
                        % (epoch + 1, nepochs, running_loss / run_count, val_loss)
                    )
            if early_stopping:
                if val_loss < best_loss:
                    best_loss = val_loss
                    best_net = copy.deepcopy(net)
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter > patience:
                        net = best_net
                        val_loss = best_loss
                        if verbose:
                            print("early stopping!")
                        break

            prev_loss = running_loss
            running_loss = 0.0

    if "test" in data_loaders:
        key = "test"
    elif "val" in data_loaders:
        key = "val"
    else:
        key = "train"
    test_loss = evaluate(net, data_loaders[key], criterion, device).item()

    if verbose:
        print("Finished Training. Test loss: ", test_loss)

    return net, test_loss


def preprocess_weights(weights):
    w_later = np.abs(weights[-1])
    w_input = np.abs(weights[0])

    for i in range(len(weights) - 2, 0, -1):
        w_later = np.matmul(w_later, np.abs(weights[i]))

    return w_input, w_later


def make_one_indexed(interaction_ranking):
    return [(tuple(np.array(i) + 1), s) for i, s in interaction_ranking]


def interpret_interactions(w_input, w_later, get_main_effects=False):
    interaction_strengths = {}
    for i in range(w_later.shape[1]):
        sorted_hweights = sorted(
            enumerate(w_input[i]), key=lambda x: x[1], reverse=True
        )
        interaction_candidate = []
        candidate_weights = []
        for j in range(w_input.shape[1]):
            insort(interaction_candidate, sorted_hweights[j][0])
            candidate_weights.append(sorted_hweights[j][1])

            if not get_main_effects and len(interaction_candidate) == 1:
                continue
            interaction_tup = tuple(interaction_candidate)
            if interaction_tup not in interaction_strengths:
                interaction_strengths[interaction_tup] = 0
            interaction_strength = (min(candidate_weights)) * (np.sum(w_later[:, i]))
            interaction_strengths[interaction_tup] += interaction_strength

    interaction_ranking = sorted(
        interaction_strengths.items(), key=operator.itemgetter(1), reverse=True
    )

    return interaction_ranking


def interpret_pairwise_interactions(w_input, w_later):
    p = w_input.shape[1]

    interaction_ranking = []
    for i in range(p):
        for j in range(p):
            if i < j:
                strength = (np.minimum(w_input[:, i], w_input[:, j]) * w_later).sum()
                interaction_ranking.append(((i, j), strength))

    interaction_ranking.sort(key=lambda x: x[1], reverse=True)
    return interaction_ranking


def get_interactions(weights, pairwise=False, one_indexed=False):
    w_input, w_later = preprocess_weights(weights)

    if pairwise:
        interaction_ranking = interpret_pairwise_interactions(w_input, w_later)
    else:
        interaction_ranking = interpret_interactions(w_input, w_later)
        interaction_ranking = prune_redundant_interactions(interaction_ranking)

    if one_indexed:
        return make_one_indexed(interaction_ranking)
    else:
        return interaction_ranking


def prune_redundant_interactions(interaction_ranking, max_interactions=100):
    interaction_ranking_pruned = []
    current_superset_inters = []
    for inter, strength in interaction_ranking:
        set_inter = set(inter)
        if len(interaction_ranking_pruned) >= max_interactions:
            break
        subset_inter_skip = False
        update_superset_inters = []
        for superset_inter in current_superset_inters:
            if set_inter < superset_inter:
                subset_inter_skip = True
                break
            elif not (set_inter > superset_inter):
                update_superset_inters.append(superset_inter)
        if subset_inter_skip:
            continue
        current_superset_inters = update_superset_inters
        current_superset_inters.append(set_inter)
        interaction_ranking_pruned.append((inter, strength))

    return interaction_ranking_pruned

