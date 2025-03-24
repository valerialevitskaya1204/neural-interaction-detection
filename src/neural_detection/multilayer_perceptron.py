# import torch
# import torch.nn as nn
# import torch.optim as optim
# import copy

# def get_weights(model):
#     weights = []
#     for name, param in model.named_parameters():
#         if "interaction_mlp" in name and "weight" in name:
#             weights.append(param.cpu().detach().numpy())
#     return weights

# class MLP(nn.Module):
#     def __init__(
#         self,
#         num_features,
#         hidden_units,
#         use_main_effect_nets=False,
#         main_effect_net_units=[10, 10, 10],
#     ):
#         super(MLP, self).__init__()

#         self.use_main_effect_nets = use_main_effect_nets
#         self.interaction_mlp = create_mlp([num_features] + hidden_units + [1])
#         self.use_linear = main_effect_net_units == [1]

#         if self.use_main_effect_nets:
#             if self.use_linear:
#                 self.linear = nn.Linear(num_features, 1, bias=False)
#             else:
#                 self.univariate_mlps = self.create_main_effect_nets(num_features, main_effect_net_units, "uni")

#     def forward(self, x):
#         y = self.interaction_mlp(x)
#         if self.use_main_effect_nets:
#             y += self.forward_main_effect_nets(x)
#         return y

#     def create_main_effect_nets(self, num_features, hidden_units, name):
#         return nn.ModuleList([create_mlp([1] + hidden_units + [1]) for _ in range(num_features)])

#     def forward_main_effect_nets(self, x):
#         return sum(mlp(x[:, [i]]) for i, mlp in enumerate(self.univariate_mlps))

# class MLP_M(MLP):
#     def __init__(self, num_features, hidden_units, main_effect_net_units=[10, 10, 10]):
#         super().__init__(num_features, hidden_units, use_main_effect_nets=True, main_effect_net_units=main_effect_net_units)

# class MLP_Cutoff(nn.Module):
#     def __init__(self, num_features, interaction_list, hidden_units=[140, 100, 60, 20]):
#         super(MLP_Cutoff, self).__init__()
#         self.main_effects = nn.ModuleList([create_mlp([1] + hidden_units + [1]) for _ in range(num_features)])
#         self.interactions = nn.ModuleList([create_mlp([len(interaction)] + hidden_units + [1]) for interaction in interaction_list])

#     def forward(self, x):
#         y_main = sum(self.main_effects[i](x[:, [i]]) for i in range(len(self.main_effects)))
#         y_inter = sum(inter_mlp(x[:, interaction]) for inter_mlp, interaction in zip(self.interactions, interaction_list))
#         return y_main + y_inter

# def create_mlp(layer_sizes):
#     layers = []
#     for i in range(1, len(layer_sizes) - 1):
#         layers.append(nn.Linear(layer_sizes[i - 1], layer_sizes[i]))
#         layers.append(nn.ReLU())
#     layers.append(nn.Linear(layer_sizes[-2], layer_sizes[-1]))
#     return nn.Sequential(*layers)



# def evaluate(net, data_loader, criterion, device):
#     losses = []
#     for inputs, labels in data_loader:
#         inputs = inputs.to(device)
#         labels = labels.to(device)
#         loss = criterion(net(inputs), labels).cpu().data
#         losses.append(loss)
#     return torch.stack(losses).mean()

# def train(
#     net,
#     data_loaders,
#     criterion=nn.MSELoss(reduction="mean"),
#     nepochs=100,
#     verbose=False,
#     early_stopping=True,
#     patience=5,
#     l1_const=1e-4,
#     l2_const=0,
#     learning_rate=0.01,
#     opt_func=optim.Adam,
#     device=torch.device("cpu"),
# ):
#     optimizer = opt_func(net.parameters(), lr=learning_rate, weight_decay=l2_const)

#     best_loss = float("inf")
#     best_net = None

#     if "val" not in data_loaders:
#         early_stopping = False

#     patience_counter = 0

#     if verbose:
#         print("starting to train")
#         if early_stopping:
#             print("early stopping enabled")

#     for epoch in range(nepochs):
#         running_loss = 0.0
#         run_count = 0
#         for _, data in enumerate(data_loaders["train"], 0):
#             inputs, labels = data
#             inputs = inputs.to(device)
#             labels = labels.to(device)
#             optimizer.zero_grad()
#             outputs = net(inputs)
#             loss = criterion(outputs, labels).mean()

#             reg_loss = 0
#             for name, param in net.named_parameters():
#                 if "interaction_mlp" in name and "weight" in name:
#                     reg_loss += torch.sum(torch.abs(param))
#             (loss + reg_loss * l1_const).backward()
#             optimizer.step()
#             running_loss += loss.item()
#             run_count += 1

#         if epoch % 1 == 0:
#             key = "val" if "val" in data_loaders else "train"
#             val_loss = evaluate(net, data_loaders[key], criterion, device)

#             if epoch % 2 == 0:
#                 if verbose:
#                     print(
#                         "[epoch %d, total %d] train loss: %.4f, val loss: %.4f"
#                         % (epoch + 1, nepochs, running_loss / run_count, val_loss)
#                     )
#             if early_stopping:
#                 if val_loss < best_loss:
#                     best_loss = val_loss
#                     best_net = copy.deepcopy(net)
#                     patience_counter = 0
#                 else:
#                     patience_counter += 1
#                     if patience_counter > patience:
#                         net = best_net
#                         val_loss = best_loss
#                         if verbose:
#                             print("early stopping!")
#                         break

#             prev_loss = running_loss
#             running_loss = 0.0

#     if "test" in data_loaders:
#         key = "test"
#     elif "val" in data_loaders:
#         key = "val"
#     else:
#         key = "train"
#     test_loss = evaluate(net, data_loaders[key], criterion, device).item()

#     if verbose:
#         print("Finished Training. Test loss: ", test_loss)

#     return net, test_loss

import torch
import torch.nn as nn
import torch.optim as optim
import copy

# Utility to extract specific weights
def get_weights(model):
    weights = []
    for name, param in model.named_parameters():
        if "interaction_mlp" in name and "weight" in name:
            weights.append(param.cpu().detach().numpy())
    return weights

# Simple MLP creator
def create_mlp(layer_sizes):
    layers = []
    for i in range(1, len(layer_sizes) - 1):
        layers.append(nn.Linear(layer_sizes[i - 1], layer_sizes[i]))
        layers.append(nn.ReLU())
    layers.append(nn.Linear(layer_sizes[-2], layer_sizes[-1]))
    return nn.Sequential(*layers)

# Main MLP model
class MLP(nn.Module):
    def __init__(
        self,
        num_features,
        hidden_units,
        use_main_effect_nets=False,
        main_effect_net_units=[10, 10, 10],
    ):
        super(MLP, self).__init__()

        self.use_main_effect_nets = use_main_effect_nets
        self.interaction_mlp = create_mlp([num_features] + hidden_units + [1])
        self.use_linear = main_effect_net_units == [1]

        if self.use_main_effect_nets:
            if self.use_linear:
                self.linear = nn.Linear(num_features, 1, bias=False)
            else:
                self.univariate_mlps = self.create_main_effect_nets(
                    num_features, main_effect_net_units, "uni"
                )

    def forward(self, x):
        y = self.interaction_mlp(x)
        if self.use_main_effect_nets:
            y += self.forward_main_effect_nets(x)
        return y

    def create_main_effect_nets(self, num_features, hidden_units, name):
        return nn.ModuleList([create_mlp([1] + hidden_units + [1]) for _ in range(num_features)])

    def forward_main_effect_nets(self, x):
        return sum(mlp(x[:, [i]]) for i, mlp in enumerate(self.univariate_mlps))

# Main effect + interaction model variant
class MLP_M(MLP):
    def __init__(self, num_features, hidden_units, main_effect_net_units=[10, 10, 10]):
        super().__init__(
            num_features,
            hidden_units,
            use_main_effect_nets=True,
            main_effect_net_units=main_effect_net_units,
        )

# MLP model with cutoff interactions
class MLP_Cutoff(nn.Module):
    def __init__(self, num_features, interaction_list, hidden_units=[140, 100, 60, 20]):
        super(MLP_Cutoff, self).__init__()
        self.interaction_list = interaction_list  # FIXED: store interaction_list
        self.main_effects = nn.ModuleList([
            create_mlp([1] + hidden_units + [1]) for _ in range(num_features)
        ])
        self.interactions = nn.ModuleList([
            create_mlp([len(interaction)] + hidden_units + [1])
            for interaction in interaction_list
        ])

    def forward(self, x):
        y_main = sum(self.main_effects[i](x[:, [i]]) for i in range(len(self.main_effects)))
        y_inter = sum(inter_mlp(x[:, interaction]) for inter_mlp, interaction in zip(self.interactions, self.interaction_list))
        return y_main + y_inter

# Model evaluation utility
def evaluate(net, data_loader, criterion, device):
    losses = []
    for inputs, labels in data_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        loss = criterion(net(inputs), labels).cpu().data
        losses.append(loss)
    return torch.stack(losses).mean()

# Training function
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

    best_loss = float("inf")
    best_net = None

    if "val" not in data_loaders:
        early_stopping = False

    patience_counter = 0

    if verbose:
        print("Starting training...")
        if early_stopping:
            print("Early stopping enabled")


            for epoch in range(nepochs):
                running_loss = 0.0
                run_count = 0
                for _, data in enumerate(data_loaders["train"], 0):
                    inputs, labels = data
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    optimizer.zero_grad()

                    outputs = net(inputs)
                    loss = criterion(outputs, labels).mean()

                    # L1 regularization
                    if l1_const > 0:
                        l1_penalty = 0
                        for param in net.parameters():
                            l1_penalty += torch.norm(param, 1)
                        loss += l1_const * l1_penalty

                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()
                    run_count += 1

                if verbose:
                    print(f"Epoch {epoch+1}/{nepochs}, Training Loss: {running_loss / run_count:.4f}")

                # Validation & early stopping
                if early_stopping and "val" in data_loaders:
                    val_loss = evaluate(net, data_loaders["val"], criterion, device).item()
                    if verbose:
                        print(f"Validation Loss: {val_loss:.4f}")

                    if val_loss < best_loss:
                        best_loss = val_loss
                        best_net = copy.deepcopy(net)
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        if patience_counter >= patience:
                            if verbose:
                                print(f"Early stopping at epoch {epoch+1}")
                            break

            if early_stopping and best_net is not None:
                net.load_state_dict(best_net.state_dict())

            return net