import functools
import torch


def handle_errors(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        X1, X2, X3, X4, X5, X6, X7, X8, X9, X10 = args[0].transpose(0, 1)
        try:
            print(f"Processing {args[0] if args else ''}")
            result = func(*args, **kwargs)
            print(f"Successfully Loaded {args[0] if args else ''}!")
            return result
        except Exception as e:
            print(f"Something went wrong: {e}")

    return wrapper


def F1(X):
    X1, X2, X3, X4, X5, X6, X7, X8, X9, X10 = X.transpose(0, 1)

    part1 = torch.pow(torch.pi, X1 * X2) * torch.sqrt(
        torch.clamp(2 * X3, min=1e-6)
    )  # π^(x1*x2) * sqrt(2*x3)
    part2 = -torch.arcsin(torch.clamp(X4, min=-1 + 1e-6, max=1 - 1e-6))  # -arcsin(x4)
    part3 = torch.log(torch.clamp(X3 + X5, min=1e-6))  # log(x3 + x5)
    part4 = -(X9 / torch.clamp(X10, min=1e-6)) * torch.sqrt(
        torch.clamp(X7 / torch.clamp(X8, min=1e-6), min=1e-6)
    )  # -(x9/x10)*sqrt(x7/x8)
    part5 = -X2 * X7  # -x2 * x7

    result = part1 + part2 + part3 + part4 + part5
    ground_truth = [{1, 2, 3}, {3, 5}, {7, 8, 9, 10}, {2, 7}]
    return result, ground_truth


def F2(X):
    X1, X2, X3, X4, X5, X6, X7, X8, X9, X10 = X.transpose(0, 1)

    part1 = torch.pow(torch.pi, X1 * X2) * torch.sqrt(
        torch.clamp(2 * torch.abs(X3), min=1e-6)
    )
    part2 = -torch.arcsin(torch.clamp(0.5 * X4, min=-1 + 1e-6, max=1 - 1e-6))
    part3 = torch.log(torch.abs(X3 + X5) + 1)
    part4 = -(X9 / (1 + torch.abs(X10))) * torch.sqrt(
        torch.clamp(X7 / (1 + torch.abs(X8)), min=1e-6)
    )
    part5 = -X2 * X7

    result = part1 + part2 + part3 + part4 + part5
    ground_truth = [{1, 2, 3}, {3, 5}, {7, 8, 9, 10}, {2, 7}]
    return result, ground_truth


def F3(X):
    X1, X2, X3, X4, X5, X6, X7, X8, X9, X10 = X.transpose(0, 1)

    part1 = torch.exp(torch.abs(X1 - X2))  # exp(|x1 - x2|)
    part2 = torch.abs(X2 * X3)  # |x2 * x3|
    part3 = -torch.pow(X3, 2 * torch.abs(X4))  # -x3^2*|x4|
    part4 = torch.log(
        X4**2 + X5**2 + X7**2 + X8**2 + 1e-6
    )  # log(x4^2 + x5^2 + x7^2 + x8^2)
    part5 = X9 + 1 / (1 + X10**2)  # x9 + 1/(1+x10^2)

    result = part1 + part2 + part3 + part4 + part5
    ground_truth = [{1, 2}, {2, 3}, {3, 4}, {4, 5, 7, 8}]
    return result, ground_truth


def F4(X):
    X1, X2, X3, X4, X5, X6, X7, X8, X9, X10 = X.transpose(0, 1)

    part1 = torch.exp(torch.abs(X1 - X2))
    part2 = torch.abs(X2 * X3)
    part3 = -torch.pow(X3, 2 * torch.abs(X4))
    part4 = (X1 * X4) ** 2
    part5 = torch.log(X4**2 + X5**2 + X7**2 + X8**2 + 1e-6)
    part6 = X9 + 1 / (1 + X10**2)

    result = part1 + part2 + part3 + part4 + part5 + part6
    ground_truth = [{1, 2}, {2, 3}, {3, 4}, {4, 5, 7, 8}]
    return result, ground_truth


def F5(X):
    X1, X2, X3, X4, X5, X6, X7, X8, X9, X10 = X.transpose(0, 1)

    part1 = 1 / (1 + X1**2 + X2**2 + X3**2)
    part2 = torch.sqrt(torch.clamp(torch.exp(X4 + X5), max=1e6))
    part3 = torch.abs(X6 + X7)
    part4 = X8 * X9 * X10

    result = part1 + part2 + part3 + part4
    ground_truth = [{1, 2, 3}, {4, 5}, {8, 9, 10}]
    return result, ground_truth


def F6(X):
    X1, X2, X3, X4, X5, X6, X7, X8, X9, X10 = X.transpose(0, 1)

    part1 = torch.exp(torch.abs(X1 * X2) + 1)
    part2 = -torch.exp(torch.abs(X3 + X4))
    part3 = torch.cos(X5 + X6 - X8)
    part4 = torch.sqrt(X8**2 + X9**2 + X10**2 + 1e-6)

    result = part1 + part2 + part3 + part4
    ground_truth = [{1, 2}, {3, 4}, {5, 6, 8}, {8, 9, 10}]
    return result, ground_truth


def F7(X):
    X1, X2, X3, X4, X5, X6, X7, X8, X9, X10 = X.transpose(0, 1)

    part1 = (torch.atan(X1) + torch.atan(X2)) ** 2
    part2 = torch.max(X3 * X4 + X6, torch.tensor(0.0, device=X.device))
    part3 = -1 / (1 + (X4 * X5 * X6 * X7 * X8) ** 2 + 1e-6)
    part4 = (torch.abs(X7) / (1 + torch.abs(X9))) ** 5
    part5 = torch.sum(X, dim=1)

    result = part1 + part2 + part3 + part4 + part5
    ground_truth = [{4, 5, 6, 7, 8}, {7, 9}]
    return result, ground_truth


def F8(X):
    X1, X2, X3, X4, X5, X6, X7, X8, X9, X10 = X.transpose(0, 1)

    part1 = X1 * X2
    part2 = 2 ** (X3 + X5 + X6)
    part3 = 2 ** (X3 + X4 + X5 + X7)
    part4 = torch.sin(X7 * torch.sin(X8 + X9))
    part5 = torch.acos(torch.clamp(0.9 * X10, min=-1 + 1e-6, max=1 - 1e-6))

    result = part1 + part2 + part3 + part4 + part5
    ground_truth = [{1, 2}, {3, 5, 6}, {3, 4, 5, 7}, {7, 8, 9}]
    return result, ground_truth


def F9(X):
    X1, X2, X3, X4, X5, X6, X7, X8, X9, X10 = X.transpose(0, 1)

    part1 = torch.tanh(X1 * X2 + X3 * X4) * torch.sqrt(torch.abs(X5) + 1e-6)
    part2 = torch.exp(X5 + X6)
    part3 = torch.log((X6 * X7 * X8) ** 2 + 1)
    part4 = X9 * X10
    part5 = 1 / (1 + torch.abs(X10))

    result = part1 + part2 + part3 + part4 + part5
    ground_truth = [{1, 2, 3, 4, 5, 6}, {5, 6}, {6, 7, 8}, {9, 10}]
    return result, ground_truth


def F10(X):
    X1, X2, X3, X4, X5, X6, X7, X8, X9, X10 = X.transpose(0, 1)

    part1 = torch.sinh(X1 + X2)
    part2 = torch.acos(
        torch.clamp(torch.tanh(X3 + X5 + X7), min=-1 + 1e-6, max=1 - 1e-6)
    )
    part3 = torch.cos(X4 + X5)
    part4 = 1 / torch.clamp(torch.cos(X7 * X9), min=1e-6)

    result = part1 + part2 + part3 + part4
    ground_truth = [{1, 2}, {4, 5}, {3, 5, 7}, {7, 9}]
    return result, ground_truth


def F11(X):
    ground_truth = []
    return torch.sum(X, dim=1), ground_truth # No interaction



def F12(X):
    ground_truth = [{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}]
    return torch.prod(X, dim=1), ground_truth  # All features interact


