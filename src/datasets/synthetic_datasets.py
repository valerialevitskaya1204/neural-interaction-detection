# import numpy as np


# def F1(x):
#     return (
#         np.pi * x[:, 0] * x[:, 1] * np.sqrt(2 * x[:, 2])
#         - np.arcsin(x[:, 3])
#         + np.log(x[:, 4] + x[:, 5])
#         - (x[:, 8] / x[:, 9]) * np.sqrt(x[:, 6] / x[:, 7])
#         - x[:, 1] * x[:, 6]
#     )


# def F2(x):
#     return (
#         np.pi ** (x[:, 0] * x[:, 1] * 2) * np.sqrt(2 * np.abs(x[:, 2]))
#         - np.arcsin(0.5 * x[:, 3])
#         + np.log(np.abs(x[:, 2] + x[:, 4]) + 1)
#         + (x[:, 8] / (1 + np.abs(x[:, 9]))) * np.sqrt(x[:, 6] / (1 + np.abs(x[:, 7])))
#         - x[:, 1] * x[:, 6]
#     )


# def F3(x):
#     return (
#         np.exp(np.abs(x[:, 0] - x[:, 1]))
#         + np.abs(x[:, 1] * x[:, 2])
#         - (x[:, 3] ** 2 * np.abs(x[:, 3]))
#         + np.log(x[:, 3] ** 2 + x[:, 4] ** 2 + x[:, 6] ** 2 + x[:, 7] ** 2)
#         + x[:, 8]
#         + 1 / (1 + x[:, 9] ** 2)
#     )


# def F4(x):
#     return (
#         np.exp(np.abs(x[:, 0] - x[:, 1]))
#         + np.abs(x[:, 1] * x[:, 2])
#         - (x[:, 3] ** 2 * np.abs(x[:, 3]))
#         + (x[:, 0] * x[:, 3]) ** 2
#         + np.log(x[:, 3] ** 2 + x[:, 4] ** 2 + x[:, 6] ** 2 + x[:, 7] ** 2)
#         + x[:, 8]
#         + 1 / (1 + x[:, 9] ** 2)
#     )


# def F5(x):
#     return (
#         1 / (1 + x[:, 0] ** 2 + x[:, 1] ** 2 + x[:, 2] ** 2)
#         + np.sqrt(np.exp(x[:, 3] + x[:, 4]))
#         + np.abs(x[:, 5] + x[:, 6])
#         + x[:, 7] * x[:, 8] * x[:, 9]
#     )


# def F6(x):
#     return (
#         np.exp(np.abs(x[:, 0] * x[:, 1]) + 1)
#         - np.exp(np.abs(x[:, 2] + x[:, 3]) + 1)
#         + np.cos(x[:, 4] + x[:, 5] - x[:, 7])
#         + np.sqrt(x[:, 7] ** 2 + x[:, 8] ** 2 + x[:, 9] ** 2)
#     )


# def F7(x):
#     return (
#         (np.arctan(x[:, 0]) + np.arctan(x[:, 1])) ** 2
#         + np.maximum(x[:, 2] * x[:, 3] + x[:, 5], 0)
#         - 1 / (1 + (x[:, 3] * x[:, 4] * x[:, 5] * x[:, 6] * x[:, 7]) ** 2)
#         + (np.abs(x[:, 6]) / (1 + np.abs(x[:, 8]))) ** 5
#         + np.sum(x, axis=1)
#     )


# def F8(x):
#     return (
#         x[:, 0] * x[:, 1]
#         + 2 ** (x[:, 2] + x[:, 4] + x[:, 5])
#         + 2 ** (x[:, 2] + x[:, 3] + x[:, 4] + x[:, 6])
#         + np.sin(x[:, 6] * np.sin(x[:, 7] + x[:, 8]))
#         + np.arccos(0.9 * x[:, 9])
#     )


# def F9(x):
#     return (
#         np.tanh(x[:, 0] * x[:, 1] + x[:, 2] * x[:, 3]) * np.sqrt(np.abs(x[:, 4]))
#         + np.exp(x[:, 4] + x[:, 5])
#         + np.log((x[:, 5] * x[:, 6] * x[:, 7]) ** 2 + 1)
#         + x[:, 8] * x[:, 9]
#         + 1 / (1 + np.abs(x[:, 9]))
#     )


# def F10(x):
#     return (
#         np.sinh(x[:, 0] + x[:, 1])
#         + np.arccos(np.tanh(x[:, 2] + x[:, 4] + x[:, 6]))
#         + np.cos(x[:, 3] + x[:, 4])
#         + 1 / np.cos(x[:, 6] * x[:, 8])
#     )


# def F11(x):
#     return (
#         x[:, 0]
#         + x[:, 1]
#         + x[:, 2]
#         + x[:, 3]
#         + x[:, 4]
#         + x[:, 5]
#         + x[:, 6]
#         + x[:, 7]
#         + x[:, 8]
#         + x[:, 9]
#     )


# def F12(x):
#     return np.prod(x, axis=1)


# def generate_synthetic_data(func, num_samples=10000, epsilon=1e-8):
#     x = np.zeros((num_samples, 10))

#     # Function-specific constraints
#     if func is F1:
#         # F1 constraints
#         x[:, 0] = np.random.uniform(-1, 1, num_samples)
#         x[:, 1] = np.random.uniform(-1, 1, num_samples)
#         x[:, 2] = np.random.uniform(0, 1, num_samples)  # x2 >=0
#         x[:, 3] = np.random.uniform(-1, 1, num_samples)

#         # Ensure x4 + x5 >0
#         x4 = np.random.uniform(-1, 1, num_samples)
#         lower = np.maximum(-x4, -1)
#         x5 = lower + np.random.uniform(0, 1, num_samples) * (1 - lower)
#         x[:, 4] = x4
#         x[:, 5] = x5

#         # x6 and x7 same sign and non-zero
#         signs = np.random.choice([1, -1], num_samples)
#         x[:, 6] = signs * np.random.uniform(epsilon, 1, num_samples)
#         x[:, 7] = signs * np.random.uniform(epsilon, 1, num_samples)

#         x[:, 8] = np.random.uniform(-1, 1, num_samples)

#         # x9 non-zero
#         x9_signs = np.random.choice([1, -1], num_samples)
#         x[:, 9] = x9_signs * np.random.uniform(epsilon, 1, num_samples)

#     elif func is F2:
#         # F2 constraints: x6 >=0
#         x = np.random.uniform(-1, 1, (num_samples, 10))
#         x[:, 6] = np.random.uniform(0, 1, num_samples)  # x6 >=0

#     elif func is F8:
#         # F8 constraints: x9 in [-1/0.9, 1/0.9] ≈ [-1.111, 1.111]
#         x = np.random.uniform(-1, 1, (num_samples, 10))
#         x[:, 9] = np.random.uniform(-1.111, 1.111, num_samples)

#     elif func is F10:
#         # F10 constraints: x6*x8 != (2k+1)*π/2
#         x = np.random.uniform(-1, 1, (num_samples, 10))
#         # Ensure cos(x6*x8) != 0 by keeping product < π/2
#         product = np.random.uniform(-np.pi / 2 + 0.01, np.pi / 2 - 0.01, num_samples)
#         # Solve for x6 and x8 maintaining their individual ranges
#         x[:, 6] = np.random.uniform(-1, 1, num_samples)
#         x[:, 8] = product / (x[:, 6] + np.where(x[:, 6] == 0, 0.1, 0))

#     else:
#         # Default case for F3-F7, F9
#         x = np.random.uniform(-1, 1, (num_samples, 10))

#     # Post-processing for common constraints
#     if func in [F3, F4]:
#         # Ensure sum(x3^2 + x4^2 + x6^2 + x7^2) > 0
#         mask = (x[:, 3] ** 2 + x[:, 4] ** 2 + x[:, 6] ** 2 + x[:, 7] ** 2) <= 0
#         if np.any(mask):
#             x[mask, 3] = 0.1  # Set minimal non-zero value

#     if func in [F11, F12]:
#         x = np.random.normal(0, 1, (num_samples, 10))

#     y = func(x)
#     return x, y

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


# Full set of F1-F12 functions, NaN-safe and formatted as requested


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
    return result


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
    return result


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
    return result


def F4(X):
    X1, X2, X3, X4, X5, X6, X7, X8, X9, X10 = X.transpose(0, 1)

    part1 = torch.exp(torch.abs(X1 - X2))
    part2 = torch.abs(X2 * X3)
    part3 = -torch.pow(X3, 2 * torch.abs(X4))
    part4 = (X1 * X4) ** 2
    part5 = torch.log(X4**2 + X5**2 + X7**2 + X8**2 + 1e-6)
    part6 = X9 + 1 / (1 + X10**2)

    result = part1 + part2 + part3 + part4 + part5 + part6
    return result


def F5(X):
    X1, X2, X3, X4, X5, X6, X7, X8, X9, X10 = X.transpose(0, 1)

    part1 = 1 / (1 + X1**2 + X2**2 + X3**2)
    part2 = torch.sqrt(torch.clamp(torch.exp(X4 + X5), max=1e6))
    part3 = torch.abs(X6 + X7)
    part4 = X8 * X9 * X10

    result = part1 + part2 + part3 + part4
    return result


def F6(X):
    X1, X2, X3, X4, X5, X6, X7, X8, X9, X10 = X.transpose(0, 1)

    part1 = torch.exp(torch.abs(X1 * X2) + 1)
    part2 = -torch.exp(torch.abs(X3 + X4))
    part3 = torch.cos(X5 + X6 - X8)
    part4 = torch.sqrt(X8**2 + X9**2 + X10**2 + 1e-6)

    result = part1 + part2 + part3 + part4
    return result


def F7(X):
    X1, X2, X3, X4, X5, X6, X7, X8, X9, X10 = X.transpose(0, 1)

    part1 = (torch.atan(X1) + torch.atan(X2)) ** 2
    part2 = torch.max(X3 * X4 + X6, torch.tensor(0.0, device=X.device))
    part3 = -1 / (1 + (X4 * X5 * X6 * X7 * X8) ** 2 + 1e-6)
    part4 = (torch.abs(X7) / (1 + torch.abs(X9))) ** 5
    part5 = torch.sum(X, dim=1)

    result = part1 + part2 + part3 + part4 + part5
    return result


def F8(X):
    X1, X2, X3, X4, X5, X6, X7, X8, X9, X10 = X.transpose(0, 1)

    part1 = X1 * X2
    part2 = 2 ** (X3 + X5 + X6)
    part3 = 2 ** (X3 + X4 + X5 + X7)
    part4 = torch.sin(X7 * torch.sin(X8 + X9))
    part5 = torch.acos(torch.clamp(0.9 * X10, min=-1 + 1e-6, max=1 - 1e-6))

    result = part1 + part2 + part3 + part4 + part5
    return result


def F9(X):
    X1, X2, X3, X4, X5, X6, X7, X8, X9, X10 = X.transpose(0, 1)

    part1 = torch.tanh(X1 * X2 + X3 * X4) * torch.sqrt(torch.abs(X5) + 1e-6)
    part2 = torch.exp(X5 + X6)
    part3 = torch.log((X6 * X7 * X8) ** 2 + 1)
    part4 = X9 * X10
    part5 = 1 / (1 + torch.abs(X10))

    result = part1 + part2 + part3 + part4 + part5
    return result


def F10(X):
    X1, X2, X3, X4, X5, X6, X7, X8, X9, X10 = X.transpose(0, 1)

    part1 = torch.sinh(X1 + X2)
    part2 = torch.acos(
        torch.clamp(torch.tanh(X3 + X5 + X7), min=-1 + 1e-6, max=1 - 1e-6)
    )
    part3 = torch.cos(X4 + X5)
    part4 = 1 / torch.clamp(torch.cos(X7 * X9), min=1e-6)

    result = part1 + part2 + part3 + part4
    return result


def F11(X):
    return torch.sum(X, dim=1)  # No interaction


def F12(X):
    return torch.prod(X, dim=1)  # All features interact
