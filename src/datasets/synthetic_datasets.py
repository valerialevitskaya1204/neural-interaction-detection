import numpy as np

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

import numpy as np


def F1(X):
    # Unpack the input X by transposing it (assuming X is a 2D numpy array where each row corresponds to one variable)
    X1, X2, X3, X4, X5, X6, X7, X8, X9, X10 = X.T

    # π is a constant
    pi = np.pi

    # Compute each part of the expression using NumPy operations
    part1 = np.power(pi, X1 * X2) * np.sqrt(2 * X3)  # π * x1 * x2√(2 * x3)
    main_effects = -np.arcsin(X4)  # sin^(-1)(x4)
    part3 = np.log(X3 + X5)  # log(x3 + x5)
    part4 = -(X9 / X10) * np.sqrt(X7 / X8)  # x9/x10 * rx7 * x8
    part5 = -X2 * X7  # x2 * x7

    # Combine all parts
    result = part1 + part3 + part4 + part5 + main_effects
    ground_truth = [{1, 2, 3}, {3, 5}, {7, 8, 9, 10}, {2, 7}]

    return result, ground_truth


def F2(X):
    # Unpack the input X by transposing it (assuming X is a 2D numpy array where each row corresponds to one variable)
    X1, X2, X3, X4, X5, X6, X7, X8, X9, X10 = X.T

    # π is a constant
    pi = np.pi

    # Compute each part of the expression using NumPy operations
    part1 = np.power(pi, X1 * X2) * np.sqrt(2 * np.abs(X3))  # π * x1 * x2√(2 * x3)
    main_effects = -np.arcsin(0.5 * X4)  # sin^(-1)(x4)
    part3 = np.log(np.abs(X3 + X5) + 1)  # log(x3 + x5)
    abs10 = 1 + np.abs(X10)
    abs8 = 1 + np.abs(X8)
    ratio = X7 / abs8
    part4 = -(X9 / abs10) * np.sqrt(ratio)  # x9/x10 * rx7 * x8
    part5 = -X2 * X7  # x2 * x7

    # Combine all parts
    result = part1 + part3 + part4 + part5 + main_effects
    ground_truth = [{1, 2, 3}, {3, 5}, {7, 8, 9, 10}, {2, 7}]

    return result, ground_truth


def F3(X):
    # Unpack the input X by transposing it (assuming X is a 2D numpy array where each row corresponds to one variable)
    X1, X2, X3, X4, X5, X6, X7, X8, X9, X10 = X.T

    # Ensure no invalid values before performing power operations
    # Add small epsilon values to avoid division by zero or issues with negative values
    epsilon = 1e-8

    part1 = np.exp(np.abs(X1 - X2))  # exp(|X1 - X2|)
    part2 = np.abs(X2 * X3)  # |X2 * X3|

    # Safe power operation: Add epsilon to X3 and X4 to avoid issues with 0
    part3 = np.power(X3 + epsilon, 2 * np.abs(X4))  # X^2 * |X4|^3

    part4 = np.log(
        np.power(X4, 2) + np.power(X5, 2) + np.power(X7, 2) + np.power(X8, 2)
    )  # log(X4^2 + X5^2 + X7^2 + X8^2)
    main_effects = X9 + (1 / (1 + np.power(X10, 2)))

    # Combine all parts
    result = part1 + part2 - part3 + part4 + main_effects
    ground_truth = [{1, 2}, {2, 3}, {3, 4}, {4, 5, 7, 8}]

    return result, ground_truth


def F4(X):
    # Unpack the input X by transposing it (assuming X is a 2D numpy array where each row corresponds to one variable)
    X1, X2, X3, X4, X5, X6, X7, X8, X9, X10 = X.T

    part1 = np.exp(np.abs(X1 - X2))  # exp(|X1 - X2|)
    part2 = np.abs(X2 * X3)  # |X2 * X3|
    part3 = -np.power(X3, 2 * np.abs(X4))  # X^2 * |X4|^3
    part4 = np.power(X1 * X4, 2)
    part4 = np.log(
        np.power(X4, 2) + np.power(X5, 2) + np.power(X7, 2) + np.power(X8, 2)
    )  # log(X4^2 + X5^2 + X7^2 + X8^2)
    main_effects = X9 + (1 / (1 + np.power(X10, 2)))

    # Combine all parts
    result = part1 + part2 + part3 + part4 + main_effects
    ground_truth = [{1, 2}, {2, 3}, {3, 4}, {4, 5, 7, 8}]

    return result, ground_truth


def F5(X):
    # Unpack the input X by transposing it (assuming X is a 2D numpy array where each row corresponds to one variable)
    X1, X2, X3, X4, X5, X6, X7, X8, X9, X10 = X.T

    # Compute each part of the expression
    part1 = 1 / (
        1 + np.power(X1, 2) + np.power(X2, 2) + np.power(X3, 2)
    )  # 1 / (1 + x1^2 + x2^2 + x3^2)
    part2 = np.sqrt(np.exp(X4 + X5))  # sqrt(exp(x4 + x5))
    main_effects = np.abs(X6 + X7)  # |x6 + x7|
    part4 = X8 * X9 * X10  # x8 * x9 * x10

    # Combine all parts to compute the result
    result = part1 + part2 + main_effects + part4
    ground_truth = [{1, 2, 3}, {4, 5}, {8, 9, 10}]

    return result, ground_truth


def F6(X):
    # Unpack the input X by transposing it (assuming X is a 2D numpy array where each row corresponds to one variable)
    X1, X2, X3, X4, X5, X6, X7, X8, X9, X10 = X.T

    # Compute each part of the expression
    part1 = np.exp(np.abs(X1 * X2) + 1)  # exp(|X1 * X2| + 1)
    part2 = -np.exp(np.abs(X3 + X4) + 1)  # exp(|X3 + X4| + 1)
    part3 = np.cos(X5 + X6 - X8)  # cos(X5 + X6 - X8)
    part4 = np.sqrt(
        np.power(X8, 2) + np.power(X9, 2) + np.power(X10, 2)
    )  # sqrt(X8^2 + X9^2 + X10^2)

    # Combine all parts to compute the result
    result = part1 + part2 + part3 + part4
    ground_truth = [{1, 2}, {3, 4}, {5, 6, 8}, {8, 9, 10}]

    return result, ground_truth


def F7(X):
    # Unpack the input X by transposing it (assuming X is a 2D numpy array where each row corresponds to one variable)
    X1, X2, X3, X4, X5, X6, X7, X8, X9, X10 = X.T

    # Compute each part of the expression
    part1 = np.power(np.arctan(X1) + np.arctan(X2), 2)  # (arctan(x1) + arctan(x2))^2
    part2 = np.maximum(X3 * X4 + X6, 0)  # max(x3 * x4 + x6, 0)
    part3 = 1 / (
        1 + np.power(X4 * X5 * X6 * X7 * X8, 2)
    )  # 1 / (1 + (x4 * x5 * x6 * x7 * x8)^2)
    part4 = np.power(np.abs(X7) / (1 + np.abs(X9)), 5)  # (|x7| / (1 + |x9|))^5
    part5 = np.sum(X, axis=1)  # sum(x1 to x10)

    # Combine all parts to compute the result
    result = part1 + part2 - part3 + part4 + part5
    ground_truth = [{4, 5, 6, 7, 8}, {7, 9}]

    return result, ground_truth


def F8(X):
    # Unpack the input X by transposing it (assuming X is a 2D numpy array where each row corresponds to one variable)
    X1, X2, X3, X4, X5, X6, X7, X8, X9, X10 = X.T

    # Compute each part of the expression
    part1 = X1 * X2  # x1 * x2
    part2 = 2 * np.power(X3, 3) + X5 + X6  # 2 * x3^3 + x5 + x6
    part3 = 2 * np.power(X3, 3) + X4 + X5 + X7  # 2 * x3^3 + x4 + x5 + x7
    part4 = np.sin(X7 * np.sin(X8 + X9))  # sin(x7 * sin(x8 + x9))
    part5 = np.arccos(0.9 * X10)  # arccos(0.9 * x10)

    # Combine all parts to compute the result
    result = part1 + part2 + part3 + part4 + part5
    ground_truth = [{1, 2},  {3}, {7, 9}]
    return result


def F9(X):
    # Unpack the input X by transposing it (assuming X is a 2D numpy array where each row corresponds to one variable)
    X1, X2, X3, X4, X5, X6, X7, X8, X9, X10 = X.T

    # Compute each part of the expression
    part1 = np.tanh(X1 * X2 + X3 * X4) * np.sqrt(
        np.abs(X5)
    )  # tanh(x1 * x2 + x3 * x4) * sqrt(|x5|)
    part2 = np.exp(X5 + X6)  # exp(x5 + x6)
    part3 = np.log(np.power(X6 * X7 * X8, 2) + 1)  # log((x6 * x7 * x8)^2 + 1)
    part4 = X9 * X10  # x9 * x10
    part5 = 1 / (1 + np.abs(X10))  # 1 / (1 + |x10|)

    # Combine all parts to compute the result
    result = part1 + part2 + part3 + part4 + part5

    return result


def F10(X):
    # Unpack the input X by transposing it (assuming X is a 2D numpy array where each row corresponds to one variable)
    X1, X2, X3, X4, X5, X6, X7, X8, X9, X10 = X.T

    # Compute each part of the expression
    part1 = np.sinh(X1 + X2)  # sinh(x1 + x2)
    part2 = np.arccos(np.tanh(X3 + X5 + X7))  # arccos(tanh(x3 + x5 + x7))
    part3 = np.cos(X4 + X5)  # cos(x4 + x5)
    part4 = 1 / np.cos(X7 * X9)  # sec(x7 * x9), which is 1 / cos(x7 * x9)

    # Combine all parts to compute the result
    result = part1 + part2 + part3 + part4

    return result


def F11(X):
    # Unpack the input X by transposing it (assuming X is a 2D numpy array where each row corresponds to one variable)
    X1, X2, X3, X4, X5, X6, X7, X8, X9, X10 = X.T

    # Compute the result (sum of all variables)
    result = X1 + X2 + X3 + X4 + X5 + X6 + X7 + X8 + X9 + X10
    ground_truth = []  # No specific ground truth set, sum of all

    return result, ground_truth


def F12(X):
    # Unpack the input X by transposing it (assuming X is a 2D numpy array where each row corresponds to one variable)
    X1, X2, X3, X4, X5, X6, X7, X8, X9, X10 = X.T

    # Compute the result (product of all variables)
    result = X1 * X2 * X3 * X4 * X5 * X6 * X7 * X8 * X9 * X10
    ground_truth = [
        {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
    ]  # All features interact with each other (multiplication of all)

    return result, ground_truth
