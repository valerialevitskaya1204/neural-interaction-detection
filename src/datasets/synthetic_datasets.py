import numpy as np

def F1(x):
    return np.pi * x[:, 0] * x[:, 1] * np.sqrt(2 * x[:, 2]) - np.arcsin(x[:, 3]) + np.log(x[:, 4] + x[:, 5]) - (x[:, 8] / x[:, 9]) * np.sqrt(x[:, 6] / x[:, 7]) - x[:, 1] * x[:, 6]

def F2(x):
    return np.pi**(x[:, 0] * x[:, 1] * 2) * np.sqrt(2 * np.abs(x[:, 2])) - np.arcsin(0.5 * x[:, 3]) + np.log(np.abs(x[:, 2] + x[:, 4]) + 1) + (x[:, 8] / (1 + np.abs(x[:, 9]))) * np.sqrt(x[:, 6] / (1 + np.abs(x[:, 7]))) - x[:, 1] * x[:, 6]

def F3(x):
    return np.exp(np.abs(x[:, 0] - x[:, 1])) + np.abs(x[:, 1] * x[:, 2]) - (x[:, 3]**2 * np.abs(x[:, 3])) + np.log(x[:, 3]**2 + x[:, 4]**2 + x[:, 6]**2 + x[:, 7]**2) + x[:, 8] + 1 / (1 + x[:, 9]**2)

def F4(x):
    return np.exp(np.abs(x[:, 0] - x[:, 1])) + np.abs(x[:, 1] * x[:, 2]) - (x[:, 3]**2 * np.abs(x[:, 3])) + (x[:, 0] * x[:, 3])**2 + np.log(x[:, 3]**2 + x[:, 4]**2 + x[:, 6]**2 + x[:, 7]**2) + x[:, 8] + 1 / (1 + x[:, 9]**2)

def F5(x):
    return 1 / (1 + x[:, 0]**2 + x[:, 1]**2 + x[:, 2]**2) + np.sqrt(np.exp(x[:, 3] + x[:, 4])) + np.abs(x[:, 5] + x[:, 6]) + x[:, 7] * x[:, 8] * x[:, 9]

def F6(x):
    return np.exp(np.abs(x[:, 0] * x[:, 1]) + 1) - np.exp(np.abs(x[:, 2] + x[:, 3]) + 1) + np.cos(x[:, 4] + x[:, 5] - x[:, 7]) + np.sqrt(x[:, 7]**2 + x[:, 8]**2 + x[:, 9]**2)

def F7(x):
    return (np.arctan(x[:, 0]) + np.arctan(x[:, 1]))**2 + np.maximum(x[:, 2] * x[:, 3] + x[:, 5], 0) - 1 / (1 + (x[:, 3] * x[:, 4] * x[:, 5] * x[:, 6] * x[:, 7])**2) + (np.abs(x[:, 6]) / (1 + np.abs(x[:, 8])))**5 + np.sum(x, axis=1)

def F8(x):
    return x[:, 0] * x[:, 1] + 2**(x[:, 2] + x[:, 4] + x[:, 5]) + 2**(x[:, 2] + x[:, 3] + x[:, 4] + x[:, 6]) + np.sin(x[:, 6] * np.sin(x[:, 7] + x[:, 8])) + np.arccos(0.9 * x[:, 9])

def F9(x):
    return np.tanh(x[:, 0] * x[:, 1] + x[:, 2] * x[:, 3]) * np.sqrt(np.abs(x[:, 4])) + np.exp(x[:, 4] + x[:, 5]) + np.log((x[:, 5] * x[:, 6] * x[:, 7])**2 + 1) + x[:, 8] * x[:, 9] + 1 / (1 + np.abs(x[:, 9]))

def F10(x):
    return np.sinh(x[:, 0] + x[:, 1]) + np.arccos(np.tanh(x[:, 2] + x[:, 4] + x[:, 6])) + np.cos(x[:, 3] + x[:, 4]) + 1 / np.cos(x[:, 6] * x[:, 8])

def generate_synthetic_data(func, num_samples=10000, epsilon=1e-8):
    x = np.zeros((num_samples, 10))
    
    # Function-specific constraints
    if func is F1:
        # F1 constraints
        x[:, 0] = np.random.uniform(-1, 1, num_samples)
        x[:, 1] = np.random.uniform(-1, 1, num_samples)
        x[:, 2] = np.random.uniform(0, 1, num_samples)  # x2 >=0
        x[:, 3] = np.random.uniform(-1, 1, num_samples)
        
        # Ensure x4 + x5 >0
        x4 = np.random.uniform(-1, 1, num_samples)
        lower = np.maximum(-x4, -1)
        x5 = lower + np.random.uniform(0, 1, num_samples) * (1 - lower)
        x[:, 4] = x4
        x[:, 5] = x5
        
        # x6 and x7 same sign and non-zero
        signs = np.random.choice([1, -1], num_samples)
        x[:, 6] = signs * np.random.uniform(epsilon, 1, num_samples)
        x[:, 7] = signs * np.random.uniform(epsilon, 1, num_samples)
        
        x[:, 8] = np.random.uniform(-1, 1, num_samples)
        
        # x9 non-zero
        x9_signs = np.random.choice([1, -1], num_samples)
        x[:, 9] = x9_signs * np.random.uniform(epsilon, 1, num_samples)

    elif func is F2:
        # F2 constraints: x6 >=0
        x = np.random.uniform(-1, 1, (num_samples, 10))
        x[:, 6] = np.random.uniform(0, 1, num_samples)  # x6 >=0

    elif func is F8:
        # F8 constraints: x9 in [-1/0.9, 1/0.9] ≈ [-1.111, 1.111]
        x = np.random.uniform(-1, 1, (num_samples, 10))
        x[:, 9] = np.random.uniform(-1.111, 1.111, num_samples)

    elif func is F10:
        # F10 constraints: x6*x8 != (2k+1)*π/2
        x = np.random.uniform(-1, 1, (num_samples, 10))
        # Ensure cos(x6*x8) != 0 by keeping product < π/2
        product = np.random.uniform(-np.pi/2 + 0.01, np.pi/2 - 0.01, num_samples)
        # Solve for x6 and x8 maintaining their individual ranges
        x[:, 6] = np.random.uniform(-1, 1, num_samples)
        x[:, 8] = product / (x[:, 6] + np.where(x[:, 6]==0, 0.1, 0))

    else:
        # Default case for F3-F7, F9
        x = np.random.uniform(-1, 1, (num_samples, 10))
    
    # Post-processing for common constraints
    if func in [F3, F4]:
        # Ensure sum(x3^2 + x4^2 + x6^2 + x7^2) > 0
        mask = (x[:, 3]**2 + x[:, 4]**2 + x[:, 6]**2 + x[:, 7]**2) <= 0
        if np.any(mask):
            x[mask, 3] = 0.1  # Set minimal non-zero value
    
    y = func(x)
    return x, y