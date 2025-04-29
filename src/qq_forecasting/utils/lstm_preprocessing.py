import torch


def create_sliding_windows(series, window_size):
    X, y = [], []
    for i in range(len(series) - window_size):
        X.append(series[i:i+window_size].values)
        y.append(series[i+window_size])
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    return X.unsqueeze(-1), y.unsqueeze(-1)  # Add feature dimension
