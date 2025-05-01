import torch
import numpy as np


def create_sliding_windows(series, window_size):
    X, y = [], []
    for i in range(len(series) - window_size):
        X.append(series[i:i+window_size])
        y.append(series[i+window_size])
    X = torch.tensor(np.array(X), dtype=torch.float32).unsqueeze(-1)  # (N, window, 1)
    y = torch.tensor(np.array(y), dtype=torch.float32).unsqueeze(-1)  # (N, 1)
    return X, y


def get_data_split(data, split_ratio, window_size, device):
    split = int(len(data) * split_ratio)
    train_data = data[:split]
    test_data = data[split:]

    X_train, y_train = create_sliding_windows(train_data, window_size)
    X_test, y_test = create_sliding_windows(test_data, window_size)

    train_seq = torch.utils.data.TensorDataset(X_train.to(device), y_train.to(device))
    test_seq = torch.utils.data.TensorDataset(X_test.to(device), y_test.to(device))

    return train_seq, test_seq


def get_batch(dataset, i, batch_size):
    seq_len = min(batch_size, len(dataset) - i)
    X_batch = torch.stack([dataset[j][0] for j in range(i, i + seq_len)])
    y_batch = torch.stack([dataset[j][1] for j in range(i, i + seq_len)])

    # Transformer expects input shape: (input_window, batch_size, 1)
    X_batch = X_batch.transpose(0, 1)  # (input_window, batch_size, 1)
    return X_batch, y_batch



