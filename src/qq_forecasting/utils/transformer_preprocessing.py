import torch
import numpy as np


def create_inout_sequences(data, input_window, output_window):
    seqs = []
    L = len(data)
    for i in range(L - input_window):
        x = data[i:i + input_window]
        y = data[i + output_window:i + input_window + output_window]
        seqs.append((x, y))
    return torch.FloatTensor(seqs)


def get_data_split(data, split_ratio, input_window, output_window, device):
    split = int(len(data) * split_ratio)
    train_data = data[:split]
    test_data = data[split:]

    train_seq = create_inout_sequences(train_data, input_window, output_window)[:-output_window]
    test_seq = create_inout_sequences(test_data, input_window, output_window)[:-output_window]

    return train_seq.to(device), test_seq.to(device)


def get_batch(source, i, batch_size, input_window):
    seq_len = min(batch_size, len(source) - 1 - i)
    data = source[i:i + seq_len]
    x = torch.stack(torch.stack([item[0] for item in data]).chunk(input_window, 1))
    y = torch.stack(torch.stack([item[1] for item in data]).chunk(input_window, 1))
    return x, y
