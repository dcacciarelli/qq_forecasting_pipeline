import torch


def predict_lstm(model, X):
    model.eval()
    with torch.no_grad():
        predictions = model(X)
    return predictions.squeeze()
