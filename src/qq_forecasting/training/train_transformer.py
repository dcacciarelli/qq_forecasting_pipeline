import torch
import time

from qq_forecasting.utils.transformer_preprocessing import get_batch

def train(model, train_data, optimizer, criterion, scheduler, input_window, batch_size, epoch):
    model.train()
    total_loss, start_time = 0.0, time.time()

    for batch, i in enumerate(range(0, len(train_data) - 1, batch_size)):
        x, y = get_batch(train_data, i, batch_size, input_window)
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.7)
        optimizer.step()
        total_loss += loss.item()

        if batch % 10 == 0 and batch > 0:
            print(f"| epoch {epoch} | batch {batch} | loss {loss.item():.6f}")

    scheduler.step()


def evaluate(model, val_data, criterion, input_window, batch_size):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for i in range(0, len(val_data) - 1, batch_size):
            x, y = get_batch(val_data, i, batch_size, input_window)
            output = model(x)
            total_loss += len(x[0]) * criterion(output, y).cpu().item()
    return total_loss / len(val_data)
