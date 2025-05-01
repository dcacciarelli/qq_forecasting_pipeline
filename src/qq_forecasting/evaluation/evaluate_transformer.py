import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from qq_forecasting.data.load_demand import load_demand_data
from qq_forecasting.models.transformer_model import TransformerEncoder
from qq_forecasting.utils.transformer_preprocessing import get_data_split, get_batch
from qq_forecasting.training.train_transformer import train, evaluate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on {device}")

input_window = 10
output_window = 1
batch_size = 64
lr = 0.00005
epochs = 10

df = load_demand_data("data/raw", years=[2019, 2020, 2021, 2022, 2023, 2024])
series = df["ND"].values[:1_000]
scaler = MinMaxScaler()
series_scaled = scaler.fit_transform(series.reshape(-1, 1)).flatten()

train_data, val_data = get_data_split(series_scaled, 0.8, input_window, output_window, device)
model = TransformerEncoder().to(device)

criterion = torch.nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

for epoch in range(1, epochs + 1):
    train(model, train_data, optimizer, criterion, scheduler, batch_size, epoch)
    if epoch == epochs:
        val_loss = evaluate(model, val_data, criterion, input_window, batch_size)
        print(f"Validation loss at final epoch: {val_loss:.6f}")

# Forecast
model.eval()
pred, actual = torch.Tensor(0), torch.Tensor(0)
with torch.no_grad():
    for i in range(len(val_data) - 1):
        x, y = get_batch(val_data, i, 1)
        out = model(x)
        pred = torch.cat((pred, out[-1].view(-1).cpu()), 0)
        actual = torch.cat((actual, y[-1].view(-1).cpu()), 0)

plt.plot(actual, label="Actual", color="red", alpha=0.7)
plt.plot(pred, label="Forecast", color="blue", linestyle="dashed")
plt.title("Transformer Forecast on Demand")
plt.xlabel("Time Steps")
plt.legend()
plt.tight_layout()
plt.show()
