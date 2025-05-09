import os
import yaml
import torch
import joblib
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset

from qq_forecasting.models.transformer_model import TransformerEncoder, train_transformer
from qq_forecasting.utils import create_sliding_windows

# ========== LOAD CONFIG ==========
with open("config/transformer_demand.yaml") as f:
    config = yaml.safe_load(f)

DATA_PATH = config["paths"]["data_path"]
MODEL_SAVE_PATH = config["paths"]["model_path"]

hp = config["training"]
model_cfg = config["model"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========== LOAD ==========
train = pd.read_csv(os.path.join(DATA_PATH, "train.csv")).squeeze()
if hp["max_training_samples"]:
    train = train.iloc[-hp["max_training_samples"]:]

# ========== SLIDING WINDOWS ==========
X_train, y_train = create_sliding_windows(train.values, window_size=hp["window_size"])
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=hp["batch_size"], shuffle=False)

# ========== TRAIN ==========
model = TransformerEncoder(**model_cfg).to(device)
train_transformer(model, train_loader, num_epochs=hp["num_epochs"], lr=hp["learning_rate"])

# ========== SAVE ==========
os.makedirs(os.path.dirname(config["paths"]["model_path"]), exist_ok=True)
torch.save(model.state_dict(), config["paths"]["model_path"])
print(f"✅ Transformer model saved to {config['paths']['model_path']}")
