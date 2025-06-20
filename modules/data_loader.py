import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import polars as pl
import json


def load_csv_to_tensor(file_path):
    data = pl.read_csv(file_path)
    required_columns = {'I', 'Q'}
    if not required_columns.issubset(data.columns):
        raise ValueError(f"CSV file must contain columns: {required_columns}")
    if data.select(pl.all().is_null().any()).to_series(0).any():
        raise ValueError("CSV file contains missing values.")
    i_values = data['I'].to_numpy()
    q_values = data['Q'].to_numpy()
    return torch.tensor(i_values + 1j * q_values, dtype=torch.cfloat)


def load_data(file_path):
    with open(f'{file_path}/spec.json') as json_file:
        config = json.load(json_file)
    return {
        "config": config,
        "train_input": load_csv_to_tensor(f"{file_path}/train_input.csv"),
        "train_output": load_csv_to_tensor(f"{file_path}/train_output.csv"),
        "val_input": load_csv_to_tensor(f"{file_path}/val_input.csv"),
        "val_output": load_csv_to_tensor(f"{file_path}/val_output.csv"),
    }

class SequenceDataset(Dataset):
    def __init__(self,
                 iq_data: torch.Tensor,
                 target: torch.Tensor,
                 window_size: int,
                 stride: int = 1
                 ):
        assert iq_data.shape[0] == target.shape[0], "iq_data и target должны быть одной длины"
        self.iq_data = iq_data
        self.target = target
        self.window_size = window_size
        self.stride = stride
        self.num_windows = max((len(iq_data) - window_size) // stride + 1, 0)

    def __len__(self):
        return self.num_windows

    def __getitem__(self, idx):
        start = idx * self.stride
        x = self.iq_data[start:start + self.window_size]
        tgt_idx = start + self.window_size
        if tgt_idx < len(self.target):
            y = self.target[tgt_idx]
        else:
            y = self.target[-1]
        return x, y

def make_dataloader(iq_tensor, target_tensor,
                    window_size, stride,
                    batch_size=64, shuffle=True, num_workers=0):
    ds = SequenceDataset(iq_tensor, target_tensor, window_size, stride)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=True)


# 4) Пример цикла train / eval
def train_epoch(model: nn.Module, loader: DataLoader,
                optimizer, criterion):
    model.train()
    total_loss = 0.0
    for xb, yb in loader:
        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
    return total_loss / len(loader.dataset)


def eval_epoch(model: nn.Module, loader: DataLoader,
               criterion):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for xb, yb in loader:
            out = model(xb)
            loss = criterion(out, yb)
            total_loss += loss.item() * xb.size(0)
    return total_loss / len(loader.dataset)