import torch
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

def load_data(folder_path):
    with open(f'{folder_path}/spec.json') as json_file:
        config = json.load(json_file)
    return {
        "config": config,
        "train_input": load_csv_to_tensor(f"{folder_path}/train_input.csv"),
        "train_output": load_csv_to_tensor(f"{folder_path}/train_output.csv"),
        "val_input": load_csv_to_tensor(f"{folder_path}/val_input.csv"),
        "val_output": load_csv_to_tensor(f"{folder_path}/val_output.csv"),
    }
