import torch
from torch.utils.data import Dataset, DataLoader
import polars as pl
import json
from modules import utils, metrics


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


class IQDataset(Dataset):
    def __init__(self,
                 features: torch.Tensor,
                 targets: torch.Tensor,
                 nperseg: int,
                 frame_length: int = None,
                 stride: int = 1):
        """
        Dataset для комплексных IQ-сигналов.
        :param features: (N,) комплексный тензор
        :param targets:  (N,) комплексный тензор
        :param nperseg: длина сегмента
        :param frame_length: длина фрейма (если None — используется сегментирование)
        :param stride: шаг фреймирования
        """
        self.nperseg = nperseg
        self.frame_length = frame_length
        self.stride = stride

        seg_f = self._split_segments(features)
        seg_t = self._split_segments(targets)

        if frame_length is None:
            self.features = seg_f
            self.targets = seg_t
        else:
            self.features = self._extract_frames(seg_f)
            self.targets = self._extract_frames(seg_t)

    def _split_segments(self, data: torch.Tensor) -> torch.Tensor:
        N = data.shape[0]
        segments = []
        for i in range(0, N, self.nperseg):
            seg = data[i:i + self.nperseg]
            if seg.shape[0] < self.nperseg:
                pad = torch.zeros(self.nperseg - seg.shape[0], dtype=data.dtype)
                seg = torch.cat([seg, pad], dim=0)
            segments.append(seg)
        return torch.stack(segments)  # -> (num_segments, nperseg)

    def _extract_frames(self, segments: torch.Tensor) -> torch.Tensor:
        frames = []
        for seg in segments:
            for i in range(0, seg.shape[0] - self.frame_length + 1, self.stride):
                frame = seg[i:i + self.frame_length]
                frames.append(frame)
        return torch.stack(frames)  # -> (num_frames, frame_length)

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]


def build_dataloaders(data_dict, frame_length, batch_size, batch_size_eval, arch=None):
    nperseg = data_dict["config"]["nperseg"]

    x_train = utils.complex_to_iq(data_dict["train_input"])
    y_train = utils.complex_to_iq(data_dict["train_output"])
    x_val = utils.complex_to_iq(data_dict["val_input"])
    y_val = utils.complex_to_iq(data_dict["val_output"])

    if arch == "dla":
        gain = metrics.calculate_gain_complex(x_train, y_train)
        y_train = gain * x_train
        y_val = gain * x_val
    elif arch == "ila":
        y_train = x_train
        y_val = x_val
    elif arch == "ilc":
        y_train = utils.complex_to_iq(data_dict["ilc_train_output"])
        # y_val = utils.complex_to_iq(data_dict["ilc_val_output"])
        y_val = y_train
        
    train_set = IQDataset(x_train, y_train, nperseg=nperseg, frame_length=frame_length)
    val_set = IQDataset(x_val, y_val, nperseg=nperseg, frame_length=None)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size_eval, shuffle=False)

    return train_loader, val_loader
