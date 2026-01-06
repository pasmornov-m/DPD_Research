import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import json
from typing import Dict, Any
from modules import utils, metrics


def load_csv_to_tensor(file_path: str) -> torch.Tensor:
    data = pd.read_csv(file_path)
    required_columns = {'I', 'Q'}
    if not required_columns.issubset(data.columns):
        raise ValueError(f"CSV file must contain columns: {required_columns}")
    if data[['I', 'Q']].isnull().any().any():
        raise ValueError("CSV file contains missing values.")
    i_values = data['I'].to_numpy(dtype='float32')
    q_values = data['Q'].to_numpy(dtype='float32')
    i = torch.from_numpy(i_values)
    q = torch.from_numpy(q_values)
    return torch.complex(i, q)


def load_data(file_path: str) -> Dict[str, Any]:
    with open(f'{file_path}/spec.json') as json_file:
        config = json.load(json_file)
    return {
        "config": config,
        "train_input": load_csv_to_tensor(f"{file_path}/train_input.csv"),
        "train_output": load_csv_to_tensor(f"{file_path}/train_output.csv"),
        "val_input": load_csv_to_tensor(f"{file_path}/val_input.csv"),
        "val_output": load_csv_to_tensor(f"{file_path}/val_output.csv"),
    }


class DataContainer:
    def __init__(self):
        self.train_input = None
        self.train_input_orig = None        
        self.train_output = None
        self.val_input = None
        self.val_input_orig = None
        self.val_output = None
        self.train_output_target = None
        self.val_output_target = None
        self.ilc_train_output = None
        self.ilc_val_output = None
        
        self.input_signal_fs = None
        self.bw_main_ch = None
        self.nperseg = None
        self.bw_sub_ch = None
        self.n_sub_ch = None
        self.sub_ch = None
        
        self.gain = None
        
    def load_data(self, file_path: str):
        with open(f'{file_path}/spec.json') as json_file:
            config = json.load(json_file)
        
        self.input_signal_fs = config['input_signal_fs']
        self.bw_main_ch = config['bw_main_ch']
        self.nperseg = config['nperseg']
        self.bw_sub_ch = config['bw_sub_ch']
        self.n_sub_ch = config['n_sub_ch']
        self.sub_ch = config['sub_ch']
            
        self.train_input = load_csv_to_tensor(f"{file_path}/train_input.csv")
        self.train_output = load_csv_to_tensor(f"{file_path}/train_output.csv")
        self.val_input = load_csv_to_tensor(f"{file_path}/val_input.csv")
        self.val_output = load_csv_to_tensor(f"{file_path}/val_output.csv")
        
        self.train_input_orig = self.train_input
        self.val_input_orig = self.val_input
        
        assert self.train_input.shape == self.train_input.shape
        assert self.val_input.shape == self.val_output.shape
    
    def get_gained_signals(self):
        self.gain = metrics.calculate_gain_complex(self.train_input, self.train_output)
        self.train_output_target = self.gain * self.train_input
        self.val_output_target = self.gain * self.val_input
    
    def cut_signals(self, M: int):
        if M <= 0:
            return
        for attr in ["train_input", "train_output", "val_input", "val_output"]:
            tensor = getattr(self, attr)
            if tensor is not None:
                setattr(self, attr, tensor[M:])
        
    def to_dict(self) -> dict:
        """Возвращает все сигналы и config как словарь (по необходимости)"""
        config_dict = {k: getattr(self, k) for k in ['input_signal_fs', 'bw_main_ch', 'nperseg', 'bw_sub_ch', 'n_sub_ch', 'sub_ch']}
        return {
            "config": config_dict,
            "train_input": self.train_input,
            "train_output": self.train_output,
            "val_input": self.val_input,
            "val_output": self.val_output,
            "train_output_target": self.train_output_target,
            "val_output_target": self.val_output_target,
            "ilc_train_output": self.ilc_train_output,
            "ilc_val_output": self.ilc_val_output
        }
    
    def reset_ilc(self):
        self.u_k_train = None
        self.u_k_val = None


class IQDataset(Dataset):
    def __init__(self,
                 features: torch.Tensor,
                 targets: torch.Tensor,
                 nperseg: int,
                 frame_length: int = None,
                 stride: int = 1):
        """
        Dataset для комплексных IQ-сигналов.
        :param features: (N,2) тензор
        :param targets:  (N,2) тензор
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
                pad = torch.zeros(self.nperseg - seg.shape[0], seg.shape[1], dtype=seg.dtype)
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

    def __getitem__(self, idx: int):
        return self.features[idx], self.targets[idx]


def build_dataloaders(data_dict, frame_length, batch_size, batch_size_eval, arch=None):
    nperseg = data_dict["config"]["nperseg"]

    x_train = utils.complex_to_iq(data_dict["train_input"])
    y_train = utils.complex_to_iq(data_dict["train_output"])
    x_val = utils.complex_to_iq(data_dict["val_input"])
    y_val = utils.complex_to_iq(data_dict["val_output"])
    
    if arch == "dla":
        y_train = utils.complex_to_iq(data_dict["train_output_target"])
        y_val = utils.complex_to_iq(data_dict["val_output_target"])
    elif arch == "ila":
        y_train = x_train
        y_val = x_val
    elif arch == "ilc":
        y_train = utils.complex_to_iq(data_dict["ilc_train_output"])
        ilc_val_output = data_dict.get("ilc_val_output")
        if ilc_val_output is not None:
            y_val = utils.complex_to_iq(ilc_val_output)
        else:
            y_val = y_train

    train_set = IQDataset(x_train, y_train, nperseg=nperseg, frame_length=frame_length)
    val_set = IQDataset(x_val, y_val, nperseg=nperseg, frame_length=None)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size_eval, shuffle=False)

    return train_loader, val_loader





def build_X_in(iq_signal: torch.Tensor, M: int = 5, P: int = 4) -> torch.Tensor:
    """
    Формирует входной тензор X_in для RTVDNN/RTDTNN полностью векторизованно.

    Parameters:
        iq_signal: torch.Tensor комплексной формы (N,) dtype=torch.complex64 или torch.complex128
        M: memory depth
        P: max amplitude power

    Returns:
        X: тензор формы (N-M, M+1, 2+P), dtype=torch.float32
    """
    eps = 1e-8

    T = M + 1

    if iq_signal.is_complex():
        I = iq_signal.real
        Q = iq_signal.imag
        amp = torch.abs(iq_signal + eps)
    elif len(iq_signal.shape) == 2 and iq_signal.shape[-1] == 2:
        I = iq_signal[..., 0]
        Q = iq_signal[..., 1]
        amp = torch.sqrt(I**2 + Q**2) + eps
    else:
        raise ValueError(f"Unsupported iq_signal shape: {iq_signal.shape}")

    # Создаем тензор с окнами: shape = (N-M, T)
    I_windows = I.unfold(0, T, 1)       # (N-M, M+1)
    Q_windows = Q.unfold(0, T, 1)
    amp_windows = amp.unfold(0, T, 1)

    features = [I_windows, Q_windows] + [amp_windows**p for p in range(1, P+1)]

    X = torch.cat([f.unsqueeze(-1) for f in features], dim=-1)
    X = X.float()

    return X


class RTDTNNDataset(torch.utils.data.Dataset):
    def __init__(self, X_in: torch.Tensor, Y_out: torch.Tensor):
        """
        X_in: (N-M, M+1, 2+P)
        Y_out: (N-M, 2)
        """
        self.X = X_in

        if Y_out.is_complex():
            self.Y = utils.complex_to_iq(Y_out)
        else:
            self.Y = Y_out
    
    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


def build_RTDTNN_dataloaders(container,
                            batch_size: int,
                            batch_size_eval: int,
                            M: int,
                            P: int,
                            arch: str = None):
    
    if arch is None:
        raise ValueError("arch must be specified: 'pa', 'dla', 'ila', or 'ilc'")
    arch = arch.lower()
    
    x_train = build_X_in(container.train_input_orig, M, P)
    x_val = build_X_in(container.val_input_orig, M, P)
    
    if arch == "pa":
        y_train = container.train_output
        y_val = container.val_output
    elif arch == "dla":
        y_train = container.train_output_target
        y_val = container.val_output_target
    elif arch == "ila":
        y_train = container.train_input_orig
        y_val = container.val_input_orig
    elif arch == "ilc":
        y_train = utils.complex_to_iq(container.ilc_train_output)
        ilc_val_output = container.ilc_val_output
        if ilc_val_output is not None:
            y_val = utils.complex_to_iq(ilc_val_output)
        else:
            y_val = y_train
    else:
        raise ValueError(f"Unknown arch '{arch}'")
    
    train_dataset = RTDTNNDataset(x_train, y_train)
    val_dataset = RTDTNNDataset(x_val, y_val)
    
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size_eval, shuffle=True)
    
    return train_dataloader, val_dataloader