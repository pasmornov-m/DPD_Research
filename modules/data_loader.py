import torch
import numpy as np
import json
from typing import Dict, Any, Callable
from modules import utils, metrics


def load_csv_to_tensor(file_path: str) -> torch.Tensor:
    data = np.genfromtxt(file_path, delimiter=',', names=True, dtype=np.float32)

    if 'I' not in data.dtype.names or 'Q' not in data.dtype.names:
        raise ValueError("CSV must contain columns: I, Q")

    i = data['I']
    q = data['Q']

    if np.isnan(i).any() or np.isnan(q).any():
        raise ValueError("CSV file contains missing values.")

    iq = torch.from_numpy(np.stack((i, q), axis=1))
    return iq


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
        self.gain = metrics.calculate_gain(self.train_input, self.train_output)
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
        """Reset ILC signals (u_k_train & u_k_val) to None"""
        self.u_k_train = None
        self.u_k_val = None


class IQDataset(torch.utils.data.Dataset):
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


def build_dataloaders(container: DataContainer, 
                      frame_length: int, 
                      batch_size: int, 
                      batch_size_eval: int, 
                      arch: str,
                      normalize_method: str | None = None):
        
    arch = arch.lower()
    assert arch in ('pa', 'dla', 'ila', 'ilc'), \
            "arch must be specified: 'pa', 'dla', 'ila' or 'ilc'"
    
    input_norm = utils.Normalizer(normalize_method)
    target_norm = utils.Normalizer(normalize_method)

    nperseg = container.nperseg

    if arch == "pa":
        x_train = container.train_input
        y_train = container.train_output
        x_val = container.val_input
        y_val = container.val_output
    
    elif arch == "dla":
        x_train = container.train_input
        y_train = container.train_output_target
        x_val = container.val_input
        y_val = container.val_output_target
        
    elif arch == "ila":
        x_train = container.train_output / container.gain
        y_train = container.train_input
        x_val = container.val_output / container.gain
        y_val = container.val_input
    
    elif arch == "ilc":
        x_train = container.train_input
        y_train = container.ilc_train_output
        x_val = container.val_input
        ilc_val_output = container.ilc_val_output
        if ilc_val_output is not None:
            y_val = ilc_val_output
        else:
            y_val = y_train
    
    x_train = input_norm.fit_transform(x_train)
    x_val = input_norm.transform(x_val)
    y_train = target_norm.fit_transform(y_train)
    y_val = target_norm.transform(y_val)
    
    train_set = IQDataset(x_train, y_train, nperseg=nperseg, frame_length=frame_length)
    val_set = IQDataset(x_val, y_val, nperseg=nperseg, frame_length=None)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size_eval, shuffle=False)

    return train_loader, val_loader, input_norm, target_norm


def build_RTDTNN_features(iq_signal: torch.Tensor, M: int = 5, P: int = 4) -> torch.Tensor:
    """
    Parameters:
        iq_signal: torch.Tensor комплексной формы (N,) dtype=torch.complex64 или torch.complex128
        M: memory depth
        P: max amplitude power

    Returns:
        X: тензор формы (N-M, M+1, 2+P), dtype=torch.float32
    """
    eps = 1e-8

    T = M + 1

    if len(iq_signal.shape) == 2 and iq_signal.shape[-1] == 2:
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


def build_LEANN_features(x: torch.Tensor, M: int):
    
    # re/im
    if len(x.shape) == 2 and x.shape[-1] == 2:
        x_I = x[..., 0]
        x_Q = x[..., 1]
    else:
        raise ValueError(f"Unsupported iq_signal shape: {x.shape}")
    
    # moving window
    X_I = x_I.unfold(0, M+1, 1)
    X_Q = x_Q.unfold(0, M+1, 1)
    
    # reverse
    X_I = torch.flip(X_I, dims=[1])
    X_Q = torch.flip(X_Q, dims=[1])
    
    X = torch.cat([X_I, X_Q], dim=1)

    return X


class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, x: torch.Tensor, y: torch.Tensor):
        self.x = x
        self.y = y
    
    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def build_nn_dataloaders(container,
                         batch_size: int,
                         batch_size_eval: int,
                         arch: str,
                         features_extractor: Callable = lambda x, **kwargs: x,
                         normalize_method: str | None = 'standard',
                         **kwargs):
    
    arch = arch.lower()
    assert arch in ('pa', 'dla', 'ila', 'ilc'), \
            "arch must be specified: 'pa', 'dla', 'ila' or 'ilc'"
    
    input_norm = utils.Normalizer(normalize_method)
    target_norm = utils.Normalizer(normalize_method)
    
    match arch:
        case "pa":
            x_train = container.train_input_orig
            x_val = container.val_input_orig
            y_train = container.train_output
            y_val = container.val_output
        case "dla":
            x_train = container.train_input_orig
            x_val = container.val_input_orig
            y_train = container.train_output_target
            y_val = container.val_output_target
        case "ila":
            x_train = container.train_output / container.gain
            x_val = container.val_output / container.gain
            y_train = container.train_input_orig
            y_val = container.val_input_orig
        case "ilc":
            x_train = container.train_input_orig
            x_val = container.val_input_orig
            y_train = container.ilc_train_output
            ilc_val_output = container.ilc_val_output
            if ilc_val_output is not None:
                y_val = ilc_val_output
            else:
                y_val = y_train
        case _:
            raise ValueError(f"Unknown arch '{arch}'")
    
    norm_x_train = input_norm.fit_transform(x_train)
    norm_x_val = input_norm.transform(x_val)
    norm_y_train = target_norm.fit_transform(y_train)
    norm_y_val = target_norm.transform(y_val)
    
    x_train = features_extractor(norm_x_train, **kwargs)
    x_val = features_extractor(norm_x_val, **kwargs)
    
    train_dataset = BaseDataset(x_train, norm_y_train)
    val_dataset = BaseDataset(x_val, norm_y_val)
    
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size_eval, shuffle=False)
    
    return train_dataloader, val_dataloader, input_norm, target_norm