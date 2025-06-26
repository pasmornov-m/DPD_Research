import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from modules.metrics import calculate_gain_complex


class IQSegmentDataset(Dataset):
    def __init__(self, features, targets, nperseg=16384):
        self.nperseg = nperseg

        features = self.split_segments(features)
        targets = self.split_segments(targets)
        self.features = torch.Tensor(features)
        self.targets = torch.Tensor(targets)

    def split_segments(self, sequence):
        num_samples = len(sequence)
        segments = []
        for i in range(0, num_samples, self.nperseg):
            segment = sequence[i:i + self.nperseg]
            if len(segment) < self.nperseg:
                padding_shape = (self.nperseg - segment.shape[0], 2)
                segment = torch.vstack((segment, torch.zeros(padding_shape)))
            segments.append(segment)
        return np.array(segments)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        features = self.features[idx, ...]
        targets = self.targets[idx, ...]
        return features, targets


class IQFrameDataset(Dataset):
    def __init__(self, features, targets, frame_length, stride=1):
        # Convert segments into frames
        self.features = torch.Tensor(self.get_frames(features, frame_length, stride))
        self.targets = torch.Tensor(self.get_frames(targets, frame_length, stride))

    @staticmethod
    def get_frames(sequence, frame_length, stride_length):
            frames = []
            sequence_length = len(sequence)
            num_frames = (sequence_length - frame_length) // stride_length + 1
            for i in range(num_frames):
                frame = sequence[i * stride_length: i * stride_length + frame_length]
                frames.append(frame)
            return np.stack(frames)
    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]


#------------------------------------------------------------------------------------------------------------------------------------------

# def build_dataloaders(data_dict, frame_length, batch_size, batch_size_eval, step=None):
#     """step == 'train_dpd' or nothing"""

#     nperseg = data_dict["config"]["nperseg"]
#     x_train = data_dict["train_input"]
#     y_train = data_dict["train_output"]
#     x_val = data_dict["val_input"]
#     y_val = data_dict["val_output"]


#     # Apply the PA Gain if training DPD
#     target_gain = calculate_gain_complex(x_train, y_train)

#     if step == 'train_dpd':
#         y_train = target_gain * x_train
#         y_val = target_gain * x_val

#     # Extract Features
#     input_size = x_train.shape[-1]

#     # Define PyTorch Datasets
#     train_set = IQFrameDataset(x_train, y_train, frame_length=frame_length)
#     val_set = IQSegmentDataset(x_val, y_val, nperseg=nperseg)

#     # Define PyTorch Dataloaders
#     train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
#     val_loader = DataLoader(val_set, batch_size=batch_size_eval, shuffle=False)

#     return (train_loader, val_loader), input_size

#------------------------------------------------------------------------------------------------------
class IQDataset(Dataset):
    def __init__(self, 
                 features: torch.Tensor, 
                 targets: torch.Tensor,
                 nperseg: int, 
                 frame_length: int = None, 
                 stride: int = 1):
        """
        Dataset, поддерживающий режим сегментов и фреймов.
        :param features: (N, 2) тензор с I/Q входом
        :param targets:  (N, 2) тензор с I/Q выходом
        :param nperseg: длина сегмента
        :param frame_length: если None — используется режим сегментов, иначе фреймы
        :param stride: шаг между фреймами
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
                pad = torch.zeros((self.nperseg - seg.shape[0], data.shape[1]), dtype=data.dtype)
                seg = torch.cat([seg, pad], dim=0)
            segments.append(seg)
        return torch.stack(segments)

    def _extract_frames(self, segments: torch.Tensor) -> torch.Tensor:
        frames = []
        for seg in segments:
            for i in range(0, seg.shape[0] - self.frame_length + 1, self.stride):
                frames.append(seg[i:i + self.frame_length])
        return torch.stack(frames)

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]
    

def build_dataloaders(data_dict, frame_length, batch_size, batch_size_eval, step=None):
    nperseg = data_dict["config"]["nperseg"]

    x_train = data_dict["train_input"]
    y_train = data_dict["train_output"]
    x_val = data_dict["val_input"]
    y_val = data_dict["val_output"]

    if step == "train_dpd":
        gain = calculate_gain_complex(x_train, y_train)  # можно переписать на torch позже
        y_train = gain * x_train
        y_val = gain * x_val

    # Создаём датасеты
    train_set = IQDataset(x_train, y_train, nperseg=nperseg, frame_length=frame_length)
    val_set = IQDataset(x_val, y_val, nperseg=nperseg, frame_length=None)

    # Dataloaders
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size_eval, shuffle=False)

    input_size = x_train.shape[-1]
    return (train_loader, val_loader), input_size
