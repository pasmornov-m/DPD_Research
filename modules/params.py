from dataclasses import dataclass, asdict
from typing import Optional, Callable
import torch


@dataclass
class ModelParams:
    model_type: str
    Ka: int
    La: int
    Kb: int
    Lb: int
    Mb: int
    Kc: int
    Lc: int
    Mc: int
    epochs: int
    lr: float


@dataclass
class gmp_params:
    Ka: int
    La: int
    Kb: int
    Lb: int
    Mb: int
    Kc: int
    Lc: int
    Mc: int

def make_gmp_params(Ka: int,
                    La: int,
                    Kb: int,
                    Lb: int,
                    Mb: int,
                    Kc: int,
                    Lc: int,
                    Mc: int
                ):
    return asdict(gmp_params(
                Ka=Ka,
                La=La,
                Kb=Kb,
                Lb=Lb,
                Mb=Mb,
                Kc=Kc,
                Lc=Lc,
                Mc=Mc
                ))


@dataclass
class snr_params:
    snr_range: list
    num_realizations: int
    fs: float
    bw_main_ch: float
    epochs: int
    learning_rate: float
    acpr_meter: Callable
    pa_model: Optional[Callable] = None
    gain: Optional[float] = None

def make_snr_params(snr_range: list,
                    num_realizations: int,
                    fs: float,
                    bw_main_ch: float,
                    epochs: int,
                    learning_rate: float,
                    acpr_meter: Callable,
                    pa_model: Optional[Callable] = None,
                    gain: Optional[float] = None):
    return asdict(snr_params(
                            snr_range=snr_range,
                            num_realizations=num_realizations,
                            fs=fs,
                            bw_main_ch=bw_main_ch,
                            epochs=epochs,
                            learning_rate=learning_rate,
                            acpr_meter=acpr_meter,
                            pa_model=pa_model,
                            gain=gain
                            ))


@dataclass
class data_params:
    x_train: torch.Tensor
    y_train_target: torch.Tensor
    x_val: torch.Tensor
    y_val_target: torch.Tensor

def make_data_params(x_train: torch.Tensor,
                     y_train_target: torch.Tensor,
                     x_val: torch.Tensor,
                     y_val_target: torch.Tensor):
    return asdict(data_params(
                            x_train=x_train,
                            y_train_target=y_train_target,
                            x_val=x_val,
                            y_val_target=y_val_target
                            ))