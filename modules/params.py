from dataclasses import dataclass, asdict
from typing import Optional, Callable, Tuple, Dict, Type, Any
import torch
from modules.nn_model import GRU, LSTM, DenseNetRegressor, CustomTCN, DiffESN, RTDTNN
from modules.gmp_model import ClassicGMP, BatchGMP




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


# @dataclass
# class data_params:
#     x_train: torch.Tensor
#     y_train_target: torch.Tensor
#     x_val: torch.Tensor
#     y_val_target: torch.Tensor

# def make_data_params(x_train: torch.Tensor,
#                      y_train_target: torch.Tensor,
#                      x_val: torch.Tensor,
#                      y_val_target: torch.Tensor):
#     return asdict(data_params(
#                             x_train=x_train,
#                             y_train_target=y_train_target,
#                             x_val=x_val,
#                             y_val_target=y_val_target
#                             ))




MODEL_REGISTRY: Dict[Type[torch.nn.Module], Callable[[Dict], Dict]] = {
    ClassicGMP: lambda tp: make_gmp_params(
        Ka=tp["gmp_degree"],
        La=tp["gmp_degree"],
        Kb=tp["gmp_degree"],
        Lb=tp["gmp_degree"],
        Mb=tp["gmp_degree"],
        Kc=tp["gmp_degree"],
        Lc=tp["gmp_degree"],
        Mc=tp["gmp_degree"],
    ),

    BatchGMP: lambda tp: make_gmp_params(
        Ka=tp["gmp_degree"],
        La=tp["gmp_degree"],
        Kb=tp["gmp_degree"],
        Lb=tp["gmp_degree"],
        Mb=tp["gmp_degree"],
        Kc=tp["gmp_degree"],
        Lc=tp["gmp_degree"],
        Mc=tp["gmp_degree"],
    ),

    GRU: lambda tp: {
        "hidden_size": tp["hidden_size"],
        "num_layers": tp["num_layers"],
        "bidirectional": tp["bidirectional"],
    },

    LSTM: lambda tp: {
        "hidden_size": tp["hidden_size"],
        "num_layers": tp["num_layers"],
        "bidirectional": tp["bidirectional"],
    },

    DenseNetRegressor: lambda tp: {
        "blocks": tp["blocks"],
    },

    CustomTCN: lambda tp: {
        "num_channels": tp["num_channels"],
        "kernel_size": tp["kernel_size"],
        "dropout": tp["dropout"],
    },

    DiffESN: lambda tp: {
        "n_reservoir": tp["n_reservoir"],
        "sparsity": tp["sparsity"],
    },

    RTDTNN: lambda tp: {
        "M": tp["M"],
        "d_in": tp["d_in"],
        "d_model": tp["d_model"],
        "n_heads": tp["n_heads"],
        "d_ff": tp["d_ff"],
        "n_fc": tp["n_fc"],
    },
}
