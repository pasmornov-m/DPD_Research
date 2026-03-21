from dataclasses import dataclass, asdict
from typing import Optional, Callable, Dict, Type
import torch
from models.gmp import GMP
from models.gru import GRU
from models.lstm import LSTM
from models.rtdtnn import RTDTNN
from models.leann import LEANN


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


MODEL_REGISTRY: Dict[Type[torch.nn.Module], Callable[[Dict], Dict]] = {

    GMP: lambda tp: make_gmp_params(
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

    RTDTNN: lambda tp: {
        "M": tp["M"],
        "d_in": tp["d_in"],
        "d_model": tp["d_model"],
        "n_heads": tp["n_heads"],
        "d_ff": tp["d_ff"],
        "n_fc": tp["n_fc"],
    },
    
    LEANN: lambda tp: {
        "M": tp["M"],
        "L1": tp["L1"],
        "L2": tp["L2"],
        "L3": tp["L3"],
        "K": tp["K"],
    },
}
