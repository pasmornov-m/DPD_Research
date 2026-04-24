from dataclasses import dataclass, asdict
from typing import Optional, Callable, Dict, Type
import torch
from models.gmp import GMP
from models.gru import GRU, KPGRU
from models.lstm import LSTM, KPLSTM
from models.rtdtnn import RTDTNN
from models.leann import LEANN
from models.tcn import TCN, KPTCN
from modules import data_loader
from modules.config import FRAME_LENGTH, BATCH_SIZE, BATCH_SIZE_EVAL


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
        "num_blocks": tp["num_blocks"],
    },
    
    LEANN: lambda tp: {
        "M": tp["M"],
        "L1": tp["L1"],
        "L2": tp["L2"],
        "L3": tp["L3"],
        "K": tp["K"],
    },
    
    TCN: lambda tp: {
        "hidden_channels": tp["hidden_channels"],
        "kernel_size": tp["kernel_size"],
    },
    
    KPLSTM: lambda tp: {
        "M": tp["M"],
        "hidden_size": tp["hidden_size"],
        "num_layers": tp["num_layers"],
        "bidirectional": tp["bidirectional"],
    },
    
    KPGRU: lambda tp: {
        "M": tp["M"],
        "hidden_size": tp["hidden_size"],
        "num_layers": tp["num_layers"],
        "bidirectional": tp["bidirectional"],
    },
    
    KPTCN: lambda tp: {
        "M": tp["M"],
        "hidden_channels": tp["hidden_channels"],
        "kernel_size": tp["kernel_size"],
    },
}


EXTRACTORS_REGISTRY: Dict[Type[torch.nn.Module], Callable[[Dict], Dict]] = {
    GMP: None,
    GRU: None,
    LSTM: None,
    TCN: None,
    KPGRU: None,
    KPLSTM: None,
    KPTCN: None,
    RTDTNN: data_loader.build_RTDTNN_features,
    LEANN: data_loader.build_LEANN_features,
}


DATALOADERS_REGISTRY: Dict[Type[torch.nn.Module], Callable[[Dict], Dict]] = {
    GMP: data_loader.build_dataloaders,
    GRU: data_loader.build_dataloaders,
    LSTM: data_loader.build_dataloaders,
    TCN: data_loader.build_dataloaders,
    KPGRU: data_loader.build_dataloaders,
    KPLSTM: data_loader.build_dataloaders,
    KPTCN: data_loader.build_dataloaders,
    RTDTNN: data_loader.build_nn_dataloaders,
    LEANN: data_loader.build_nn_dataloaders,
}

DATALOADERS_PROPS: Dict[Type[torch.nn.Module], Callable[[Dict], Dict]] = {
    GMP: lambda features_extractor, *args, **kwargs: {
        "batch_size": BATCH_SIZE,
        "batch_size_eval": BATCH_SIZE_EVAL,
        "frame_length": FRAME_LENGTH,
        "features_extractor": features_extractor,
        "normalize_method": None},
    GRU: lambda features_extractor, *args, **kwargs: {
        "batch_size": BATCH_SIZE,
        "batch_size_eval": BATCH_SIZE_EVAL,
        "frame_length": FRAME_LENGTH,
        "features_extractor": features_extractor,
        "normalize_method": 'standard'},
    LSTM: lambda features_extractor, *args, **kwargs: {
        "batch_size": BATCH_SIZE,
        "batch_size_eval": BATCH_SIZE_EVAL,
        "frame_length": FRAME_LENGTH,
        "features_extractor": features_extractor,
        "normalize_method": 'standard'},
    RTDTNN: lambda features_extractor, train_props, *args, **kwargs: {
        "batch_size": BATCH_SIZE,
        "batch_size_eval": BATCH_SIZE_EVAL,
        "features_extractor": features_extractor,
        "normalize_method": 'standard',
        "M": train_props.get("M"),
        "P": train_props.get("P")},
    LEANN: lambda features_extractor, train_props, *args, **kwargs: {
        "batch_size": BATCH_SIZE,
        "batch_size_eval": BATCH_SIZE_EVAL,
        "features_extractor": features_extractor,
        "normalize_method": 'standard',
        "M": train_props.get("M")},
    TCN: lambda features_extractor, *args, **kwargs: {
        "batch_size": BATCH_SIZE,
        "batch_size_eval": BATCH_SIZE_EVAL,
        "frame_length": FRAME_LENGTH,
        "features_extractor": features_extractor,
        "normalize_method": 'standard'},
    KPGRU: lambda features_extractor, *args, **kwargs: {
        "batch_size": BATCH_SIZE,
        "batch_size_eval": BATCH_SIZE_EVAL,
        "frame_length": FRAME_LENGTH,
        "features_extractor": features_extractor,
        "normalize_method": 'standard'},
    KPLSTM: lambda features_extractor, *args, **kwargs: {
        "batch_size": BATCH_SIZE,
        "batch_size_eval": BATCH_SIZE_EVAL,
        "frame_length": FRAME_LENGTH,
        "features_extractor": features_extractor,
        "normalize_method": 'standard'},
    KPTCN: lambda features_extractor, *args, **kwargs: {
        "batch_size": BATCH_SIZE,
        "batch_size_eval": BATCH_SIZE_EVAL,
        "frame_length": FRAME_LENGTH,
        "features_extractor": features_extractor,
        "normalize_method": 'standard'},
}