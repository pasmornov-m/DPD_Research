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
        "num_blocks": tp["num_blocks"],
    },
    
    LEANN: lambda tp: {
        "M": tp["M"],
        "L1": tp["L1"],
        "L2": tp["L2"],
        "L3": tp["L3"],
        "K": tp["K"],
    },
}




DATALOADERS_REGISTRY: Dict[Type[torch.nn.Module], Callable[[Dict], Dict]] = {

    GMP: lambda p:{},

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

    RTDTNN: {"dataloader_func": data_loader.build_nn_dataloaders},
    
    LEANN: lambda tp: {
        "M": tp["M"],
        "L1": tp["L1"],
        "L2": tp["L2"],
        "L3": tp["L3"],
        "K": tp["K"],
    },
}

try:
    builder = params.MODEL_REGISTRY[self.base_model]
except KeyError:
    raise ValueError(f"Unsupported base_model: {self.base_model}")

self.model_params = builder(self.train_props)


def _prepare_loaders(model, dataloader, container, batch_size, batch_size_eval, feature_extractor, train_props, arch):
    dataloaders_builder = DATALOADERS_REGISTRY[model]
    dataloader = dataloaders_builder(dataloader, container, batch_size, batch_size_eval, feature_extractor, train_props)
    
    if issubclass(base_model, RTDTNN):
        pa_train_loader, pa_val_loader, pa_input_norm, pa_target_norm = data_loader.build_nn_dataloaders(container, 
                                                                                        batch_size=batch_size, 
                                                                                        batch_size_eval=batch_size_eval,
                                                                                        arch=arch,
                                                                                        features_extractor=feature_extractor,
                                                                                        M=train_props["M"], 
                                                                                        P=train_props["P"])
        dla_train_loader, dla_val_loader, dla_input_norm, dla_target_norm = data_loader.build_nn_dataloaders(container, 
                                                                                        batch_size=batch_size, 
                                                                                        batch_size_eval=batch_size_eval, 
                                                                                        arch=arch,
                                                                                        features_extractor=feature_extractor,
                                                                                        M=train_props["M"], 
                                                                                        P=train_props["P"])
        ila_train_loader, ila_val_loader, ila_input_norm, ila_target_norm = data_loader.build_nn_dataloaders(container, 
                                                                                        batch_size=batch_size, 
                                                                                        batch_size_eval=batch_size_eval,
                                                                                        arch=arch,
                                                                                        features_extractor=feature_extractor,
                                                                                        M=train_props["M"], 
                                                                                        P=train_props["P"])
    elif issubclass(base_model, LEANN):
        feature_extractor = data_loader.build_LEANN_features
        pa_train_loader, pa_val_loader, pa_input_norm, pa_target_norm = data_loader.build_nn_dataloaders(container, 
                                                                                        batch_size=batch_size, 
                                                                                        batch_size_eval=batch_size_eval, 
                                                                                        arch=arch,
                                                                                        features_extractor=feature_extractor,
                                                                                        normalize_method='standard',
                                                                                        M=train_props["M"])
        dla_train_loader, dla_val_loader, dla_input_norm, dla_target_norm = data_loader.build_nn_dataloaders(container, 
                                                                                        batch_size=batch_size, 
                                                                                        batch_size_eval=batch_size_eval,
                                                                                        arch=arch,
                                                                                        features_extractor=feature_extractor,
                                                                                        normalize_method='standard',
                                                                                        M=train_props["M"])
        ila_train_loader, ila_val_loader, ila_input_norm, ila_target_norm = data_loader.build_nn_dataloaders(container, 
                                                                                        batch_size=batch_size, 
                                                                                        batch_size_eval=batch_size_eval,
                                                                                        arch=arch,
                                                                                        features_extractor=feature_extractor,
                                                                                        normalize_method='standard',
                                                                                        M=train_props["M"])
    elif issubclass(base_model, (GRU, LSTM)):
        pa_train_loader, pa_val_loader, pa_input_norm, pa_target_norm = data_loader.build_dataloaders(container=container, 
                                                                        frame_length=frame_length, 
                                                                        batch_size=batch_size, 
                                                                        batch_size_eval=batch_size_eval,
                                                                        arch=arch,
                                                                        normalize_method='standard')
        dla_train_loader, dla_val_loader, dla_input_norm, dla_target_norm = data_loader.build_dataloaders(container=container, 
                                                                        frame_length=frame_length, 
                                                                        batch_size=batch_size, 
                                                                        batch_size_eval=batch_size_eval, 
                                                                        arch=arch,
                                                                        normalize_method='standard')
        ila_train_loader, ila_val_loader, ila_input_norm, ila_target_norm = data_loader.build_dataloaders(container=container, 
                                                                        frame_length=frame_length, 
                                                                        batch_size=batch_size, 
                                                                        batch_size_eval=batch_size_eval, 
                                                                        arch=arch,
                                                                        normalize_method='standard')
    elif issubclass(base_model, GMP):
        pa_train_loader, pa_val_loader, pa_input_norm, pa_target_norm = data_loader.build_dataloaders(container=container, 
                                                                        frame_length=frame_length, 
                                                                        batch_size=batch_size, 
                                                                        batch_size_eval=batch_size_eval,
                                                                        arch=arch)
        dla_train_loader, dla_val_loader, dla_input_norm, dla_target_norm = data_loader.build_dataloaders(container=container, 
                                                                        frame_length=frame_length, 
                                                                        batch_size=batch_size, 
                                                                        batch_size_eval=batch_size_eval, 
                                                                        arch=arch)
        ila_train_loader, ila_val_loader, ila_input_norm, ila_target_norm = data_loader.build_dataloaders(container=container, 
                                                                        frame_length=frame_length, 
                                                                        batch_size=batch_size, 
                                                                        batch_size_eval=batch_size_eval, 
                                                                        arch=arch)