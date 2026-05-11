import torch
import torch.nn as nn
from models.base_model import BaseModel
from models.kp_module import KPConvModule, KPModule
from modules import utils


class GRU(BaseModel, torch.nn.Module):
    def __init__(self, 
                 input_size: int = 2, 
                 hidden_size: int = 64, 
                 num_layers: int = 1, 
                 output_size: int = 2, 
                 bidirectional: bool = False, 
                 batch_first: bool = True,
                 bias: bool = True,
                 model_name: str = ""):
        
        torch.nn.Module.__init__(self)
        BaseModel.__init__(self, model_name=model_name)
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.bidirectional = bidirectional
        self.batch_first = batch_first
        self.bias = bias
        self.num_directions = 2 if bidirectional else 1
        
        self.gru = torch.nn.GRU(input_size=self.input_size, 
                                hidden_size=self.hidden_size, 
                                num_layers=self.num_layers, 
                                batch_first=self.batch_first, 
                                bidirectional=self.bidirectional,
                                bias=self.bias)
        
        self.fc = torch.nn.Linear(in_features=hidden_size*self.num_directions, 
                                  out_features=self.output_size,
                                  bias=self.bias)

    @utils.complex_handler
    def forward(self, x, h_0=None):
        batch_size = x.size(0)
        
        if h_0 is None:
            if x.dim() == 2:
                h_0 = torch.zeros(self.num_directions * self.num_layers, 
                                  self.hidden_size)
            else:
                h_0 = torch.zeros(self.num_directions * self.num_layers, 
                                  batch_size, 
                                  self.hidden_size)
        
        out, _ = self.gru(x, h_0)
        y = self.fc(out)
        return y
    
    def _get_filename(self):
        return (
            f"{self.model_name}_{self.class_name}_"
            f"hs{self.hidden_size}_nl{self.num_layers}_"
            f"in{self.input_size}_out{self.output_size}_bi{int(self.bidirectional)}.pt"
        )


class KPGRU(BaseModel, torch.nn.Module):
    def __init__(self, 
                 M: int = 5,
                 hidden_size: int = 5, 
                 num_layers: int = 1, 
                 output_size: int = 2, 
                 reduced_dim: int = 4,
                 bidirectional: bool = False, 
                 batch_first: bool = True,
                 bias: bool = True, 
                 kp_type: str = 'conv',
                 model_name: str = ""):
        
        torch.nn.Module.__init__(self)
        BaseModel.__init__(self, model_name=model_name)
        
        self.kp_type = kp_type
        
        match kp_type:
            case 'conv':
                self.kp_module = KPConvModule(M=M)
            case 'linear':
                self.kp_module = KPModule(M=M)
            case _:
                raise ValueError(f"Unsupported kp module type: {kp_type}")
        
        self.feature_dim = self.kp_module.get_feature_dim()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.bidirectional = bidirectional
        self.batch_first = batch_first
        self.bias = bias
        self.num_directions = 2 if bidirectional else 1
        
        self.reduced_dim = reduced_dim
        self.kp_proj = torch.nn.Linear(self.feature_dim, reduced_dim)
        
        self.gru_model = GRU(input_size=reduced_dim,
                               hidden_size=hidden_size,
                               num_layers=num_layers,
                               output_size=output_size,
                               bidirectional=bidirectional,
                               batch_first=batch_first,
                               bias=bias)
    
    @utils.complex_handler
    def forward(self, x, h_0=None):
        """
        x: [batch, seq_len, 2]
        """
                
        kp_features = self.kp_module(x)  # [B, T, F, 2]
        B, T, F, C = kp_features.shape
        kp = kp_features.reshape(B, T, F * C)
        kp = self.kp_proj(kp)
                
        output = self.gru_model(kp, h_0)
        
        return output
    
    def _get_filename(self):
        return (
            f"{self.model_name}_{self.class_name}_"
            f"hs{self.hidden_size}_nl{self.num_layers}_"
            f"out{self.output_size}_bi{int(self.bidirectional)}_"
            f"M{self.kp_module.M}_K{self.kp_module.K}_rd{self.reduced_dim}_"
            f"fd{self.feature_dim}_kp{self.kp_type}.pt"
        )
