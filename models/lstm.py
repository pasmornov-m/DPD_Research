import torch
import torch.nn as nn
from models.base_model import BaseModel
from models.kp_module import KPConvModule
from modules import utils


class LSTM(BaseModel, torch.nn.Module):
    def __init__(self, 
                 input_size=2, 
                 hidden_size=64, 
                 num_layers=1, 
                 output_size=2, 
                 bidirectional=False, 
                 batch_first=True,
                 bias=False, 
                 model_name=""):
        
        torch.nn.Module.__init__(self)
        BaseModel.__init__(self, model_name=model_name)
        
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.batch_first = batch_first
        self.bias = bias
        self.num_directions = 2 if bidirectional else 1

        self.lstm = nn.LSTM(input_size=input_size,
                          hidden_size=hidden_size,
                          num_layers=num_layers,
                          bidirectional=self.bidirectional,
                          batch_first=self.batch_first,
                          bias=self.bias)
        self.fc_out = nn.Linear(in_features=hidden_size*self.num_directions,
                                out_features=self.output_size,
                                bias=self.bias)

    @utils.complex_handler
    def forward(self, x, h_0=None):
        batch_size = x.size(0)
        
        if h_0 is None or c_0 is None:
            h_0 = torch.zeros(self.num_directions * self.num_layers, 
                            batch_size, 
                            self.hidden_size)
            c_0 = torch.zeros(self.num_directions * self.num_layers, 
                            batch_size, 
                            self.hidden_size)
        
        y, (h_n, c_n) = self.lstm(x, (h_0, c_0))
        y = self.fc_out(y)
        return y
    
    def _get_filename(self):
        return (
            f"{self.model_name}_{self.class_name}_"
            f"hs{self.hidden_size}_nl{self.num_layers}_"
            f"in{self.input_size}_out{self.output_size}_bi{int(self.bidirectional)}.pt")




class KPLSTM(BaseModel, torch.nn.Module):
    def __init__(self, 
                 M: int = 5,
                 hidden_size: int = 5, 
                 num_layers: int = 1, 
                 output_size: int = 2, 
                 reduced_dim: int = 4,
                 bidirectional: bool = False, 
                 batch_first: bool = True,
                 bias: bool = False, 
                 model_name: str = ""):
        
        torch.nn.Module.__init__(self)
        BaseModel.__init__(self, model_name=model_name)
        
        self.kp_module = KPConvModule(M=M)
        
        feature_dim = self.kp_module.get_feature_dim()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.bidirectional = bidirectional
        self.batch_first = batch_first
        self.bias = bias
        self.num_directions = 2 if bidirectional else 1
        
        self.reduced_dim = reduced_dim
        self.kp_proj = torch.nn.Linear(feature_dim, reduced_dim)

        self.lstm = torch.nn.LSTM(input_size=reduced_dim,
                                  hidden_size=hidden_size,
                                  num_layers=num_layers,
                                  bidirectional=bidirectional,
                                  batch_first=batch_first,
                                  bias=bias)
                
        self.fc_out = torch.nn.Linear(in_features=hidden_size * self.num_directions,
                                      out_features=output_size,
                                      bias=bias)
    
    @utils.complex_handler
    def forward(self, x, h_0=None, c_0=None):
        """
        x: [batch, seq_len, 2]
        """
        
        B = x.shape[0]
        
        if h_0 is None:
            h_0 = torch.zeros(self.num_directions * self.num_layers,
                              B, 
                              self.hidden_size)
        if c_0 is None:
            c_0 = torch.zeros(self.num_directions * self.num_layers,
                              B, 
                              self.hidden_size)
        
        kp_features = self.kp_module(x)  # [B, T, F, 2]
        B, T, F, C = kp_features.shape
        kp = kp_features.reshape(B, T, F * C)
        kp = self.kp_proj(kp)
                
        lstm_out, (h_n, c_n) = self.lstm(kp, (h_0, c_0))

        output = self.fc_out(lstm_out)
        
        return output
    
    def _get_filename(self):
        return (
            f"{self.model_name}_{self.class_name}_"
            f"hs{self.hidden_size}_nl{self.num_layers}_"
            f"M{self.kp_module.M}_K{self.kp_module.K}_"
            f"bi{int(self.bidirectional)}.pt"
        )
