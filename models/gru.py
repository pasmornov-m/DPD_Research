import torch
import torch.nn as nn
from models.base_model import BaseModel
from modules import utils


class GRU(BaseModel, torch.nn.Module):
    def __init__(self, 
                 input_size=2, 
                 hidden_size=64, 
                 num_layers=1, 
                 output_size=2, 
                 bidirectional=False, 
                 batch_first=True, 
                 model_name=""):
        
        torch.nn.Module.__init__(self)
        BaseModel.__init__(self, model_name=model_name)
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.bidirectional = bidirectional
        self.batch_first = batch_first
        self.num_directions = 2 if bidirectional else 1
        
        self.gru = nn.GRU(input_size=self.input_size, 
                          hidden_size=self.hidden_size, 
                          num_layers=self.num_layers, 
                          batch_first=self.batch_first, 
                          bidirectional=self.bidirectional)
        
        self.fc = nn.Linear(in_features=hidden_size*self.num_directions, 
                            out_features=self.output_size)

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
            f"in{self.input_size}_out{self.output_size}_bi{int(self.bidirectional)}.pt")
