import torch
from modules import utils
from models.base_model import BaseModel
from models.kp_module import KPConvModule, KPModule



class TCN(BaseModel, torch.nn.Module):
    def __init__(self, 
                 in_channels: int = 2,
                 hidden_channels: int = 32,
                 out_channels: int = 2,
                 kernel_size: int = 5,
                 model_name: str = ""):
        torch.nn.Module.__init__(self)
        BaseModel.__init__(self, model_name=model_name)
        
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dilation = 1
        self.stride = 1

        pad1 = (kernel_size - 1) * self.dilation * 1
        pad2 = (kernel_size - 1) * self.dilation * 2
        pad3 = (kernel_size - 1) * self.dilation * 4
        pad4 = (kernel_size - 1) * self.dilation * 8
        
        self.network = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=self.in_channels, 
                            out_channels=self.hidden_channels, 
                            kernel_size=1),
            torch.nn.Hardswish(),
            torch.nn.Conv1d(self.hidden_channels, 
                            self.hidden_channels, 
                            self.kernel_size, 
                            stride=self.stride, 
                            padding=pad1, 
                            dilation=self.dilation, 
                            groups=self.hidden_channels, 
                            bias=False),
            torch.nn.Hardswish(),
            torch.nn.Conv1d(self.hidden_channels, 
                            self.hidden_channels, 
                            self.kernel_size, 
                            stride=self.stride, 
                            padding=pad2, 
                            dilation=self.dilation*2, 
                            groups=self.hidden_channels, 
                            bias=False),
            torch.nn.Hardswish(),
            torch.nn.Conv1d(self.hidden_channels, 
                            self.hidden_channels, 
                            self.kernel_size, 
                            stride=self.stride, 
                            padding=pad3, 
                            dilation=self.dilation*4, 
                            groups=self.hidden_channels, 
                            bias=False),
            torch.nn.Hardswish(),
            torch.nn.Conv1d(self.hidden_channels, 
                            self.hidden_channels, 
                            self.kernel_size, 
                            stride=self.stride,
                            padding=pad4, 
                            dilation=self.dilation*8, 
                            groups=self.hidden_channels, 
                            bias=False),
            torch.nn.Hardswish(),
            torch.nn.Conv1d(self.hidden_channels, 
                            self.out_channels, 
                            kernel_size=1, 
                            bias=False),
            )
        
        if in_channels != out_channels:
            self.residual_proj = torch.nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual_proj = torch.nn.Identity()
    
    def forward(self, x):

        x_transpose = x.transpose(1, 2)
        out = self.network(x_transpose)
        out = out[:, :, :x_transpose.shape[2]]
        
        res = self.residual_proj(x_transpose)
        res = res[:, :, :out.shape[2]]
        out = out + res
        
        out = out.transpose(1, 2)
        
        return out
    
    def _get_filename(self):
        return (
            f"{self.model_name}_{self.class_name}_"
            f"ic{self.in_channels}_hc{self.hidden_channels}_"
            f"oc{self.out_channels}_ks{self.kernel_size}.pt"
        )





class KPTCN(BaseModel, torch.nn.Module):
    def __init__(self,
                 M: int,
                 hidden_channels: int = 32,
                 out_channels: int = 2,
                 kernel_size: int = 5,
                 reduced_dim: int = 4,
                 kp_type: str = 'conv',
                 model_name: str = ""):
        
        torch.nn.Module.__init__(self)
        BaseModel.__init__(self, model_name=model_name)
        
        self.M = M
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.kp_type = kp_type
        
        match kp_type:
            case 'conv':
                self.kp_module = KPConvModule(M=M)
            case 'linear':
                self.kp_module = KPModule(M=M)
            case _:
                raise ValueError(f"Unsupported kp module type: {kp_type}")
        
        self.feature_dim = self.kp_module.get_feature_dim()
        self.reduced_dim = reduced_dim

        self.kp_proj = torch.nn.Linear(self.feature_dim, reduced_dim)

        self.tcn = TCN(in_channels=reduced_dim,
                       hidden_channels=hidden_channels,
                       out_channels=out_channels,
                       kernel_size=kernel_size,
                       model_name=model_name)

    @utils.complex_handler
    def forward(self, x):
        """
        x: [batch, seq_len, 2]
        """
        
        kp_features = self.kp_module(x)  # [B, T, F, 2]
        
        B, T, F, C = kp_features.shape
        kp = kp_features.reshape(B, T, F * C)
        
        # kp = kp.transpose(1, 2)
        kp = self.kp_proj(kp)
        # kp = kp.transpose(1, 2)
        
        out = self.tcn(kp)
        
        return out
    
    def _get_filename(self):
        return (
            f"{self.model_name}_{self.class_name}_"
            f"hc{self.hidden_channels}_oc{self.out_channels}_ks{self.kernel_size}_"
            f"M{self.kp_module.M}_K{self.kp_module.K}_rd{self.reduced_dim}_"
            f"fd{self.feature_dim}_kp{self.kp_type}.pt"
        )
