import torch
from models.base_model import BaseModel
from modules import utils


class LEANN(BaseModel, torch.nn.Module):
    def __init__(self, 
                 M: int, 
                 L2: int, 
                 L3: int,
                 K: int, 
                 L1: int = 2,
                 model_name: str = ""):
        
        torch.nn.Module.__init__(self)
        BaseModel.__init__(self, model_name=model_name)
        
        self.M = M
        self.L2 = L2
        self.L3 = L3
        self.K = K
        self.L1 = L1
        self._EPS = 1e-12

        self.fir = torch.nn.Linear(in_features=2*(self.M+1), 
                                   out_features=2*self.L1, 
                                   bias=True)
        
        self.lea_1_weight = torch.nn.Parameter(torch.randn(self.L1, L2) * 1)
        self.lea_1_bias = torch.nn.Parameter(torch.zeros(self.L1, L2))
        
        self.lea_k_weight = torch.nn.Parameter(torch.randn(L2, L3) * 1)
        self.lea_k_bias = torch.nn.Parameter(torch.zeros(L2, L3))
        
        torch.nn.init.xavier_uniform_(self.lea_1_weight)
        torch.nn.init.xavier_uniform_(self.lea_k_weight)
            
    def vdm(self, u):
        B = u.shape[0]
        u = u.view(B, self.L1, 2)
        
        I = u[..., 0]
        Q = u[..., 1]
        
        amplitude = torch.sqrt(I**2 + Q**2 + self._EPS)
        phase = torch.atan2(I, Q)
        
        return amplitude, phase

    def lea_1(self, u):
        u_exp = u.unsqueeze(-1)

        weight = self.lea_1_weight.unsqueeze(0)
        bias = self.lea_1_bias.unsqueeze(0)

        # edge activation
        f = weight * torch.abs(torch.abs(u_exp) - bias)

        # суммирование по L1
        v = f.sum(dim=1)
        return v
    
    def lea_k(self, v):
        v_exp = v.unsqueeze(-1)
        weight = self.lea_k_weight.unsqueeze(0)
        bias = self.lea_k_bias.unsqueeze(0)

        h = weight * torch.abs(torch.abs(v_exp) - bias) ** self.K

        return h
    
    def prb(self, h, phase):

        phase_exp = phase.repeat(1, self.L2 // self.L1 + 1)[:, :self.L2].unsqueeze(-1)

        I = h * torch.sin(phase_exp)
        Q = h * torch.cos(phase_exp)

        return I, Q
    
    def row_sum(self, I, Q):
        yI = I.sum(dim=(1, 2))
        yQ = Q.sum(dim=(1, 2))
        return torch.stack([yI, yQ], dim=-1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, 2M]
        returns y: [B, 2]
        """
        u = self.fir(x)
        amplitude, phase = self.vdm(u)
        v = self.lea_1(amplitude)
        h = self.lea_k(v)
        I, Q = self.prb(h, phase)
        y = self.row_sum(I, Q)
 
        return y
    
    def _get_filename(self):
        return (
            f"{self.model_name}_{self.class_name}_"
            f"M_{self.M}_L1_{self.L1}_"
            f"L2_{self.L2}_L3_{self.L3}_K_{self.K}.pt")