import torch
import torch.nn as nn
from modules.gmp_ar_model import GMP_AR

class MoE_GMP_AR(nn.Module):
    def __init__(self, num_experts, gmp_ar_kwargs):
        super().__init__()
        self.experts = nn.ModuleList([
            GMP_AR(**gmp_ar_kwargs) for _ in range(num_experts)
        ])

        self.logits = nn.Parameter(torch.zeros(num_experts))
        self.Dy = gmp_ar_kwargs['Dy']
        self.d = torch.nn.Parameter(0.001 * torch.randn(self.Dy, dtype=torch.cfloat))
        self.logit_weights = torch.nn.Parameter(torch.tensor([0.01, 0.01]))

    def forward(self, x):
        
        weights = torch.softmax(self.logits, dim=0)

        ys = [expert(x) for expert in self.experts]

        y_stack = torch.stack(ys, dim=-1)
        y_weighted = (y_stack * weights).mean(dim=-1)

        w_gmp, w_ar = torch.softmax(self.logit_weights, dim=0)
        y_ar = self._compute_autoregression(y_weighted)
        y = w_gmp * y_weighted + w_ar * y_ar

        return y
    
    def _compute_autoregression(self, y_gmp):
        N = y_gmp.shape[0]
        if N < self.Dy:
            raise ValueError(f"Для Dy={self.Dy} требуется хотя бы {self.Dy+1} отсчетов в y_gmp.")

        X_ar = y_gmp.unfold(0, self.Dy, 1)
        y_ar_part = X_ar @ self.d

        y_ar = torch.zeros_like(y_gmp)
        y_ar[self.Dy:] = y_ar_part[:-1]

        return y_ar


class MoE_GMP_AR(nn.Module):
    def __init__(self, num_experts, gmp_ar_kwargs):
        super().__init__()
        self.experts = nn.ModuleList([
            GMP_AR(**gmp_ar_kwargs) for _ in range(num_experts)
        ])

        self.logits = nn.Parameter(torch.zeros(num_experts))
        self.Dy = gmp_ar_kwargs['Dy']
        self.logit_weight = torch.nn.Parameter(torch.tensor([0.01]))

    def forward(self, x):
        
        weights = torch.softmax(self.logits, dim=0)

        ys = [expert(x) for expert in self.experts]

        y_stack = torch.stack(ys, dim=-1)
        y_weighted = (y_stack * weights).mean(dim=-1)

        w_gmp = torch.softmax(self.logit_weight, dim=0)
        y = w_gmp * y_weighted

        return y
    