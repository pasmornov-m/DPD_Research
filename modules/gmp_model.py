import torch
import torch.nn as nn
import os
from typing import Tuple
from modules import utils


class ClassicGMP(nn.Module):
    """
    Generalized Memory Polynomial with complex numbers stored as [N].
    Input: x : [N]  (cfloat)
    Output: y : [N] (cfloat)
    Coefficients: a: [K_a, L_a], b: [K_b, L_b, M_b], c: [K_c, L_c, M_c]
    """
    def __init__(self, 
                 Ka: int, 
                 La: int, 
                 Kb: int, 
                 Lb: int, 
                 Mb: int, 
                 Kc: int, 
                 Lc: int, 
                 Mc: int, 
                 model_name: str = 'model'):
        super().__init__()
        self.Ka, self.La = Ka, La
        self.Kb, self.Lb, self.Mb = Kb, Lb, Mb
        self.Kc, self.Lc, self.Mc = Kc, Lc, Mc
        self.model_name = model_name
        self.class_name = self.__class__.__name__

        self.a = torch.nn.Parameter(0.001 * torch.randn((self.Ka, self.La), dtype=torch.cfloat))
        self.b = torch.nn.Parameter(0.001 * torch.randn((self.Kb, self.Lb, self.Mb), dtype=torch.cfloat))
        self.c = torch.nn.Parameter(0.001 * torch.randn((self.Kc, self.Lc, self.Mc), dtype=torch.cfloat))

        self.powers_Ka = torch.arange(self.Ka)
        self.powers_Kb = torch.arange(self.Kb)
        self.powers_Kc = torch.arange(self.Kc)

        self.indices_La = torch.arange(self.La).unsqueeze(1)
        self.indices_Lb = torch.arange(self.Lb).unsqueeze(1)
        self.indices_Lc = torch.arange(self.Lc).unsqueeze(1)

        self.indices_Mb = torch.arange(self.Mb).unsqueeze(0).unsqueeze(2)
        self.indices_Mc = torch.arange(self.Mc).unsqueeze(0).unsqueeze(2)
        

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


    @utils.iq_handler
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self._compute_terms(x)
        return y


    def save_weights(self, directory="model_params"):
        os.makedirs(directory, exist_ok=True)
        filename = f"{directory}/{self.model_name}_{self.class_name}_Ka{self.Ka}_La{self.La}_Kb{self.Kb}_Lb{self.Lb}_Mb{self.Mb}_Kc{self.Kc}_Lc{self.Lc}_Mc{self.Mc}.pt"
        torch.save(self.state_dict(), filename)
        print(f"Coefficients saved to {filename}")


    def load_weights(self, directory="model_params"):
        filename = f"{directory}/{self.model_name}_{self.class_name}_Ka{self.Ka}_La{self.La}_Kb{self.Kb}_Lb{self.Lb}_Mb{self.Mb}_Kc{self.Kc}_Lc{self.Lc}_Mc{self.Mc}.pt"
        if os.path.isfile(filename):
            state_dict = torch.load(filename)
            self.load_state_dict(state_dict)
            print(f"Coefficients loaded from {filename}")
            return True
        else:
            print(f"No saved coefficients found at {filename}, initializing new parameters.")
            return False
        
    
    def _compute_terms(self, x: torch.Tensor) -> torch.Tensor:
        N = x.shape[0]
        indices_N = torch.arange(N).unsqueeze(0)

        indices_delayed_La = (indices_N - self.indices_La).clamp(min=0, max=N-1)
        indices_delayed_Lb = (indices_N - self.indices_Lb).clamp(min=0, max=N-1)
        indices_delayed_Lc = (indices_N - self.indices_Lc).clamp(min=0, max=N-1)

        indices_delayed_Ma = indices_delayed_La
        indices_delayed_Mb = (indices_N.unsqueeze(1) - self.indices_Lb.unsqueeze(2) - self.indices_Mb).clamp(min=0, max=N-1)
        indices_delayed_Mc = (indices_N.unsqueeze(1) - self.indices_Lc.unsqueeze(2) + self.indices_Mc).clamp(min=0, max=N-1)

        x_truncated_La = x[indices_delayed_La]
        x_truncated_Lb = x[indices_delayed_Lb]
        x_truncated_Lc = x[indices_delayed_Lc]

        x_truncated_Ma = x[indices_delayed_Ma]
        x_truncated_Mb = x[indices_delayed_Mb]
        x_truncated_Mc = x[indices_delayed_Mc]

        abs_powers_a = (torch.abs(x_truncated_Ma).unsqueeze(-1) ** self.powers_Ka).to(x.dtype)
        abs_powers_b = (torch.abs(x_truncated_Mb).unsqueeze(-1) ** self.powers_Kb).to(x.dtype)
        abs_powers_c = (torch.abs(x_truncated_Mc).unsqueeze(-1) ** self.powers_Kc).to(x.dtype)

        x_scaled_a = x_truncated_La.unsqueeze(-1) * abs_powers_a
        x_scaled_b = x_truncated_Lb.unsqueeze(1).unsqueeze(-1) * abs_powers_b
        x_scaled_c = x_truncated_Lc.unsqueeze(1).unsqueeze(-1) * abs_powers_c

        term_a = torch.einsum('kln,lnk->n', self.a.unsqueeze(-1), x_scaled_a)
        term_b = torch.einsum('klm,lmnk->n', self.b, x_scaled_b)
        term_c = torch.einsum('klm,lmnk->n', self.c, x_scaled_c)

        y = term_a + term_b + term_c
        
        return y


class BatchGMP(nn.Module):
    """
    Generalized Memory Polynomial (batched) with complex numbers stored as [..., 2] (re, im).
    Input: x : [B, T, 2]  (float32)  last dim: [Re, Im]
    Output: y : [B, T, 2]
    Coefficients: a: [K_a, L_a, 2], b: [K_b, L_b, M_b, 2], c: [K_c, L_c, M_c, 2]
    """

    def __init__(self, 
                 Ka: int, 
                 La: int, 
                 Kb: int, 
                 Lb: int, 
                 Mb: int, 
                 Kc: int, 
                 Lc: int, 
                 Mc: int, 
                 model_name: str = 'model'):
        super().__init__()
        self.Ka, self.La = Ka, La
        self.Kb, self.Lb, self.Mb = Kb, Lb, Mb
        self.Kc, self.Lc, self.Mc = Kc, Lc, Mc
        self.model_name = model_name
        self.class_name = self.__class__.__name__
        
        self.dtype = torch.float32

        self.a = nn.Parameter(0.01 * torch.randn((Ka, La, 2), dtype=self.dtype))
        self.b = nn.Parameter(0.01 * torch.randn((Kb, Lb, Mb, 2), dtype=self.dtype))
        self.c = nn.Parameter(0.01 * torch.randn((Kc, Lc, Mc, 2), dtype=self.dtype))

        self.register_buffer('powers_Ka', torch.arange(Ka, dtype=self.dtype))
        self.register_buffer('powers_Kb', torch.arange(Kb, dtype=self.dtype))
        self.register_buffer('powers_Kc', torch.arange(Kc, dtype=self.dtype))
        
        self.register_buffer('arange_La', torch.arange(La, dtype=self.dtype)[:, None])
        self.register_buffer('arange_Lb', torch.arange(Lb, dtype=self.dtype)[:, None])
        self.register_buffer('arange_Lc', torch.arange(Lc, dtype=self.dtype)[:, None])
        
        self.register_buffer('arange_Mb', torch.arange(Mb, dtype=self.dtype)[None, None, :])
        self.register_buffer('arange_Mc', torch.arange(Mc, dtype=self.dtype)[None, None, :])
    
    def save_weights(self, directory="model_params"):
        os.makedirs(directory, exist_ok=True)
        filename = f"{directory}/{self.model_name}_{self.class_name}_Ka{self.Ka}_La{self.La}_Kb{self.Kb}_Lb{self.Lb}_Mb{self.Mb}_Kc{self.Kc}_Lc{self.Lc}_Mc{self.Mc}.pt"
        torch.save(self.state_dict(), filename)
        print(f"Coefficients saved to {filename}")

    def load_weights(self, directory="model_params"):
        filename = f"{directory}/{self.model_name}_{self.class_name}_Ka{self.Ka}_La{self.La}_Kb{self.Kb}_Lb{self.Lb}_Mb{self.Mb}_Kc{self.Kc}_Lc{self.Lc}_Mc{self.Mc}.pt"
        if os.path.isfile(filename):
            state_dict = torch.load(filename)
            self.load_state_dict(state_dict)
            print(f"Coefficients loaded from {filename}")
            return True
        else:
            print(f"No saved coefficients found at {filename}, initializing new parameters.")
            return False

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    @staticmethod
    def abs_complex(x: torch.Tensor) -> torch.Tensor:
        eps = 1e-8
        return torch.sqrt(x[..., 0] ** 2 + x[..., 1] ** 2 + eps)
    
    @staticmethod
    def _to_complex(x: torch.Tensor) -> torch.Tensor:
        return torch.view_as_complex(x.contiguous().view(*x.shape[:-1], 2))
    
    def _gather_with_delay(self, x_padded: torch.Tensor, t_idx: torch.Tensor, 
                           arange_L: torch.Tensor, L: int, M: int = None) -> torch.Tensor:
        """
        Возвращает x[n-l] или x[n-l-m] с правильной размерностью.
        """
        if M is None:
            indices = (t_idx[None, :] - arange_L).long()
            indices = indices.unsqueeze(0).unsqueeze(-1).expand(x_padded.size(0), -1, -1, 2)
            x_exp = x_padded.unsqueeze(1).expand(-1, L, -1, -1)
            return torch.gather(x_exp, dim=2, index=indices)
        else:
            l_broadcast = arange_L.view(-1, 1, 1)
            m_broadcast = torch.arange(M, device=t_idx.device).view(1, -1, 1)
            lm_indices = (t_idx[None, None, :] - l_broadcast - m_broadcast).long()
            lm_indices = lm_indices.unsqueeze(0).unsqueeze(-1).expand(x_padded.size(0), -1, -1, -1, 2)
            x_exp = x_padded.unsqueeze(1).unsqueeze(1).expand(-1, L, M, -1, -1)
            return torch.gather(x_exp, dim=3, index=lm_indices)
    
    def _pad_input(self, x, T):
        max_delay = max(self.La, self.Lb + self.Mb - 1, self.Lc + self.Mc - 1, 1)
        if max_delay > 1:
            pad = x[:, :1].expand(x.size(0), max_delay - 1, 2)
            x_padded = torch.cat([pad, x], dim=1)
        else:
            x_padded = x
        t_idx = torch.arange(max_delay - 1, max_delay - 1 + T, device=x.device)
        return x_padded, t_idx

    def _term_a(self, x_padded, t_idx):
        if self.La == 0: 
            return None
        x_La = self._gather_with_delay(x_padded, t_idx, self.arange_La, self.La)
        abs_x_La = self.abs_complex(x_La)
        abs_powers = abs_x_La.unsqueeze(-1) ** self.powers_Ka
        x_cmplx = self._to_complex(x_La)
        return x_cmplx.unsqueeze(-1) * abs_powers.to(torch.complex64)

    def _term_b(self, x_padded, t_idx):
        if not (self.Lb > 0 and self.Mb > 0): 
            return None
        x_Lb = self._gather_with_delay(x_padded, t_idx, self.arange_Lb, self.Lb)
        x_LMb = self._gather_with_delay(x_padded, t_idx, self.arange_Lb, self.Lb, self.Mb)
        abs_x_LMb = self.abs_complex(x_LMb)
        abs_powers = abs_x_LMb.unsqueeze(-1) ** self.powers_Kb
        x_Lb_cmplx = self._to_complex(x_Lb)
        x_Lb_exp = x_Lb_cmplx.unsqueeze(2).unsqueeze(-1)
        return x_Lb_exp * abs_powers.to(torch.complex64)

    def _term_c(self, x_padded, t_idx):
        if not (self.Lc > 0 and self.Mc > 0): 
            return None
        x_Lc_abs = self._gather_with_delay(x_padded, t_idx, self.arange_Lc, self.Lc)
        abs_x_Lc = self.abs_complex(x_Lc_abs)
        abs_powers = abs_x_Lc.unsqueeze(2).unsqueeze(-1) ** self.powers_Kc
        x_LMc = self._gather_with_delay(x_padded, t_idx, self.arange_Lc, self.Lc, self.Mc)
        x_LMc_cmplx = self._to_complex(x_LMc)
        return abs_powers.to(torch.complex64) * x_LMc_cmplx.unsqueeze(-1)
    
    @utils.complex_handler
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, 2] (float32), last dim: [Re, Im]
        returns y: [B, T, 2]
        """
        if x.dim() != 3 or x.shape[-1] != 2:
            raise ValueError(f"Input must be [B, T, 2] (re,im), but got {x.shape}")
        
        B, T = x.shape[0], x.shape[1]
        x_padded, t_idx = self._pad_input(x, T)

        term_a = self._term_a(x_padded, t_idx)
        term_b = self._term_b(x_padded, t_idx)
        term_c = self._term_c(x_padded, t_idx)

        a_cmplx = self._to_complex(self.a)
        b_cmplx = self._to_complex(self.b)
        c_cmplx = self._to_complex(self.c)

        y = torch.zeros(B, T, dtype=torch.complex64, device=x.device)

        if term_a is not None:
            y += torch.einsum('bltk,k l->bt', term_a, a_cmplx)
        if term_b is not None:
            y += torch.einsum('blmtk,k l m->bt', term_b, b_cmplx)
        if term_c is not None:
            y += torch.einsum('blmtk,k l m->bt', term_c, c_cmplx)

        return torch.view_as_real(y)
