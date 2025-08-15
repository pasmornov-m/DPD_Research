import torch
import torch.nn as nn
import os
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

        self.a = nn.Parameter(0.001 * torch.randn((Ka, La, 2), dtype=torch.float32))
        self.b = nn.Parameter(0.001 * torch.randn((Kb, Lb, Mb, 2), dtype=torch.float32))
        self.c = nn.Parameter(0.001 * torch.randn((Kc, Lc, Mc, 2), dtype=torch.float32))

        self.register_buffer('powers_Ka', torch.arange(Ka, dtype=torch.float32))
        self.register_buffer('powers_Kb', torch.arange(Kb, dtype=torch.float32))
        self.register_buffer('powers_Kc', torch.arange(Kc, dtype=torch.float32))
    
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

    @utils.complex_handler
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, 2]
        returns y: [B, T, 2]
        """
        if x.dim() != 3 or x.shape[-1] != 2:
            raise ValueError(f"Input must be [B, T, 2] (re,im), but expected {x.shape}")

        device = x.device
        B, T, _ = x.shape

        t_idx = torch.arange(T, device=device)
        idx_La = (t_idx[None, :] - torch.arange(self.La, device=device)[:, None]).clamp(0, T - 1).long()
        idx_Lb = (t_idx[None, :] - torch.arange(self.Lb, device=device)[:, None]).clamp(0, T - 1).long()
        idx_Lc = (t_idx[None, :] - torch.arange(self.Lc, device=device)[:, None]).clamp(0, T - 1).long()

        idx_Mb = (
            t_idx[None, :, None]
            - torch.arange(self.Lb, device=device)[:, None, None]
            - torch.arange(self.Mb, device=device)[None, None, :]
        ).clamp(0, T - 1).long()

        idx_Mc = (
            t_idx[None, :, None]
            - torch.arange(self.Lc, device=device)[:, None, None]
            + torch.arange(self.Mc, device=device)[None, None, :]
        ).clamp(0, T - 1).long()
        
        idx_La_b = idx_La.unsqueeze(0).expand(B, -1, -1)
        idx_Lb_b = idx_Lb.unsqueeze(0).expand(B, -1, -1)
        idx_Lc_b = idx_Lc.unsqueeze(0).expand(B, -1, -1)

        idx_Mb_b = idx_Mb.unsqueeze(0).expand(B, -1, -1, -1)
        idx_Mc_b = idx_Mc.unsqueeze(0).expand(B, -1, -1, -1)

        x_unsqz_1 = x.unsqueeze(1)
        x_exp_La = x_unsqz_1.expand(-1, self.La, -1, -1)
        x_exp_Lb = x_unsqz_1.expand(-1, self.Lb, -1, -1)
        x_exp_Lc = x_unsqz_1.expand(-1, self.Lc, -1, -1)

        idx_La_idx = idx_La_b.unsqueeze(-1).expand(-1, -1, -1, 2)
        idx_Lb_idx = idx_Lb_b.unsqueeze(-1).expand(-1, -1, -1, 2)
        idx_Lc_idx = idx_Lc_b.unsqueeze(-1).expand(-1, -1, -1, 2)
        
        x_La = torch.gather(x_exp_La, dim=2, index=idx_La_idx)
        x_Lb = torch.gather(x_exp_Lb, dim=2, index=idx_Lb_idx)
        x_Lc = torch.gather(x_exp_Lc, dim=2, index=idx_Lc_idx)

        x_exp_for_Mb = x_exp_Lb.unsqueeze(3).expand(-1, -1, -1, self.Mb, -1)

        idx_Mb_idx = idx_Mb_b.unsqueeze(-1).expand(-1, -1, -1, -1, 2)

        idx_Mb_idx = idx_Mb_idx.clamp(0, x_exp_for_Mb.size(3) - 1)
        x_Mb = torch.gather(x_exp_for_Mb, dim=3, index=idx_Mb_idx)

        x_exp_for_Mc = x_exp_Lc.unsqueeze(3)
        idx_Mc_idx = idx_Mc_b.unsqueeze(-1).expand(-1, -1, -1, -1, 2)
        idx_Mc_idx = idx_Mc_idx.clamp(0, x_exp_for_Mc.size(3) - 1)
        x_Mc = torch.gather(x_exp_for_Mc, dim=3, index=idx_Mc_idx)

        abs_a = torch.sqrt(x_La[..., 0] ** 2 + x_La[..., 1] ** 2)
        abs_b = torch.sqrt(x_Mb[..., 0] ** 2 + x_Mb[..., 1] ** 2)
        abs_c = torch.sqrt(x_Mc[..., 0] ** 2 + x_Mc[..., 1] ** 2)

        abs_powers_a = (abs_a.unsqueeze(-1) ** self.powers_Ka.to(device))
        abs_powers_b = (abs_b.unsqueeze(-1) ** self.powers_Kb.to(device))
        abs_powers_c = (abs_c.unsqueeze(-1) ** self.powers_Kc.to(device))

        x_La = x_La.unsqueeze(3)
        x_scaled_a = x_La * abs_powers_a.unsqueeze(-1)

        x_Lb = x_Lb.unsqueeze(3).unsqueeze(4)
        abs_powers_b = abs_powers_b.unsqueeze(-1)
        x_scaled_b = x_Lb * abs_powers_b

        x_Lc = x_Lc.unsqueeze(3).unsqueeze(4)
        abs_powers_c = abs_powers_c.unsqueeze(-1)
        x_scaled_c = x_Lc * abs_powers_c

        a_re = self.a[..., 0]
        a_im = self.a[..., 1]

        x_a_re = x_scaled_a[..., 0]
        x_a_im = x_scaled_a[..., 1]

        term_a_re = torch.einsum('kl,bltk->bt', a_re, x_a_re) - torch.einsum('kl,bltk->bt', a_im, x_a_im)
        term_a_im = torch.einsum('kl,bltk->bt', a_re, x_a_im) + torch.einsum('kl,bltk->bt', a_im, x_a_re)

        b_re = self.b[..., 0]
        b_im = self.b[..., 1]
        x_b_re = x_scaled_b[..., 0]
        x_b_im = x_scaled_b[..., 1]

        term_b_re = torch.einsum('klm,bltmk->bt', b_re, x_b_re) - torch.einsum('klm,bltmk->bt', b_im, x_b_im)
        term_b_im = torch.einsum('klm,bltmk->bt', b_re, x_b_im) + torch.einsum('klm,bltmk->bt', b_im, x_b_re)

        c_re = self.c[..., 0]
        c_im = self.c[..., 1]
        x_c_re = x_scaled_c[..., 0]
        x_c_im = x_scaled_c[..., 1]

        term_c_re = torch.einsum('klm,bltmk->bt', c_re, x_c_re) - torch.einsum('klm,bltmk->bt', c_im, x_c_im)
        term_c_im = torch.einsum('klm,bltmk->bt', c_re, x_c_im) + torch.einsum('klm,bltmk->bt', c_im, x_c_re)

        y_re = term_a_re + term_b_re + term_c_re
        y_im = term_a_im + term_b_im + term_c_im

        y = torch.stack([y_re, y_im], dim=-1)
        return y