import torch
import torch.nn as nn
from modules import utils
from models.base_model import BaseModel


class GMP(BaseModel, nn.Module):
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
                 model_name: str = ""):
        
        torch.nn.Module.__init__(self)
        BaseModel.__init__(self, model_name=model_name)
        
        self.Ka, self.La = Ka, La
        self.Kb, self.Lb, self.Mb = Kb, Lb, Mb
        self.Kc, self.Lc, self.Mc = Kc, Lc, Mc
        
        self.a = nn.Parameter(0.01 * torch.randn((Ka, La), dtype=torch.complex64))
        self.b = nn.Parameter(0.01 * torch.randn((Kb, Lb, Mb), dtype=torch.complex64))
        self.c = nn.Parameter(0.01 * torch.randn((Kc, Lc, Mc), dtype=torch.complex64))

        self.register_buffer('powers_Ka', torch.arange(Ka, dtype=torch.long))
        self.register_buffer('powers_Kb', torch.arange(Kb, dtype=torch.long))
        self.register_buffer('powers_Kc', torch.arange(Kc, dtype=torch.long))
        
        self.register_buffer('arange_La', torch.arange(La, dtype=torch.long)[:, None])
        self.register_buffer('arange_Lb', torch.arange(Lb, dtype=torch.long)[:, None])
        self.register_buffer('arange_Lc', torch.arange(Lc, dtype=torch.long)[:, None])
        
        self.register_buffer('arange_Mb', torch.arange(Mb, dtype=torch.long)[None, None, :])
        self.register_buffer('arange_Mc', torch.arange(Mc, dtype=torch.long)[None, None, :])
    
    def _get_filename(self):
        return (f"{self.model_name}_{self.class_name}_"
                f"Ka{self.Ka}_La{self.La}_"
                f"Kb{self.Kb}_Lb{self.Lb}_Mb{self.Mb}_"
                f"Kc{self.Kc}_Lc{self.Lc}_Mc{self.Mc}.pt")
    
    @staticmethod
    def abs_complex(x: torch.Tensor) -> torch.Tensor:
        eps = 1e-8
        return torch.sqrt(x[..., 0] ** 2 + x[..., 1] ** 2 + eps)
    
    @staticmethod
    def _to_complex(x: torch.Tensor) -> torch.Tensor:
        return torch.view_as_complex(x.contiguous().view(*x.shape[:-1], 2))
    
    def _gather_with_delay(self, 
                           x_padded: torch.Tensor, 
                           t_idx: torch.Tensor, 
                           arange_L: torch.Tensor, 
                           L: int, 
                           M: int = None) -> torch.Tensor:
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
            m_broadcast = torch.arange(M).view(1, -1, 1)
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
        t_idx = torch.arange(max_delay - 1, max_delay - 1 + T)
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

        y = torch.zeros(B, T, dtype=torch.complex64)

        if term_a is not None:
            y += torch.einsum('bltk,k l->bt', term_a, self.a)
        if term_b is not None:
            y += torch.einsum('blmtk,k l m->bt', term_b, self.b)
        if term_c is not None:
            y += torch.einsum('blmtk,k l m->bt', term_c, self.c)

        return torch.view_as_real(y)

    def count_flops(self) -> int:
        """
        Approximate FLOPs estimation for GMP model.

        FLOPs are counted for:
        - one output sample (one time step)

        Notes
        -----

        Assumptions:
        - add/sub/mul        : 1 FLOP
        - complex add        : 2 FLOPs
        - complex mul        : 6 FLOPs
        - abs(complex)       : 3 FLOPs
        - sqrt               : 1 FLOP (absorbed in abs)
        - power x**K         : (K - 1) multiplications
        - einsum             : 2 FLOPs per real multiply-add

        Returns
        -------
        int: FLOPs per one output sample
        """

        Ka, La = self.Ka, self.La
        Kb, Lb, Mb = self.Kb, self.Lb, self.Mb
        Kc, Lc, Mc = self.Kc, self.Lc, self.Mc

        # TERM A
        # y = x[n-l] * |x[n-l]|^k

        # abs(x)
        abs_a = La * 3

        # power
        pow_a = La * Ka

        # multiplication complex * real power
        mul_a = La * Ka * 2   # complex * real scalar

        term_a = La * (abs_a + pow_a) + mul_a

        # TERM B
        # x[n-l] * x[n-l-m] * |x[n-l-m]|^k

        abs_b = Lb * Mb * 3
        pow_b = Lb * Mb * Kb

        # complex multiplication:
        # x * delayed_x -> 6 FLOPs
        cross_mul = Lb * Mb * 6

        mul_b = Lb * Mb * Kb * 2

        term_b = abs_b + pow_b + cross_mul + mul_b

        # TERM C
        # |x[n-l]|^k * x[n-l-m]

        abs_c = Lc * Mc * 3
        pow_c = Lc * Mc * Kc

        cross_mul_c = Lc * Mc * 6

        term_c = abs_c + pow_c + cross_mul_c

        # EINSUM + COEFFICIENT SUM

        # einsum reductions:
        # term_a: k,l -> t  => approx 2 ops per accumulation
        einsum_a = Ka * La * 2

        # term_b: k,l,m -> t
        einsum_b = Kb * Lb * Mb * 2

        # term_c: k,l,m -> t
        einsum_c = Kc * Lc * Mc * 2

        # coefficient multiplications (complex)
        coeff_a = Ka * La * 6
        coeff_b = Kb * Lb * Mb * 6
        coeff_c = Kc * Lc * Mc * 6

        # TOTAL

        total_flops = (
            term_a + term_b + term_c +
            einsum_a + einsum_b + einsum_c +
            coeff_a + coeff_b + coeff_c
        )

        return int(total_flops)
    
    def count_macs(self, sequence_length: int = 1) -> int:
        """
        Approximate MACs estimation for GMP model.

        Counts only:
        - multiplications
        - accumulations

        Does NOT count:
        - abs()
        - sqrt()
        - tensor indexing/gather
        - padding
        - reshape/view
        - memory access overhead

        Notes
        -----
        Complexity is estimated for:
        - one forward pass
        - one batch sample

        Parameters
        ----------
        sequence_length : int
            Temporal sequence length T.

        Returns
        -------
        int
            MACs per forward pass.
        """

        T = sequence_length

        Ka, La = self.Ka, self.La
        Kb, Lb, Mb = self.Kb, self.Lb, self.Mb
        Kc, Lc, Mc = self.Kc, self.Lc, self.Mc

        total_mac = 0

        # TERM A
        #
        # x[n-l] * |x[n-l]|^k
        #
        # For each element:
        # - power: k multiplications (approx)
        # - multiply by complex sample: 1
        #
        # Total:
        # T * La * Ka * (k + 1)
        #
        # einsum accumulation:
        # T * La * Ka

        if La > 0 and Ka > 0:

            # power multiplications
            power_mac_a = T * La * sum(range(Ka))

            # multiply with x
            mult_mac_a = T * La * Ka

            # einsum accumulation
            reduce_mac_a = T * (La * Ka - 1)

            term_a_mac = (
                power_mac_a
                + mult_mac_a
                + reduce_mac_a
            )

            total_mac += term_a_mac

        # TERM B
        #
        # x[n-l] * |x[n-l-m]|^k
        #
        # Shape:
        # [Lb, Mb, Kb]

        if Lb > 0 and Mb > 0 and Kb > 0:

            power_mac_b = T * Lb * Mb * sum(range(Kb))

            mult_mac_b = T * Lb * Mb * Kb

            reduce_mac_b = T * (Lb * Mb * Kb - 1)

            term_b_mac = (
                power_mac_b
                + mult_mac_b
                + reduce_mac_b
            )

            total_mac += term_b_mac

        # TERM C
        #
        # x[n-l] * |x[n-l+m]|^k
        #
        # Shape:
        # [Lc, Mc, Kc]

        if Lc > 0 and Mc > 0 and Kc > 0:

            power_mac_c = T * Lc * Mc * sum(range(Kc))

            mult_mac_c = T * Lc * Mc * Kc

            reduce_mac_c = T * (Lc * Mc * Kc - 1)

            term_c_mac = (
                power_mac_c
                + mult_mac_c
                + reduce_mac_c
            )

            total_mac += term_c_mac

        return int(total_mac)


class GMPTensor(torch.nn.Module, BaseModel):

    def __init__(self, M1, M2, P, model_name: str = ""):
        torch.nn.Module.__init__(self)
        BaseModel.__init__(self, model_name=model_name)

        self.M1=M1
        self.M2=M2
        self.P=P
                
        self.register_buffer("S", 0.001 * torch.randn(M1, M2, P, dtype=torch.complex64))
        self.register_buffer('powers', (2*torch.arange(P, dtype=torch.int))[None,None,:])
        self.register_buffer('arange_M1', torch.arange(M1, dtype=torch.int)[None,:])
        self.register_buffer('arange_M2', torch.arange(M2, dtype=torch.int)[None,:])
        self.register_buffer('delay', torch.tensor(max(self.M1, self.M2)-1, dtype=torch.int))
    
    def set_parameters(self, S_new):
        """Update parameters"""
        self.S.copy_(S_new)
    
    def get_delay(self):
            return self.delay

    def forward(self, x):
        N = len(x)-self.delay

        n_idx = torch.arange(N)[:, None]
        i_idx = self.arange_M1
        j_idx = self.arange_M2

        x_i = x[self.delay+n_idx-i_idx] # (N,M1)
        x_j = x[self.delay+n_idx-j_idx] # (N,M2)

        abs_xj = torch.abs(x_j)
        abs_powers = (abs_xj[:,:,None]**self.powers) # (N,M2,P)

        Phi = (x_i[:, :, None, None] * abs_powers[:, None, :, :]) # (N,M1,M2,P)
        Phi = Phi.reshape(N, self.M1 * self.M2 * self.P)
        S_vec = self.S.reshape(-1)
        y = Phi @ S_vec

        return y


class GMPCP(torch.nn.Module, BaseModel):
    def __init__(self, R, M1, M2, P, model_name: str = "", *args, **kwargs):
        torch.nn.Module.__init__(self)
        BaseModel.__init__(self, model_name=model_name)
        
        self.R, self.M1, self.M2, self.P = R, M1, M2, P
        
        self.register_buffer('a', 0.0001 * torch.randn(M1, R, dtype=torch.complex64))
        self.register_buffer('b', 0.0001 * torch.randn(M2, R, dtype=torch.complex64))
        self.register_buffer('c', 0.0001 * torch.randn(P, R, dtype=torch.complex64))
        
        self.max_delay = max(self.M1, self.M2) - 1
        self.register_buffer('M1_arange', torch.arange(self.M1))
        self.register_buffer('M2_arange', torch.arange(self.M2))
        
        self.register_buffer('bc_real', None)
        self.register_buffer('bc_imag', None)
        self._get_cache_bc()
            
    def _get_cache_bc(self):
        bc = self.b[:, None, :] * self.c[None, :, :]
        self.bc_real = bc.real
        self.bc_imag = bc.imag
    
    def _get_filename(self):
        return (f"{self.model_name}_{self.class_name}_"
                f"R({self.R})_M1({self.M1})_"
                f"M2({self.M2})_P({self.P}).pt")
        
    def get_delay(self):
        return self.max_delay
    
    def set_parameters(self, a_new, b_new, c_new):
        """Update parameters"""
        self.a.copy_(a_new)
        self.b.copy_(b_new)
        self.c.copy_(c_new)
        self._get_cache_bc()
    
    def compute_powers(self, x, P):
        powers = [torch.ones_like(x)]
        
        for _ in range(1, P):
            powers.append(powers[-1] * x)

        return torch.stack(powers, dim=-1)
    
    @utils.iq_handler
    def forward(self, x):
        """
        x: (T,) — комплексный входной сигнал
        returns: (T,) — комплексный выходной сигнал
        """
        T = x.shape[0]
        T_cut = T - self.max_delay
        T_cut_processed_range = torch.arange(T_cut)[:, None] + self.max_delay

        if (self.bc_real is None) or (self.bc_imag is None):
            self._get_cache_bc()
            
        
        indices_m1 = (T_cut_processed_range - self.M1_arange)
        delays_m1 = x[indices_m1]
        sum_i = delays_m1 @ self.a
        
        indices_m2 = (T_cut_processed_range - self.M2_arange)
        abs_x = torch.abs(x)
        abs_delays_m2 = abs_x[indices_m2]
        abs_powers = self.compute_powers(abs_delays_m2, self.P)
                
        sum_jp = torch.einsum('tmp,mpr->tr', abs_powers, self.bc_real) + 1j * torch.einsum('tmp,mpr->tr', abs_powers, self.bc_imag)
        
        y = (sum_i * sum_jp).sum(dim=1)
        
        return y