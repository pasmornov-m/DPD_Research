import torch
from models.base_model import BaseModel


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

        self.fir = torch.nn.Linear(in_features=2*(self.M+1), 
                                   out_features=2*self.L1, 
                                   bias=True)
        
        self.lea_1_weight = torch.nn.Parameter(torch.randn(self.L1, L2))
        self.lea_1_bias = torch.nn.Parameter(torch.zeros(self.L1, L2))
        
        self.lea_k_weight = torch.nn.Parameter(torch.randn(L2, L3))
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
    
    def count_flops(self) -> int:
        """
        Approximate FLOPs estimation for LEANN model.

        FLOPs are counted for:
        - one forward pass
        - one sample

        Notes
        -----

        Assumptions:
        - add/sub/mul/abs : 1 FLOP
        - sqrt            : 10 FLOPs
        - atan2           : 20 FLOPs
        - sin/cos         : 20 FLOPs
        - power x**K      : (K - 1) muls

        Returns
        -------
        int: FLOPs per one output sample
        """

        M = self.M
        L1 = self.L1
        L2 = self.L2
        L3 = self.L3
        K = self.K

        # FIR
        #
        # Linear:
        # y = Wx + b
        #
        # FLOPs:
        # 2 * in_features * out_features

        in_features = 2 * (M + 1)
        out_features = 2 * L1

        fir_flops = 2 * in_features * out_features

        # VDM
        #
        # amplitude:
        # I^2              -> 1 mul
        # Q^2              -> 1 mul
        # add              -> 1 add
        # sqrt             -> 10
        #
        # phase:
        # atan2            -> 20
        #
        # total per element:
        # 13 + 20 = 33

        vdm_flops = 33 * L1

        # LEA-1
        #
        # f = weight * abs(abs(u) - bias)
        #
        # Per element:
        # abs(u)           -> 1
        # subtract bias    -> 1
        # abs              -> 1
        # multiply weight  -> 1
        #
        # total = 4
        #
        # Reduction:
        # sum(dim=1)

        lea1_elementwise = 4 * L1 * L2
        lea1_reduction = (L1 - 1) * L2

        lea1_flops = lea1_elementwise + lea1_reduction

        # LEA-K
        #
        # h = weight * abs(abs(v) - bias) ** K
        #
        # Per element:
        # abs(v)           -> 1
        # subtract bias    -> 1
        # abs              -> 1
        # power            -> (K - 1)
        # multiply weight  -> 1
        #
        # total = K + 3

        leak_flops = (K + 3) * L2 * L3

        # PRB
        #
        # I = h * sin(phase)
        # Q = h * cos(phase)
        #
        # Per element:
        # sin              -> 20
        # mul              -> 1
        # cos              -> 20
        # mul              -> 1
        #
        # total = 42

        prb_flops = 42 * L2 * L3

        # ROW SUM
        #
        # sum over [L2, L3]

        row_sum_flops = 2 * (L2 * L3 - 1)

        # Per sample

        one_sample_flops = (
            fir_flops
            + vdm_flops
            + lea1_flops
            + leak_flops
            + prb_flops
            + row_sum_flops
        )

        return int(one_sample_flops)

    def count_macs(self) -> int:
        """
        Approximate MACs estimation for LEANN model.

        Counts only:
        - multiplications
        - accumulations (additions in reductions / dot products)

        Does NOT count:
        - abs
        - sqrt
        - atan2
        - sin/cos
        - comparisons
        - indexing/view/reshape

        Notes
        -----
        MAC complexity is usually more representative for:
        - FPGA/ASIC implementations
        - DSP hardware
        - neural DPD literature

        Returns
        -------
        int
            MACs per one output sample.
        """

        M = self.M
        L1 = self.L1
        L2 = self.L2
        L3 = self.L3
        K = self.K

        total_mac = 0

        # FIR layer
        #
        # Linear:
        # y = Wx + b
        #
        # For each output:
        # - in_features multiplications
        # - in_features - 1 accumulations
        #
        # MAC ~= in_features * out_features

        in_features = 2 * (M + 1)
        out_features = 2 * L1

        fir_mac = in_features * out_features

        total_mac += fir_mac

        # LEA-1
        #
        # f = weight * abs(abs(u) - bias)
        #
        # Count only:
        # - multiply weight
        #
        # Shape:
        # [L1, L2]
        #
        # Plus reduction over dim=1

        lea1_mul = L1 * L2
        lea1_acc = (L1 - 1) * L2

        lea1_mac = lea1_mul + lea1_acc

        total_mac += lea1_mac

        # LEA-K
        #
        # h = weight * abs(abs(v) - bias) ** K
        #
        # Count:
        # - power via repeated multiplications
        # - final multiply by weight
        #
        # Per element:
        # - (K - 1) muls for power
        # - 1 mul for weight
        #
        # Shape:
        # [L2, L3]

        leak_mul = K * L2 * L3

        total_mac += leak_mul

        # PRB
        #
        # I = h * sin(phase)
        # Q = h * cos(phase)
        #
        # Ignore sin/cos evaluation.
        # Count only output multiplications.
        #
        # Per element:
        # - 2 multiplications

        prb_mac = 2 * L2 * L3

        total_mac += prb_mac

        # Row sum
        #
        # sum over [L2, L3]
        #
        # Two outputs:
        # - I accumulation
        # - Q accumulation

        row_sum_mac = 2 * (L2 * L3 - 1)

        total_mac += row_sum_mac

        return int(total_mac)