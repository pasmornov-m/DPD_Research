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
    
    def count_flops(self) -> int:
        """
        Approximate FLOPs estimation for TCN model.

        FLOPs are counted for:
        - one output sample

        Notes
        -----

        Assumptions:
        - add/sub/mul : 1 FLOP
        - Hardswish   : ~6 FLOPs (approx)
        - Conv1d:
            * dense conv: 2 * Cin * Cout * K
            * depthwise: 2 * Cin * K

        Returns
        -------
        int: FLOPs per one output sample
        """

        C_in = self.in_channels
        C_h = self.hidden_channels
        C_out = self.out_channels
        K = self.kernel_size

        # 1) Initial 1x1 conv: C_in -> C_h

        conv1_flops = 2 * C_in * C_h

        # Hardswish after conv1

        hswish1 = 6 * C_h

        # Depthwise Conv blocks
        #
        # Each:
        # - depthwise conv: 2 * C_h * K
        # - hardswish: ~6 ops per channel

        dw_flops = 0

        for _ in range(4):
            dw_flops += 2 * C_h * K   # depthwise conv
            dw_flops += 6 * C_h       # activation

        # Final 1x1 conv: C_h -> C_out

        conv_final_flops = 2 * C_h * C_out

        # Residual projection (if needed)

        if self.in_channels != self.out_channels:
            residual_flops = 2 * C_in * C_out
        else:
            residual_flops = 0

        # Residual addition

        residual_add = C_out

        # TOTAL

        total_flops = (
            conv1_flops
            + hswish1
            + dw_flops
            + conv_final_flops
            + residual_flops
            + residual_add
        )

        return int(total_flops)

    def count_macs(self, sequence_length: int = 1) -> int:
        """
        Approximate MACs estimation for TCN model.

        Counts only:
        - multiplications
        - accumulations

        Does NOT count:
        - activations (Hardswish)
        - padding/cropping
        - transpose/view operations
        - memory access overhead

        Parameters
        ----------
        sequence_length : int
            Input temporal length T.

        Returns
        -------
        int
            MACs for one forward pass.
        """

        T = sequence_length

        ic = self.in_channels
        hc = self.hidden_channels
        oc = self.out_channels
        ks = self.kernel_size

        total_mac = 0

        # Input projection
        #
        # Conv1d:
        # in_channels=ic
        # out_channels=hc
        # kernel_size=1
        #
        # MACs per output element:
        # ic multiplications + accumulations

        input_proj_mac = T * ic * hc

        total_mac += input_proj_mac

        # Depthwise convolutions
        #
        # groups = hidden_channels
        #
        # Each channel has independent convolution:
        #
        # MACs:
        # T * hc * ks
        #
        # Four depthwise blocks

        depthwise_mac = 4 * T * hc * ks

        total_mac += depthwise_mac

        # Output projection
        #
        # Conv1d:
        # hc -> oc
        # kernel_size=1

        output_proj_mac = T * hc * oc

        total_mac += output_proj_mac

        # Residual projection
        #
        # Only if in_channels != out_channels

        if ic != oc:
            residual_mac = T * ic * oc
            total_mac += residual_mac

        return int(total_mac)



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
        
        kp = self.kp_proj(kp)
        
        out = self.tcn(kp)
        
        return out
    
    def _get_filename(self):
        return (
            f"{self.model_name}_{self.class_name}_"
            f"hc{self.hidden_channels}_oc{self.out_channels}_ks{self.kernel_size}_"
            f"M{self.kp_module.M}_K{self.kp_module.K}_rd{self.reduced_dim}_"
            f"fd{self.feature_dim}_kp{self.kp_type}.pt"
        )

    def count_flops(self) -> int:
        """
        Approximate FLOPs estimation for KPTCN model.

        FLOPs are counted for:
        - one output sample (one sequence)

        Notes
        -----

        Assumptions:
        - Linear: 2 * in * out
        - reshape/concat: ignored
        - KP module: uses its own count_flops()
        - TCN module: uses its own count_flops()

        Returns
        -------
        int: FLOPs per one output sample
        """

        F = self.feature_dim
        R = self.reduced_dim

        # KP MODULE

        # already implemented inside KP module
        kp_flops = self.kp_module.count_flops()

        # RESHAPE (B,T,F,2) -> (B,T,2F)

        reshape_flops = 0  # no arithmetic operations

        # LINEAR PROJECTION (F -> reduced_dim)
        # applied per timestep

        # per timestep: 2 * F * R
        proj_flops = 2 * F * R

        # TCN MODULE

        # TCN already defines its own FLOPs model
        tcn_flops = self.tcn.count_flops()

        # TOTAL

        total_flops = (
            kp_flops
            + reshape_flops
            + proj_flops
            + tcn_flops
        )

        return int(total_flops)
    
    def count_macs(self, sequence_length: int = 1) -> int:
        """
        Approximate MACs estimation for KPTCN model.

        Counts only multiplications + accumulations.

        Assumptions:
        - KP module already implements count_macs()
        - TCN already implements count_macs()
        - Linear projection counted explicitly
        - reshape/concat ignored

        Parameters
        ----------
        sequence_length : int
            Temporal length T.

        Returns
        -------
        int
            MACs per forward pass.
        """

        T = sequence_length

        # ==========================================================
        # KP FEATURE EXTRACTOR
        # ==========================================================

        kp_mac = self.kp_module.count_macs(sequence_length=T)

        # ==========================================================
        # FEATURE FLATTEN (ignored)
        # ==========================================================

        # ==========================================================
        # LINEAR PROJECTION
        #
        # feature_dim -> reduced_dim
        # applied per timestep
        # ==========================================================

        proj_mac = T * self.feature_dim * self.reduced_dim

        # ==========================================================
        # TEMPORAL CONVOLUTION BACKBONE (TCN)
        # ==========================================================

        tcn_mac = self.tcn.count_macs(sequence_length=T)

        # ==========================================================
        # TOTAL
        # ==========================================================

        total_mac = kp_mac + proj_mac + tcn_mac

        return int(total_mac)