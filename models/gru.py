import torch
import torch.nn as nn
from models.base_model import BaseModel
from models.kp_module import KPConvModule, KPModule
from modules import utils


class GRU(BaseModel, torch.nn.Module):
    def __init__(self, 
                 input_size: int = 2, 
                 hidden_size: int = 64, 
                 num_layers: int = 1, 
                 output_size: int = 2, 
                 bidirectional: bool = False, 
                 batch_first: bool = True,
                 bias: bool = True,
                 model_name: str = ""):
        
        torch.nn.Module.__init__(self)
        BaseModel.__init__(self, model_name=model_name)
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.bidirectional = bidirectional
        self.batch_first = batch_first
        self.bias = bias
        self.num_directions = 2 if bidirectional else 1
        
        self.gru = torch.nn.GRU(input_size=self.input_size, 
                                hidden_size=self.hidden_size, 
                                num_layers=self.num_layers, 
                                batch_first=self.batch_first, 
                                bidirectional=self.bidirectional,
                                bias=self.bias)
        
        self.fc = torch.nn.Linear(in_features=hidden_size*self.num_directions, 
                                  out_features=self.output_size,
                                  bias=self.bias)

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
            f"in{self.input_size}_out{self.output_size}_bi{int(self.bidirectional)}.pt"
        )
    
    def count_flops(self) -> int:
        """
        Approximate FLOPs estimation for GRU model.

        FLOPs are counted for:
        - one output sample

        Notes
        -----

        Assumptions:
        - add/sub/mul      : 1 FLOP
        - sigmoid          : 4 FLOPs
        - tanh             : 4 FLOPs

        Returns
        -------
        int: FLOPs per one output sample
        """

        H = self.hidden_size
        I = self.input_size
        O = self.output_size
        L = self.num_layers
        D = self.num_directions

        # GRU CELL
        #
        # Gates:
        # z_t : update gate
        # r_t : reset gate
        # n_t : candidate hidden state
        #
        # Each gate:
        #
        # x_t @ W_ih
        # h_t @ W_hh
        #
        # FLOPs:
        #
        # 2 * input_dim * H
        # +
        # 2 * H * H
        #
        # multiplied by 3 gates

        gru_flops = 0

        for layer in range(L):

            # Input dimension

            if layer == 0:
                input_dim = I
            else:
                input_dim = H * D

            # Matrix multiplications
            #
            # 3 GRU gates

            gate_flops = (
                3
                * (
                    2 * input_dim * H
                    + 2 * H * H
                )
            )

            # Bias additions

            if self.bias:
                gate_flops += 3 * H

            # Activations
            #
            # 2 sigmoid:
            # z_t, r_t
            #
            # 1 tanh:
            # n_t

            activation_flops = (
                2 * H * 4
                + 1 * H * 4
            )

            # Hidden state update
            #
            # n_t = tanh(...)
            #
            # h_t = (1 - z_t) * n_t + z_t * h_prev
            #
            # elementwise ops

            elementwise_flops = (
                5 * H
            )

            layer_flops = (
                gate_flops
                + activation_flops
                + elementwise_flops
            )

            gru_flops += layer_flops * D

        # OUTPUT FC
        #
        # Linear(H*D -> O)

        fc_flops = (
            2
            * (H * D)
            * O
        )

        if self.bias:
            fc_flops += O

        # Per sample

        one_sample_flops = (
            gru_flops
            + fc_flops
        )

        return int(one_sample_flops)

    def count_macs(self, sequence_length: int = 1) -> int:
        """
        Approximate MACs estimation for GRU model.

        Counts only:
        - multiplications
        - accumulations

        Does NOT count:
        - sigmoid/tanh activations
        - bias additions
        - memory access
        - tensor reshaping

        Notes
        -----
        Complexity is estimated for:
        - one forward pass
        - one sample

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

        input_size = self.input_size
        hidden_size = self.hidden_size
        num_layers = self.num_layers
        output_size = self.output_size
        num_directions = self.num_directions

        total_mac = 0

        # GRU CELL
        #
        # Gates:
        # - update gate z
        # - reset gate r
        # - candidate state n
        #
        # Each gate contains:
        #
        # x_t @ W_ih
        # h_t @ W_hh
        #
        # Per timestep:
        #
        # 3 * (
        #     input_size * hidden_size
        #     +
        #     hidden_size * hidden_size
        # )
        #
        # For stacked layers:
        # input_size becomes:
        # hidden_size * num_directions

        layer_input_size = input_size

        for _ in range(num_layers):

            gru_layer_mac = (
                3
                * T
                * num_directions
                * (
                    layer_input_size * hidden_size
                    + hidden_size * hidden_size
                )
            )

            total_mac += gru_layer_mac

            layer_input_size = hidden_size * num_directions

        # Output projection
        #
        # Linear:
        # hidden_size * directions -> output_size
        #
        # Applied for each timestep

        fc_mac = (
            T
            * hidden_size
            * num_directions
            * output_size
        )

        total_mac += fc_mac

        return int(total_mac)



class KPGRU(BaseModel, torch.nn.Module):
    def __init__(self, 
                 M: int = 5,
                 hidden_size: int = 5, 
                 num_layers: int = 1, 
                 output_size: int = 2, 
                 reduced_dim: int = 4,
                 bidirectional: bool = False, 
                 batch_first: bool = True,
                 bias: bool = True, 
                 kp_type: str = 'conv',
                 model_name: str = ""):
        
        torch.nn.Module.__init__(self)
        BaseModel.__init__(self, model_name=model_name)
        
        self.kp_type = kp_type
        
        match kp_type:
            case 'conv':
                self.kp_module = KPConvModule(M=M)
            case 'linear':
                self.kp_module = KPModule(M=M)
            case _:
                raise ValueError(f"Unsupported kp module type: {kp_type}")
        
        self.feature_dim = self.kp_module.get_feature_dim()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.bidirectional = bidirectional
        self.batch_first = batch_first
        self.bias = bias
        self.num_directions = 2 if bidirectional else 1
        
        self.reduced_dim = reduced_dim
        self.kp_proj = torch.nn.Linear(self.feature_dim, reduced_dim)
        
        self.gru_model = GRU(input_size=reduced_dim,
                               hidden_size=hidden_size,
                               num_layers=num_layers,
                               output_size=output_size,
                               bidirectional=bidirectional,
                               batch_first=batch_first,
                               bias=bias)
    
    @utils.complex_handler
    def forward(self, x, h_0=None):
        """
        x: [batch, seq_len, 2]
        """
                
        kp_features = self.kp_module(x)  # [B, T, F, 2]
        B, T, F, C = kp_features.shape
        kp = kp_features.reshape(B, T, F * C)
        kp = self.kp_proj(kp)
                
        output = self.gru_model(kp, h_0)
        
        return output
    
    def _get_filename(self):
        return (
            f"{self.model_name}_{self.class_name}_"
            f"hs{self.hidden_size}_nl{self.num_layers}_"
            f"out{self.output_size}_bi{int(self.bidirectional)}_"
            f"M{self.kp_module.M}_K{self.kp_module.K}_rd{self.reduced_dim}_"
            f"fd{self.feature_dim}_kp{self.kp_type}.pt"
        )

    def count_flops(self) -> int:
        """
        Approximate FLOPs estimation for KPGRU model.

        FLOPs are counted for:
        - one output sample (one sequence)

        Notes
        -----

        Assumptions:
        - KP module has its own count_flops()
        - Linear: 2 * in * out
        - GRU has its own count_flops() (per sample)
        - reshape/concat: free

        Returns
        -------
        int: FLOPs per one output sample
        """

        F = self.feature_dim
        R = self.reduced_dim

        # KP MODULE

        kp_flops = self.kp_module.count_flops()

        # RESHAPE (B,T,F,2) -> (B,T,2F)

        reshape_flops = 0

        # LINEAR PROJECTION (feature reduction)

        proj_flops = 2 * F * R

        # GRU MODULE

        gru_flops = self.gru_model.count_flops()

        # TOTAL

        total_flops = (
            kp_flops
            + reshape_flops
            + proj_flops
            + gru_flops
        )

        return int(total_flops)

    def count_macs(self, sequence_length: int = 1) -> int:
        """
        Approximate MACs estimation for KPGRU model.

        Counts only multiplications + accumulations.

        Assumptions:
        - KP module already implements count_macs()
        - GRU already implements count_macs()
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

        # KP FEATURE EXTRACTOR

        kp_mac = self.kp_module.count_macs(sequence_length=T)

        # FEATURE FLATTEN (ignored)

        # LINEAR PROJECTION
        #
        # feature_dim -> reduced_dim
        # applied per timestep

        proj_mac = T * self.feature_dim * self.reduced_dim

        # GRU BACKBONE

        gru_mac = self.gru_model.count_macs(sequence_length=T)

        # TOTAL

        total_mac = kp_mac + proj_mac + gru_mac

        return int(total_mac)