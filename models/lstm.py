import torch
from models.base_model import BaseModel
from models.kp_module import KPConvModule, KPModule
from modules import utils


class LSTM(BaseModel, torch.nn.Module):
    def __init__(self, 
                 input_size=2, 
                 hidden_size=64, 
                 num_layers=1, 
                 output_size=2, 
                 bidirectional=False, 
                 batch_first=True,
                 bias=True, 
                 model_name=""):
        
        torch.nn.Module.__init__(self)
        BaseModel.__init__(self, model_name=model_name)
        
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.batch_first = batch_first
        self.bias = bias
        self.num_directions = 2 if bidirectional else 1

        self.lstm = torch.nn.LSTM(input_size=input_size,
                          hidden_size=hidden_size,
                          num_layers=num_layers,
                          bidirectional=self.bidirectional,
                          batch_first=self.batch_first,
                          bias=self.bias)
        self.fc = torch.nn.Linear(in_features=hidden_size*self.num_directions,
                                  out_features=self.output_size,
                                  bias=self.bias)

    @utils.complex_handler
    def forward(self, x, hx=None):
        batch_size = x.size(0)
                        
        if hx is None:
            h_0 = torch.zeros(self.num_directions * self.num_layers, 
                            batch_size, 
                            self.hidden_size)
            c_0 = torch.zeros(self.num_directions * self.num_layers, 
                            batch_size, 
                            self.hidden_size)
        
        y, (h_n, c_n) = self.lstm(x, (h_0, c_0))
        y = self.fc(y)
        return y
    
    def _get_filename(self):
        return (
            f"{self.model_name}_{self.class_name}_"
            f"hs{self.hidden_size}_nl{self.num_layers}_"
            f"in{self.input_size}_out{self.output_size}_bi{int(self.bidirectional)}.pt"
        )
    
    def count_flops(self) -> int:
        """
        Approximate FLOPs estimation for LSTM model.

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

        # LSTM CELL
        #
        # Gates:
        # i_t, f_t, g_t, o_t
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
        # multiplied by 4 gates

        lstm_flops = 0

        for layer in range(L):

            # Input dimension

            if layer == 0:
                input_dim = I
            else:
                input_dim = H * D

            # Matrix multiplications

            gate_flops = (
                4
                * (
                    2 * input_dim * H
                    + 2 * H * H
                )
            )

            # Bias additions

            if self.bias:
                gate_flops += 4 * H

            # Activations
            #
            # 3 sigmoid:
            # i_t, f_t, o_t
            #
            # 2 tanh:
            # g_t, tanh(c_t)

            activation_flops = (
                3 * H * 4
                + 2 * H * 4
            )

            # Cell update
            #
            # c_t = f_t * c_prev + i_t * g_t
            # h_t = o_t * tanh(c_t)
            #
            # elementwise ops

            elementwise_flops = (
                3 * H
                + 2 * H
            )

            layer_flops = (
                gate_flops
                + activation_flops
                + elementwise_flops
            )

            lstm_flops += layer_flops * D

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
            lstm_flops
            + fc_flops
        )

        return int(one_sample_flops)

    def count_macs(self, sequence_length: int = 1) -> int:
        """
        Approximate MACs estimation for LSTM model.

        Counts only:
        - multiplications
        - accumulations

        Does NOT count:
        - sigmoid/tanh activations
        - bias additions

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

        H = self.hidden_size
        I = self.input_size
        L = self.num_layers
        D = self.num_directions

        total_mac = 0

        # LSTM CELL
        #
        # Gates:
        # i_t, f_t, g_t, o_t  -> 4 gates
        #
        # Each gate has:
        # x_t @ W_ih
        # h_t @ W_hh
        #
        # MACs per timestep:
        #
        # 4 * (input_dim * H + H * H)

        layer_input_size = I

        for _ in range(L):

            lstm_layer_mac = (
                4
                * T
                * D
                * (
                    layer_input_size * H
                    + H * H
                )
            )

            total_mac += lstm_layer_mac

            # next layer input
            layer_input_size = H * D

        # OUTPUT FC
        #
        # Linear(H*D -> O) applied per timestep

        O = self.output_size

        fc_mac = (
            T
            * H
            * D
            * O
        )

        total_mac += fc_mac

        return int(total_mac)



class KPLSTM(BaseModel, torch.nn.Module):
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
        
        self.lstm_model = LSTM(input_size=reduced_dim,
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
                
        output = self.lstm_model(kp)
        
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
        Approximate FLOPs estimation for KPLSTM model.

        FLOPs are counted for:
        - one output sample (one sequence)

        Notes
        -----

        Assumptions:
        - KP module has its own count_flops()
        - Linear: 2 * in * out
        - LSTM has its own count_flops() returning per-sample FLOPs
        - reshape: free

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

        # per timestep: 2 * F * R
        proj_flops = 2 * F * R

        # LSTM MODULE

        lstm_flops = self.lstm_model.count_flops()

        # TOTAL

        total_flops = (
            kp_flops
            + reshape_flops
            + proj_flops
            + lstm_flops
        )

        return int(total_flops)

    def count_macs(self, sequence_length: int = 1) -> int:
        """
        Approximate MACs estimation for KPLSTM model.

        Counts only multiplications + accumulations.

        Assumptions:
        - KP module already implements count_macs()
        - LSTM already implements count_macs()
        - Linear projection is counted explicitly
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

        # KP MODULE (black-box)

        kp_mac = self.kp_module.count_macs(sequence_length=T)

        # FEATURE FLATTEN
        #
        # (B,T,F,2) -> (B,T,F*2)
        # ignored (no MACs)

        # LINEAR PROJECTION
        #
        # feature_dim -> reduced_dim
        # applied per timestep

        linear_mac = T * self.feature_dim * self.reduced_dim

        # LSTM MODULE (black-box)

        lstm_mac = self.lstm_model.count_macs(sequence_length=T)

        # TOTAL

        total_mac = kp_mac + linear_mac + lstm_mac

        return int(total_mac)