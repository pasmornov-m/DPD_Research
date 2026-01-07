import torch
from torch import nn
import numpy as np
import os
from modules import utils, data_loader
from pytorch_tcn import TCN


class BaseModel:
    def __init__(self, model_name: str = ""):
        self.model_name = model_name
        self.class_name = self.__class__.__name__
        self._filename = None
    
    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    @property
    def filename(self):
        if self._filename is None:
            self._filename = self._get_filename()
        return self._filename
    
    def _get_filename(self):
        raise NotImplementedError
    
    def save_weights(self, directory: str = "model_params"):
        os.makedirs(directory, exist_ok=True)
        full_path = f"{directory}/{self.filename}"
        torch.save(self.state_dict(), full_path)
        print(f"Model weights saved to {full_path}")

    def load_weights(self, directory: str = "model_params") -> bool:
        full_path = f"{directory}/{self.filename}"
        if os.path.isfile(full_path):
            state_dict = torch.load(full_path, map_location="cpu")
            self.load_state_dict(state_dict)
            print(f"Model weights loaded from {full_path}")
            return True
        else:
            print(f"No saved weights found at {full_path}, initializing new parameters.")
            return False



class GRU(BaseModel, torch.nn.Module):
    def __init__(self, 
                 input_size=2, 
                 hidden_size=64, 
                 num_layers=1, 
                 output_size=2, 
                 bidirectional=False, 
                 batch_first=True, 
                 model_name=""):
        
        torch.nn.Module.__init__(self)
        BaseModel.__init__(self, model_name=model_name)
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.bidirectional = bidirectional
        self.batch_first = batch_first
        self.num_directions = 2 if bidirectional else 1
        
        self.gru = nn.GRU(input_size=self.input_size, 
                          hidden_size=self.hidden_size, 
                          num_layers=self.num_layers, 
                          batch_first=self.batch_first, 
                          bidirectional=self.bidirectional)
        
        self.fc = nn.Linear(in_features=hidden_size*self.num_directions, 
                            out_features=self.output_size)

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
            f"in{self.input_size}_out{self.output_size}_bi{int(self.bidirectional)}.pt")




class LSTM(BaseModel, torch.nn.Module):
    def __init__(self, 
                 input_size=2, 
                 hidden_size=64, 
                 num_layers=1, 
                 output_size=2, 
                 bidirectional=False, 
                 batch_first=True,
                 bias=False, 
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

        self.lstm = nn.LSTM(input_size=input_size,
                          hidden_size=hidden_size,
                          num_layers=num_layers,
                          bidirectional=self.bidirectional,
                          batch_first=self.batch_first,
                          bias=self.bias)
        self.fc_out = nn.Linear(in_features=hidden_size*self.num_directions,
                                out_features=self.output_size,
                                bias=self.bias)

    @utils.complex_handler
    def forward(self, x, h_0=None):
        batch_size = x.size(0)
        
        if h_0 is None or c_0 is None:
            h_0 = torch.zeros(self.num_directions * self.num_layers, 
                            batch_size, 
                            self.hidden_size)
            c_0 = torch.zeros(self.num_directions * self.num_layers, 
                            batch_size, 
                            self.hidden_size)
        
        y, (h_n, c_n) = self.lstm(x, (h_0, c_0))
        y = self.fc_out(y)
        return y
    
    def _get_filename(self):
        return (
            f"{self.model_name}_{self.class_name}_"
            f"hs{self.hidden_size}_nl{self.num_layers}_"
            f"in{self.input_size}_out{self.output_size}_bi{int(self.bidirectional)}.pt")


class CustomTCN(TCN):
    def __init__(self, model_name="custom_tcn", **kwargs):
        num_inputs = kwargs.pop("num_inputs", 2)
        output_projection = kwargs.pop("output_projection", 2)
        dropout = kwargs.pop("dropout", 0.0)
        input_shape = kwargs.pop("input_shape", "NLC")
        super().__init__(num_inputs=num_inputs,
                         output_projection=output_projection,
                         dropout=dropout,
                         input_shape=input_shape,
                         **kwargs)
        self.num_inputs = num_inputs
        self.num_channels = kwargs.get("num_channels")
        self.output_projection = output_projection
        self.kernel_size = kwargs.get("kernel_size")
        self.dropout = dropout
        self.input_shape = input_shape
        self.model_name = model_name
        self.class_name = self.__class__.__name__

    @utils.complex_handler
    def forward(self, x, *args, **kwargs):
        out = super().forward(x, *args, **kwargs)
        return out
    
    def count_params(self):
            return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def save_weights(self, directory="model_params"):
        os.makedirs(directory, exist_ok=True)
        filename = (
            f"{directory}/{self.model_name}_{self.class_name}_"
            f"ch{self.num_channels}_ks{self.kernel_size}_"
            f"in{self.num_inputs}_out{self.output_projection}.pt"
        )
        torch.save(self.state_dict(), filename)
        print(f"Model weights saved to {filename}")

    def load_weights(self, directory="model_params"):
        filename = (
            f"{directory}/{self.model_name}_{self.class_name}_"
            f"ch{self.num_channels}_ks{self.kernel_size}_"
            f"in{self.num_inputs}_out{self.output_projection}.pt"
        )
        if os.path.isfile(filename):
            state_dict = torch.load(filename, map_location='cpu')
            self.load_state_dict(state_dict)
            print(f"Model weights loaded from {filename}")
            return True
        else:
            print(f"No saved weights found at {filename}, initializing new parameters.")
            return False




class TorchESN(nn.Module):
    def __init__(self, 
                 n_inputs, 
                 n_outputs, 
                 n_reservoir=200,
                 spectral_radius=0.95, 
                 sparsity=0, 
                 noise=0.001, 
                 input_shift=None,
                 input_scaling=None, 
                 teacher_forcing=True, 
                 feedback_scaling=None,
                 teacher_scaling=None, 
                 teacher_shift=None,
                 random_state=None):
        """
        Args:
            n_inputs: nr of input dimensions
            n_outputs: nr of output dimensions
            n_reservoir: nr of reservoir neurons
            spectral_radius: spectral radius of the recurrent weight matrix
            sparsity: proportion of recurrent weights set to zero
            noise: noise added to each neuron (regularization)
            input_shift: scalar or vector of length n_inputs to add to each
                        input dimension before feeding it to the network.
            input_scaling: scalar or vector of length n_inputs to multiply
                        with each input dimension before feeding it to the netw.
            teacher_forcing: if True, feed the target back into output units
            teacher_scaling: factor applied to the target signal
            teacher_shift: additive term applied to the target signal
            random_state: positive integer seed, np.rand.RandomState object,
                          or None to use numpy's builting RandomState.
        """
        super().__init__()
        self.n_inputs = n_inputs
        self.n_reservoir = n_reservoir
        self.n_outputs = n_outputs
        self.spectral_radius = spectral_radius
        self.sparsity = sparsity
        self.noise = noise
        self.input_shift = self._correct_dimensions(input_shift, n_inputs)
        self.input_scaling = self._correct_dimensions(input_scaling, n_inputs)

        self.teacher_scaling = teacher_scaling
        self.teacher_shift = teacher_shift

        self.out_activation = self._identity
        self.inverse_out_activation = self._identity
        self.random_state = random_state

        if isinstance(random_state, int):
            torch.manual_seed(random_state)

        self.teacher_forcing = teacher_forcing
        self.initweights()

    def initweights(self):
        W = torch.rand(self.n_reservoir, self.n_reservoir) - 0.5
        mask = torch.rand_like(W) < self.sparsity
        W[mask] = 0.0
        eigvals = torch.linalg.eigvals(W).abs()
        radius = eigvals.max().item()
        self.W = (self.spectral_radius / radius) * W
        self.W_in = torch.rand(self.n_reservoir, self.n_inputs) * 2 - 1
        self.W_feedb = torch.rand(self.n_reservoir, self.n_outputs) * 2 - 1
        
        self.W_comb = torch.cat([self.W, self.W_in, self.W_feedb], dim=1)
    
    def _identity(self, x):
        return x
    
    def _correct_dimensions(self, s, targetlength):
        """checks the dimensionality of some numeric argument s, broadcasts it
        to the specified length if possible.

        Args:
            s: None, scalar or 1D array
            targetlength: expected length of s

        Returns:
            None if s is None, else numpy vector of length targetlength
        """
        if s is not None:
            s = np.array(s)
            if s.ndim == 0:
                s = np.array([s] * targetlength)
            elif s.ndim == 1:
                if not len(s) == targetlength:
                    raise ValueError("Arg must have length " + str(targetlength))
            else:
                raise ValueError("Invalid argument")
        return s

    def _update(self, state, input_pattern, output_pattern, noise):
        """performs one update step.

        i.e., computes the next network state by applying the recurrent weights
        to the last state & and feeding in the current input and output patterns
        """
        inp = torch.cat([state, input_pattern, output_pattern], dim=0)
        preactivation = self.W_comb @ inp
        return torch.tanh(preactivation) + noise

    def _scale_inputs(self, inputs):
        """for each input dimension j: multiplies by the j'th entry in the
        input_scaling argument, then adds the j'th entry of the input_shift
        argument."""
        if self.input_scaling is not None:
            inputs = inputs * self.input_scaling
        if self.input_shift is not None:
            inputs = inputs + self.input_shift
        return inputs

    def _scale_teacher(self, teacher):
        """multiplies the teacher/target signal by the teacher_scaling argument,
        then adds the teacher_shift argument to it."""
        if self.teacher_scaling is not None:
            teacher = teacher * self.teacher_scaling
        if self.teacher_shift is not None:
            teacher = teacher + self.teacher_shift
        return teacher

    def _unscale_teacher(self, teacher_scaled):
        """inverse operation of the _scale_teacher method."""
        if self.teacher_shift is not None:
            teacher_scaled = teacher_scaled - self.teacher_shift
        if self.teacher_scaling is not None:
            teacher_scaled = teacher_scaled / self.teacher_scaling
        return teacher_scaled

    @utils.complex_handler
    def fit(self, inputs, outputs):
        inputs, outputs = map(torch.squeeze, [inputs, outputs])
        N = inputs.shape[0]
        
        inputs = self._scale_inputs(inputs)
        outputs = self._scale_teacher(outputs)
        noise_vec = self.noise * (torch.rand(N, self.n_reservoir) - 0.5)

        states = torch.zeros(N, self.n_reservoir)

        for n in range(1, N):
            states[n] = self._update(states[n - 1], inputs[n], outputs[n - 1], noise_vec[n])
        
        self.laststate = states[-1].detach()
        self.lastinput = inputs[-1].detach()
        self.lastoutput = outputs[-1].detach()

        extended = torch.cat([states, inputs], dim=1)
        transient = min(inputs.shape[1] // 10, 100)
        A = extended[transient:]
        B = self.inverse_out_activation(outputs[transient:])

        W_out_T, *_ = torch.linalg.lstsq(A, B)
        self.W_out = nn.Parameter(W_out_T.T.detach(), requires_grad=True)


    @utils.complex_handler
    def forward(self, inputs, continuation=True):
        inputs = torch.squeeze(inputs)
        N = inputs.shape[0]
        noise_vec = self.noise * (torch.rand(N, self.n_reservoir) - 0.5)

        if continuation:
            prev_state = self.laststate.clone()
            prev_input = self.lastinput.clone()
            prev_output = self.lastoutput.clone()
        else:
            prev_state = torch.zeros(self.n_reservoir)
            prev_input = torch.zeros(self.n_inputs)
            prev_output = torch.zeros(self.n_outputs)

        inputs = torch.cat([prev_input.unsqueeze(0), self._scale_inputs(inputs)], dim=0)
        states = torch.zeros(N + 1, self.n_reservoir)
        outputs = torch.zeros(N + 1, self.n_outputs)

        states[0] = prev_state
        outputs[0] = prev_output

        for n in range(N):
            states[n + 1] = self._update(states[n], inputs[n + 1], outputs[n], noise_vec[n])
            x = torch.cat([states[n + 1], inputs[n + 1]])
            outputs[n + 1] = self.out_activation(self.W_out @ x)

        return self._unscale_teacher(self.out_activation(outputs[1:]))
    



class DiffESN(nn.Module):
    def __init__(self,
                 n_inputs: int = 2,
                 n_outputs: int = 2,
                 n_reservoir: int = 200,
                 spectral_radius: float = 0.95,
                 sparsity: float = 0.0,
                 noise: float = 0.001,
                 input_scaling: float | list[float] = 1.0,
                 input_shift: float | list[float] = 0.0,
                 random_state: int | None = None,
                 model_name: str = "esn"):
        super().__init__()
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.n_reservoir = n_reservoir
        self.noise = noise
        self.model_name = model_name
        self.class_name = self.__class__.__name__

        self.register_buffer('input_scaling', self._make_vector(input_scaling, n_inputs))
        self.register_buffer('input_shift', self._make_vector(input_shift, n_inputs))
        
        if isinstance(random_state, int):
            torch.manual_seed(random_state)
        W = torch.rand(n_reservoir, n_reservoir) - 0.5
        mask = torch.rand_like(W) < sparsity
        W[mask] = 0.0
        eigs = torch.linalg.eigvals(W).abs()
        
        W *= (spectral_radius / eigs.max().item())
        self.register_buffer('W', W)
        W_in = torch.rand(n_reservoir, n_inputs)*2 - 1
        self.register_buffer('W_in', W_in)

        self.W_out = nn.Linear(n_reservoir + n_inputs, n_outputs)
        
        self.state = None
        
        # self.readout = self._make_readout(
        #     input_dim=self.n_reservoir + self.n_inputs,
        #     output_dim=self.n_outputs,
        #     hidden_layers=[16]
        #     )

    def _make_readout(self, input_dim: int, output_dim: int, hidden_layers: list[int]):
        layers = []
        dims = [input_dim] + hidden_layers + [output_dim]
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(dims[-2], dims[-1]))
        return nn.Sequential(*layers)

    def _make_vector(self, v, length):
        t = torch.tensor(v, dtype=torch.float32)
        if t.ndim == 0:
            return t.repeat(length)
        if t.shape[0] != length:
            raise ValueError(f'Expected length {length}, got {t.shape[0]}')
        return t

    def reset_state(self, batch_size: int):
        self.state = torch.zeros(self.n_reservoir, batch_size)
        
    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def named_params(self):
        return [(name, param.shape) for name, param in self.named_parameters()]

    @utils.complex_handler
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch, seq_len, n_inputs]
        returns: [batch, seq_len, n_outputs]
        """
        B, T, _ = x.shape
        self.reset_state(B)
        h = self.state
        noise_vec = self.noise * torch.randn(self.n_reservoir, B, T)
        u_all = (x * self.input_scaling + self.input_shift).transpose(1, 2)
        u_all = u_all.permute(1, 0, 2)

        outputs = []
        for t in range(T):
            u_t = u_all[:, :, t]
            preact = self.W @ h + self.W_in @ u_t
            noise_t = noise_vec[:, :, t]
            h = torch.tanh(preact + noise_t)
            lin_in = torch.cat([h.T, u_t.T], dim=-1)
            # y_t = self.readout(lin_in)
            y_t = self.W_out(lin_in)
            outputs.append(y_t)

        self.state = h.detach()
        result_outputs = torch.stack(outputs, dim=1)

        return result_outputs

    def save_weights(self, directory="model_params"):
        os.makedirs(directory, exist_ok=True)
        filename = (
            f"{directory}/{self.model_name}_{self.class_name}_"
            f"res{self.n_reservoir}_in{self.n_inputs}_out{self.n_outputs}.pt"
        )
        torch.save(self.state_dict(), filename)
        print(f"Model weights saved to {filename}")

    def load_weights(self, directory="model_params"):
        filename = (
            f"{directory}/{self.model_name}_{self.class_name}_"
            f"res{self.n_reservoir}_in{self.n_inputs}_out{self.n_outputs}.pt"
        )
        if os.path.isfile(filename):
            state_dict = torch.load(filename)
            self.load_state_dict(state_dict)
            print(f"Model weights loaded from {filename}")
            return True
        else:
            print(f"No saved weights found at {filename}, initializing new parameters.")
            return False




class BuildingBlock(nn.Module):
    """
    Один «строительный» блок DenseNet:
    (BN → FC → ReLU) × 3 + конкатенация входа и выхода
    Поддерживает входы размера (B, T, N)
    """
    def __init__(self, in_features: int):
        super().__init__()
        self.bn1 = nn.BatchNorm1d(in_features)
        self.fc1 = nn.Linear(in_features, in_features)
        self.bn2 = nn.BatchNorm1d(in_features)
        self.fc2 = nn.Linear(in_features, in_features)
        self.bn3 = nn.BatchNorm1d(in_features)
        self.fc3 = nn.Linear(in_features, in_features)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, N = x.shape
        x_ = x.reshape(B * T, N)

        out = self.relu(self.fc1(self.bn1(x_)))
        out = self.relu(self.fc2(self.bn2(out)))
        out = self.relu(self.fc3(self.bn3(out)))

        out = out.view(B, T, N)
        return torch.cat([x, out], dim=2)


class DenseNetRegressor(nn.Module):
    def __init__(self, blocks: int, input_dim: int = 2, model_name: str = "densenet"):
        super().__init__()
        self.input_blocks = blocks
        self.input_dim = input_dim
        self.model_name = model_name
        self.class_name = self.__class__.__name__
        
        self.init_fc = nn.Linear(input_dim, input_dim)
        self.init_bn = nn.BatchNorm1d(input_dim)
        self.relu = nn.ReLU(inplace=True)

        self.blocks = nn.ModuleList()
        features = input_dim
        for _ in range(blocks):
            self.blocks.append(BuildingBlock(features))
            features *= 2

        self.final_bn = nn.BatchNorm1d(features)
        self.final_fc = nn.Linear(features, input_dim)

    @utils.complex_handler
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, N = x.shape
        x_ = x.reshape(B * T, N)

        x_ = self.relu(self.init_fc(self.init_bn(x_)))
        x = x_.view(B, T, N)

        for block in self.blocks:
            x = block(x)

        B, T, N = x.shape
        x_ = x.reshape(B * T, N)
        x_ = self.final_fc(self.final_bn(x_))

        return x_.view(B, T, -1)

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def named_params(self):
        return [(name, param.shape) for name, param in self.named_parameters()]
    
    def save_weights(self, directory="model_params"):
        os.makedirs(directory, exist_ok=True)
        filename = f"{directory}/{self.model_name}_{self.class_name}_" \
                   f"blocks{self.input_blocks}_in{self.input_dim}.pt"
        torch.save(self.state_dict(), filename)
        print(f"Model weights saved to {filename}")

    def load_weights(self, directory="model_params"):
        filename = f"{directory}/{self.model_name}_{self.class_name}_" \
                   f"blocks{self.input_blocks}_in{self.input_dim}.pt"
        if os.path.isfile(filename):
            state_dict = torch.load(filename)
            self.load_state_dict(state_dict)
            print(f"Model weights loaded from {filename}")
            return True
        else:
            print(f"No saved weights found at {filename}, initializing new parameters.")
            return 




class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


class TransformerEncoderBlock(torch.nn.Module):
    def __init__(self, d_model, nhead, d_ff):
        super().__init__()
        self.attn = torch.nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.ln1 = torch.nn.LayerNorm(d_model)
        
        # FFN заменяем на две 1D-CNN с kernel=1
        self.conv1 = torch.nn.Conv1d(d_model, d_ff, kernel_size=1)
        self.conv2 = torch.nn.Conv1d(d_ff, d_model, kernel_size=1)
        self.ln2 = torch.nn.LayerNorm(d_model)
        
        self.activation = torch.nn.Tanh()

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        attn_out, _ = self.attn(x, x, x)
        x = x + attn_out
        x = self.ln1(x)

        # 1D-CNN: надо поменять размерность для Conv1d
        x_cnn = x.transpose(1, 2)  # (batch, d_model, seq_len)
        x_cnn = self.conv1(x_cnn)
        x_cnn = self.activation(x_cnn)
        x_cnn = self.conv2(x_cnn)
        x_cnn = self.activation(x_cnn)
        x_cnn = x_cnn.transpose(1, 2)  # (batch, seq_len, d_model)

        x = self.ln2(x + x_cnn)        
        return x




class RTDTNN(BaseModel, torch.nn.Module):
    def __init__(self, 
                 d_in: int,
                 d_model: int = 6, 
                 n_heads: int = 2, 
                 d_ff: int = 10, 
                 n_fc: int = 8, 
                 M: int = 5, 
                 num_blocks: int = 1, 
                 model_name: str = "model"):
        
        torch.nn.Module.__init__(self)
        BaseModel.__init__(self, model_name=model_name)
        
        self.M = M
        self.T = M + 1
        self.d_in = d_in
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.n_fc = n_fc
        self.num_blocks = num_blocks
        
        self.in_norm = nn.LayerNorm(d_in)
        
        self.input_fc = torch.nn.Linear(d_in, d_model)
        
        self.encoders = torch.nn.ModuleList(
            [TransformerEncoderBlock(d_model, n_heads, d_ff) for _ in range(num_blocks)]
        )
        
        self.fc = torch.nn.Linear(self.T * d_model, n_fc)
        self.activation = torch.nn.Tanh()
        
        self.out = torch.nn.Linear(n_fc, 2)
        
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

        
    def forward(self, x):
        # x: (batch, seq_len, d_in)
        x = self.input_fc(x)                    # (batch, seq_len, d_model)

        for encoder in self.encoders:           # transformer encoder
            x = encoder(x)
        
        x = x.reshape(x.size(0), -1)          # (batch, seq_len*d_model)
        x = self.fc(x)
        x = self.activation(x)                     # FC + tanh
        x = self.out(x)                       # (batch, 2)
        return x
    
    def _get_filename(self):
        return (
            f"{self.model_name}_{self.class_name}"
            f"_din{self.d_in}_dmodel{self.d_model}_heads{self.n_heads}"
            f"_dff{self.d_ff}_nfc{self.n_fc}_M{self.M}_blocks{self.num_blocks}.pt"
            )




class CustomLSTM(nn.Module):
    def __init__(self,
                 incoming,
                 num_units,
                 ingate=None,
                 forgetgate=None,
                 cell=None,
                 outgate=None,
                 hid_init=0.0,
                 cell_init=0.0,
                 learn_init=True,
                 nonlinearity=torch.tanh,
                 backwards=False,
                 gradient_steps=-1,
                 mask_input=None,
                 only_return_final=False,
                 hid_prop=False):
        
        incoming = incoming if hid_prop else [incoming]
        self.mask_incoming_index = -1
        if mask_input is not None:
            incoming.append(mask_input)
            self.mask_incoming_index = len(incoming)-1
        super().__init__()

        self.nonlinearity = nonlinearity
        self.num_units = num_units
        
        if isinstance(incoming, int):
            self.incoming = incoming
        elif isinstance(incoming, (tuple, list)):
            self.incoming = int(np.prod(incoming[2:])) if len(incoming) > 2 else incoming[-1]
        else:
            raise ValueError("incoming must be int or shape-like (tuple/list)")
        
        self.backwards = backwards
        self.gradient_steps = gradient_steps
        self.only_return_final = only_return_final
        self.hidden_noise = torch.ones(1, dtype=torch.float32)
        self.hidden_clip = torch.ones(1, dtype=torch.float32)
        self.mu_hid = torch.ones(1, dtype=torch.float32)
        self.log_sigma2_hid = torch.ones(1, dtype=torch.float32)
        self.learn_init = learn_init
        self.hid_prop = hid_prop

        if ingate is None:
            self.W_in_to_ingate = nn.Parameter(torch.empty(self.incoming, self.num_units))
            self.W_hid_to_ingate = nn.Parameter(torch.empty(self.num_units, self.num_units))
            self.b_ingate = nn.Parameter(torch.full((self.num_units,), 0.0, dtype=torch.float32))
            nn.init.xavier_uniform_(self.W_in_to_ingate) # GlorotUniform
            nn.init.orthogonal_(self.W_hid_to_ingate, gain=1.1) # Orthogonal
            self.nonlinearity_ingate = utils.hard_sigmoid
        
        if forgetgate is None:
            self.W_in_to_forgetgate = nn.Parameter(torch.empty(self.incoming, self.num_units))
            self.W_hid_to_forgetgate = nn.Parameter(torch.empty(self.num_units, self.num_units))
            self.b_forgetgate = nn.Parameter(torch.full((self.num_units,), 1.0, dtype=torch.float32))
            nn.init.xavier_uniform_(self.W_in_to_forgetgate)
            nn.init.orthogonal_(self.W_hid_to_forgetgate, gain=1.1)
            self.nonlinearity_forgetgate = utils.hard_sigmoid

        if cell is None:
            self.W_in_to_cell = nn.Parameter(torch.empty(self.incoming, self.num_units))
            self.W_hid_to_cell = nn.Parameter(torch.empty(self.num_units, self.num_units))
            self.b_cell = nn.Parameter(torch.full((self.num_units,), 0.0, dtype=torch.float32))
            nn.init.xavier_uniform_(self.W_in_to_cell)
            nn.init.orthogonal_(self.W_hid_to_cell, gain=1.1)
            self.nonlinearity_cell = torch.tanh

        if outgate is None:
            self.W_in_to_outgate = nn.Parameter(torch.empty(self.incoming, self.num_units))
            self.W_hid_to_outgate = nn.Parameter(torch.empty(self.num_units, self.num_units))
            self.b_outgate = nn.Parameter(torch.full((self.num_units,), 0.0, dtype=torch.float32))
            nn.init.xavier_uniform_(self.W_in_to_outgate)
            nn.init.orthogonal_(self.W_hid_to_outgate, gain=1.1)
            self.nonlinearity_outgate = utils.hard_sigmoid

        self.hid_init = nn.Parameter(
            torch.full((1, self.num_units), hid_init, dtype=torch.float32),
            requires_grad=learn_init
        )

        self.cell_init = nn.Parameter(
            torch.full((1, self.num_units), cell_init, dtype=torch.float32),
            requires_grad=learn_init
        )
    
    def input_preactivation(self, input: torch.Tensor, gate_type: str) -> torch.Tensor:
        if gate_type == 'input':
            return input @ self.W_in_to_ingate
        elif gate_type == 'forget':
            return input @ self.W_in_to_forgetgate
        elif gate_type == 'cell':
            return input @ self.W_in_to_cell
        elif gate_type == 'output':
            return input @ self.W_in_to_outgate
        else:
            raise ValueError(f"Unknown gate_type: {gate_type}")
    
    def generate_noise_and_clip(self, num_batch, seq_len):
        return
    
    def forward(self, inputs, deterministic: bool = False, clip: bool = False):
        """
        inputs: либо тензор (batch, seq_len, input_dim) либо список/tuple, где inputs[0] - вход,
                и при self.mask_incoming_index > 0 mask находится в inputs[self.mask_incoming_index].
        Возвращает:
        - если self.only_return_final: (batch, num_units)
        - elif self.hid_prop: (2, batch, seq_len, num_units)
        - else: (batch, seq_len, num_units)
        """
        if isinstance(inputs, (list, tuple)):
            input = inputs[0]
        else:
            input = inputs

        mask = None
        if getattr(self, "mask_incoming_index", -1) > 0:
            if isinstance(inputs, (list, tuple)) and len(inputs) > self.mask_incoming_index:
                mask = inputs[self.mask_incoming_index]
            else:
                mask = None

        input = input.transpose(0, 1)
        seq_len, num_batch, _ = input.shape

        try:
            self.generate_noise_and_clip(num_batch, deterministic, clip)
        except TypeError:
            self.generate_noise_and_clip(num_batch, seq_len)

        input_i = self.input_preactivation(input, 'input', deterministic=deterministic, clip=clip) + self.b_ingate
        input_f = self.input_preactivation(input, 'forget', deterministic=deterministic, clip=clip) + self.b_forgetgate
        input_c = self.input_preactivation(input, 'cell', deterministic=deterministic, clip=clip) + self.b_cell
        input_o = self.input_preactivation(input, 'output', deterministic=deterministic, clip=clip) + self.b_outgate

        if self.hid_prop:
            hid_init, cell_init = inputs[1][0], inputs[1][1]
        else:
            hid_init = self.hid_init.expand(num_batch, -1).to(dtype=input.dtype)
            cell_init = self.cell_init.expand(num_batch, -1).to(dtype=input.dtype)

        hid = hid_init
        cell = cell_init

        hid_seq = []
        cell_seq = []

        if mask is not None:
            if mask.ndim == 3:
                mask_seq = mask.permute(1, 0, 2)
            else:
                mask_seq = mask.unsqueeze(-1).permute(1, 0, 2)
            mask_seq = mask_seq.to(dtype=input.dtype)
        else:
            mask_seq = None

        for t in range(seq_len):
            input_n_i = input_i[t]
            input_n_f = input_f[t]
            input_n_c = input_c[t]
            input_n_o = input_o[t]

            hid_preact_i = self.hidden_preactivation(hid, 'input')
            hid_preact_f = self.hidden_preactivation(hid, 'forget')
            hid_preact_c = self.hidden_preactivation(hid, 'cell')
            hid_preact_o = self.hidden_preactivation(hid, 'output')

            ingate = self.nonlinearity_ingate(input_n_i + hid_preact_i)
            forgetgate = self.nonlinearity_forgetgate(input_n_f + hid_preact_f)
            cell_candidate = self.nonlinearity_cell(input_n_c + hid_preact_c)
            cell = forgetgate * cell + ingate * cell_candidate
            outgate = self.nonlinearity_outgate(input_n_o + hid_preact_o)
            hid = outgate * self.nonlinearity(cell)

            if mask_seq is not None:
                m = mask_seq[t]
                cell = m * cell + (1.0 - m) * cell_init
                hid  = m * hid  + (1.0 - m) * hid_init

            hid_seq.append(hid)
            cell_seq.append(cell)

        cell_out = torch.stack(cell_seq, dim=0)
        hid_out = torch.stack(hid_seq, dim=0)

        if self.only_return_final:
            return hid_out[:, -1, :]

        if self.backwards:
            hid_out = hid_out.flip(dims=[0])
            cell_out = cell_out.flip(dims=[0])

        hid_out = hid_out.transpose(0, 1)
        cell_out = cell_out.transpose(0, 1)

        if self.hid_prop:
            return torch.cat([hid_out.unsqueeze(0), cell_out.unsqueeze(0)], dim=0)
        else:
            return hid_out




class Dense(nn.Module):
    def __init__(self, incoming, num_units, nonlinearity=nn.Identity()):
        super().__init__()
        self.num_units = num_units
        self.nonlinearity = nonlinearity
        
        self.W = nn.Parameter(torch.empty(incoming, num_units))
        self.b = nn.Parameter(torch.zeros(num_units))
        
        nn.init.xavier_uniform_(self.W)

    def pre_activation(self, input):
        return torch.matmul(input, self.W)
    
    def get_output_shape_for(self, input_shape):
        return tuple(input_shape[:-1]) + (self.num_units,)

    def get_output_for(self, input):
        return self.nonlinearity(self.pre_activation(input) + self.b)

    def get_ard(self):
        return {"w": torch.ones_like(self.W)}
    
    def forward(self, input, **kwargs):
        """
        input: tensor with last dim == incoming, arbitrary leading dims allowed
        returns: tensor with same leading dims and last dim == num_units
        """
        out = self.pre_activation(input) + self.b
        out = self.nonlinearity(out)
        return out


# class BayesianDense(Dense):
#     def __init__(self, 
#                  incoming, 
#                  num_units, 
#                  log_sigma_init = -3.0,
#                  thresh=3.0,
#                  W_initializer=None, 
#                  b_init=0.0, 
#                  nonlinearity=lambda x: x):
#         super().__init__(incoming, num_units, nonlinearity)

#         if isinstance(incoming, int):
#             self.num_inputs = int(incoming)
#         else:
#             try:
#                 self.num_inputs = int(incoming[-1])
#             except Exception:
#                 raise ValueError("incoming must be int or shape-like")

#         self.num_units = int(num_units)
#         self.nonlinearity = nonlinearity
#         self.thresh = thresh
#         self.dtype = torch.float32

#         self.W = nn.Parameter(torch.empty(self.num_inputs, self.num_units, dtype=self.dtype))
#         self.b = nn.Parameter(torch.full((self.num_units,), float(b_init), dtype=self.dtype))
#         self.logsig = nn.Parameter(torch.full((self.num_inputs, self.num_units), float(log_sigma_init), dtype=self.dtype))

#         if W_initializer is None:
#             nn.init.xavier_uniform_(self.W)
#         else:
#             W_initializer(self.W)

#     def pre_activation(self, input: torch.Tensor, deterministic: bool = False, clip: bool = False):
#         """
#         input: либо 2D (batch, input_dim) либо 3D (batch, seq_len, input_dim)
#         Возвращает: mu + шум*si (или только mu в deterministic режиме)
#         """
#         W_eff = self.W
#         sigma2 = torch.exp(2.0 * self.logsig)

#         if clip:
#             log_alpha = utils.clip_func(2.0 * self.logsig - utils.safe_torch_log(W_eff.pow(2)))
#             clip_mask = log_alpha.ge(self.thresh)
#             W_eff = torch.where(clip_mask, torch.zeros_like(W_eff), W_eff)
#             sigma2 = torch.where(clip_mask, torch.zeros_like(sigma2), sigma2)

#         if deterministic:
#             return input @ W_eff
        
#         mu = input @ W_eff
#         si = torch.sqrt((input * input) @ sigma2 + 1e-8)

#         if input.ndim == 2:
#             noise = torch.randn_like(mu)
#         else:
#             noise = torch.randn((mu.shape[0], 1, mu.shape[2]))

#         return mu + noise * si

#     def eval_reg(self, train_size: float):
#         """
#         alpha regularization: utils.alpha_regf(clip_func(2*logsig - log(W^2))).sum() / train_size
#         Возвращаем torch scalar
#         """
#         log_alpha = utils.clip_func(2.0 * self.logsig - utils.safe_torch_log(self.W.pow(2)))
#         reg = utils.alpha_regf(log_alpha).sum() / float(train_size)
#         return reg

#     def get_ard(self) -> dict[str, torch.Tensor]:
#         """
#         Возвращаем torch-маску (bool tensor).
#         Маска не требует градиентов и используется для sparsification.
#         """
#         W = self.W.detach()
#         logsig = self.logsig.detach()
#         log_alpha = 2.0 * logsig - 2.0 * utils.safe_torch_log(torch.abs(W))
#         mask = (log_alpha < self.thresh)

#         return {"w": mask}
    
#     def forward(self, input: torch.Tensor, deterministic: bool = False, clip: bool = False, **kwargs):
#         """
#         input: tensor with last dim == num_inputs, can be 2D (batch, in) or 3D (batch, seq_len, in)
#         deterministic, clip: передаются в pre_activation и управляют режимом
#         Возвращает: nonlinearity( pre_activation(input, deterministic, clip) + b )
#         """
#         out = self.pre_activation(input, deterministic=deterministic, clip=clip)
#         out = out + self.b
#         return self.nonlinearity(out)


class BayesianDense(nn.Module):
    """
    Dense layer with Sparse Variational Dropout.
    
    Использует Local Reparameterization Trick:
    Вместо сэмплирования весов W = μ + σ·ε и вычисления y = x·W,
    напрямую сэмплируем активации: y = x·μ + √(x²·σ²)·ε
        
    Параметры:
        μ (self.mu_w): среднее весов, shape (in, out)
        log σ (self.logsig_w): логарифм стд. отклонения, shape (in, out)
        b: bias, shape (out,)
    """
    
    def __init__(self, 
                 incoming, 
                 num_units, 
                 log_sigma_init: float = -3.0,
                 thresh: float = 3.0,
                 W_initializer=None, 
                 b_init: float = 0.0, 
                 nonlinearity=None):
        super().__init__()

        if isinstance(incoming, int):
            self.num_inputs = int(incoming)
        else:
            try:
                self.num_inputs = int(incoming[-1])
            except Exception:
                raise ValueError("incoming must be int or shape-like")

        self.num_units = int(num_units)
        self.nonlinearity = nonlinearity if nonlinearity is not None else (lambda x: x)
        self.thresh = float(thresh)
        self.log_sigma_init = float(log_sigma_init)
        self.dtype = torch.float32
        
        self.mu_w = nn.Parameter(torch.empty(self.num_inputs, self.num_units, dtype=self.dtype))
        self.logsig_w = nn.Parameter(torch.full((self.num_inputs, self.num_units), log_sigma_init, dtype=self.dtype))
        self.b = nn.Parameter(torch.full((self.num_units,), float(b_init), dtype=self.dtype))

        if W_initializer is None:
            nn.init.xavier_uniform_(self.mu_w)
        else:
            W_initializer(self.mu_w)
    
    def _compute_log_alpha(self, logsig, mu):
        """
        log α = log(σ²/μ²) = 2·logsig - 2·log|μ|
        """
        log_alpha = 2.0 * logsig - 2.0 * utils.safe_torch_log(mu)
        return utils.clip_func(log_alpha)

    def forward(self, 
                input: torch.Tensor, 
                deterministic: bool = False, 
                clip: bool = False, 
                **kwargs):
        """
        Forward pass с Local Reparameterization Trick.
        
        Args:
            input: (batch, in) или (batch, seq_len, in)
            deterministic: если True, используем только μ
            clip: если True, обнуляем "мёртвые" веса
            
        Returns:
            output: nonlinearity(pre_activation + bias)
        """
        
        device = input.device
        mu_w = self.mu_w

        if clip:
            log_alpha = self._compute_log_alpha(self.logsig_w, mu_w)
            dead_mask = log_alpha >= self.thresh
            mu_w = torch.where(dead_mask, torch.zeros_like(mu_w), mu_w)
            sigma2 = torch.where(
                dead_mask, 
                torch.zeros_like(self.logsig_w), 
                torch.exp(2.0 * self.logsig_w)
            )
        else:
            sigma2 = torch.exp(2.0 * self.logsig_w)
        
        # Local Reparameterization Trick
        # y = x·μ + √(x²·σ²)·ε
        
        mu_activation = input @ mu_w
        
        if deterministic:
            pre_activation = mu_activation
        else:
            var_activation = (input * input) @ sigma2
            
            std_activation = torch.sqrt(var_activation)
            
            if input.ndim == 2:
                noise = torch.randn_like(mu_activation)
            else:
                noise = torch.randn(
                    (input.shape[0], 1, self.num_units), 
                    device=device, 
                    dtype=self.dtype
                )
            
            # y = μ + σ·ε
            pre_activation = mu_activation + std_activation * noise
        
        output = self.nonlinearity(pre_activation + self.b)
        
        return output

    def eval_reg(self, train_size: int):
        """
        KL divergence regularization для ELBO.
        
        Returns:
            KL / train_size
        """
        
        log_alpha = self._compute_log_alpha(self.logsig_w, self.mu_w)
        kl = utils.alpha_regf(log_alpha).sum()
        
        return kl / float(train_size)

    def get_ard(self) -> dict:
        """
        Получение ARD масок.
        
        Returns:
            dict с маской "w": True = вес активен, False = можно отбросить
        """
        
        with torch.no_grad():
            log_alpha = self._compute_log_alpha(self.logsig_w, self.mu_w)
            # True = активный вес (log_alpha < thresh)
            mask = log_alpha < self.thresh
        
        return {"w": mask}



class BayesianDense_noLRT(Dense):
    def __init__(self, 
                 incoming, 
                 num_units, 
                 log_sigma_init=-3.0,
                 W_initializer=None, 
                 b_init=0.0, 
                 nonlinearity=lambda x: x):
        """
        incoming: int (input size) or shape-like with last dim = input size.
        """
        super().__init__(incoming, num_units, nonlinearity)
        if isinstance(incoming, int):
            self.num_inputs = int(incoming)
        else:
            try:
                self.num_inputs = int(incoming[-1])
            except Exception:
                raise ValueError("incoming must be int or shape-like")

        self.num_units = int(num_units)
        self.nonlinearity = nonlinearity
        self.thresh = 3.0

        self.W = nn.Parameter(torch.empty(self.num_inputs, self.num_units, dtype=torch.float32))
        self.b = nn.Parameter(torch.full((self.num_units,), float(b_init), dtype=torch.float32))
        self.logsig = nn.Parameter(torch.full((self.num_inputs, self.num_units), float(log_sigma_init), dtype=torch.float32))

        if W_initializer is None:
            nn.init.xavier_uniform_(self.W)
        else:
            try:
                W_initializer(self.W)
            except Exception:
                self.W.data.copy_(torch.tensor(W_initializer(self.W.shape), dtype=self.W.dtype))

    def pre_activation(self, input: torch.Tensor, deterministic: bool = False, clip: bool = False):
        """
        input: 2D (batch, in) или 3D (batch, seq_len, in) (и др. формы с последним измерением in)
        """
        sigma2 = torch.exp(2 * self.logsig)
        W_eff = self.W
        
        if clip:
            log_alpha = utils.clip_func(2 * self.logsig - utils.safe_torch_log(self.W.pow(2)))
            clip_mask = log_alpha.ge(self.thresh)
            W_eff = torch.where(clip_mask, torch.zeros_like(W_eff), W_eff)
            sigma2 = torch.where(clip_mask, torch.zeros_like(sigma2), sigma2)

        if deterministic:
            return input @ W_eff
        
        if input.ndim == 2:
            mu = input @ W_eff
            si = torch.sqrt(input.pow(2) @ sigma2 + 1e-8)
            return mu + torch.randn_like(mu) * si
        else:
            W_noisy = W_eff + torch.randn_like(W_eff) * torch.exp(self.logsig)
            return input @ W_noisy

    def eval_reg(self, train_size: float):
        log_alpha = utils.clip_func(2 * self.logsig - utils.safe_torch_log(self.W.pow(2)))
        reg = utils.alpha_regf(log_alpha).sum() / float(train_size)
        return reg
    
    def get_ard(self) -> dict[str, torch.Tensor]:
        W = self.W.detach()
        logsig = self.logsig.detach()
        log_alpha = 2 * logsig - 2 * utils.safe_torch_log(torch.abs(W))
        mask = (log_alpha < self.thresh)
        return {"w": mask}
    
    def forward(self, input: torch.Tensor, deterministic: bool = None, clip: bool = False) -> torch.Tensor:
        if deterministic is None:
            deterministic = not self.training
        out = self.pre_activation(input, deterministic=deterministic, clip=clip)
        out = out + self.b
        return self.nonlinearity(out)
