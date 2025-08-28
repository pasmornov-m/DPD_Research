import torch
from torch import nn
import numpy as np
import os
from modules import utils
from pytorch_tcn import TCN


class GRU(nn.Module):
    def __init__(self, input_size=2, hidden_size=64, num_layers=1, output_size=2, bidirectional=False, batch_first=True, model_name=""):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.bidirectional = bidirectional
        self.batch_first = batch_first
        self.model_name = model_name
        self.gru = nn.GRU(input_size=self.input_size, 
                          hidden_size=self.hidden_size, 
                          num_layers=self.num_layers, 
                          batch_first=self.batch_first, 
                          bidirectional=self.bidirectional)
        self.fc = nn.Linear(hidden_size, output_size)

    @utils.complex_handler
    def forward(self, x, h_0=None):
        if h_0 is None:
            if x.dim() == 2:
                h_0 = torch.zeros(self.num_layers, self.hidden_size)
            else:
                batch_size = x.size(0)
                h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        out, _ = self.gru(x, h_0)
        y = self.fc(out)
        return y
    
    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def save_weights(self, directory="model_params"):
        os.makedirs(directory, exist_ok=True)
        filename = (
            f"{directory}/{self.model_name}_gru_model_"
            f"hs{self.hidden_size}_nl{self.num_layers}_"
            f"in{self.input_size}_out{self.output_size}.pt"
        )
        torch.save(self.state_dict(), filename)
        print(f"Model weights saved to {filename}")

    def load_weights(self, directory="model_params"):
        filename = (
            f"{directory}/{self.model_name}_gru_model_"
            f"hs{self.hidden_size}_nl{self.num_layers}_"
            f"in{self.input_size}_out{self.output_size}.pt")
        if os.path.isfile(filename):
            state_dict = torch.load(filename)
            self.load_state_dict(state_dict)
            print(f"Model weights loaded from {filename}")
            return True
        else:
            print(f"No saved weights found at {filename}, initializing new parameters.")
            return False




class LSTM(nn.Module):
    def __init__(self, input_size=2, hidden_size=64, num_layers=1, output_size=2, bidirectional=False, batch_first=True,
                 bias=False, model_name=""):
        super().__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.batch_first = batch_first
        self.bias = bias
        self.model_name = model_name

        self.lstm = nn.LSTM(input_size=input_size,
                          hidden_size=hidden_size,
                          num_layers=num_layers,
                          bidirectional=self.bidirectional,
                          batch_first=self.batch_first,
                          bias=self.bias)
        self.fc_out = nn.Linear(in_features=self.hidden_size,
                                out_features=self.output_size,
                                bias=self.bias)


    @utils.complex_handler
    def forward(self, x, h_0=None):
        if h_0 is None:
            batch_size = x.size(0)
            h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        out, (_, _) = self.lstm(x, (h_0, h_0))
        y = self.fc_out(out)
        return y

    def count_params(self):
            return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def save_weights(self, directory="model_params"):
        os.makedirs(directory, exist_ok=True)
        filename = (
            f"{directory}/{self.model_name}_lstm_model_"
            f"hs{self.hidden_size}_nl{self.num_layers}_"
            f"in{self.input_size}_out{self.output_size}.pt"
        )
        torch.save(self.state_dict(), filename)
        print(f"Model weights saved to {filename}")

    def load_weights(self, directory="model_params"):
        filename = (
            f"{directory}/{self.model_name}_lstm_model_"
            f"hs{self.hidden_size}_nl{self.num_layers}_"
            f"in{self.input_size}_out{self.output_size}.pt")
        if os.path.isfile(filename):
            state_dict = torch.load(filename)
            self.load_state_dict(state_dict)
            print(f"Model weights loaded from {filename}")
            return True
        else:
            print(f"No saved weights found at {filename}, initializing new parameters.")
            return False




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
        self.model_name = model_name
        self.num_inputs = num_inputs
        self.num_channels = kwargs.get("num_channels")
        self.output_projection = output_projection
        self.kernel_size = kwargs.get("kernel_size")
        self.dropout = dropout
        self.input_shape = input_shape


    @utils.complex_handler
    def forward(self, x, *args, **kwargs):
        out = super().forward(x, *args, **kwargs)
        return out
    
    def count_params(self):
            return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def save_weights(self, directory="model_params"):
        os.makedirs(directory, exist_ok=True)
        filename = (
            f"{directory}/{self.model_name}_tcn_model_"
            f"ch{self.num_channels}_ks{self.kernel_size}_"
            f"in{self.num_inputs}_out{self.output_projection}.pt"
        )
        torch.save(self.state_dict(), filename)
        print(f"Model weights saved to {filename}")

    def load_weights(self, directory="model_params"):
        filename = (
            f"{directory}/{self.model_name}_tcn_model_"
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
        filename = f"{directory}/{self.model_name}_{self.class_name}_" \
                   f"res{self.n_reservoir}_in{self.n_inputs}_out{self.n_outputs}.pt"
        torch.save(self.state_dict(), filename)
        print(f"Model weights saved to {filename}")

    def load_weights(self, directory="model_params"):
        filename = f"{directory}/{self.model_name}_{self.class_name}_" \
                   f"res{self.n_reservoir}_in{self.n_inputs}_out{self.n_outputs}.pt"
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
            return False