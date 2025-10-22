import torch
from torch import nn
from torch.autograd import Variable
import numpy as np
import os
from modules import utils
from pytorch_tcn import TCN


class GRU(nn.Module):
    def __init__(self, 
                 input_size=2, 
                 hidden_size=64, 
                 num_layers=1, 
                 output_size=2, 
                 bidirectional=False, 
                 batch_first=True, 
                 model_name=""):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.bidirectional = bidirectional
        self.batch_first = batch_first
        self.model_name = model_name
        self.class_name = self.__class__.__name__
        
        self.gru = nn.GRU(input_size=self.input_size, 
                          hidden_size=self.hidden_size, 
                          num_layers=self.num_layers, 
                          batch_first=self.batch_first, 
                          bidirectional=self.bidirectional)
        
        self.fc = nn.Linear(self.hidden_size, self.output_size)

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
            f"{directory}/{self.model_name}_{self.class_name}_"
            f"hs{self.hidden_size}_nl{self.num_layers}_"
            f"in{self.input_size}_out{self.output_size}.pt"
        )
        torch.save(self.state_dict(), filename)
        print(f"Model weights saved to {filename}")

    def load_weights(self, directory="model_params"):
        filename = (
            f"{directory}/{self.model_name}_{self.class_name}_"
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
        self.class_name = self.__class__.__name__

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
            f"{directory}/{self.model_name}_{self.class_name}_"
            f"hs{self.hidden_size}_nl{self.num_layers}_"
            f"in{self.input_size}_out{self.output_size}.pt"
        )
        torch.save(self.state_dict(), filename)
        print(f"Model weights saved to {filename}")

    def load_weights(self, directory="model_params"):
        filename = (
            f"{directory}/{self.model_name}_{self.class_name}_"
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
        
        incomings = incoming if hid_prop else [incoming]
        self.mask_incoming_index = -1
        if mask_input is not None:
            incomings.append(mask_input)
            self.mask_incoming_index = len(incomings)-1
        super().__init__()

        self.nonlinearity = nonlinearity
        self.num_units = num_units
        
        if isinstance(incoming, int):
            self.num_inputs = incoming
        elif isinstance(incoming, (tuple, list)):
            self.num_inputs = int(np.prod(incoming[2:])) if len(incoming) > 2 else incoming[-1]
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
            self.W_in_to_ingate = nn.Parameter(torch.empty(self.num_inputs, self.num_units))
            self.W_hid_to_ingate = nn.Parameter(torch.empty(self.num_units, self.num_units))
            self.b_ingate = nn.Parameter(torch.full((self.num_units,), 0.0, dtype=torch.float32))
            nn.init.xavier_uniform_(self.W_in_to_ingate) # GlorotUniform
            nn.init.orthogonal_(self.W_hid_to_ingate, gain=1.1) # Orthogonal
            self.nonlinearity_ingate = utils.hard_sigmoid
        
        if forgetgate is None:
            self.W_in_to_forgetgate = nn.Parameter(torch.empty(self.num_inputs, self.num_units))
            self.W_hid_to_forgetgate = nn.Parameter(torch.empty(self.num_units, self.num_units))
            self.b_forgetgate = nn.Parameter(torch.full((self.num_units,), 1.0, dtype=torch.float32))
            nn.init.xavier_uniform_(self.W_in_to_forgetgate)
            nn.init.orthogonal_(self.W_hid_to_forgetgate, gain=1.1)
            self.nonlinearity_forgetgate = utils.hard_sigmoid

        if cell is None:
            self.W_in_to_cell = nn.Parameter(torch.empty(self.num_inputs, self.num_units))
            self.W_hid_to_cell = nn.Parameter(torch.empty(self.num_units, self.num_units))
            self.b_cell = nn.Parameter(torch.full((self.num_units,), 0.0, dtype=torch.float32))
            nn.init.xavier_uniform_(self.W_in_to_cell)
            nn.init.orthogonal_(self.W_hid_to_cell, gain=1.1)
            self.nonlinearity_cell = torch.tanh

        if outgate is None:
            self.W_in_to_outgate = nn.Parameter(torch.empty(self.num_inputs, self.num_units))
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


class BayesianLSTM(CustomLSTM):
    """
    config: L probabilistic weight with lognormal prior, N probabilistic weight with standart normal prior, 
           D deterministic learnable weight, 
           C constant weight (1 for multiplicative and 0 for additive weights)\
           
           config[0]: W input_to_hidden, W hidden_to_hidden
                     L N D (C is not supported)
           config[1]: hat Z preactivation multiplicative weights
                     L N D C
           config[2]: Z input and hidden multiplicative weights
                     L N D C I R
    """
    def __init__(self, 
                 incoming, 
                 num_units,
                 log_sigma_in_init = -3.0, 
                 log_sigma_hid_init = -3.0,
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
                 thresh=3.0,
                 mask_input=None,
                 only_return_final=False,
                 hid_prop = False,
                 config="DCC"):
 
        super().__init__(incoming, 
                         num_units, 
                         ingate, 
                         forgetgate, 
                         cell, 
                         outgate,
                         hid_init, 
                         cell_init, 
                         learn_init, 
                         nonlinearity, 
                         backwards, 
                         gradient_steps, 
                         mask_input,
                         only_return_final,
                         hid_prop)
        
        self.reg = True
        self.config = config
        self.log_sigma_in_init = log_sigma_in_init
        self.log_sigma_hid_init = log_sigma_hid_init
        self.dtype = torch.float32

        if self.config[0] in {"L", "N"}:
            self.logsig_w_in = nn.Parameter(torch.full((4, incoming, num_units), log_sigma_in_init))
            self.logsig_w_hid = nn.Parameter(torch.full((4, num_units, num_units), log_sigma_hid_init))
        else:
            self.register_buffer("logsig_w_in", torch.zeros(4))
            self.register_buffer("logsig_w_hid", torch.zeros(4))
        
        if self.config[2] in {"L", "N", "D", "I"}:
            self.mu_in = nn.Parameter(torch.ones(incoming))
        if self.config[2] in {"L", "N", "I"}:
            self.logsig_in = nn.Parameter(torch.full((incoming,), log_sigma_in_init))
        if self.config[2] in {"L", "N", "D", "R"}:
            self.mu_hid = nn.Parameter(torch.ones(num_units))
        if self.config[2] in {"L", "N", "R"}:
            self.logsig_hid = nn.Parameter(torch.full((num_units,), log_sigma_hid_init))

        if self.config[1] in {"L", "N", "D"}:
            self.mu_gates = nn.Parameter(torch.ones(4, num_units))
        if self.config[1] in {"L", "N"}:
            self.logsig_gates = nn.Parameter(torch.full((4, num_units), log_sigma_hid_init))

        self.input_noise = None
        self.hidden_noise = None
        self.input_clip = None
        self.hidden_clip = None

        self.thresh = thresh
    
    def generate_noise_and_clip(self, num_batch, deterministic=False, clip=False):
        if not deterministic:
            if self.config[0] in {"L", "N"}:
                self.input_w_noise = torch.randn(4, self.num_inputs, self.num_units) * torch.exp(self.logsig_w_in)
                self.hidden_w_noise = torch.randn(4, self.num_units, self.num_units) * torch.exp(self.logsig_w_hid)
            else:
                self.input_w_noise = torch.zeros(4)
                self.hidden_w_noise = torch.zeros(4)

            if self.config[2] in {"L", "N"}:
                self.input_noise = torch.randn(num_batch, self.num_inputs) * torch.exp(self.logsig_in) + self.mu_in
                self.hidden_noise = torch.randn(num_batch, self.num_units) * torch.exp(self.logsig_hid) + self.mu_hid
            elif self.config[2] == "I":
                self.input_noise = torch.randn(num_batch, self.num_inputs) * torch.exp(self.logsig_in) + self.mu_in
                self.hidden_noise = torch.ones(1)
            elif self.config[2] == "R":
                self.input_noise = torch.ones(1)
                self.hidden_noise = torch.randn(num_batch, self.num_units) * torch.exp(self.logsig_hid) + self.mu_hid
            elif self.config[2] == "D":
                self.input_noise = self.mu_in
                self.hidden_noise = self.mu_hid
            else:
                self.input_noise = torch.ones(1)
                self.hidden_noise = torch.ones(1)
        
            if self.config[1] in {"L", "N"}:
                self.gates_noise = torch.randn(4, num_batch, self.num_units) * torch.exp(self.logsig_gates)[:, None, :] + self.mu_gates[:, None, :]
            elif self.config[1] == "D":
                self.gates_noise = self.mu_gates
            else:
                self.gates_noise = torch.ones(4)
        
        else:
            self.input_w_noise = torch.zeros(4)
            self.hidden_w_noise = torch.zeros(4)

            if self.config[2] in {"L", "N", "D"}:
                self.input_noise = self.mu_in
                self.hidden_noise = self.mu_hid
            elif self.config[2] == "I":
                self.input_noise = self.mu_in
                self.hidden_noise = torch.ones(1)
            elif self.config[2] == "R":
                self.input_noise = torch.ones(1)
                self.hidden_noise = self.mu_hid
            else:
                self.input_noise = torch.ones(1)
                self.hidden_noise = torch.ones(1)

            if self.config[1] in {"L", "N", "D"}:
                self.gates_noise = self.mu_gates
            else:
                self.gates_noise = torch.ones(4, dtype=self.dtype)
        
        if clip:
            if self.config[0] == "L":
                W_in_cat = torch.cat([self.W_in_to_ingate[None,:,:],
                                    self.W_in_to_forgetgate[None,:,:],
                                    self.W_in_to_cell[None,:,:],
                                    self.W_in_to_outgate[None,:,:]], dim=0)
                log_alpha_w_in = utils.clip_func(2 * self.logsig_w_in - torch.log(W_in_cat**2 + self.epsilon))
                self.input_w_clip = log_alpha_w_in <= self.thresh

                W_hid_cat = torch.cat([self.W_hid_to_ingate[None,:,:],
                                    self.W_hid_to_forgetgate[None,:,:],
                                    self.W_hid_to_cell[None,:,:],
                                    self.W_hid_to_outgate[None,:,:]], dim=0)
                log_alpha_w_hid = utils.clip_func(2 * self.logsig_w_hid - utils.safe_torch_log(W_hid_cat**2))
                self.hidden_w_clip = log_alpha_w_hid <= self.thresh
            else:
                self.input_w_clip = torch.ones(4)
                self.hidden_w_clip = torch.ones(4)

            if self.config[2] == "L":
                log_alpha_in = utils.clip_func(2 * self.logsig_in - utils.safe_torch_log(self.mu_in**2))
                self.input_clip = log_alpha_in <= self.thresh
                log_alpha_hid = utils.clip_func(2 * self.logsig_hid - utils.safe_torch_log(self.mu_hid**2))
                self.hidden_clip = log_alpha_hid <= self.thresh
            elif self.config[2] == "I":
                log_alpha_in = utils.clip_func(2 * self.logsig_in - utils.safe_torch_log(self.mu_in**2))
                self.input_clip = log_alpha_in <= self.thresh
                self.hidden_clip = torch.ones(1)
            elif self.config[2] == "R":
                self.input_clip = torch.ones(1)
                log_alpha_hid = utils.clip_func(2 * self.logsig_hid - utils.safe_torch_log(self.mu_hid**2))
                self.hidden_clip = log_alpha_hid <= self.thresh
            else:
                self.input_clip = torch.ones(1)
                self.hidden_clip = torch.ones(1)

            if self.config[1] == "L":
                log_alpha_gates = utils.clip_func(2 * self.logsig_gates - utils.safe_torch_log(self.mu_gates**2))
                self.gates_clip = log_alpha_gates <= self.thresh
            else:
                self.gates_clip = torch.ones(4)
        
        else:
            self.input_w_clip = torch.ones(4)
            self.hidden_w_clip = torch.ones(4)
            self.input_clip = torch.ones(1)
            self.hidden_clip = torch.ones(1)
            self.gates_clip = torch.ones(4)
        
        self.W_hid = torch.cat([
            self.W_hid_to_ingate + self.hidden_w_noise[0],
            self.W_hid_to_forgetgate + self.hidden_w_noise[1],
            self.W_hid_to_cell + self.hidden_w_noise[2],
            self.W_hid_to_outgate + self.hidden_w_noise[3]
        ], dim=1)

        self.W_in = torch.cat([
            self.W_in_to_ingate + self.input_w_noise[0],
            self.W_in_to_forgetgate + self.input_w_noise[1],
            self.W_in_to_cell + self.input_w_noise[2],
            self.W_in_to_outgate + self.input_w_noise[3],
        ], dim=1)

        return
    
    def eval_reg(self, train_size):
        W_in = torch.cat([
            self.W_in_to_ingate.unsqueeze(0),
            self.W_in_to_forgetgate.unsqueeze(0),
            self.W_in_to_cell.unsqueeze(0),
            self.W_in_to_outgate.unsqueeze(0)
        ], dim=0)

        if self.config[0] == "N":
            KL_element_in = -self.logsig_w_in + 0.5 * (torch.exp(2 * self.logsig_w_in) + W_in**2) - 0.5
            KL = KL_element_in.sum()
        elif self.config[0] == "L":
            log_alpha_w_in = utils.clip_func(2 * self.logsig_w_in - utils.safe_torch_log(W_in**2))
            KL = utils.alpha_regf(log_alpha_w_in).sum()
        else:
            KL = torch.zeros(1, dtype=self.dtype).sum()

        W_hid = torch.cat([
            self.W_hid_to_ingate.unsqueeze(0),
            self.W_hid_to_forgetgate.unsqueeze(0),
            self.W_hid_to_cell.unsqueeze(0),
            self.W_hid_to_outgate.unsqueeze(0)
        ], dim=0)

        if self.config[0] == "N":
            KL_element_hid = -self.logsig_w_hid + 0.5 * (torch.exp(2 * self.logsig_w_hid) + W_hid**2) - 0.5
            KL += KL_element_hid.sum()
        elif self.config[0] == "L":
            log_alpha_w_hid = utils.clip_func(2 * self.logsig_w_hid - utils.safe_torch_log(W_hid**2))
            KL += utils.alpha_regf(log_alpha_w_hid).sum()

        if self.config[2] in {"L", "R", "I"}:
            if self.config[2] in {"L", "R"}:
                log_alpha_hid = utils.clip_func(2 * self.logsig_hid - utils.safe_torch_log(self.mu_hid**2))
                KL += utils.alpha_regf(log_alpha_hid).sum()
            if self.config[2] in {"L", "I"}:
                log_alpha_in = utils.clip_func(2 * self.logsig_in - utils.safe_torch_log(self.mu_in**2))
                KL += utils.alpha_regf(log_alpha_in).sum()
        elif self.config[2] == "N":
            KL += (-self.logsig_hid + 0.5 * (torch.exp(2 * self.logsig_hid) + self.mu_hid**2) - 0.5).sum()
            KL += (-self.logsig_in + 0.5 * (torch.exp(2 * self.logsig_in) + self.mu_in**2) - 0.5).sum()

        if self.config[1] == "L":
            log_alpha_gates = utils.clip_func(2 * self.logsig_gates - utils.safe_torch_log(self.mu_gates**2))
            KL += utils.alpha_regf(log_alpha_gates).sum()
        elif self.config[1] == "N":
            KL_element = -self.logsig_gates + 0.5 * (torch.exp(2 * self.logsig_gates) + self.mu_gates**2) - 0.5
            KL += KL_element.sum()
        
        reg = KL / train_size
        return reg
    
    def get_ard(self):
        if self.config[0] == "L":
            W_in = torch.cat([
                self.W_in_to_ingate.unsqueeze(0),
                self.W_in_to_forgetgate.unsqueeze(0),
                self.W_in_to_cell.unsqueeze(0),
                self.W_in_to_outgate.unsqueeze(0)
            ], dim=0)
            log_alpha_w_in = 2 * self.logsig_w_in - 2 * utils.safe_torch_log(torch.abs(W_in))
            mask_w_in = log_alpha_w_in < self.thresh

            W_hid = torch.cat([
                self.W_hid_to_ingate.unsqueeze(0),
                self.W_hid_to_forgetgate.unsqueeze(0),
                self.W_hid_to_cell.unsqueeze(0),
                self.W_hid_to_outgate.unsqueeze(0)
            ], dim=0)
            log_alpha_w_hid = 2 * self.logsig_w_hid - 2 * utils.safe_torch_log(torch.abs(W_hid))
            mask_w_hid = log_alpha_w_hid < self.thresh
        else:
            mask_w_in = torch.ones((4,) + self.W_in_to_ingate.shape, dtype=torch.bool)
            mask_w_hid = torch.ones((4,) + self.W_hid_to_ingate.shape, dtype=torch.bool)

        mask_in = mask_w_in.any(dim=2).any(dim=0)
        mask_hid_by_w = mask_w_hid.any(dim=2).any(dim=0)
        mask_hid_by_z = torch.ones_like(mask_hid_by_w, dtype=torch.bool)
        
        def log_alpha_calc(logsig, mu):
            return 2 * logsig - 2 * utils.safe_torch_log(torch.abs(mu))

        if self.config[2] == "L":
            log_alpha_hid = log_alpha_calc(self.logsig_hid, self.mu_hid)
            log_alpha_in = log_alpha_calc(self.logsig_in, self.mu_in)
            mask_in = torch.logical_and(log_alpha_in < self.thresh, mask_in)
            mask_hid_by_z = log_alpha_hid < self.thresh
        elif self.config[2] == "I":
            log_alpha_in = log_alpha_calc(self.logsig_in, self.mu_in)
            mask_in = torch.logical_and(log_alpha_in < self.thresh, mask_in)
        elif self.config[2] == "R":
            log_alpha_hid = log_alpha_calc(self.logsig_hid, self.mu_hid)
            mask_hid_by_z = log_alpha_hid < self.thresh

        # --- gates ---
        mask = torch.cat([mask_w_in, mask_w_hid], dim=1)
        if self.config[1] == "L":
            log_alpha_gates = log_alpha_calc(self.logsig_gates, self.mu_gates)
            mask_gates = torch.logical_and(log_alpha_gates < self.thresh, mask.any(dim=1))
        else:
            mask_gates = mask.any(dim=1)

        return {
            "w_input": mask_w_in,
            "w_hidden": mask_w_hid,
            "gates": mask_gates,
            "z_input": mask_in,
            "z_hidden_by_w": mask_hid_by_w,
            "z_hidden": mask_hid_by_z,
        }
        
    def prepare_gates_noise(self, num_batch):
        dtype = self.dtype
        num_units = self.num_units

        if self.gates_noise is None:
            gates_noise = torch.ones((num_batch, 4, num_units), dtype=dtype)
        else:
            g = self.gates_noise
            if g.ndim == 1:
                gates_noise = g.unsqueeze(0).unsqueeze(-1)
                gates_noise = gates_noise.expand(num_batch, -1, num_units)
            elif g.ndim == 2:
                gates_noise = g.unsqueeze(0).expand(num_batch, -1, -1)
            elif g.ndim == 3:
                gates_noise = g.permute(1, 0, 2).to(dtype=dtype)
            else:
                raise ValueError(f"Unexpected shape for gates_noise: {g.shape}")

        if self.gates_clip is None:
            gates_clip = torch.ones((1, 4, num_units), dtype=dtype)
        else:
            gc = self.gates_clip
            if gc.ndim == 0:
                gates_clip = gc.expand(1, 4, num_units)
            elif gc.ndim == 1:
                gates_clip = gc.unsqueeze(0).unsqueeze(-1).expand(1, -1, num_units)
            elif gc.ndim == 2:
                gates_clip = gc.unsqueeze(0)
            else:
                raise ValueError(f"Unexpected shape for gates_clip: {gc.shape}")

        gates_noise_clipped = gates_noise * gates_clip

        return (gates_noise_clipped[:, 0, :], 
                gates_noise_clipped[:, 1, :], 
                gates_noise_clipped[:, 2, :], 
                gates_noise_clipped[:, 3, :])

    def forward(self, inputs, hid_init=None, deterministic: bool = False, clip: bool = False):
        """
        inputs: либо тензор (batch, seq_len, input_dim) либо список/tuple:
                inputs[0] - input tensor,
                при self.mask_incoming_index > 0 mask ожидается в inputs[self.mask_incoming_index],
                при self.hid_prop ожидается inputs[1] с hid_init и cell_init.
        Возвращает:
        - если self.only_return_final: (batch, num_units)
        - elif self.hid_prop: (2, batch, seq_len, num_units)
        - else: (batch, seq_len, num_units)
        """

        if isinstance(inputs, (list, tuple)):
            x = inputs[0]
        else:
            x = inputs

        mask = None
        if getattr(self, "mask_incoming_index", -1) > 0 and isinstance(inputs, (list, tuple)):
            if len(inputs) > self.mask_incoming_index:
                mask = inputs[self.mask_incoming_index]
        
        if mask is not None:
            if mask.ndim == 3:
                mask_seq = mask
            else:
                mask_seq = mask.unsqueeze(-1)
        else:
            mask_seq = None

        num_batch, seq_len, _ = x.shape
        hid_out = torch.empty((num_batch, seq_len, self.num_units), dtype=self.dtype)
        cell_out = torch.empty_like(hid_out)
 
        self.generate_noise_and_clip(num_batch, deterministic, clip)
        hidden_w_clip = self.hidden_w_clip.repeat_interleave(self.num_units)

        g0_gc0, g1_gc1, g2_gc2, g3_gc3 = self.prepare_gates_noise(num_batch)
        
        x_eff = x * self.input_noise * self.input_clip
        input_preact = torch.matmul(x_eff, self.W_in)
        input_i, input_f, input_c, input_o = torch.chunk(input_preact, 4, dim=-1)

        input_i = input_i * self.input_w_clip[0] + self.b_ingate
        input_f = input_f * self.input_w_clip[1] + self.b_forgetgate
        input_c = input_c * self.input_w_clip[2] + self.b_cell
        input_o = input_o * self.input_w_clip[3] + self.b_outgate
        
        if self.hid_prop and hid_init is not None:
            hid = hid_init[0].to(dtype=self.dtype)
            cell = hid_init[1].to(dtype=self.dtype)
        else:
            hid = torch.zeros(num_batch, self.num_units, dtype=self.dtype)
            cell = torch.zeros(num_batch, self.num_units, dtype=self.dtype)
                
        hn = self.hidden_noise if self.hidden_noise is not None else torch.ones(1, dtype=self.dtype)
        hc = self.hidden_clip if self.hidden_clip is not None else torch.ones(1, dtype=self.dtype)
        hn_hc = hn * hc
        
        t_range = range(seq_len - 1, -1, -1) if self.backwards else range(seq_len)

        for t in t_range:
            cell_prev = cell
            hid_prev = hid
            
            input_n_i = input_i[:, t, :]
            input_n_f = input_f[:, t, :]
            input_n_c = input_c[:, t, :]
            input_n_o = input_o[:, t, :]
            
            hid_preact = hid @ self.W_hid
            
            hid_preact = hid_preact * hidden_w_clip
            hid_preact_i, hid_preact_f, hid_preact_c, hid_preact_o = torch.chunk(hid_preact, 4, dim=1)

            ingate = self.nonlinearity_ingate((input_n_i + hid_preact_i) * g0_gc0 + self.b_ingate)
            forgetgate = self.nonlinearity_forgetgate((input_n_f + hid_preact_f) * g1_gc1 + self.b_forgetgate)
            cell_candidate = self.nonlinearity_cell((input_n_c + hid_preact_c) * g2_gc2 + self.b_cell)
            
            cell_new = forgetgate * cell_prev + ingate * cell_candidate
            outgate = self.nonlinearity_outgate((input_n_o + hid_preact_o) * g3_gc3 + self.b_outgate)
            hid_new = outgate * self.nonlinearity(cell_new)

            hid_new = hid_new * hn_hc

            if mask_seq is not None:
                m = mask_seq[:, t, :]
                if m.ndim == 2 and m.shape[1] == 1:
                    m = m.expand(-1, self.num_units)
                cell = m * cell_new + (1.0 - m) * cell_prev
                hid = m * hid_new + (1.0 - m) * hid_prev
            else:
                cell = cell_new
                hid = hid_new

            hid_out[:, t, :] = hid
            cell_out[:, t, :] = cell

        if self.only_return_final:
            return hid_out[:, -1, :]

        if self.backwards:
            hid_out = hid_out.flip(dims=[1])
            cell_out = cell_out.flip(dims=[1])

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
    
    def forward(self, input):
        """
        input: tensor with last dim == incoming, arbitrary leading dims allowed
        returns: tensor with same leading dims and last dim == num_units
        """
        lin = self.pre_activation(input)
        lin = lin + self.b
        return self.nonlinearity(lin)

class BayesianDense(Dense):
    def __init__(self, 
                 incoming, 
                 num_units, 
                 log_sigma_init = -3.0,
                 thresh=3.0,
                 W_initializer=None, 
                 b_init=0.0, 
                 nonlinearity=lambda x: x):
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
        self.thresh = thresh
        self.dtype = torch.float32

        self.W = nn.Parameter(torch.empty(self.num_inputs, self.num_units, dtype=self.dtype))
        self.b = nn.Parameter(torch.full((self.num_units,), float(b_init), dtype=self.dtype))
        self.log_sigma = nn.Parameter(torch.full((self.num_inputs, self.num_units), float(log_sigma_init), dtype=self.dtype))

        if W_initializer is None:
            nn.init.xavier_uniform_(self.W)
        else:
            W_initializer(self.W)

    def pre_activation(self, input: torch.Tensor, deterministic: bool = False, clip: bool = False):
        """
        input: либо 2D (batch, input_dim) либо 3D (batch, seq_len, input_dim)
        Возвращает: mu + шум*si (или только mu в deterministic режиме)
        """
        W_eff = self.W
        sigma2 = torch.exp(2.0 * self.log_sigma)

        if clip:
            log_alpha = utils.clip_func(2.0 * self.log_sigma - utils.safe_torch_log(W_eff.pow(2)))
            clip_mask = log_alpha.ge(self.thresh)
            W_eff = torch.where(clip_mask, torch.zeros_like(W_eff), W_eff)
            sigma2 = torch.where(clip_mask, torch.zeros_like(sigma2), sigma2)

        if deterministic:
            return input @ W_eff
        
        mu = input @ W_eff
        si = torch.sqrt((input * input) @ sigma2 + 1e-8)

        if input.ndim == 2:
            noise = torch.randn_like(mu)
        else:
            noise = torch.randn((mu.shape[0], 1, mu.shape[2]))

        return mu + noise * si
        

    def eval_reg(self, train_size: float):
        """
        alpha regularization: utils.alpha_regf(clip_func(2*log_sigma - log(W^2))).sum() / train_size
        Возвращаем torch scalar
        """
        log_alpha = utils.clip_func(2.0 * self.log_sigma - utils.safe_torch_log(self.W.pow(2)))
        reg = utils.alpha_regf(log_alpha).sum() / float(train_size)
        return reg

    def get_ard(self) -> dict[str, torch.Tensor]:
        """
        Возвращаем torch-маску (bool tensor).
        Маска не требует градиентов и используется для sparsification.
        """
        W = self.W.detach()
        log_sigma = self.log_sigma.detach()
        log_alpha = 2.0 * log_sigma - 2.0 * utils.safe_torch_log(torch.abs(W))
        mask = (log_alpha < self.thresh)

        return {"w": mask}
    
    def forward(self, input: torch.Tensor, deterministic: bool = False, clip: bool = False, **kwargs):
        """
        input: tensor with last dim == num_inputs, can be 2D (batch, in) or 3D (batch, seq_len, in)
        deterministic, clip: передаются в pre_activation и управляют режимом
        Возвращает: nonlinearity( pre_activation(input, deterministic, clip) + b )
        """
        out = self.pre_activation(input, deterministic=deterministic, clip=clip)
        out = out + self.b
        return self.nonlinearity(out)


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
        self.log_sigma = nn.Parameter(torch.full((self.num_inputs, self.num_units), float(log_sigma_init), dtype=torch.float32))

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
        sigma2 = torch.exp(2 * self.log_sigma)
        W_eff = self.W
        
        if clip:
            log_alpha = utils.clip_func(2 * self.log_sigma - utils.safe_torch_log(self.W.pow(2)))
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
            W_noisy = W_eff + torch.randn_like(W_eff) * torch.exp(self.log_sigma)
            return input @ W_noisy

    def eval_reg(self, train_size: float):
        log_alpha = utils.clip_func(2 * self.log_sigma - utils.safe_torch_log(self.W.pow(2)))
        reg = utils.alpha_regf(log_alpha).sum() / float(train_size)
        return reg
    
    def get_ard(self) -> dict[str, torch.Tensor]:
        W = self.W.detach()
        log_sigma = self.log_sigma.detach()
        log_alpha = 2 * log_sigma - 2 * utils.safe_torch_log(torch.abs(W))
        mask = (log_alpha < self.thresh)
        return {"w": mask}
    
    def forward(self, input: torch.Tensor, deterministic: bool = None, clip: bool = False) -> torch.Tensor:
        if deterministic is None:
            deterministic = not self.training
        out = self.pre_activation(input, deterministic=deterministic, clip=clip)
        out = out + self.b
        return self.nonlinearity(out)
