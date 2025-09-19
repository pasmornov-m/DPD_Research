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




class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias

        self.xh = nn.Linear(input_size, hidden_size * 4, bias=bias)
        self.hh = nn.Linear(hidden_size, hidden_size * 4, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / np.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, input, hx=None):

        # Inputs:
        #       input: of shape (batch_size, input_size)
        #       hx: of shape (batch_size, hidden_size)
        # Outputs:
        #       hy: of shape (batch_size, hidden_size)
        #       cy: of shape (batch_size, hidden_size)

        if hx is None:
            hx = Variable(input.new_zeros(input.size(0), self.hidden_size))
            hx = (hx, hx)

        hx, cx = hx

        gates = self.xh(input) + self.hh(hx)

        # Get gates (i_t, f_t, g_t, o_t)
        input_gate, forget_gate, cell_gate, output_gate = gates.chunk(4, 1)

        i_t = torch.sigmoid(input_gate)
        f_t = torch.sigmoid(forget_gate)
        g_t = torch.tanh(cell_gate)
        o_t = torch.sigmoid(output_gate)

        cy = cx * f_t + i_t * g_t

        hy = o_t * torch.tanh(cy)


        return (hy, cy)


class CustomLSTM(nn.Module):
    def __init__(self, input_size=2, hidden_size=64, num_layers=1, output_size=2,
                 bias=False, batch_first=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first

        # Строим стек LSTMCell
        self.rnn_cell_list = nn.ModuleList()
        self.rnn_cell_list.append(LSTMCell(input_size, hidden_size, bias))
        for l in range(1, num_layers):
            self.rnn_cell_list.append(LSTMCell(hidden_size, hidden_size, bias))

        # Выходной слой
        self.fc_out = nn.Linear(hidden_size, output_size, bias=bias)

    @utils.complex_handler
    def forward(self, x, hx=None):
        """
        x:  (batch, seq_len, input_size)
        hx: (h0, c0), где h0 и c0 имеют форму (num_layers, batch, hidden_size)
        """
        batch_size, seq_len, _ = x.size()
        device = x.device

        if hx is None:
            h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
            c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        else:
            h0, c0 = hx

        hidden = [(h0[layer], c0[layer]) for layer in range(self.num_layers)]

        outputs = []
        for t in range(seq_len):
            x_t = x[:, t, :]
            for layer in range(self.num_layers):
                h, c = hidden[layer]
                if layer == 0:
                    hidden[layer] = self.rnn_cell_list[layer](x_t, (h, c))
                else:
                    hidden[layer] = self.rnn_cell_list[layer](hidden[layer - 1][0], (h, c))
            outputs.append(hidden[-1][0].unsqueeze(1))

        outputs = torch.cat(outputs, dim=1)  # (batch, seq_len, hidden_size)

        y = self.fc_out(outputs)  # (batch, seq_len, output_size)

        return y



# --- вспомогательная alpha-regularizer (приближение / placeholder)
def alpha_regf(log_alpha):
    # Приближение для KL в варианте "logalpha" (Molchanov et al. 2017 использует аппрокс.)
    # Это placeholder; при желании замените на точную функцию из вашей реализации.
    # Здесь даём гладкую положительную функцию, растущую с log_alpha.
    return 0.5 * torch.log1p(torch.exp(log_alpha))  # простая гладкая аппроксимация

# def alpha_regf(alpha: torch.Tensor) -> torch.Tensor:
#     """
#     Аппроксимация KL регуляризатора для вариационного dropout (Molchanov et al., 2017).
    
#     alpha: torch.Tensor, отношение σ^2 / θ^2 (log alpha или просто alpha)
#     возвращает: torch.Tensor с теми же размерами, представляющий регуляризацию
#     """
#     # Чтобы избежать log(0)
#     eps = 1e-8
#     alpha = torch.clamp(alpha, min=eps)
    
#     term1 = 0.64 * torch.sigmoid(1.87 + 1.49 * torch.log(alpha))
#     term2 = 0.5 * torch.log1p(1.0 / alpha)
    
#     return term1 - term2

# --- Предполагаем, что есть CustomLSTM (или LSTM) из предыдущих сообщений.
# --- BayesianLSTM наследует CustomLSTM и переопределяет части поведения.
class BayesianLSTM(CustomLSTM):   # CustomLSTM — предыдущая реализация LSTM
    def __init__(self, input_size, hidden_size, num_layers, output_size,
                 bias=True, config="LLL",
                 log_sigma_in_init=-3.0, log_sigma_hid_init=-3.0,
                 thresh=3.0,
                 **kwargs):
        """
        config: строка длины >=3, описана в комментариях в исходном коде Theano.
        config[0] — поведение весов W (L,N,D,...)
        config[1] — поведение preactivation multiplicative weights (gates)
        config[2] — поведение Z (input/hidden multiplicative weights)
        """
        super().__init__(input_size=input_size, hidden_size=hidden_size,
                         num_layers=num_layers, output_size=output_size, **kwargs)

        self.config = config
        self.log_sigma_in_init = log_sigma_in_init
        self.log_sigma_hid_init = log_sigma_hid_init
        self.thresh = thresh
        
        # Веса input → gates
        self.W_in_to_ingate = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.W_in_to_forgetgate = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.W_in_to_cell = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.W_in_to_outgate = nn.Parameter(torch.Tensor(input_size, hidden_size))

        # Веса hidden → gates
        self.W_hid_to_ingate = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.W_hid_to_forgetgate = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.W_hid_to_cell = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.W_hid_to_outgate = nn.Parameter(torch.Tensor(hidden_size, hidden_size))

        # Смещения
        self.b_ingate = nn.Parameter(torch.Tensor(hidden_size))
        self.b_forgetgate = nn.Parameter(torch.Tensor(hidden_size))
        self.b_cell = nn.Parameter(torch.Tensor(hidden_size))
        self.b_outgate = nn.Parameter(torch.Tensor(hidden_size))

        self.reset_parameters()
        
        self.fc_out = nn.Linear(in_features=hidden_size,
                                out_features=output_size,
                                bias=bias)

        # --- Параметры лог-сигм для весов (W_in, W_hid) если выбран L или N
        if self.config[0] in {"L", "N"}:
            self.logsig_w_in = nn.Parameter(torch.full(
                (4, self.input_size, self.hidden_size), log_sigma_in_init, dtype=torch.float32))
            self.logsig_w_hid = nn.Parameter(torch.full(
                (4, self.hidden_size, self.hidden_size), log_sigma_hid_init, dtype=torch.float32))
        else:
            # буферы нулей — чтобы код не ломался
            self.register_buffer('logsig_w_in', torch.zeros(4, self.input_size, self.hidden_size))
            self.register_buffer('logsig_w_hid', torch.zeros(4, self.hidden_size, self.hidden_size))

        # --- Z (input/hidden neuron multiplicative params)
        if self.config[2] in {"L", "N", "D", "I"}:
            self.mu_in = nn.Parameter(torch.ones(self.input_size))
        else:
            self.register_buffer('mu_in', torch.ones(self.input_size))

        if self.config[2] in {"L", "N", "I"}:
            self.logsig_in = nn.Parameter(torch.full((self.input_size,), log_sigma_in_init))
        else:
            self.register_buffer('logsig_in', torch.zeros(self.input_size))

        if self.config[2] in {"L", "N", "D", "R"}:
            self.mu_hid = nn.Parameter(torch.ones(self.hidden_size))
        else:
            self.register_buffer('mu_hid', torch.ones(self.hidden_size))

        if self.config[2] in {"L", "N", "R"}:
            self.logsig_hid = nn.Parameter(torch.full((self.hidden_size,), log_sigma_hid_init))
        else:
            self.register_buffer('logsig_hid', torch.zeros(self.hidden_size))

        # --- gates multiplicative parameters
        if self.config[1] in {"L", "N", "D"}:
            self.mu_gates = nn.Parameter(torch.ones(4, self.hidden_size))
        else:
            self.register_buffer('mu_gates', torch.ones(4, self.hidden_size))

        if self.config[1] in {"L", "N"}:
            self.logsig_gates = nn.Parameter(torch.full((4, self.hidden_size), log_sigma_hid_init))
        else:
            self.register_buffer('logsig_gates', torch.zeros(4, self.hidden_size))

        # internal noise placeholders (будут заполнены в generate_noise_and_clip)
        self.input_w_noise = None
        self.hidden_w_noise = None
        self.input_noise = None
        self.hidden_noise = None
        self.gates_noise = None

        # clip masks
        self.input_w_clip = None
        self.hidden_w_clip = None
        self.input_clip = None
        self.hidden_clip = None
        self.gates_clip = None

    def clip_func(self, mtx, to=8.0):
        return torch.clamp(mtx, min=-to, max=to)
    
    def reset_parameters(self):
        std = 1.0 / (self.hidden_size ** 0.5)
        for w in self.parameters():
            nn.init.uniform_(w, -std, std)

    def generate_noise_and_clip(self, batch_size, device=None, deterministic=False, clip=False):
        """
        Заполняет self.input_w_noise, self.hidden_w_noise, input_noise, hidden_noise, gates_noise.
        Если deterministic=True — шум заменяется на нулевые шумы, а mu используются как значения.
        clip=True — вычисляются маски клиппинга на основе logalpha.
        """
        if device is None:
            device = next(self.parameters()).device

        # --- веса шумы
        if not deterministic and self.config[0] in {"L", "N"}:
            self.input_w_noise = torch.randn(4, self.input_size, self.hidden_size, device=device) * torch.exp(self.logsig_w_in)
            self.hidden_w_noise = torch.randn(4, self.hidden_size, self.hidden_size, device=device) * torch.exp(self.logsig_w_hid)
        else:
            self.input_w_noise = torch.zeros(4, self.input_size, self.hidden_size, device=device)
            self.hidden_w_noise = torch.zeros(4, self.hidden_size, self.hidden_size, device=device)

        # --- z (input/hidden) noise per example
        if not deterministic and self.config[2] in {"L", "N"}:
            self.input_noise = torch.randn(batch_size, self.input_size, device=device) * torch.exp(self.logsig_in) + self.mu_in
            self.hidden_noise = torch.randn(batch_size, self.hidden_size, device=device) * torch.exp(self.logsig_hid) + self.mu_hid
        elif not deterministic and self.config[2] == "I":
            self.input_noise = torch.randn(batch_size, self.input_size, device=device) * torch.exp(self.logsig_in) + self.mu_in
            self.hidden_noise = torch.ones(1, device=device)
        elif not deterministic and self.config[2] == "R":
            self.input_noise = torch.ones(1, device=device)
            self.hidden_noise = torch.randn(batch_size, self.hidden_size, device=device) * torch.exp(self.logsig_hid) + self.mu_hid
        elif not deterministic and self.config[2] == "D":
            self.input_noise = self.mu_in
            self.hidden_noise = self.mu_hid
        else:
            # deterministic or config not related
            if self.config[2] in {"L", "N", "D"}:
                self.input_noise = self.mu_in.unsqueeze(0).expand(batch_size, -1).to(device)
                self.hidden_noise = self.mu_hid.unsqueeze(0).expand(batch_size, -1).to(device)
            elif self.config[2] == "I":
                self.input_noise = self.mu_in.unsqueeze(0).expand(batch_size, -1).to(device)
                self.hidden_noise = torch.ones(1, device=device)
            elif self.config[2] == "R":
                self.input_noise = torch.ones(1, device=device)
                self.hidden_noise = self.mu_hid.unsqueeze(0).expand(batch_size, -1).to(device)
            else:
                self.input_noise = torch.ones(1, device=device)
                self.hidden_noise = torch.ones(1, device=device)

        # --- gates noise
        if not deterministic and self.config[1] in {"L", "N"}:
            # shape (4, batch, hidden)
            self.gates_noise = (torch.randn(4, batch_size, self.hidden_size, device=device) *
                                torch.exp(self.logsig_gates).unsqueeze(1) + self.mu_gates.unsqueeze(1))
        elif self.config[1] == "D":
            self.gates_noise = self.mu_gates.unsqueeze(1)  # shape (4,1,hidden) - will broadcast
        else:
            self.gates_noise = torch.ones(4, 1, self.hidden_size, device=device)

        # --- clip masks
        if clip:
            # W tensors: stack 4 matrices for in and hid
            W_in = torch.stack([self.W_in_to_ingate, self.W_in_to_forgetgate,
                                self.W_in_to_cell, self.W_in_to_outgate], dim=0)  # (4, in, hid)
            if self.config[0] == "L":
                log_alpha_w_in = self.clip_func(2.0 * self.logsig_w_in - torch.log(W_in ** 2 + 1e-12))
                self.input_w_clip = (log_alpha_w_in <= self.thresh).float()
            else:
                self.input_w_clip = torch.ones_like(W_in)

            W_hid = torch.stack([self.W_hid_to_ingate, self.W_hid_to_forgetgate,
                                 self.W_hid_to_cell, self.W_hid_to_outgate], dim=0)  # (4,hid,hid)
            if self.config[0] == "L":
                log_alpha_w_hid = self.clip_func(2.0 * self.logsig_w_hid - torch.log(W_hid ** 2 + 1e-12))
                self.hidden_w_clip = (log_alpha_w_hid <= self.thresh).float()
            else:
                self.hidden_w_clip = torch.ones_like(W_hid)

            # input/hidden z clip
            if self.config[2] == "L":
                log_alpha_in = self.clip_func(2.0 * self.logsig_in - torch.log(self.mu_in ** 2 + 1e-12))
                self.input_clip = (log_alpha_in <= self.thresh).float()
                log_alpha_hid = self.clip_func(2.0 * self.logsig_hid - torch.log(self.mu_hid ** 2 + 1e-12))
                self.hidden_clip = (log_alpha_hid <= self.thresh).float()
            elif self.config[2] == "I":
                log_alpha_in = self.clip_func(2.0 * self.logsig_in - torch.log(self.mu_in ** 2 + 1e-12))
                self.input_clip = (log_alpha_in <= self.thresh).float()
                self.hidden_clip = torch.ones(1, device=device)
            elif self.config[2] == "R":
                self.input_clip = torch.ones(1, device=device)
                log_alpha_hid = self.clip_func(2.0 * self.logsig_hid - torch.log(self.mu_hid ** 2 + 1e-12))
                self.hidden_clip = (log_alpha_hid <= self.thresh).float()
            else:
                self.input_clip = torch.ones(1, device=device)
                self.hidden_clip = torch.ones(1, device=device)

            # gates
            if self.config[1] == "L":
                log_alpha_gates = self.clip_func(2.0 * self.logsig_gates - torch.log(self.mu_gates ** 2 + 1e-12))
                self.gates_clip = (log_alpha_gates <= self.thresh).float()
            else:
                self.gates_clip = torch.ones_like(self.mu_gates)
        else:
            # no clipping: ones
            self.input_w_clip = torch.ones(4, self.input_size, self.hidden_size, device=device)
            self.hidden_w_clip = torch.ones(4, self.hidden_size, self.hidden_size, device=device)
            self.input_clip = torch.ones(1, device=device)
            self.hidden_clip = torch.ones(1, device=device)
            self.gates_clip = torch.ones(4, self.hidden_size, device=device)

    def input_preactivation(self, x_b, gate_type, deterministic=False, clip=False):
        # x_b: (batch, input_size)
        gate_idx = {'input': 0, 'forget': 1, 'cell': 2, 'output': 3}[gate_type]
        # Use per-example input_noise if available
        input_noise = self.input_noise if (isinstance(self.input_noise, torch.Tensor) and self.input_noise.dim() == 2) else self.input_noise
        # apply input multiplicative z and clip
        x_mod = x_b * input_noise
        # weight with potential weight-noise and clip
        W = [self.W_in_to_ingate, self.W_in_to_forgetgate, self.W_in_to_cell, self.W_in_to_outgate][gate_idx]  # (in, hid)
        W_noise = self.input_w_noise[gate_idx] if self.input_w_noise is not None else 0.0
        W_clip = self.input_w_clip[gate_idx] if self.input_w_clip is not None else 1.0
        W_eff = (W + W_noise) * W_clip
        return x_mod @ W_eff  # (batch, hidden)

    def hidden_preactivation(self, h_b, gate_type, deterministic=False, clip=False):
        gate_idx = {'input': 0, 'forget': 1, 'cell': 2, 'output': 3}[gate_type]
        W = [self.W_hid_to_ingate, self.W_hid_to_forgetgate, self.W_hid_to_cell, self.W_hid_to_outgate][gate_idx]  # (hid, hid)
        W_noise = self.hidden_w_noise[gate_idx] if self.hidden_w_noise is not None else 0.0
        W_clip = self.hidden_w_clip[gate_idx] if self.hidden_w_clip is not None else 1.0
        W_eff = (W + W_noise) * W_clip
        return h_b @ W_eff

    @utils.complex_handler
    def forward(self, x, hx=None, deterministic=False, clip=False, mask=None):
        """
        x: (batch, seq_len, input_size)
        returns: outputs (batch, seq_len, hidden)  -- like standard nn.LSTM output (before final fc)
        hx: optional initial tuple (h0, c0) with shape (num_layers, batch, hidden)
        deterministic (bool): if True -> use mu params (no noise)
        clip (bool): whether to compute clip masks
        mask: optional (batch, seq_len) boolean Tensor to support masking (like theano's mask)
        """
        batch_size, seq_len, _ = x.size()
        device = x.device
        # prepare noises & clips
        self.generate_noise_and_clip(batch_size, device=device, deterministic=deterministic, clip=clip)

        # initialize h0,c0
        if hx is None:
            h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
            c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        else:
            h0, c0 = hx

        hidden = [(h0[layer], c0[layer]) for layer in range(self.num_layers)]

        outputs = []
        for t in range(seq_len):
            x_t = x[:, t, :]  # (batch, input)
            for layer in range(self.num_layers):
                h_prev, c_prev = hidden[layer]
                if layer == 0:
                    in_act = self.input_preactivation(x_t, 'input', deterministic, clip)
                    f_act = self.input_preactivation(x_t, 'forget', deterministic, clip)
                    c_act = self.input_preactivation(x_t, 'cell', deterministic, clip)
                    o_act = self.input_preactivation(x_t, 'output', deterministic, clip)
                else:
                    # previous top-layer hidden used as "input"
                    prev_h = hidden[layer - 1][0]
                    in_act = self.input_preactivation(prev_h, 'input', deterministic, clip)  # note: in original code Z applies to input & hidden; we reuse same function
                    f_act = self.input_preactivation(prev_h, 'forget', deterministic, clip)
                    c_act = self.input_preactivation(prev_h, 'cell', deterministic, clip)
                    o_act = self.input_preactivation(prev_h, 'output', deterministic, clip)

                # add hidden contributions
                in_pre = in_act + self.hidden_preactivation(h_prev, 'input', deterministic, clip)
                f_pre = f_act + self.hidden_preactivation(h_prev, 'forget', deterministic, clip)
                c_pre = c_act + self.hidden_preactivation(h_prev, 'cell', deterministic, clip)
                o_pre = o_act + self.hidden_preactivation(h_prev, 'output', deterministic, clip)

                # apply gate multiplicative noise and bias
                # gates_noise shape (4, batch, hidden) or (4,1,hidden) for broadcast
                g0 = self.gates_noise[0] if self.gates_noise is not None else 1.0
                g1 = self.gates_noise[1] if self.gates_noise is not None else 1.0
                g2 = self.gates_noise[2] if self.gates_noise is not None else 1.0
                g3 = self.gates_noise[3] if self.gates_noise is not None else 1.0

                # add biases (we rely on biases existing as self.b_ingate etc.)
                i_t = torch.sigmoid(in_pre * g0 + self.b_ingate)
                f_t = torch.sigmoid(f_pre * g1 + self.b_forgetgate)
                g_t = torch.tanh(c_pre * g2 + self.b_cell)
                o_t = torch.sigmoid(o_pre * g3 + self.b_outgate)

                c_new = f_t * c_prev + i_t * g_t

                # hidden multiplicative z
                hidden_noise_per_example = self.hidden_noise if (isinstance(self.hidden_noise, torch.Tensor) and self.hidden_noise.dim() == 2) else self.hidden_noise
                h_new = o_t * torch.tanh(c_new) * hidden_noise_per_example

                # masking if present
                if mask is not None:
                    m_t = mask[:, t].unsqueeze(-1).to(device)  # (batch,1)
                    c_new = m_t * c_new + (1 - m_t) * c_prev
                    h_new = m_t * h_new + (1 - m_t) * h_prev

                hidden[layer] = (h_new, c_new)

            outputs.append(hidden[-1][0].unsqueeze(1))

        outputs = torch.cat(outputs, dim=1)  # (batch, seq_len, hidden)
        last_out = self.fc_out(outputs)      # (batch, output_size)
        # return same output shape as builtin nn.LSTM (but before fc)
        return last_out

    def eval_reg(self, train_size):
        """
        Compute KL regularization term per training set (scalar tensor).
        Mirrors Theano eval_reg: supports 'N' (normal) and 'L' (log-alpha) cases.
        """
        KL = torch.tensor(0.0, device=next(self.parameters()).device)

        # W_in
        W_in = torch.stack([self.W_in_to_ingate, self.W_in_to_forgetgate,
                            self.W_in_to_cell, self.W_in_to_outgate], dim=0)  # (4,in,hidden)
        if self.config[0] == "N":
            KL_element_in = - self.logsig_w_in + 0.5 * (torch.exp(2.0 * self.logsig_w_in) + W_in ** 2) - 0.5
            KL = KL + KL_element_in.sum()
        elif self.config[0] == "L":
            log_alpha_w_in = self.clip_func(2.0 * self.logsig_w_in - torch.log(W_in ** 2 + 1e-12))
            KL = KL + alpha_regf(log_alpha_w_in).sum()

        # W_hid
        W_hid = torch.stack([self.W_hid_to_ingate, self.W_hid_to_forgetgate,
                             self.W_hid_to_cell, self.W_hid_to_outgate], dim=0)
        if self.config[0] == "N":
            KL_element_hid = - self.logsig_w_hid + 0.5 * (torch.exp(2.0 * self.logsig_w_hid) + W_hid ** 2) - 0.5
            KL = KL + KL_element_hid.sum()
        elif self.config[0] == "L":
            log_alpha_w_hid = self.clip_func(2.0 * self.logsig_w_hid - torch.log(W_hid ** 2 + 1e-12))
            KL = KL + alpha_regf(log_alpha_w_hid).sum()

        # neurons (z)
        if self.config[2] == "L":
            log_alpha_hid = self.clip_func(2.0 * self.logsig_hid - torch.log(self.mu_hid ** 2 + 1e-12))
            KL = KL + alpha_regf(log_alpha_hid).sum()
            log_alpha_in = self.clip_func(2.0 * self.logsig_in - torch.log(self.mu_in ** 2 + 1e-12))
            KL = KL + alpha_regf(log_alpha_in).sum()
        elif self.config[2] == "I":
            log_alpha_in = self.clip_func(2.0 * self.logsig_in - torch.log(self.mu_in ** 2 + 1e-12))
            KL = KL + alpha_regf(log_alpha_in).sum()
        elif self.config[2] == "R":
            log_alpha_hid = self.clip_func(2.0 * self.logsig_hid - torch.log(self.mu_hid ** 2 + 1e-12))
            KL = KL + alpha_regf(log_alpha_hid).sum()
        elif self.config[2] == "N":
            KL_element = - self.logsig_hid + 0.5 * (torch.exp(2.0 * self.logsig_hid) + self.mu_hid ** 2) - 0.5
            KL = KL + KL_element.sum()
            KL_element = - self.logsig_in + 0.5 * (torch.exp(2.0 * self.logsig_in) + self.mu_in ** 2) - 0.5
            KL = KL + KL_element.sum()

        # gates
        if self.config[1] == "L":
            log_alpha_gates = self.clip_func(2.0 * self.logsig_gates - torch.log(self.mu_gates ** 2 + 1e-12))
            KL = KL + alpha_regf(log_alpha_gates).sum()
        elif self.config[1] == "N":
            KL_element = - self.logsig_gates + 0.5 * (torch.exp(2.0 * self.logsig_gates) + self.mu_gates ** 2) - 0.5
            KL = KL + KL_element.sum()

        return KL / float(train_size)

    def get_ard(self):
        """
        Вычисляет маски ARD (аналог get_ard в Theano).
        Возвращает dict с булевыми масками (в numpy).
        """
        # W masks
        if self.config[0] == "L":
            W_in = torch.stack([self.W_in_to_ingate, self.W_in_to_forgetgate,
                                self.W_in_to_cell, self.W_in_to_outgate], dim=0).detach()
            log_alpha_w_in = (2.0 * self.logsig_w_in.detach() - 2.0 * torch.log(torch.abs(W_in) + 1e-12))
            mask_w_in = (log_alpha_w_in < self.thresh).cpu().numpy()
            W_hid = torch.stack([self.W_hid_to_ingate, self.W_hid_to_forgetgate,
                                 self.W_hid_to_cell, self.W_hid_to_outgate], dim=0).detach()
            log_alpha_w_hid = (2.0 * self.logsig_w_hid.detach() - 2.0 * torch.log(torch.abs(W_hid) + 1e-12))
            mask_w_hid = (log_alpha_w_hid < self.thresh).cpu().numpy()
        else:
            mask_w_in = torch.ones((4, self.input_size, self.hidden_size), dtype=torch.bool).cpu().numpy()
            mask_w_hid = torch.ones((4, self.hidden_size, self.hidden_size), dtype=torch.bool).cpu().numpy()

        # neurons
        mask_in = mask_w_in.any(axis=2).any(axis=0)  # reduce
        mask_hid_by_w = mask_w_hid.any(axis=2).any(axis=0)
        mask_hid_by_z = mask_hid_by_w.copy()

        if self.config[2] == "L":
            log_alpha_hid = (2.0 * self.logsig_hid.detach().cpu().numpy() - 2.0 * np.log(np.abs(self.mu_hid.detach().cpu().numpy()) + 1e-12))
            log_alpha_in = (2.0 * self.logsig_in.detach().cpu().numpy() - 2.0 * np.log(np.abs(self.mu_in.detach().cpu().numpy()) + 1e-12))
            mask_in = np.logical_and(log_alpha_in < self.thresh, mask_in)
            mask_hid_by_z = log_alpha_hid < self.thresh
        elif self.config[2] == "I":
            log_alpha_in = (2.0 * self.logsig_in.detach().cpu().numpy() - 2.0 * np.log(np.abs(self.mu_in.detach().cpu().numpy()) + 1e-12))
            mask_in = np.logical_and(log_alpha_in < self.thresh, mask_in)
        elif self.config[2] == "R":
            log_alpha_hid = (2.0 * self.logsig_hid.detach().cpu().numpy() - 2.0 * np.log(np.abs(self.mu_hid.detach().cpu().numpy()) + 1e-12))
            mask_hid_by_z = log_alpha_hid < self.thresh

        # gates
        if self.config[1] == "L":
            log_alpha_gates = (2.0 * self.logsig_gates.detach().cpu().numpy() - 2.0 * np.log(np.abs(self.mu_gates.detach().cpu().numpy()) + 1e-12))
            mask = np.concatenate([mask_w_in, mask_w_hid], axis=1)
            mask_gates = np.logical_and(log_alpha_gates < self.thresh, mask.any(axis=1))
        else:
            mask = np.concatenate([mask_w_in, mask_w_hid], axis=1)
            mask_gates = mask.any(axis=1)

        return {
            "w_input": mask_w_in,
            "w_hidden": mask_w_hid,
            "gates": mask_gates,
            "z_input": mask_in,
            "z_hidden_by_w": mask_hid_by_w,
            "z_hidden": mask_hid_by_z
        }




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
            return False