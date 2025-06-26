import torch
from torch import nn
import os
from modules import utils


class GRU(nn.Module):
    def __init__(self, input_size=2, hidden_size=64, num_layers=1, output_size=2, bidirectional=False, model_name="gru_model"):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.bidirectional = bidirectional
        self.model_name = model_name
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, bidirectional=bidirectional)
        self.fc = nn.Linear(hidden_size, output_size)

    @utils.complex_handler
    def forward(self, x, h_0=None):
        batch_size = x.size(0)
        if h_0 is None:
            h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        out, _ = self.gru(x, h_0)
        # y = out[:, -1, :]
        y = self.fc(out)
        return y

    def save_weights(self, directory="model_params"):
        os.makedirs(directory, exist_ok=True)
        filename = os.path.join(directory, f"{self.model_name}.pt")
        torch.save(self.state_dict(), filename)
        print(f"Model weights saved to {filename}")

    def load_weights(self, directory="model_params"):
        filename = os.path.join(directory, f"{self.model_name}.pt")
        if os.path.isfile(filename):
            self.load_state_dict(torch.load(filename))
            print(f"Model weights loaded from {filename}")
            return True
        else:
            print(f"No saved weights found at {filename}")
            return False


class LSTM(nn.Module):
    def __init__(self, input_size=2, hidden_size=64, num_layers=1, output_size=2, bidirectional=False, batch_first=True,
                 bias=False, model_name="lstm_model"):
        super().__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.batch_first = batch_first
        self.bias = bias
        self.model_name = model_name

        self.rnn = nn.LSTM(input_size=input_size,
                          hidden_size=hidden_size,
                          num_layers=num_layers,
                          bidirectional=self.bidirectional,
                          batch_first=self.batch_first,
                          bias=self.bias)
        self.fc_out = nn.Linear(in_features=self.hidden_size*self.num_layers,
                                out_features=self.output_size,
                                bias=self.bias)

    @utils.complex_handler
    def forward(self, x):
        out, h = self.rnn(x)
        hh = torch.cat((h[0][-2, :, :], h[0][-1, :, :]), dim=1)
        y = self.fc_out(hh)
        return y

    def save_weights(self, directory="model_params"):
        os.makedirs(directory, exist_ok=True)
        filename = os.path.join(directory, f"{self.model_name}.pt")
        torch.save(self.state_dict(), filename)
        print(f"Model weights saved to {filename}")

    def load_weights(self, directory="model_params"):
        filename = os.path.join(directory, f"{self.model_name}.pt")
        if os.path.isfile(filename):
            self.load_state_dict(torch.load(filename))
            print(f"Model weights loaded from {filename}")
            return True
        else:
            print(f"No saved weights found at {filename}")
            return False