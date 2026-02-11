import torch
import numpy as np
from models.base_model import BaseModel


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
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


class TransformerEncoderBlock(torch.nn.Module):
    def __init__(self, d_model, nhead, d_ff):
        super().__init__()
        self.attn = torch.nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.ln1 = torch.nn.LayerNorm(d_model)

        self.conv1 = torch.nn.Conv1d(d_model, d_ff, kernel_size=1)
        self.conv2 = torch.nn.Conv1d(d_ff, d_model, kernel_size=1)
        self.ln2 = torch.nn.LayerNorm(d_model)

        self.activation = torch.nn.Tanh()

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        x = x + attn_out
        x = self.ln1(x)

        x_cnn = x.transpose(1, 2)
        x_cnn = self.conv1(x_cnn)
        x_cnn = self.activation(x_cnn)
        x_cnn = self.conv2(x_cnn)
        x_cnn = self.activation(x_cnn)
        x_cnn = x_cnn.transpose(1, 2)

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
                 model_name: str = "model",
                 use_pe=False):
        
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
        
        self.use_pe = use_pe
        
        self.in_norm = torch.nn.LayerNorm(d_in)
        
        self.input_fc = torch.nn.Linear(d_in, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        self.encoders = torch.nn.ModuleList(
            [TransformerEncoderBlock(d_model, n_heads, d_ff) for _ in range(num_blocks)]
        )
        
        self.fc = torch.nn.Linear(self.T * d_model, n_fc)
        self.activation = torch.nn.Tanh()
        
        self.out = torch.nn.Linear(n_fc, 2)
        
        torch.nn.init.zeros_(self.out.weight)
        torch.nn.init.zeros_(self.out.bias)

        
    def forward(self, x):
        x = self.input_fc(x)
        if self.use_pe:
            x = self.pos_encoder(x)

        for encoder in self.encoders:
            x = encoder(x)
        
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        x = self.activation(x)
        x = self.out(x)
        return x

    def _get_filename(self):
        return (
            f"{self.model_name}_{self.class_name}"
            f"_din{self.d_in}_dmodel{self.d_model}_heads{self.n_heads}"
            f"_dff{self.d_ff}_nfc{self.n_fc}_M{self.M}_blocks{self.num_blocks}.pt"
            )
