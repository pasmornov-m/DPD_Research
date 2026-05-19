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
        # self.ln1 = torch.nn.LayerNorm(d_model)

        self.conv1 = torch.nn.Conv1d(d_model, d_ff, kernel_size=1)
        self.conv2 = torch.nn.Conv1d(d_ff, d_model, kernel_size=1)
        # self.ln2 = torch.nn.LayerNorm(d_model)

        # self.activation = torch.nn.Tanh()
        self.activation = torch.nn.Hardswish()

    def forward(self, x):
        # x = self.ln1(x)
        attn_out, _ = self.attn(x, x, x)
        x = x + attn_out
        # x = self.ln1(x)

        x_cnn = x.transpose(1, 2)
        x_cnn = self.conv1(x_cnn)
        x_cnn = self.activation(x_cnn)
        x_cnn = self.conv2(x_cnn)
        x_cnn = self.activation(x_cnn)
        x_cnn = x_cnn.transpose(1, 2)
        
        x = x + x_cnn
        # x = self.ln2(x)
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

        # self.in_norm = torch.nn.LayerNorm(d_in)
        
        self.input_fc = torch.nn.Linear(d_in, d_model)
        # self.pos_encoder = PositionalEncoding(d_model)
                
        self.encoders = torch.nn.ModuleList(
            [TransformerEncoderBlock(d_model, n_heads, d_ff) for _ in range(num_blocks)]
        )
        
        self.fc = torch.nn.Linear(self.T * d_model, n_fc)

        # self.activation = torch.nn.Tanh()
        self.activation = torch.nn.Hardswish()
        
        self.out = torch.nn.Linear(n_fc, 2)
        
        torch.nn.init.xavier_uniform_(self.out.weight)
        torch.nn.init.zeros_(self.out.bias)

        
    def forward(self, x):
        x = self.input_fc(x)
        
        # x = self.pos_encoder(x)

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

    def count_flops(self) -> int:
        """
        Approximate FLOPs estimation for RTDTNN (Transformer-based model).

        FLOPs are counted for:
        - one output sample

        Notes
        -----

        Assumptions:
        - add/sub/mul            : 1 FLOP
        - tanh                   : 4 FLOPs
        - Linear                 : 2 * in * out
        - Conv1d (1x1)           : 2 * in * out
        - LayerNorm              : ~5 * d_model (approx)
        - Attention:
            * QK^T              : 2 * T * d_model
            * softmax           : ~5 * T * T
            * AV                : 2 * T * d_model
        - Multihead scaling ignored in detail (aggregated per model dim)

        Returns
        -------
        int: FLOPs per one output sample
        """

        T = self.T
        d_in = self.d_in
        d_model = self.d_model
        d_ff = self.d_ff
        n_heads = self.n_heads
        n_blocks = self.num_blocks
        n_fc = self.n_fc

        # INPUT PROJECTION

        input_fc_flops = 2 * d_in * d_model

        # TRANSFORMER BLOCKS

        block_flops = 0

        for _ in range(n_blocks):

            # MULTIHEAD ATTENTION (approx aggregated)

            # QK^T
            qk_flops = 2 * T * d_model

            # softmax
            softmax_flops = 5 * T * T

            # AV
            av_flops = 2 * T * d_model

            attn_flops = qk_flops + softmax_flops + av_flops

            # residual + layernorm1
            ln1_flops = 5 * d_model * T

            # CNN part (1x1 convs)

            conv1_flops = 2 * d_model * d_ff * T
            conv2_flops = 2 * d_ff * d_model * T

            tanh_flops = 4 * T * d_ff + 4 * T * d_model

            # residual + layernorm2
            ln2_flops = 5 * d_model * T

            block_flops += (
                attn_flops
                + ln1_flops
                + conv1_flops
                + conv2_flops
                + tanh_flops
                + ln2_flops
            )

        # FINAL FC HEAD

        fc_flops = 2 * (T * d_model) * n_fc

        # activation
        act_flops = 4 * n_fc

        # output layer
        out_flops = 2 * n_fc * 2

        # TOTAL

        total_flops = (
            input_fc_flops
            + block_flops
            + fc_flops
            + act_flops
            + out_flops
        )

        return int(total_flops)
    
    def count_macs(self) -> int:
        """
        Approximate MACs estimation for RTDTNN model.

        Counts only:
        - multiplications
        - accumulations

        Does NOT count:
        - softmax
        - LayerNorm
        - Tanh
        - reshape/transpose
        - residual additions
        - positional encoding

        Notes
        -----
        MACs are estimated for:
        - one forward pass
        - one sample

        Returns
        -------
        int
            MACs per one output sample.
        """

        T = self.T
        d_in = self.d_in
        d_model = self.d_model
        d_ff = self.d_ff
        n_fc = self.n_fc
        n_heads = self.n_heads
        num_blocks = self.num_blocks

        total_mac = 0

        # Input projection
        #
        # Linear:
        # d_in -> d_model
        #
        # Applied for each timestep

        input_fc_mac = T * d_in * d_model

        total_mac += input_fc_mac

        # Transformer encoder blocks

        for _ in range(num_blocks):

            # -----------------------------------------------------
            # Multihead Attention
            #
            # Q, K, V projections:
            #
            # 3 * (T * d_model * d_model)
            # -----------------------------------------------------

            qkv_mac = 3 * T * d_model * d_model

            # -----------------------------------------------------
            # Attention scores
            #
            # Q @ K^T
            #
            # Shape:
            # [T, d_model] x [d_model, T]
            #
            # ≈ T^2 * d_model
            # -----------------------------------------------------

            attention_scores_mac = T * T * d_model

            # -----------------------------------------------------
            # Attention-weighted values
            #
            # Attn @ V
            #
            # ≈ T^2 * d_model
            # -----------------------------------------------------

            attention_value_mac = T * T * d_model

            # -----------------------------------------------------
            # Output projection
            #
            # d_model -> d_model
            # -----------------------------------------------------

            attention_out_mac = T * d_model * d_model

            # -----------------------------------------------------
            # Feed-forward Conv1d block
            #
            # Conv1d(kernel=1)
            #
            # d_model -> d_ff
            # d_ff -> d_model
            # -----------------------------------------------------

            ffn_mac = (
                T * d_model * d_ff
                + T * d_ff * d_model
            )

            block_mac = (
                qkv_mac
                + attention_scores_mac
                + attention_value_mac
                + attention_out_mac
                + ffn_mac
            )

            total_mac += block_mac

        # Fully-connected layer
        #
        # Flatten:
        # T * d_model -> n_fc

        fc_mac = (T * d_model) * n_fc

        total_mac += fc_mac

        # Output layer
        #
        # n_fc -> 2

        out_mac = n_fc * 2

        total_mac += out_mac

        return int(total_mac)