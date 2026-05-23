import torch
import numpy as np
from models.base_model import BaseModel


class TransformerEncoderBlock(torch.nn.Module):
    def __init__(self, d_model, nhead, d_ff):
        super().__init__()
        self.attn = torch.nn.MultiheadAttention(d_model, nhead, batch_first=True)

        self.conv1 = torch.nn.Conv1d(d_model, d_ff, kernel_size=1)
        self.conv2 = torch.nn.Conv1d(d_ff, d_model, kernel_size=1)

        self.activation = torch.nn.Hardswish()

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        x = x + attn_out

        x_cnn = x.transpose(1, 2)
        x_cnn = self.conv1(x_cnn)
        x_cnn = self.activation(x_cnn)
        x_cnn = self.conv2(x_cnn)
        x_cnn = self.activation(x_cnn)
        x_cnn = x_cnn.transpose(1, 2)
        
        x = x + x_cnn
        
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
        
        self.input_fc = torch.nn.Linear(d_in, d_model)
                
        self.encoders = torch.nn.ModuleList(
            [TransformerEncoderBlock(d_model, n_heads, d_ff) for _ in range(num_blocks)]
        )
        
        self.fc = torch.nn.Linear(self.T * d_model, n_fc)

        self.activation = torch.nn.Hardswish()
        
        self.out = torch.nn.Linear(n_fc, 2)
        
        torch.nn.init.xavier_uniform_(self.out.weight)
        torch.nn.init.zeros_(self.out.bias)

        
    def forward(self, x):
        x = self.input_fc(x)
        
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
        Approximate FLOPs estimation for RTDTNN.

        Counted per ONE output sample.

        Assumptions
        -----------
        - multiply/add                 : 1 FLOP
        - Linear / Conv1d(1x1)         : 2 * in * out
        - Hardswish                    : ~4 FLOPs
        - residual add                 : 1 FLOP
        - softmax                      : ~5 FLOPs
        - attention scaling/division   : ignored

        Returns
        -------
        int
            Approximate FLOPs per sample.
        """

        T = self.T
        d_in = self.d_in
        d_model = self.d_model
        d_ff = self.d_ff
        n_fc = self.n_fc
        n_blocks = self.num_blocks

        total_flops = 0

        # ============================================================
        # INPUT PROJECTION
        # ============================================================

        # Linear(d_in -> d_model) for every token
        #
        # input shape:
        # [B, T, d_in]
        #
        # output:
        # [B, T, d_model]

        input_fc_flops = T * (2 * d_in * d_model)

        total_flops += input_fc_flops

        # ============================================================
        # TRANSFORMER BLOCKS
        # ============================================================

        for _ in range(n_blocks):

            # --------------------------------------------------------
            # MULTIHEAD SELF-ATTENTION
            # --------------------------------------------------------

            # Q, K, V projections
            #
            # 3x Linear(d_model -> d_model)

            qkv_flops = 3 * T * (2 * d_model * d_model)

            # Attention scores: Q @ K^T
            #
            # [T, d_model] x [d_model, T]
            #
            # => [T, T]

            qk_flops = 2 * T * T * d_model

            # Softmax over attention matrix

            softmax_flops = 5 * T * T

            # Attention-weighted values
            #
            # [T, T] x [T, d_model]

            av_flops = 2 * T * T * d_model

            # Output projection

            out_proj_flops = T * (2 * d_model * d_model)

            # Residual add after attention

            attn_residual_flops = T * d_model

            attention_flops = (
                qkv_flops
                + qk_flops
                + softmax_flops
                + av_flops
                + out_proj_flops
                + attn_residual_flops
            )

            # --------------------------------------------------------
            # FFN (Conv1d 1x1)
            # --------------------------------------------------------

            # Conv1d(d_model -> d_ff)

            conv1_flops = T * (2 * d_model * d_ff)

            # Hardswish

            act1_flops = 4 * T * d_ff

            # Conv1d(d_ff -> d_model)

            conv2_flops = T * (2 * d_ff * d_model)

            # Hardswish

            act2_flops = 4 * T * d_model

            # Residual add

            ffn_residual_flops = T * d_model

            ffn_flops = (
                conv1_flops
                + act1_flops
                + conv2_flops
                + act2_flops
                + ffn_residual_flops
            )

            total_flops += attention_flops + ffn_flops

        # ============================================================
        # FINAL HEAD
        # ============================================================

        # Flatten:
        #
        # [T, d_model] -> [T*d_model]

        flatten_dim = T * d_model

        # FC

        fc_flops = 2 * flatten_dim * n_fc

        # Hardswish

        fc_act_flops = 4 * n_fc

        # Output layer

        out_flops = 2 * n_fc * 2

        total_flops += (
            fc_flops
            + fc_act_flops
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
        - Hardswish
        - softmax
        - reshape / transpose
        - residual additions
        - parameter initialization

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
        num_blocks = self.num_blocks

        total_mac = 0

        # ============================================================
        # INPUT PROJECTION
        # ============================================================

        # Linear:
        # d_in -> d_model
        #
        # Applied for each timestep

        input_fc_mac = T * d_in * d_model

        total_mac += input_fc_mac

        # ============================================================
        # TRANSFORMER BLOCKS
        # ============================================================

        for _ in range(num_blocks):

            # --------------------------------------------------------
            # MULTIHEAD SELF-ATTENTION
            # --------------------------------------------------------

            # Q, K, V projections
            #
            # 3x Linear(d_model -> d_model)

            qkv_mac = 3 * T * d_model * d_model

            # --------------------------------------------------------
            # Attention scores
            #
            # Q @ K^T
            #
            # [T, d_model] x [d_model, T]
            #
            # -> [T, T]
            # --------------------------------------------------------

            attention_scores_mac = T * T * d_model

            # --------------------------------------------------------
            # Attention-weighted values
            #
            # Attn @ V
            #
            # [T, T] x [T, d_model]
            # --------------------------------------------------------

            attention_values_mac = T * T * d_model

            # --------------------------------------------------------
            # Output projection
            #
            # Linear(d_model -> d_model)
            # --------------------------------------------------------

            attention_out_mac = T * d_model * d_model

            attention_mac = (
                qkv_mac
                + attention_scores_mac
                + attention_values_mac
                + attention_out_mac
            )

            # --------------------------------------------------------
            # FFN BLOCK
            #
            # Conv1d(kernel=1)
            # equivalent to token-wise Linear
            # --------------------------------------------------------

            conv1_mac = T * d_model * d_ff

            conv2_mac = T * d_ff * d_model

            ffn_mac = conv1_mac + conv2_mac

            # --------------------------------------------------------

            block_mac = attention_mac + ffn_mac

            total_mac += block_mac

        # ============================================================
        # FINAL HEAD
        # ============================================================

        # Flatten:
        #
        # [T, d_model] -> [T*d_model]

        flatten_dim = T * d_model

        # FC:
        #
        # T*d_model -> n_fc

        fc_mac = flatten_dim * n_fc

        total_mac += fc_mac

        # Output:
        #
        # n_fc -> 2

        out_mac = n_fc * 2

        total_mac += out_mac

        return int(total_mac)