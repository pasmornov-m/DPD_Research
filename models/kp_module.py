import torch


class KPModule(torch.nn.Module):
    def __init__(self,
                 M: int = 5):
        
        assert M % 2 == 1, "M must be odd"
        
        torch.nn.Module.__init__(self)
        
        self.M = M
        self.K= 5
        self._EPS = 1e-10
        self.feature_dim = (self.M + 3 * self.K * self.K) * 2
        
        self.linear_re = torch.nn.Linear(in_features=2*self.M, out_features=self.K)
        self.linear_im = torch.nn.Linear(in_features=2*self.M, out_features=self.K)
        self.linear_abs1 = torch.nn.Linear(in_features=self.M, out_features=self.K)
        self.linear_abs2 = torch.nn.Linear(in_features=self.M, out_features=self.K)
        self.linear_abs3 = torch.nn.Linear(in_features=self.M, out_features=self.K)

        self.activation = torch.nn.Hardswish()
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

    def get_feature_dim(self):
        return self.feature_dim
        
    def _build_memory(self, x):
        # x: (B, N, 2)
        half = self.M // 2

        x_re = x[..., 0]  # (B, N)
        x_im = x[..., 1]

        x_re = torch.cat([x_re[:, -half:], x_re, x_re[:, :half]], dim=1)
        x_im = torch.cat([x_im[:, -half:], x_im, x_im[:, :half]], dim=1)

        x_re_mem = x_re.unfold(1, self.M, 1)  # (B, N, M)
        x_im_mem = x_im.unfold(1, self.M, 1)

        return x_re_mem, x_im_mem
    
    def _split_features(self, x_re_mem, x_im_mem):
        x_concat = torch.cat([x_re_mem, x_im_mem], dim=-1)
        
        magnitude2 = x_re_mem**2 + x_im_mem**2
        x_abs1 = torch.sqrt(magnitude2 + self._EPS)
        x_abs2 = magnitude2
        x_abs3 = x_abs1 * magnitude2
        
        return (x_concat, x_abs1, x_abs2, x_abs3)
       
    def linear_projection(self, x_concat, x_abs1, x_abs2, x_abs3):
        z_re = self.activation(self.linear_re(x_concat))
        z_im = self.activation(self.linear_im(x_concat))
        z_abs1 = self.activation(self.linear_abs1(x_abs1))
        z_abs2 = self.activation(self.linear_abs2(x_abs2))
        z_abs3 = self.activation(self.linear_abs3(x_abs3))
        
        return (z_re, z_im, z_abs1, z_abs2, z_abs3)
    
    def kron_product(self, a, b):
        """
        Kronecker product for batches
        a: (B, N, K)
        b: (B, N, K)
        returns: (B, N, K*K)
        """
        B, N, K = a.shape
        
        a_expanded = a.unsqueeze(-1)  # (B, N, K, 1)
        b_expanded = b.unsqueeze(-2)  # (B, N, 1, K)
        
        kron = a_expanded * b_expanded  # (B, N, K, K)
        kron = kron.reshape(B, N, K * K)  # (B, N, K*K)
        
        return kron

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x_re_mem, x_im_mem = self._build_memory(x)
        x_concat, x_abs1, x_abs2, x_abs3 = self._split_features(x_re_mem, x_im_mem)
        z_re, z_im, z_abs1, z_abs2, z_abs3 = self.linear_projection(x_concat, x_abs1, x_abs2, x_abs3)
        
        # p = 1: (z_1 + j*z_2) ⊗ z_abs1
        kron1_real = self.kron_product(z_re, z_abs1)
        kron1_imag = self.kron_product(z_im, z_abs1)
        
        # p = 2: (z_1 + j*z_2) ⊗ z_abs2
        kron2_real = self.kron_product(z_re, z_abs2)
        kron2_imag = self.kron_product(z_im, z_abs2)
        
        # p = 3: (z_1 + j*z_2) ⊗ z_abs3
        kron3_real = self.kron_product(z_re, z_abs3)
        kron3_imag = self.kron_product(z_im, z_abs3)
        
        real_parts = [
            x_re_mem,      # (B, N, M)
            kron1_real,    # (B, N, K*K)
            kron2_real,    # (B, N, K*K)
            kron3_real     # (B, N, K*K)
        ]
        
        imag_parts = [
            x_im_mem,      # (B, N, M)
            kron1_imag,    # (B, N, K*K)
            kron2_imag,    # (B, N, K*K)
            kron3_imag     # (B, N, K*K)
        ]
        
        real_concat = torch.cat(real_parts, dim=-1)  # (B, N, M + 3*K*K)
        imag_concat = torch.cat(imag_parts, dim=-1)  # (B, N, M + 3*K*K)
        
        kp_features = torch.stack([real_concat, imag_concat], dim=-1)  # (B, N, total_features, 2)
        
        return kp_features

    def count_flops(self) -> int:
        """
        Approximate FLOPs estimation for KPModule.

        FLOPs are counted for:
        - one output sample (one sequence)

        Notes
        -----

        Assumptions:
        - add/sub/mul        : 1 FLOP
        - sqrt               : 10 FLOPs
        - Hardswish          : 6 FLOPs (approx)
        - Linear             : 2 * in * out
        - abs/square         : 1 FLOP each
        - memory build (unfold/concat): ignored or minimal cost

        Returns
        -------
        int: FLOPs per one output sample
        """

        M = self.M
        K = self.K

        # MEMORY BUILD (approx ignored)

        memory_flops = 0

        # FEATURE SPLIT

        # magnitude2 = re^2 + im^2
        mag2_flops = 2 * M  # per window element

        # sqrt
        abs1_flops = 10 * M

        # abs2 = magnitude2
        abs2_flops = 0

        # abs3 = abs1 * magnitude2
        abs3_flops = M

        split_flops = mag2_flops + abs1_flops + abs2_flops + abs3_flops

        # LINEAR PROJECTIONS

        # Linear(x_concat: 2M -> K)
        lin_re = 2 * (2 * M) * K
        lin_im = 2 * (2 * M) * K

        # Linear(abs1: M -> K)
        lin_abs1 = 2 * M * K
        lin_abs2 = 2 * M * K
        lin_abs3 = 2 * M * K

        linear_total = lin_re + lin_im + lin_abs1 + lin_abs2 + lin_abs3

        # activations (5 branches)
        activation_total = 6 * (5 * K * M)

        # KRONECKER PRODUCTS

        # 3 magnitude branches × (real + imag) = 6 kron products
        kron_size = K * K

        kron_flops = 6 * kron_size * 1  # elementwise multiplications

        # CONCAT + STACK (ignored)

        concat_flops = 0

        # TOTAL

        total_flops = (
            memory_flops
            + split_flops
            + linear_total
            + activation_total
            + kron_flops
            + concat_flops
        )

        return int(total_flops)

    def count_macs(self, sequence_length: int = 1) -> int:
        """
        Approximate MACs estimation for KPModule.

        Counts only:
        - multiplications
        - accumulations (implicit in Linear / matmul)

        Does NOT count:
        - activations (Hardswish)
        - sqrt, add, concat, reshape, unfold
        - bias additions

        Notes
        -----
        Complexity is estimated for:
        - one forward pass
        - one sample
        - full sequence length

        Parameters
        ----------
        sequence_length : int
            Temporal length N.

        Returns
        -------
        int
            MACs per forward pass.
        """

        N = sequence_length
        M = self.M
        K = self.K

        total_mac = 0

        # LINEAR PROJECTIONS (5 branches)
        #
        # Each Linear: in_features * out_features
        # applied per timestep

        mac_linear_re = N * (2 * M) * K
        mac_linear_im = N * (2 * M) * K

        mac_linear_abs1 = N * M * K
        mac_linear_abs2 = N * M * K
        mac_linear_abs3 = N * M * K

        total_mac += (
            mac_linear_re
            + mac_linear_im
            + mac_linear_abs1
            + mac_linear_abs2
            + mac_linear_abs3
        )

        # KRONECKER PRODUCTS
        #
        # kron(a,b): (B,N,K) ⊗ (B,N,K)
        # = elementwise multiplication over K×K
        #
        # cost: K^2 multiplications per element pair
        #
        # we have 3 pairs:
        # (z_re, z_abs*)
        # (z_im, z_abs*)

        kron_per_pair = N * (K * K)

        mac_kron = 6 * kron_per_pair  # 3 abs paths × real+imag

        total_mac += mac_kron

        # MEMORY BUILDING + CONCAT + SPLITS
        #
        # ignored (no MAC definition impact)

        return int(total_mac)




class KPConvModule(torch.nn.Module):
    def __init__(self,
                 M: int = 5,
                 K: int = 5):
        
        assert M % 2 == 1, "M must be odd"
        
        torch.nn.Module.__init__(self)
        
        self.M = M
        self.K = K
        self._EPS = 1e-10
        self.feature_dim = (1 + 3 * self.K * self.K) * 2
        
        self.conv_reim = torch.nn.Conv1d(in_channels=2, out_channels=2*K, kernel_size=M, padding='same')
        self.conv_abs1 = torch.nn.Conv1d(in_channels=1, out_channels=K, kernel_size=M, padding='same')
        self.conv_abs2 = torch.nn.Conv1d(in_channels=1, out_channels=K, kernel_size=M, padding='same')
        self.conv_abs3 = torch.nn.Conv1d(in_channels=1, out_channels=K, kernel_size=M, padding='same')

        self.activation = torch.nn.Hardswish()
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv1d):
                torch.nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
    
    def get_feature_dim(self):
        return self.feature_dim
        
    def _split_features(self, x):
        # x: (B, N, 2)

        x_re = x[..., 0]
        x_im = x[..., 1]

        magnitude2 = x_re**2 + x_im**2

        x_abs1 = torch.sqrt(magnitude2 + self._EPS)
        x_abs2 = magnitude2
        x_abs3 = x_abs1 * magnitude2

        return x_re, x_im, x_abs1, x_abs2, x_abs3
    
    def linear_projection(self, x_re, x_im, x_abs1, x_abs2, x_abs3):

        # ---- prepare tensors ----
        # (B, N) → (B, 1, N)
        x_re = x_re.unsqueeze(1)
        x_im = x_im.unsqueeze(1)
        x_abs1 = x_abs1.unsqueeze(1)
        x_abs2 = x_abs2.unsqueeze(1)
        x_abs3 = x_abs3.unsqueeze(1)

        # merge real + imag
        x_reim = torch.cat([x_re, x_im], dim=1)  # (B,2,N)

        # ---- convolution ----
        z_reim = self.activation(self.conv_reim(x_reim))  # (B,2K,N)
        z_abs1 = self.activation(self.conv_abs1(x_abs1))
        z_abs2 = self.activation(self.conv_abs2(x_abs2))
        z_abs3 = self.activation(self.conv_abs3(x_abs3))

        # ---- split real / imag ----
        z_re, z_im = torch.chunk(z_reim, chunks=2, dim=1)

        # ---- reshape back ----
        z_re = z_re.transpose(1, 2)
        z_im = z_im.transpose(1, 2)

        z_abs1 = z_abs1.transpose(1, 2)
        z_abs2 = z_abs2.transpose(1, 2)
        z_abs3 = z_abs3.transpose(1, 2)

        return (
            z_re,
            z_im,
            z_abs1,
            z_abs2,
            z_abs3
        )
        
    def kron_product(self, a, b):
        """
        Kronecker product for batches
        a: (B, N, K)
        b: (B, N, K)
        returns: (B, N, K*K)
        """
        B, N, K = a.shape
        
        a_expanded = a.unsqueeze(-1)  # (B, N, K, 1)
        b_expanded = b.unsqueeze(-2)  # (B, N, 1, K)
        
        kron = a_expanded * b_expanded  # (B, N, K, K)
        kron = kron.reshape(B, N, K * K)  # (B, N, K*K)
        
        return kron
    
    def forward(self, x):
        x_re, x_im, x_abs1, x_abs2, x_abs3 = self._split_features(x)

        z_re, z_im, z_abs1, z_abs2, z_abs3 = self.linear_projection(x_re, x_im, x_abs1, x_abs2, x_abs3)

        # p = 1: (z_1 + j*z_2) ⊗ z_abs1
        kron1_real = self.kron_product(z_re, z_abs1)
        kron1_imag = self.kron_product(z_im, z_abs1)
        
        # p = 2: (z_1 + j*z_2) ⊗ z_abs2
        kron2_real = self.kron_product(z_re, z_abs2)
        kron2_imag = self.kron_product(z_im, z_abs2)
        
        # p = 3: (z_1 + j*z_2) ⊗ z_abs3
        kron3_real = self.kron_product(z_re, z_abs3)
        kron3_imag = self.kron_product(z_im, z_abs3)
        
        real_parts = [
            x_re.unsqueeze(-1),          # (B, N, M)
            kron1_real,    # (B, N, K*K)
            kron2_real,    # (B, N, K*K)
            kron3_real     # (B, N, K*K)
        ]
        
        imag_parts = [
            x_im.unsqueeze(-1),          # (B, N, M)
            kron1_imag,    # (B, N, K*K)
            kron2_imag,    # (B, N, K*K)
            kron3_imag     # (B, N, K*K)
        ]
        
        real_concat = torch.cat(real_parts, dim=-1)  # (B, N, M + 3*K*K)
        imag_concat = torch.cat(imag_parts, dim=-1)  # (B, N, M + 3*K*K)
        
        kp_features = torch.stack([real_concat, imag_concat], dim=-1)  # (B, N, total_features, 2)
        
        return kp_features

    def count_flops(self) -> int:
        """
        Approximate FLOPs estimation for KPConvModule.

        FLOPs are counted for:
        - one output sample (one sequence)

        Notes
        -----

        Assumptions:
        - add/sub/mul      : 1 FLOP
        - sqrt             : 10 FLOPs
        - Hardswish        : 6 FLOPs (approx)
        - Conv1d           : 2 * Cin * Cout * K
        - elementwise mul  : 1 FLOP

        Returns
        -------
        int: FLOPs per one output sample
        """

        M = self.M
        K = self.K

        # FEATURE SPLIT

        # x_re^2 + x_im^2
        mag2_flops = 2  # mul + add

        # sqrt(mag2)
        abs1_flops = 10

        # abs2 = magnitude2
        abs2_flops = 0

        # abs3 = abs1 * magnitude2
        abs3_flops = 1

        split_flops = mag2_flops + abs1_flops + abs2_flops + abs3_flops

        # CONV1D PROJECTIONS

        # conv_reim: (2 -> 2K, kernel M)
        conv_reim = 2 * 2 * K * M

        # conv_abs1: (1 -> K)
        conv_abs1 = 2 * 1 * K * M

        # conv_abs2
        conv_abs2 = 2 * 1 * K * M

        # conv_abs3
        conv_abs3 = 2 * 1 * K * M

        conv_total = conv_reim + conv_abs1 + conv_abs2 + conv_abs3

        # activation Hardswish (4 conv outputs)
        # channels: 2K + 3K = 5K
        activation_flops = 6 * (5 * K)

        # KRONECKER PRODUCTS

        # each kron: (B, N, K) ⊗ (B, N, K)
        # elementwise multiplication + reshape

        kron_size = K * K

        # 3 magnitudes × 2 (real + imag) = 6 kron products
        kron_flops = 6 * kron_size

        # CONCAT OPERATIONS

        # concatenation is free (no FLOPs)

        # FINAL OUTPUT ASSEMBLY

        # stacking real/imag
        stack_flops = 0

        total_flops = (
            split_flops
            + conv_total
            + activation_flops
            + kron_flops
            + stack_flops
        )

        return int(total_flops)

    def count_macs(self, sequence_length: int = 1) -> int:
        """
        Approximate MACs estimation for KPConvModule.

        Counts only:
        - multiplications + accumulations in Conv1d

        Does NOT count:
        - activations (Hardswish)
        - sqrt / arithmetic feature engineering
        - concat / transpose / reshape / split
        - bias

        Notes
        -----
        Complexity is estimated for:
        - one forward pass
        - one sample
        - full sequence length

        Parameters
        ----------
        sequence_length : int
            Temporal length N.

        Returns
        -------
        int
            MACs per forward pass.
        """

        N = sequence_length
        M = self.M
        K = self.K

        total_mac = 0

        # CONV BLOCKS
        #
        # MACs Conv1d:
        # out_channels * in_channels * kernel_size * N
        #

        # conv_reim: (2 -> 2K)
        mac_reim = (
            N
            * (2 * K)
            * 2
            * M
        )

        # conv_abs1: (1 -> K)
        # conv_abs2: (1 -> K)
        # conv_abs3: (1 -> K)

        mac_abs = (
            3
            * (
                N
                * K
                * 1
                * M
            )
        )

        total_mac += mac_reim + mac_abs

        # FEATURE ENGINEERING + KRONECKER
        #
        # ignored:
        # - sqrt
        # - magnitude
        # - reshape/split/transpose
        #
        # KRONECKER:
        # 3 pairs × (real + imag) = 6 tensors
        # each: K×K multiplications per timestep

        mac_kron = 6 * N * (K * K)

        total_mac += mac_kron

        return int(total_mac)