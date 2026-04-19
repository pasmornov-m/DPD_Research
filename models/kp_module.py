import torch


class KPModule(torch.nn.Module):
    def __init__(self,
                 M: int = 5):
        
        assert M % 2 == 1, "M must be odd"
        
        torch.nn.Module.__init__(self)
        
        self.M = M
        self.K= 5
        self._eps = 1e-10
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
        x_abs1 = torch.sqrt(magnitude2 + self._eps)
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