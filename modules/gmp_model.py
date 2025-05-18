import torch
import torch.nn as nn
import os
from modules.utils import to_torch_tensor, check_early_stopping
from modules.metrics import compute_mse


class GMP(nn.Module):
    def __init__(self, Ka, La, Kb, Lb, Mb, Kc, Lc, Mc, model_type):
        super().__init__()
        self.Ka, self.La = Ka, La
        self.Kb, self.Lb, self.Mb = Kb+1, Lb, Mb+1
        self.Kc, self.Lc, self.Mc = Kc+1, Lc, Mc+1
        self.model_type = model_type

        self.a = torch.nn.Parameter(0.001 * torch.randn((self.Ka, self.La), dtype=torch.cfloat))
        self.b = torch.nn.Parameter(0.001 * torch.randn((self.Kb, self.Lb, self.Mb), dtype=torch.cfloat))
        self.c = torch.nn.Parameter(0.001 * torch.randn((self.Kc, self.Lc, self.Mc), dtype=torch.cfloat))

        self.coeffs_a = self.a.unsqueeze(-1)
        self.coeffs_b = self.b
        self.coeffs_c = self.c

        self.powers_Ka = torch.arange(self.Ka)
        self.powers_Kb = torch.arange(self.Kb)
        self.powers_Kc = torch.arange(self.Kc)

        self.indices_La = torch.arange(self.La).unsqueeze(1)
        self.indices_Lb = torch.arange(self.Lb).unsqueeze(1)
        self.indices_Lc = torch.arange(self.Lc).unsqueeze(1)

        self.indices_Mb = torch.arange(self.Mb).unsqueeze(0).unsqueeze(2)
        self.indices_Mc = torch.arange(self.Mc).unsqueeze(0).unsqueeze(2)

        self.max_length = 100000
        self.register_buffer("arange", torch.arange(self.max_length))

        self.register_buffer('idx_La_full',
            (self.arange.unsqueeze(0) - torch.arange(self.La).unsqueeze(1))
              .clamp(0, self.max_length-1)
        )
        self.register_buffer('idx_Lb_full',
            (self.arange.unsqueeze(0) - torch.arange(self.Lb).unsqueeze(1))
              .clamp(0, self.max_length-1)
        )
        self.register_buffer('idx_Lc_full',
            (self.arange.unsqueeze(0) - torch.arange(self.Lc).unsqueeze(1))
              .clamp(0, self.max_length-1)
        )

        # 3) предрасчёт индексов перекрёстных задержек M
        #    shape (len_Lb, len_Mb, max_length)
        Mb = torch.arange(self.Mb).view(1, -1, 1)
        self.register_buffer('idx_Mb_full',
            (self.arange.view(1, 1, -1) - 
             torch.arange(self.Lb).view(-1,1,1) - 
             Mb
            ).clamp(0, self.max_length-1)
        )
        Mc = torch.arange(self.Mc).view(1, -1, 1)
        self.register_buffer('idx_Mc_full',
            (self.arange.view(1, 1, -1) - 
             torch.arange(self.Lc).view(-1,1,1) + 
             Mc
            ).clamp(0, self.max_length-1)
        )

    
    def number_parameters(self):
        number_of_params = (self.Ka*self.La)+(self.Kb*self.Lb*self.Mb)+(self.Kc*self.Lc*self.Mc)
        return number_of_params

    def forward(self, x):        
        y = self._compute_terms(x)
        return y

    def optimize_coefficients_grad(self, input_data, target_data, epochs=100000, learning_rate=0.01):
        input_data, target_data = map(to_torch_tensor, (input_data, target_data))

        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate, amsgrad=True)
        for epoch in range(epochs):
            optimizer.zero_grad()
            output = self.forward(input_data)
            loss = compute_mse(output, target_data)
            loss.backward()
            optimizer.step()

            if epoch%100==0:
                print(f"Epoch [{epoch}/{epochs}], Loss: {loss.item()}")
    

    def save_coefficients(self, directory="model_params"):
        os.makedirs(directory, exist_ok=True)
        filename = f"{directory}/{self.model_type}_gmp_model_Ka{self.Ka}_La{self.La}_Kb{self.Kb}_Lb{self.Lb}_Mb{self.Mb}_Kc{self.Kc}_Lc{self.Lc}_Mc{self.Mc}.pt"
        torch.save(self.state_dict(), filename)
        print(f"Coefficients saved to {filename}")


    def load_coefficients(self, directory="model_params"):
        filename = f"{directory}/{self.model_type}_gmp_model_Ka{self.Ka}_La{self.La}_Kb{self.Kb}_Lb{self.Lb}_Mb{self.Mb}_Kc{self.Kc}_Lc{self.Lc}_Mc{self.Mc}.pt"
        if os.path.isfile(filename):
            state_dict = torch.load(filename)
            self.load_state_dict(state_dict)
            print(f"Coefficients loaded from {filename}")
            return True
        else:
            print(f"No saved coefficients found at {filename}, initializing new parameters.")
            return False
        
    
    def _compute_terms(self, x):
        N = x.shape[0]

        if N > self.max_length:
            raise ValueError(f"Длина {N} превышает max_length={self.max_length}")
        
        idx_La = self.idx_La_full[:, :N]     # (La, N)
        idx_Lb = self.idx_Lb_full[:, :N]     # (Lb, N)
        idx_Lc = self.idx_Lc_full[:, :N]     # (Lc, N)

        idx_Ma = idx_La
        idx_Mb = self.idx_Mb_full[:, :, :N]
        idx_Mc = self.idx_Mc_full[:, :, :N].clamp(0, N - 1)

        # 5) выборки сигнала по задержкам
        x_La = x[idx_La]      # (La, N)
        x_Lb = x[idx_Lb]      # (Lb, N)
        x_Lc = x[idx_Lc]      # (Lc, N)

        x_Ma = x[idx_Ma]
        x_Mb = x[idx_Mb]
        x_Mc = x[idx_Mc]

        # indices_delayed_La = (indices_N - self.indices_La).clamp(min=0, max=N-1)
        # indices_delayed_Lb = (indices_N - self.indices_Lb).clamp(min=0, max=N-1)
        # indices_delayed_Lc = (indices_N - self.indices_Lc).clamp(min=0, max=N-1)

        # indices_delayed_Ma = (indices_delayed_La).clamp(min=0, max=N-1)
        # indices_delayed_Mb = (indices_N.unsqueeze(1) - self.indices_Lb.unsqueeze(2) - self.indices_Mb).clamp(min=0, max=N-1)
        # indices_delayed_Mc = (indices_N.unsqueeze(1) - self.indices_Lc.unsqueeze(2) + self.indices_Mc).clamp(min=0, max=N-1)

        # x_truncated_1a = x[indices_delayed_La]
        # x_truncated_1b = x[indices_delayed_Lb]
        # x_truncated_1c = x[indices_delayed_Lc]

        # x_truncated_2a = x[indices_delayed_Ma]
        # x_truncated_2b = x[indices_delayed_Mb]
        # x_truncated_2c = x[indices_delayed_Mc]

        abs_powers_a = (torch.abs(x_Ma).unsqueeze(-1) ** self.powers_Ka).type_as(x_La)
        abs_powers_b = (torch.abs(x_Mb).unsqueeze(-1) ** self.powers_Kb).type_as(x_Lb)
        abs_powers_c = (torch.abs(x_Mc).unsqueeze(-1) ** self.powers_Kc).type_as(x_Lc)

        x_scaled_a = x_La.unsqueeze(-1) * abs_powers_a
        x_scaled_b = x_Lb.unsqueeze(1).unsqueeze(-1) * abs_powers_b
        x_scaled_c = x_Lc.unsqueeze(1).unsqueeze(-1) * abs_powers_c

        term_a = torch.einsum('kln,lnk->n', self.coeffs_a, x_scaled_a)
        term_b = torch.einsum('klm,lmnk->n', self.coeffs_b, x_scaled_b)
        term_c = torch.einsum('klm,lmnk->n', self.coeffs_c, x_scaled_c)

        y = sum([term_a, term_b, term_c])
        
        return y
