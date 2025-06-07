import torch
import torch.nn as nn
import os
from modules.utils import to_torch_tensor, check_early_stopping
from modules.metrics import compute_mse


class GMP(nn.Module):
    def __init__(self, Ka, La, Kb, Lb, Mb, Kc, Lc, Mc, model_type=None):
        super().__init__()
        self.Ka, self.La = Ka, La
        self.Kb, self.Lb, self.Mb = Kb, Lb, Mb
        self.Kc, self.Lc, self.Mc = Kc, Lc, Mc
        self.model_type = model_type

        self.a = torch.nn.Parameter(0.001 * torch.randn((self.Ka, self.La), dtype=torch.cfloat))
        self.b = torch.nn.Parameter(0.001 * torch.randn((self.Kb, self.Lb, self.Mb), dtype=torch.cfloat))
        self.c = torch.nn.Parameter(0.001 * torch.randn((self.Kc, self.Lc, self.Mc), dtype=torch.cfloat))

        self.powers_Ka = torch.arange(self.Ka)
        self.powers_Kb = torch.arange(self.Kb)
        self.powers_Kc = torch.arange(self.Kc)

        self.indices_La = torch.arange(self.La).unsqueeze(1)
        self.indices_Lb = torch.arange(self.Lb).unsqueeze(1)
        self.indices_Lc = torch.arange(self.Lc).unsqueeze(1)

        self.indices_Mb = torch.arange(self.Mb).unsqueeze(0).unsqueeze(2)
        self.indices_Mc = torch.arange(self.Mc).unsqueeze(0).unsqueeze(2)
    
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
        indices_N = torch.arange(N).unsqueeze(0)

        indices_delayed_La = (indices_N - self.indices_La).clamp(min=0, max=N-1)
        indices_delayed_Lb = (indices_N - self.indices_Lb).clamp(min=0, max=N-1)
        indices_delayed_Lc = (indices_N - self.indices_Lc).clamp(min=0, max=N-1)

        indices_delayed_Ma = indices_delayed_La
        indices_delayed_Mb = (indices_N.unsqueeze(1) - self.indices_Lb.unsqueeze(2) - self.indices_Mb).clamp(min=0, max=N-1)
        indices_delayed_Mc = (indices_N.unsqueeze(1) - self.indices_Lc.unsqueeze(2) + self.indices_Mc).clamp(min=0, max=N-1)

        x_truncated_1a = x[indices_delayed_La]
        x_truncated_1b = x[indices_delayed_Lb]
        x_truncated_1c = x[indices_delayed_Lc]

        x_truncated_2a = x[indices_delayed_Ma]
        x_truncated_2b = x[indices_delayed_Mb]
        x_truncated_2c = x[indices_delayed_Mc]

        abs_powers_a = (torch.abs(x_truncated_2a).unsqueeze(-1) ** self.powers_Ka).to(x.dtype)
        abs_powers_b = (torch.abs(x_truncated_2b).unsqueeze(-1) ** self.powers_Kb).to(x.dtype)
        abs_powers_c = (torch.abs(x_truncated_2c).unsqueeze(-1) ** self.powers_Kc).to(x.dtype)

        x_scaled_a = x_truncated_1a.unsqueeze(-1) * abs_powers_a
        x_scaled_b = x_truncated_1b.unsqueeze(1).unsqueeze(-1) * abs_powers_b
        x_scaled_c = x_truncated_1c.unsqueeze(1).unsqueeze(-1) * abs_powers_c

        term_a = torch.einsum('kln,lnk->n', self.a.unsqueeze(-1), x_scaled_a)
        term_b = torch.einsum('klm,lmnk->n', self.b, x_scaled_b)
        term_c = torch.einsum('klm,lmnk->n', self.c, x_scaled_c)

        y = term_a + term_b + term_c
        
        return y
