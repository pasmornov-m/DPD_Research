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
        self.coeffs_b = self.b.unsqueeze(-1)
        self.coeffs_c = self.c.unsqueeze(-1)

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
        N = len(x)
        indices_N = torch.arange(N).unsqueeze(0)
        y = self._sum_terms(x, N, indices_N)
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

            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item()}")
    

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


    def _compute_terms(self, coeffs, x, N, indices_N, powers_K, indices_L, indices_M=None, sign=0):

        indices_1 = indices_N - indices_L
        
        if indices_M is None:
            indices_2 = indices_1
        else:
            indices_2 = indices_N.unsqueeze(1) - indices_L.unsqueeze(2) + sign * indices_M

        indices_1 = indices_1.clamp(min=0, max=N-1)
        indices_2 = indices_2.clamp(min=0, max=N-1)

        x_truncated_1 = x[indices_1]
        x_truncated_2 = x[indices_2]

        abs_x_powers = torch.abs(x_truncated_2).unsqueeze(-1) ** powers_K

        if indices_M is None:
            x_scaled = x_truncated_1.unsqueeze(-1) * abs_x_powers
        else:
            x_scaled = x_truncated_1.unsqueeze(1).unsqueeze(-1) * abs_x_powers
        
        term = (coeffs * x_scaled.permute(-1, 0, 1, *([2] if indices_M is not None else []))).sum(dim=tuple(range(len(x_scaled.shape) - 1)))

        return term
    
    def _sum_terms(self, x, N, indices_N):
        y = sum([self._compute_terms(self.coeffs_a, x, N, indices_N, self.powers_Ka, self.indices_La),
        self._compute_terms(self.coeffs_b, x, N, indices_N, self.powers_Kb, self.indices_Lb, self.indices_Mb, sign=-1),
        self._compute_terms(self.coeffs_c, x, N, indices_N, self.powers_Kc, self.indices_Lc, self.indices_Mc, sign=+1)])
        return y
