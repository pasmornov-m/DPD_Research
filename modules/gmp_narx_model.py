import torch
import os
from modules.utils import to_torch_tensor, check_early_stopping
from modules.metrics import compute_mse
from modules.gmp_model import GMP


class GMP_NARX(GMP):
    def __init__(self, Ka, La, Kb, Lb, Mb, Kc, Lc, Mc, Dy, model_type):
        super().__init__(Ka, La, Kb, Lb, Mb, Kc, Lc, Mc, model_type)
        self.Dy = Dy

        self.d = torch.nn.Parameter(0.001 * torch.randn(self.Dy, dtype=torch.cfloat))
        self.logit_weights = torch.nn.Parameter(torch.tensor([0.01, 0.01]))
    
    def number_parameters(self):
        num_gmp = self.Ka * self.La + self.Kb * self.Lb * self.Mb + self.Kc * self.Lc * self.Mc
        num_ar = self.Dy
        num_params = num_gmp + num_ar
        return num_params

    def forward(self, x):
        w_gmp, w_ar = torch.softmax(self.logit_weights, dim=0)
        y_gmp = super().forward(x)
        y_ar = self._compute_autoregression(y_gmp)
        y = w_gmp * y_gmp + w_ar * y_ar
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
    

    # сохранение коэффициентов
    def save_coefficients(self, directory="model_params"):
        os.makedirs(directory, exist_ok=True)
        filename = f"{directory}/{self.model_type}_narx_gmp_model_Ka{self.Ka}_La{self.La}_Kb{self.Kb}_Lb{self.Lb}_Mb{self.Mb}_Kc{self.Kc}_Lc{self.Lc}_Mc{self.Mc}_Dy{self.Dy}.pt"
        torch.save(self.state_dict(), filename)
        print(f"Coefficients saved to {filename}")


    # загрузка коэффициентов из файла
    def load_coefficients(self, directory="model_params"):
        filename = f"{directory}/{self.model_type}_narx_gmp_model_Ka{self.Ka}_La{self.La}_Kb{self.Kb}_Lb{self.Lb}_Mb{self.Mb}_Kc{self.Kc}_Lc{self.Lc}_Mc{self.Mc}_Dy{self.Dy}.pt"
        if os.path.isfile(filename):
            self.load_state_dict(torch.load(filename))
            print(f"Coefficients loaded from {filename}")
            return True
        else:
            print(f"No saved coefficients found at {filename}, initializing new parameters.")
            return False


    def _compute_autoregression(self, y_gmp):
        N = y_gmp.shape[0]
        if N < self.Dy:
            raise ValueError(f"Для Dy={self.Dy} требуется хотя бы {self.Dy+1} отсчетов в y_gmp.")

        X_ar = y_gmp.unfold(0, self.Dy, 1)
        y_ar_part = X_ar @ self.d

        y_ar = torch.zeros_like(y_gmp)
        y_ar[self.Dy:] = y_ar_part[:-1]

        return y_ar





