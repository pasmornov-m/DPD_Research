import torch
import os
from modules import utils
from modules.utils import to_torch_tensor, check_early_stopping
from modules.metrics import compute_mse
from modules.gmp_model import GMP


class GMP_AR(GMP):
    def __init__(self, Ka, La, Kb, Lb, Mb, Kc, Lc, Mc, Dy, model_name=None):
        super().__init__(Ka, La, Kb, Lb, Mb, Kc, Lc, Mc, model_name)
        self.Dy = Dy
        self.alpha = torch.nn.Parameter(0.001 * torch.randn(Dy))

        self.d = torch.nn.Parameter(0.001 * torch.randn(self.Dy, dtype=torch.cfloat))
        self.logit_weights = torch.nn.Parameter(torch.tensor([0.01, 0.01]))
        self.ar_weights = torch.nn.Parameter(torch.tensor([0.01, 0.01]))
    
    def count_params(self):
        num_gmp = self.Ka * self.La + self.Kb * self.Lb * self.Mb + self.Kc * self.Lc * self.Mc
        num_ar = self.Dy
        num_params = num_gmp + num_ar
        return num_params

    @utils.iq_handler
    def forward(self, x):
        w_gmp, w_ar = torch.softmax(self.logit_weights, dim=0)
        y_gmp = super().forward(x)
        y_ar = self._compute_autoregression(y_gmp)
        y = w_gmp * y_gmp + w_ar * y_ar
        return y

    # сохранение коэффициентов
    def save_weights(self, directory="model_params"):
        os.makedirs(directory, exist_ok=True)
        filename = f"{directory}/{self.model_name}_ar_gmp_model_Ka{self.Ka}_La{self.La}_Kb{self.Kb}_Lb{self.Lb}_Mb{self.Mb}_Kc{self.Kc}_Lc{self.Lc}_Mc{self.Mc}_Dy{self.Dy}.pt"
        torch.save(self.state_dict(), filename)
        print(f"Coefficients saved to {filename}")


    # загрузка коэффициентов из файла
    def load_weights(self, directory="model_params"):
        filename = f"{directory}/{self.model_name}_ar_gmp_model_Ka{self.Ka}_La{self.La}_Kb{self.Kb}_Lb{self.Lb}_Mb{self.Mb}_Kc{self.Kc}_Lc{self.Lc}_Mc{self.Mc}_Dy{self.Dy}.pt"
        if os.path.isfile(filename):
            print(filename)
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





