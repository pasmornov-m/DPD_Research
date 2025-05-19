import torch
import torch.nn.functional as F
from torch.func import vmap
import os
from modules.utils import to_torch_tensor, check_early_stopping
from modules.metrics import compute_mse
from modules.gmp_model import GMP


class NARX_FracDelays_GMP(GMP):
    def __init__(self, Ka, La, Kb, Lb, Mb, Kc, Lc, Mc, Dy, P_fd, model_type):
        super().__init__(Ka, La, Kb, Lb, Mb, Kc, Lc, Mc, model_type)
        self.Dy = Dy
        self.P_fd = P_fd

        self.d = torch.nn.Parameter(0.001 * torch.randn(self.Dy, dtype=torch.cfloat))
        self.deltas = torch.nn.Parameter(0.5 * torch.ones(P_fd, dtype=torch.float))
        self.logit_alpha = torch.nn.Parameter(torch.logit(torch.tensor(0.9)))
        self.d_s = torch.nn.Parameter(torch.tensor(0.001, dtype=torch.float))
        self.logit_weights = torch.nn.Parameter(torch.tensor([0.01, 0.01]))

        deltas_transform = self.deltas.view(-1, 1)
        kernels = torch.stack([1 - deltas_transform, deltas_transform], dim=1)
        self.kernels = kernels.view(self.P_fd,1,2)
    
    def number_parameters(self):
        num_gmp = self.Ka * self.La + self.Kb * self.Lb * self.Mb + self.Kc * self.Lc * self.Mc
        num_ar = self.Dy
        num_frac = self.P_fd + 1 + 1
        num_params = num_gmp + num_ar + num_frac
        return num_params

    def forward(self, x):
        w_gmp, w_ar = torch.softmax(self.logit_weights, dim=0)
        y_gmp = super().forward(x)
        # y_frac = self._frac_delay(x, y_gmp)
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
        filename = f"{directory}/{self.model_type}_narx_fc_delays_gmp_model_Ka{self.Ka}_La{self.La}_Kb{self.Kb}_Lb{self.Lb}_Mb{self.Mb}_Kc{self.Kc}_Lc{self.Lc}_Mc{self.Mc}_Dy{self.Dy}_P{self.P_fd}.pt"
        torch.save(self.state_dict(), filename)
        print(f"Coefficients saved to {filename}")


    # загрузка коэффициентов из файла
    def load_coefficients(self, directory="model_params"):
        filename = f"{directory}/{self.model_type}_narx_fc_delays_gmp_model_Ka{self.Ka}_La{self.La}_Kb{self.Kb}_Lb{self.Lb}_Mb{self.Mb}_Kc{self.Kc}_Lc{self.Lc}_Mc{self.Mc}_Dy{self.Dy}_P{self.P_fd}.pt"
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

    
    def _frac_delay(self, x, y_gmp):
        N = x.shape[0]
        x_real = x.real.view(1,1,N)
        x_img  = x.imag.view(1,1,N)
        x_fd_real = F.conv1d(x_real, self.kernels, padding=0)
        x_fd_img  = F.conv1d(x_img,  self.kernels, padding=0)

        x_fd_real = F.pad(x_fd_real, (0,1))
        x_fd_img = F.pad(x_fd_img, (0,1))

        x_fd = (x_fd_real + 1j * x_fd_img)
        x_fd = x_fd[:, 0, :]

        batch_x = torch.cat([x.unsqueeze(0), x_fd], dim=0)
        indices = torch.arange(N).unsqueeze(0)

        # y_total = y_gmp
        # for bx in batch_x:
        #     y_total = y_total + self._sum_terms(bx, N, indices)

        sum_terms_batch = vmap(lambda bx: self._compute_terms(bx), in_dims=0, out_dims=0)
        terms = sum_terms_batch(batch_x)
        y_total = y_gmp + terms.sum(dim=0)

        return y_total





