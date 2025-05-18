import torch
import torch.nn as nn
import torch.nn.functional as F
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
    
    def number_parameters(self):
        num_gmp = self.Ka * self.La + self.Kb * self.Lb * self.Mb + self.Kc * self.Lc * self.Mc
        num_ar = self.Dy
        num_frac = self.P_fd + 1 + 1
        num_params = num_gmp + num_ar + num_frac
        return num_params

    def forward(self, x):
        N = len(x)

        y_gmp = super().forward(x)
        
        # 2) дробные задержки: единый conv1d батч
        deltas = self.deltas.view(-1, 1)            # (P_fd,1)
        kernels = torch.stack([1 - deltas, deltas], dim=1)       # (P_fd,2)
        kernels = kernels.view(self.P_fd,1,2)                        # (P_fd,1,2)

        x_real = x.real.view(1,1,N)
        x_img  = x.imag.view(1,1,N)
        x_fd_real = F.conv1d(x_real, kernels, padding=0)  # (P_fd,1,N-1)
        x_fd_img  = F.conv1d(x_img,  kernels, padding=0)
        # паддим один раз:
        x_fd_real = F.pad(x_fd_real, (0,1))
        x_fd_img = F.pad(x_fd_img, (0,1))

        x_fd = (x_fd_real + 1j * x_fd_img)[:, 0, :]  # shape: (P_fd, N)


        # 3) батчевое вычисление gmp-термов
        # соберем (P_fd+1, N) сигналы: первая строка исходный x
        batch_x = torch.cat([x.unsqueeze(0), x_fd], dim=0)     # (P_fd+1, N)
        # применяем _sum_terms к батчу
        # реализуем батч версию внутри forward
        indices = torch.arange(N).unsqueeze(0)  # (1,N)
        # отложим аккумуляцию
        y_total = y_gmp
        for bx in batch_x:
            y_total = y_total + self._sum_terms(bx, N, indices)
        y_ar = self._compute_autoregression(N, y_total, use_smoothed_ar=True)

        y = y_gmp + y_ar
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


    def _compute_autoregression(self, N, y_gmp, use_smoothed_ar=True):
        if y_gmp.shape[0] < self.Dy:
            raise ValueError(f"Для Dy={self.Dy} требуется хотя бы {self.Dy+1} отсчетов в y_gmp.")

        y_ar = torch.zeros_like(y_gmp)

        if use_smoothed_ar:
            # === Смягчённая авторегрессия (экспоненциальная память) ===
            alpha = torch.sigmoid(self.logit_alpha) if hasattr(self, 'logit_alpha') else 0.9  # по умолчанию

            s_list = [torch.zeros_like(y_gmp[0])]
            for n in range(1, N):
                s_new = alpha * s_list[-1] + (1 - alpha) * y_gmp[n - 1]
                s_list.append(s_new)

            s = torch.stack(s_list)

            # y_ar = d_s * s (где d_s — обучаемый параметр)
            d_s = self.d_s if hasattr(self, 'd_s') else 0.001  # по умолчанию
            y_ar = d_s * s

        else:
            # === Классическая AR-модель ===
            X_ar = torch.stack([y_gmp[self.Dy - j - 1:N - j - 1] for j in range(self.Dy)], dim=1)
            y_ar_part = X_ar @ self.d
            y_ar[self.Dy:] = y_ar_part

        return y_ar

    
    def _frac_delay(self, x, delta):
        N = x.shape[0]
        k = int(torch.floor(torch.tensor(delta)).item())
        alpha = delta - k

        idx_0 = torch.arange(N) - k
        idx_1 = idx_0 - 1

        idx_0 = idx_0.clamp(0, N - 1)
        idx_1 = idx_1.clamp(0, N - 1)

        x0 = x[idx_0]
        x1 = x[idx_1]

        return (1 - alpha) * x0 + alpha * x1





