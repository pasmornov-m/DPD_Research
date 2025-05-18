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
        N = x.shape[0]
        y_gmp = super().forward(x)
                
        # y_total = self._frac_delay(x, y_gmp)

        y_total = y_gmp
        
        y_ar = self._compute_autoregression(N, y_total, use_smoothed_ar=False)

        y = y_total + y_ar
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
        if N < self.Dy:
            raise ValueError(f"Для Dy={self.Dy} требуется хотя бы {self.Dy+1} отсчетов в y_gmp.")

        y_real = y_gmp.real
        y_imag = y_gmp.imag

        if use_smoothed_ar:
            # 1) экспоненциальная память:
            #    s[n] = (1-α) * sum_{k=0..n-1} α^(n-k-1) * y_gmp[k]
            alpha = torch.sigmoid(self.logit_alpha)
            one_minus = 1 - alpha
            # отклик длины N
            h = one_minus * alpha ** torch.arange(N, dtype=y_real.dtype)
            # подготавливаем сигналы (1,1,N) и пададим слева (N-1,0)
            y_r_p = F.pad(y_real.view(1,1,N), (N-1,0))
            y_i_p = F.pad(y_imag.view(1,1,N), (N-1,0))
            # свёрточное ядро (перевёрнутое h)
            kernel = h.flip(0).view(1,1,N)
            # делаем conv1d → (1,1,N)
            s_r = F.conv1d(y_r_p, kernel)
            s_i = F.conv1d(y_i_p, kernel)
            s = (s_r + 1j * s_i).view(N)
            # масштабируем
            y_ar = self.d_s * s

        else:
            # 2) классическая AR через conv1d
            # pad=(Dy,0) чтобы результат тоже длиной N
            pad = (self.Dy, 0)
            y_r_p = F.pad(y_real.view(1,1,N), pad)
            y_i_p = F.pad(y_imag.view(1,1,N), pad)
            # готовим ядра из d (переворачиваем, чтобы conv давал автоковариацию)
            dr = self.d.real.flip(0).view(1,1,self.Dy)
            di = self.d.imag.flip(0).view(1,1,self.Dy)
            # комплексная свёртка: (a+jb)*(c+jd) = (ac - bd) + j(ad + bc)
            ar_r = F.conv1d(y_r_p, dr) - F.conv1d(y_i_p, di)
            ar_i = F.conv1d(y_r_p, di) + F.conv1d(y_i_p, dr)
            y_ar = (ar_r + 1j * ar_i)[..., :N].view(N)

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

        sum_terms_batch = vmap(lambda bx: self._sum_terms(bx, N, indices), in_dims=0, out_dims=0)
        terms = sum_terms_batch(batch_x)
        y_total = y_gmp + terms.sum(dim=0)

        return y_total





