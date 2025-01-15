import torch
import os
from modules.utils import to_torch_tensor, check_early_stopping
from modules.metrics import compute_mse

class GeneralizedMemoryPolynomial:
    def __init__(self, Ka, Ma, Kb, Mb, Kc, Mc, P, Q, model_type):
        self.Ka, self.Ma = Ka, Ma
        self.Kb, self.Mb = Kb, Mb
        self.Kc, self.Mc = Kc, Mc
        self.P, self.Q = P, Q
        self.model_type = model_type

        self.am = torch.nn.Parameter(0.01 * torch.randn((self.Ma + 1, self.Ka), dtype=torch.cfloat, requires_grad=True))
        self.bm = torch.nn.Parameter(0.01 * torch.randn((self.Mb + 1, self.Kb, self.P), dtype=torch.cfloat, requires_grad=True))
        self.cm = torch.nn.Parameter(0.01 * torch.randn((self.Mc + 1, self.Kc, self.Q), dtype=torch.cfloat, requires_grad=True))
    
    def compute_output(self, iq_data):
        iq_data = to_torch_tensor(iq_data)

        N = len(iq_data)
        yGMP = torch.zeros(N, dtype=torch.cfloat, device=iq_data.device)

        for m in range(self.Ma + 1):
            max_len = N - m
            if max_len > 0:
                terms = (
                    self.am[m, :self.Ka].unsqueeze(0) * iq_data[:max_len].unsqueeze(1) *
                    torch.abs(iq_data[:max_len].unsqueeze(1)) ** torch.arange(self.Ka, device=iq_data.device)
                )
                yGMP[:max_len] += terms.sum(dim=1)

        for m in range(self.Mb + 1):
            for p in range(1, self.P + 1):
                max_len = N - (m + p)
                if max_len > 0:
                    terms = (
                        self.bm[m, :self.Kb, p - 1].unsqueeze(0) * iq_data[:max_len].unsqueeze(1) *
                        torch.abs(iq_data[p:p + max_len].unsqueeze(1)) ** torch.arange(self.Kb, device=iq_data.device)
                    )
                    yGMP[:max_len] += terms.sum(dim=1)

        for m in range(self.Mc + 1):
            for q in range(1, self.Q + 1):
                max_len = N - (m + q)
                if max_len > 0:
                    terms = (
                        self.cm[m, :self.Kc, q - 1].unsqueeze(0) * iq_data[q:q + max_len].unsqueeze(1) *
                        torch.abs(iq_data[:max_len].unsqueeze(1)) ** torch.arange(self.Kc, device=iq_data.device)
                    )
                    yGMP[:max_len] += terms.sum(dim=1)

        return yGMP

    def optimize_coefficients_grad(self, input_data, target_data, epochs=100000, learning_rate=0.01):
        input_data, target_data = map(to_torch_tensor, (input_data, target_data))
        best_loss = float('inf')
        no_improve_epochs = 0
        epoch_before_break = 25
        r_order = 7
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate, eps = 1e-16, betas=(0.9, 0.999), weight_decay=1e-2, amsgrad=True)                
        for epoch in range(epochs):
            optimizer.zero_grad()
            output = self.compute_output(input_data)
            loss = compute_mse(output, target_data)
            loss.backward()
            optimizer.step()
            current_loss = loss.item()

            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {current_loss}")
            self.save_coefficients()

            stop, best_loss, no_improve_epochs = check_early_stopping(current_loss, best_loss, r_order, epoch_before_break, no_improve_epochs, epoch)
            if stop:
                break

    def parameters(self):
        return [self.am, self.bm, self.cm]        
    
    def save_coefficients(self, directory="model_params"):
        os.makedirs(directory, exist_ok=True)
        filename = f"{directory}/{self.model_type}_gmp_model_Ka{self.Ka}_Ma{self.Ma}_Kb{self.Kb}_Mb{self.Mb}_Kc{self.Kc}_Mc{self.Mc}_P{self.P}_Q{self.Q}.pt"
        torch.save({
                'am': self.am,
                'bm': self.bm,
                'cm': self.cm
            }, filename)
        print(f"Coefficients saved to {filename}")

    def load_coefficients(self, directory="model_params"):
        filename = f"{directory}/{self.model_type}_gmp_model_Ka{self.Ka}_Ma{self.Ma}_Kb{self.Kb}_Mb{self.Mb}_Kc{self.Kc}_Mc{self.Mc}_P{self.P}_Q{self.Q}.pt"
        if os.path.isfile(filename):
            checkpoint = torch.load(filename)
            self.am = torch.nn.Parameter(checkpoint['am'])
            self.bm = torch.nn.Parameter(checkpoint['bm'])
            self.cm = torch.nn.Parameter(checkpoint['cm'])
            print(f"Coefficients loaded from {filename}")
            return True
        else:
            print(f"No saved coefficients found at {filename}, initializing new parameters.")
            return False
    