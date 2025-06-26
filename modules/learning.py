import torch
from torch import nn
from torch.optim.optimizer import Optimizer
from torch.utils.data import Dataset, DataLoader
from typing import Callable
from modules.utils import to_torch_tensor, complex_to_iq
from modules.metrics import compute_mse, add_complex_noise

def optimize_dla_grad(input_data, target_data, dpd_model, pa_model, epochs=100000, learning_rate=1e-3, 
                      add_noise=False, snr=None, fs=None, bw=None):
    input_data, target_data = map(to_torch_tensor, (input_data, target_data))

    optimizer = torch.optim.Adam(dpd_model.parameters(), lr=learning_rate, amsgrad=True)
    for param in pa_model.parameters():
        param.requires_grad = False
            
    for epoch in range(epochs):
        optimizer.zero_grad()
        # print(f"input_data.shape: {input_data.shape}")
        dpd_output = dpd_model.forward(input_data)
        # print(f"dpd_output.shape: {dpd_output.shape}")
        pa_output = pa_model.forward(dpd_output)
        # print(f"pa_output.shape: {pa_output.shape}")

        if add_noise:
            if snr is None or fs is None or bw is None:
                raise ValueError("SNR, fs, and bw must be provided when add_noise=True")
            pa_output = add_complex_noise(pa_output, snr, fs, bw)

        loss = compute_mse(pa_output, target_data)
        loss.backward()
        optimizer.step()

        if epoch%100==0:
                print(f"Epoch [{epoch}/{epochs}], Loss: {loss.item()}")

    print("DLA optimization completed.")


def optimize_ila_grad(dpd_model, input_data, output_data, gain, epochs=100000, learning_rate=0.01, pa_model=None, 
                      add_noise=False, snr=None, fs=None, bw=None):
    input_data = to_torch_tensor(input_data)
    
    if pa_model:
        print("Compute on pa_model")
        output_data = (pa_model.forward(input_data) / gain).detach()
    else:
        print("Compute on presaved output_data")
        output_data = to_torch_tensor(output_data) / gain

    optimizer = torch.optim.Adam(dpd_model.parameters(), lr=learning_rate, amsgrad=True)
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = dpd_model.forward(output_data)
        if add_noise:
            if snr is None or fs is None or bw is None:
                raise ValueError("SNR, fs, and bw must be provided when add_noise=True")
            output = add_complex_noise(output, snr, fs, bw)
        loss = compute_mse(output, input_data)
        loss.backward()
        optimizer.step()

        if epoch%100==0:
            print(f"Epoch [{epoch}/{epochs}], Loss: {loss.item()}")

    print("ILA-DPD Training Complete.")


def ilc_signal_grad(input_data, target_data, pa_model, epochs=1000000, learning_rate=0.1, 
                    add_noise=False, snr=None, fs=None, bw=None):
    input_data, target_data = map(to_torch_tensor, (input_data, target_data))
    u = torch.nn.Parameter(input_data.clone(), requires_grad=True)

    optimizer = torch.optim.AdamW([u], lr=learning_rate, amsgrad=True)
    for param in pa_model.parameters():
        param.requires_grad = False

    for epoch in range(epochs):
        optimizer.zero_grad()
        pa_output = pa_model.forward(u)

        if add_noise:
            if snr is None or fs is None or bw is None:
                raise ValueError("SNR, fs, and bw must be provided when add_noise=True")
            pa_output = add_complex_noise(pa_output, snr, fs, bw)

        loss = compute_mse(pa_output, target_data)
        loss.backward()
        optimizer.step()

        if epoch%100==0:
            print(f"Epoch [{epoch}/{epochs}], Loss: {loss.item()}")

    return u.detach()

# def model_train(model: nn.Module,
#               dataloader: DataLoader,
#               optimizer: Optimizer,
#               criterion: Callable):
#     model.train()
#     losses = []
#     for features, targets in dataloader:
#         optimizer.zero_grad()
#         out = model(features)
#         loss = criterion(out, targets)
#         loss.backward()
#         optimizer.step()
#         loss.detach()
#         losses.append(loss.item())
#     loss = sum(losses) / len(losses)
#     return loss

# def model_eval(model: nn.Module,
#              dataloader: DataLoader,
#              criterion: Callable,
#              metric_criterion=None):
#     model.eval()
#     with torch.no_grad():
#         losses = []
#         if metric_criterion:
#             metric_losses = []
#         prediction = []
#         for features, targets in dataloader:
#             outputs = model(features)
#             loss = criterion(outputs, targets)
#             if metric_criterion:
#                 metric_loss = metric_criterion(outputs, targets)
#                 metric_losses.append(metric_loss.item())
#             prediction.append(outputs)
#             losses.append(loss.item())
#     avg_loss = sum(losses) / len(losses)
#     if metric_criterion:
#         avg_metric_loss = sum(metric_losses) / len(metric_losses)
#     prediction = torch.cat(prediction, dim=0)
#     if metric_criterion:
#         return avg_loss, avg_metric_loss
#     return avg_loss


# def model_inference(model: nn.Module, input_data: torch.Tensor):
#     model.eval()
#     with torch.no_grad():
#         predict = model(input_data)
#     return predict


# def model_train(model: nn.Module,
#               input_data: torch.Tensor,
#               target_data: torch.Tensor,
#               optimizer: Optimizer,
#               criterion: Callable):
#     model.train()
#     optimizer.zero_grad()
#     out = model(input_data)
#     loss = criterion(out, target_data)
#     loss.backward()
#     optimizer.step()
#     loss.detach()
#     return loss

# def model_eval(model: nn.Module,
#              input_data: torch.Tensor,
#              target_data: torch.Tensor,
#              criterion: Callable,
#              metric_criterion=None):
#     model.eval()
#     with torch.no_grad():
#         outputs = model(input_data)
#         loss = criterion(outputs, target_data)
#         if metric_criterion:
#             metric_loss = metric_criterion(outputs, target_data)
#     if metric_criterion:
#         return loss, metric_loss
#     return loss