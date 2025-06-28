import torch
from torch import nn
import numpy as np
import functools


def to_torch_tensor(data):
    return data if isinstance(data, torch.Tensor) else torch.tensor(data, dtype=torch.cfloat)

def check_early_stopping(current_loss, best_loss, r_order, epoch_before_break, no_improve_epochs, epoch):
    if round(current_loss, r_order) < round(best_loss, r_order):
        best_loss = current_loss
        no_improve_epochs = 0
    else:
        no_improve_epochs += 1
    if no_improve_epochs >= epoch_before_break:
        print(f"Early stopping at epoch {epoch + 1}")
        return True, best_loss, no_improve_epochs
    return False, best_loss, no_improve_epochs

def moving_average(arr, freqs, fs, window_size):
    psd_smoothed = np.convolve(arr, np.ones(window_size)/window_size, mode='valid')
    f_smoothed = freqs[:len(psd_smoothed)]
    f_smoothed = np.fft.fftshift(np.fft.fftfreq(len(psd_smoothed), d=1/fs))
    return f_smoothed, psd_smoothed

def iq_to_complex(iq_signal):
    i_values = iq_signal[..., 0]
    q_values = iq_signal[..., 1]
    complex_signals = i_values + 1j * q_values
    return complex_signals

def complex_to_iq(complex_signal):
    return torch.view_as_real(complex_signal)


def complex_handler(forward_func):
    """Декоратор для автоматической обработки комплексных чисел в методах forward"""
    @functools.wraps(forward_func)
    def wrapper(self, input_x, *args, **kwargs):
        is_complex = input_x.is_complex()

        if is_complex:
            x = complex_to_iq(input_x)
            x = x.unsqueeze(0)
        else:
            x = input_x
        
        y = forward_func(self, x, *args, **kwargs)
        
        if is_complex:
            y = torch.squeeze(y)
            y = iq_to_complex(y)
        
        return y
    return wrapper

def iq_handler(forward_func):
    @functools.wraps(forward_func)
    def wrapper(self, x, *args, **kwargs):
        is_reshape = (x.ndim >= 3 and x.shape[-1] == 2 and x.dtype == torch.float32)
        if is_reshape:
            x = torch.squeeze(x)
            x_complex = iq_to_complex(x)
        else:
            x_complex = x
        y = forward_func(self, x_complex, *args, **kwargs)
        if is_reshape:
            y = y.unsqueeze(1)
            y = complex_to_iq(y)
        return y
    return wrapper


class NoiseModel():
    def __init__(self, snr, fs, bw):
        self.snr = snr
        self.fs = fs
        self.bw = bw
    
    def __call__(self, signal):
        from modules import metrics
        output = metrics.add_complex_noise(signal, self.snr, self.fs, self.bw)
        return output

def freeze_pa_model(model):
    for param in model.parameters():
        param.requires_grad = False


class CascadeModel(nn.Module):
    def __init__(self, model_1, model_2, gain=None, cascade_type=None):
        super().__init__()
        self.model_1 = model_1
        self.model_2 = model_2
        self.gain = gain
        self.cascade_type = cascade_type

    def forward(self, x):
        x = self.model_1(x)
        if self.cascade_type == "ila" and self.gain:
            x = x / self.gain
        x = self.model_2(x)
        return x