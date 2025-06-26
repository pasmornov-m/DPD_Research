import torch
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
