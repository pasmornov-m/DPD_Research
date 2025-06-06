import torch
import numpy as np

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