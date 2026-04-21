import torch
from torch import nn
import numpy as np
import functools
import time
from datetime import timedelta


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

def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False

def safe_torch_log(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Computes a stable log by returning torch.log(x + eps) to avoid log(0).
    """
    return torch.log(x + eps)

def clip_func(x: torch.Tensor, to: float = 8) -> torch.Tensor:
        return torch.clamp(x,  min=-float(to), max=float(to))

def cut_array(M, *arrays):
    return tuple(arr[M:] for arr in arrays)


def complex_handler(forward_func):
    """Декоратор для автоматической обработки комплексных чисел в методах forward"""
    @functools.wraps(forward_func)
    def wrapper(self, inputs, *args, **kwargs):
        is_complex = inputs.is_complex()
        add_batch_dim = (inputs.dim() == 2)

        if is_complex:
            inputs = complex_to_iq(inputs)
            inputs = inputs.unsqueeze(0)
            
        if add_batch_dim:
            inputs = inputs.unsqueeze(0)
        
        result = forward_func(self, inputs, *args, **kwargs)
        
        if result is None:
            return
        
        if is_complex:
            result = torch.squeeze(result)
            result = iq_to_complex(result)
            
        if add_batch_dim:
            result = result.squeeze(0)
        
        return result
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


def complex_handler_np(func):
    @functools.wraps(func)
    def wrapper(self, inputs, outputs=None, *args, **kwargs):
        is_complex = inputs.is_complex() or (outputs.is_complex() if outputs is not None else False)
        is_torch_in  = isinstance(inputs, torch.Tensor)
        is_torch_out = isinstance(outputs, torch.Tensor) if outputs is not None else False
        is_torch = is_torch_in or is_torch_out
        
        if is_complex:
            inputs = complex_to_iq(inputs)
            outputs = complex_to_iq(outputs) if outputs is not None else None
        else:
            inputs = inputs
            outputs = outputs if outputs is not None else None
        
        if is_torch:
            dtype = inputs.dtype
            inputs = inputs.detach().cpu().numpy()
            outputs = outputs.detach().cpu().numpy() if outputs is not None else None
        else:
            inputs = np.array(inputs)
            outputs = np.array(outputs) if outputs is not None else None
            
        result = func(self, inputs, outputs, *args, **kwargs)
        
        if is_torch:
            result = to_torch_tensor(result)
        
        if is_complex:
            result = iq_to_complex(result)
        
        return result
    return wrapper


def timer_decorator(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        print(f"Выполнение заняло: {timedelta(seconds=round(elapsed))}")
        return result
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


class CascadeModel(nn.Module):
    def __init__(self, model_1, model_2, gain=None, cascade_type=None, normalizer=None):
        super().__init__()
        self.model_1 = model_1
        self.model_2 = model_2
        self.gain = gain
        self.cascade_type = cascade_type
        self.normalizer = normalizer

    def forward(self, x, deterministic=None):
        if deterministic:
            x = self.model_1(x, deterministic=True)
        else:
            x = self.model_1(x)
        
        if self.normalizer:
            x = self.normalizer.inverse_transform(x)
            
        # if self.cascade_type == "ila" and self.gain:
        #     x = x / self.gain
        x = self.model_2(x)
        
        return x


class Normalizer:
    def __init__(self, method="standard"):
        """
        Args:
            method (str): 'minmax' для масштабирования в [0, 1], 
                          'standard' для стандартизации (mean=0, std=1)
                          'power' для нормализация мощности E[|x|^2]=1
        """
        assert method in ("minmax", "standard", "power", None), \
            "Метод должен быть 'minmax', 'standard', 'power' или None"
        self.method = method
        self.params = {}
        self._eps = 1e-8

    def fit(self, x: torch.Tensor):
        """Рассчитывает параметры нормализации по тензору x"""
        match self.method:
            case "minmax":
                self.params['min'] = x.min(dim=0, keepdim=True)[0]
                self.params['max'] = x.max(dim=0, keepdim=True)[0]
            case "standard":
                self.params['mean'] = x.mean(dim=0, keepdim=True)
                self.params['std'] = x.std(dim=0, keepdim=True)
            case "power":
                power = torch.mean(x[..., 0]**2 + x[..., 1]**2)
                self.params['scale'] = torch.sqrt(power)
            case None:
                ...

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        """Применяет нормализацию к x"""
        match self.method:
            case "minmax":
                return (x - self.params['min']) / (self.params['max'] - self.params['min'] + self._eps)
            case "standard":
                return (x - self.params['mean']) / (self.params['std'] + self._eps)
            case "power":
                return x / (self.params['scale'] + self._eps)
            case None:
                return x

    def inverse_transform(self, x_norm: torch.Tensor) -> torch.Tensor:
        """Денормализует x"""
        match self.method:
            case "minmax":
                return x_norm * (self.params['max'] - self.params['min']) + self.params['min']
            case "standard":
                return x_norm * self.params['std'] + self.params['mean']
            case "power":
                return x_norm * self.params['scale']
            case None:
                return x_norm

    def fit_transform(self, x: torch.Tensor) -> torch.Tensor:
        """Комбинированный вызов fit + transform"""
        self.fit(x)
        return self.transform(x)