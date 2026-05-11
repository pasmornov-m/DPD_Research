import torch
import torch.nn.functional as F
import numpy as np
from scipy.signal import welch, get_window, lfilter
from modules import utils
from modules.utils import to_torch_tensor, iq_to_complex, moving_average
from typing import Tuple


def compute_mse(y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Вычисляет среднеквадратичную ошибку (MSE) между двумя сигналами.
    """
    if torch.is_complex(y_hat):
        y_hat = torch.view_as_real(y_hat)
    if torch.is_complex(y):
        y = torch.view_as_real(y)

    if y_hat.shape != y.shape:
        raise ValueError(f"Формы входов не совпадают: y_hat {y_hat.shape}, y {y.shape}")
    if y_hat.shape[-1] != 2:
        raise ValueError("Ожидается последний размер = 2 (I, Q)")

    return F.mse_loss(y_hat, y)


def compute_nmse(y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    if torch.is_complex(y_hat):
        y_hat = torch.view_as_real(y_hat)
    if torch.is_complex(y):
        y = torch.view_as_real(y)
        
    if y_hat.shape != y.shape:
        raise ValueError(f"Формы входов не совпадают: y_hat {y_hat.shape}, y {y.shape}")
    if y_hat.shape[-1] != 2:
        raise ValueError("Ожидается последний размер = 2 (I, Q)")
    
    mse = F.mse_loss(y_hat, y)
    energy = (y ** 2).mean()
    if energy == 0:
        raise ZeroDivisionError("Energy of the ground truth is zero.")
    nmse = mse / energy
    return 10 * torch.log10(nmse)


def calculate_am_am(input_data: torch.Tensor, output_data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    assert input_data.shape == output_data.shape, "input_data and output_data must have the same shape"

    input_amplitude = torch.abs(input_data)
    output_amplitude = torch.abs(output_data)
    input_amplitude = input_amplitude / torch.max(input_amplitude)
    output_amplitude = output_amplitude / torch.max(output_amplitude)
    return input_amplitude, output_amplitude


def calculate_am_pm(input_data: torch.Tensor, output_data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    assert input_data.shape == output_data.shape, "input_data and output_data must have the same shape"
    
    input_amplitude = torch.abs(input_data)
    valid_indices = input_amplitude > 1e-20

    phase_difference = torch.zeros_like(input_amplitude)
    phase_difference[valid_indices] = torch.angle(output_data[valid_indices]) - torch.angle(input_data[valid_indices])

    two_pi = 2 * torch.pi
    for i in range(1, len(phase_difference)):
        delta = phase_difference[i] - phase_difference[i - 1]
        if delta > torch.pi:
            phase_difference[i:] -= two_pi
        elif delta < -torch.pi:
            phase_difference[i:] += two_pi

    # Приведение к диапазону [-180, 180]
    phase_difference = torch.rad2deg(phase_difference)
    phase_difference = (phase_difference + 180) % 360 - 180
    
    return input_amplitude, phase_difference


def get_amplitude(data: torch.Tensor) -> torch.Tensor:
    data = to_torch_tensor(data)
    power = data.real**2 + data.imag**2
    amplitude = torch.sqrt(power)
    return amplitude


def calculate_gain(input_data: torch.Tensor, output_data: torch.Tensor) -> torch.Tensor:
    assert input_data.shape == output_data.shape, "input_data and output_data must have the same shape"
    if not input_data.is_complex():
        input_data = iq_to_complex(input_data)
    if not output_data.is_complex():
        output_data = iq_to_complex(output_data)
    amp_in, amp_out = map(get_amplitude, (input_data, output_data))
    max_in_amp, max_out_amp = map(torch.max, (amp_in, amp_out))
    target_gain = torch.mean(max_out_amp / max_in_amp)
    return target_gain


def compute_signal_power(signal: torch.Tensor) -> torch.Tensor:
    power = torch.mean(torch.abs(signal) ** 2)
    return power


def power_spectrum(input_data, fs, nperseg, window_size=None):
    input_data = iq_to_complex(input_data)
    if isinstance(input_data, torch.Tensor):
        input_data = input_data.detach().cpu().numpy()
    _, spectrum = welch(input_data, fs=fs, nperseg=nperseg)
    freqs = np.fft.fftshift(np.fft.fftfreq(len(spectrum), d=1/fs))
    spectrum = np.fft.fftshift(spectrum)
    if window_size:
        freqs, spectrum = moving_average(spectrum, freqs, fs, window_size)
    spectrum_db = 10 * np.log10(np.abs(spectrum))
    return freqs, spectrum_db

def calculate_acpr(input_data, acpr_meter):
    input_data = iq_to_complex(input_data)
    acpr_vals, main_pw, adj_pw = acpr_meter(input_data)
    return acpr_vals


def add_complex_noise(signal, snr, fs, bw):
    """
    Добавляет комплексный шум к сигналу (поддержка: комплексный или [..., 2] real/imag).
    Возвращает сигнал в том же формате.
    """
    snr_ln = 10 ** (snr / 10)

    if signal.dtype == torch.float32 and signal.shape[-1] == 2:

        power_signal = torch.mean(signal.pow(2).sum(dim=-1), dim=-1, keepdim=True)
        noise_power = power_signal / snr_ln * (fs / bw) * 0.5
        noise = torch.randn_like(signal) * torch.sqrt(noise_power[..., None])
        return signal + noise

    elif signal.dtype.is_complex:
        power_signal = torch.mean(torch.abs(signal) ** 2, dim=-1, keepdim=True)
        noise_power = power_signal / snr_ln * (fs / bw) * 0.5
        noise = (torch.randn_like(signal.real) + 1j * torch.randn_like(signal.imag)) * torch.sqrt(noise_power)
        return signal + noise

    else:
        raise ValueError("Unsupported signal format: expected complex or real+imag in last dim.")


def noise_realizations(num_realizations, model, x, y_target, acpr_meter):
    from modules import learning
    
    nmse_values, acpr_left_values, acpr_right_values = [], [], []
    for _ in range(num_realizations):
        y_noise = learning.net_inference(net=model, x=x).detach()
        nmse_values.append(compute_nmse(y_noise, y_target))
        acpr_left, acpr_right = calculate_acpr(y_noise, acpr_meter)
        acpr_left_values.append(acpr_left)
        acpr_right_values.append(acpr_right)
    return map(lambda x: sum(x) / num_realizations, (nmse_values, acpr_left_values, acpr_right_values))


def summarize_statistics(statistics_list, r_order: int = 2):
    """
    statistics_list: List[Dict]
        [{"nmse": float, "acpr": [acpr_l, acpr_r]}, ...]

    Returns:
        Dict со статистиками
    """

    nmse_vals = []
    acpr_l_vals = []
    acpr_r_vals = []

    for stats in statistics_list:
        nmse_vals.append(stats["nmse"])

        acpr_l_vals.append(stats["acpr"][0])
        acpr_r_vals.append(stats["acpr"][1])

    nmse_vals = np.array(nmse_vals)
    acpr_l_vals = np.array(acpr_l_vals)
    acpr_r_vals = np.array(acpr_r_vals)

    summary = {
        "nmse": {
            "mean": np.round(np.mean(nmse_vals), r_order),
            "std": np.round(np.std(nmse_vals), r_order),
            "min": np.round(np.min(nmse_vals), r_order),
            "max": np.round(np.max(nmse_vals), r_order),
        },

        "acpr_l": {
            "mean": np.round(np.mean(acpr_l_vals), r_order),
            "std": np.round(np.std(acpr_l_vals), r_order),
            "min": np.round(np.min(acpr_l_vals), r_order),
            "max": np.round(np.max(acpr_l_vals), r_order),
        },

        "acpr_r": {
            "mean": np.round(np.mean(acpr_r_vals), r_order),
            "std": np.round(np.std(acpr_r_vals), r_order),
            "min": np.round(np.min(acpr_r_vals), r_order),
            "max": np.round(np.max(acpr_r_vals), r_order),
        }
    }

    return summary


class ACPR:
    
    _EPS = 1e-10
    
    def __init__(
        self,
        sample_rate: float,
        main_channel_frequency: float = 0.0,
        main_measurement_bandwidth: float = 50e3,
        adjacent_channel_offset: np.ndarray = np.array([-100e3, 100e3]),
        adjacent_measurement_bandwidth: np.ndarray = None,
        measurement_filter_source: str = 'None',   # 'None' or 'Property'
        measurement_filter: np.ndarray = np.array([1.0]),  # FIR coefficients
        spectral_estimation: str = 'welch',  # only 'welch' supported
        segment_length: int = 2560,
        overlap_percentage: float = 60.0,
        window: str = 'blackmanharris',
        fft_length: int = None,              # None → equals segment_length
        power_units: str = 'dBW',            # 'Watts', 'dBW', 'dBm'
        return_main_power: bool = False,
        return_adjacent_powers: bool = False,
    ):
        # 1) Validate and store basic parameters
        if sample_rate <= 0:
            raise ValueError("sample_rate must be > 0")
        self.fs = float(sample_rate)

        self.fc0 = float(main_channel_frequency)
        if main_measurement_bandwidth <= 0:
            raise ValueError("main_measurement_bandwidth must be > 0")
        self.bw0 = float(main_measurement_bandwidth)

        self.offsets = np.atleast_1d(adjacent_channel_offset).astype(float)
        if adjacent_measurement_bandwidth is None:
            self.bw_adj = np.full_like(self.offsets, self.bw0)
        else:
            self.bw_adj = np.atleast_1d(adjacent_measurement_bandwidth).astype(float)
            if self.bw_adj.shape not in ((len(self.offsets),), ()):
                raise ValueError(
                    "adjacent_measurement_bandwidth must be scalar or same length as adjacent_channel_offset"
                )

        # 2) Filter configuration
        if measurement_filter_source not in ('None', 'Property'):
            raise ValueError("measurement_filter_source must be 'None' or 'Property'")
        self.filter_source = measurement_filter_source
        self.fir = np.atleast_1d(measurement_filter).astype(float)

        # 3) Spectral estimation method
        if spectral_estimation.lower() != 'welch':
            raise NotImplementedError("Only 'welch' spectral_estimation is supported")
        self.method = 'welch'

        # 4) Welch parameters
        if segment_length <= 0 or not isinstance(segment_length, int):
            raise ValueError("segment_length must be a positive integer")
        self.nperseg = segment_length

        if not (0 <= overlap_percentage < 100):
            raise ValueError("overlap_percentage must be in [0, 100)")
        self.noverlap = int(self.nperseg * overlap_percentage / 100)

        self.window = window
        self.nfft = fft_length or self.nperseg

        # 5) Output options
        if power_units not in ('Watts', 'dBW', 'dBm'):
            raise ValueError("power_units must be one of 'Watts', 'dBW', 'dBm'")
        self.power_units = power_units

        self.return_main = return_main_power
        self.return_adj = return_adjacent_powers

        # 6) Precompute window, frequency grid, df and masks
        self._prepare_windows_and_masks()

    def _prepare_windows_and_masks(self):
        """Precompute window, frequency vector, df and channel masks."""
        self.window_vals = get_window(self.window, self.nperseg)

        freqs, _ = welch(
            np.zeros(self.nperseg, dtype=complex),
            fs=self.fs,
            window=self.window_vals,
            nperseg=self.nperseg,
            noverlap=self.noverlap,
            nfft=self.nfft,
            return_onesided=False,
            scaling='density'
        )
        freqs = np.fft.fftshift(freqs)
        self.freqs = freqs
        self.df = freqs[1] - freqs[0]

        # Main channel mask
        low0 = self.fc0 - self.bw0 / 2
        high0 = self.fc0 + self.bw0 / 2
        self.main_mask = (freqs >= low0) & (freqs <= high0)

        # Adjacent channel masks
        self.adj_masks = []
        for offset, bw in zip(self.offsets, self.bw_adj):
            low = self.fc0 + offset - bw / 2
            high = self.fc0 + offset + bw / 2
            self.adj_masks.append((freqs >= low) & (freqs <= high))

    def _to_numpy(self, signal):
        """Convert input to 1D complex numpy array and validate."""
        if isinstance(signal, torch.Tensor):
            signal = signal.detach().cpu().numpy()
        arr = np.asarray(signal)
        if arr.ndim != 1 or not np.iscomplexobj(arr):
            raise ValueError("signal must be a 1D complex-valued array")
        return arr

    def _apply_filter(self, signal: np.ndarray) -> np.ndarray:
        """Apply FIR filter if filter_source == 'Property'."""
        if self.filter_source == 'Property':
            return lfilter(self.fir, [1.0], signal)
        return signal

    def _compute_psd(self, signal: np.ndarray) -> np.ndarray:
        """Compute two-sided PSD via Welch and shift zero-frequency to center."""
        _, psd = welch(
            signal,
            fs=self.fs,
            window=self.window_vals,
            nperseg=self.nperseg,
            noverlap=self.noverlap,
            nfft=self.nfft,
            return_onesided=False,
            scaling='density'
        )
        return np.fft.fftshift(psd)

    def _integrate_powers(self, psd: np.ndarray):
        """Integrate PSD over main and adjacent channel masks."""
        P0 = np.sum(psd[self.main_mask]) * self.df
        P_adj = np.array([np.sum(psd[m]) * self.df for m in self.adj_masks])
        P0 = np.maximum(P0, self._EPS)
        P_adj = np.maximum(P_adj, self._EPS)
        return P0, P_adj

    def _convert_units(self, P0: float, P_adj: np.ndarray):
        """
        Compute ACPR and convert main/adjacent powers into requested units.

        Returns:
            acpr: array of ACPR values
            main_p: main channel power
            adj_p: array of adjacent channel powers
        """
        # Linear ACPR ratio
        acpr = (self.bw0 / self.bw_adj) * (P_adj / P0)

        if self.power_units == 'Watts':
            return acpr, P0, P_adj

        # Convert to dB
        acpr_db = 10 * np.log10(acpr)
        main_db = 10 * np.log10(P0)
        adj_db = 10 * np.log10(P_adj)

        if self.power_units == 'dBm':
            main_db += 30
            adj_db += 30

        return acpr_db, main_db, adj_db

    def __call__(self, signal):
        """
        Measure ACPR of the input signal.

        Parameters:
            signal: 1D complex numpy array or torch.Tensor

        Returns:
            acpr_vals: ACPR values (scalar or array)
            main_power (optional): main channel power
            adjacent_powers (optional): adjacent channel powers
        """
        sig = self._to_numpy(signal)
        sig = self._apply_filter(sig)
        psd = self._compute_psd(sig)
        P0, P_adj = self._integrate_powers(psd)
        
        acpr, main_p, adj_p = self._convert_units(P0, P_adj)

        outputs = [acpr]
        if self.return_main:
            outputs.append(main_p)
        if self.return_adj:
            outputs.append(adj_p)

        return tuple(outputs) if len(outputs) > 1 else acpr


class RegLoss(torch.nn.Module):
    def __init__(self, model, original_loss, lambda_reg: float = 1e-4):
        super().__init__()
        
        self.model = model
        self.original_loss = original_loss
        self.lambda_reg = lambda_reg
        self.params = list(model.parameters())
        self.num_params = sum(p.numel() for p in self.params)
    
    def extra_loss(self, eps=1e-2):
        reg = 0.0

        for p in self.params:
            reg += torch.log10(p.pow(2) + eps).sum()

        return reg / self.num_params
    
    def forward(self, prediction, target):
        base_loss = self.original_loss(prediction, target)
        reg = self.extra_loss()
        
        return base_loss + self.lambda_reg * reg