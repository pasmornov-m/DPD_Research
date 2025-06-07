import torch
from torch.nn import MSELoss
import numpy as np
from scipy.signal import welch
from modules.utils import to_torch_tensor
from scipy.signal import welch, get_window, lfilter


def compute_mse(x, y):
    if torch.is_complex(x):
        x = torch.view_as_real(x)
    if torch.is_complex(y):
        y = torch.view_as_real(y)

    if x.shape != y.shape:
        raise ValueError(f"Формы входов не совпадают: x {x.shape}, y {y.shape}")
    if x.shape[-1] != 2:
        raise ValueError("Ожидается последний размер = 2 (I, Q)")

    mse = MSELoss()
    return mse(x, y)


def compute_nmse(prediction, ground_truth):
    prediction, ground_truth = map(to_torch_tensor, (prediction, ground_truth))
    mse = compute_mse(prediction, ground_truth)
    energy = (ground_truth.real ** 2 + ground_truth.imag ** 2).mean()
    if energy == 0:
        raise ZeroDivisionError("Energy of the ground truth is zero.")
    return 10 * torch.log10(mse / energy)


def calculate_am_am(input_data, output_data):
    assert input_data.shape == output_data.shape, "input_data and output_data must have the same shape"
    input_data, output_data = map(to_torch_tensor, (input_data, output_data))

    input_amplitude = torch.abs(input_data)
    output_amplitude = torch.abs(output_data)
    input_amplitude = input_amplitude / torch.max(input_amplitude)
    output_amplitude = output_amplitude / torch.max(output_amplitude)
    return input_amplitude, output_amplitude


def calculate_am_pm(input_data, output_data):
    assert input_data.shape == output_data.shape, "input_data and output_data must have the same shape"
    input_data, output_data = map(to_torch_tensor, (input_data, output_data))
    
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


def get_amplitude(data):
    data = to_torch_tensor(data)
    power = data.real**2 + data.imag**2
    amplitude = torch.sqrt(power)
    return amplitude


def calculate_gain_complex(input_data, output_data):
    input_data, output_data = map(to_torch_tensor, (input_data, output_data))
    amp_in, amp_out = map(get_amplitude, (input_data, output_data))
    max_in_amp, max_out_amp = map(torch.max, (amp_in, amp_out))
    target_gain = torch.mean(max_out_amp / max_in_amp)
    return target_gain


def compute_signal_power(signal):
    power = torch.mean(torch.abs(signal) ** 2)
    return power


def power_spectrum(input_data, fs, nperseg):
    if isinstance(input_data, torch.Tensor):
        input_data = input_data.detach().cpu().numpy()
    _, spectrum = welch(input_data, fs=fs, nperseg=nperseg)
    freqs = np.fft.fftshift(np.fft.fftfreq(len(spectrum), d=1/fs))
    spectrum = np.fft.fftshift(spectrum)
    return freqs, spectrum


def calculate_acpr(input_data, acpr_meter):
    acpr_vals, main_pw, adj_pw = acpr_meter(input_data)
    return acpr_vals


def add_complex_noise(signal, snr, fs, bw):
    N = len(signal)
    snr_ln = 10 ** (snr/10)
    signal_var = compute_signal_power(signal)
    noise_var = torch.mean(signal_var / snr_ln * (fs/bw) * 0.5)
    noise = torch.sqrt(noise_var) * (torch.randn(N, dtype=signal.dtype) + 1j * torch.randn(N, dtype=signal.dtype))
    noise_signal = signal + noise
    return noise_signal


def noise_realizations(num_realizations, signal, y_target, snr, fs, bw, acpr_meter):
    nmse_values, acpr_left_values, acpr_right_values = [], [], []
    for _ in range(num_realizations):
        y_noise = add_complex_noise(signal, snr, fs, bw)
        nmse_values.append(compute_nmse(y_noise, y_target))
        acpr_left, acpr_right = calculate_acpr(y_noise, acpr_meter)
        acpr_left_values.append(acpr_left)
        acpr_right_values.append(acpr_right)
    return map(lambda x: sum(x) / num_realizations, (nmse_values, acpr_left_values, acpr_right_values))




class ACPR:
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
