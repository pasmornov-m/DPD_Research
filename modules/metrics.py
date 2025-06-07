import torch
from torch.nn import MSELoss
import numpy as np
from scipy.signal import welch
import matlab
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


# def calculate_acpr(eng, input_data, fs, bw_main_ch, nperseg):
#     signal_numpy = input_data.numpy()
#     signal_matlab = matlab.double(np.column_stack((signal_numpy.real, signal_numpy.imag)).tolist())
#     result = eng.compute_acpr(signal_matlab, fs, bw_main_ch, nperseg, nargout=1)
#     acpr_left, acpr_right = float(result[0][0]), float(result[0][1])
#     return acpr_left, acpr_right


class ACPR:
    def __init__(
        self,
        sample_rate: float,
        main_channel_frequency: float = 0.0,
        main_measurement_bandwidth: float = 50e3,
        adjacent_channel_offset: np.ndarray = np.array([-100e3, 100e3]),
        adjacent_measurement_bandwidth: np.ndarray = None,
        measurement_filter_source: str = 'None',   # 'None' или 'Property'
        measurement_filter: np.ndarray = np.array([1.0]),  # FIR-коэффициенты
        spectral_estimation: str = 'welch',  # только 'welch' сейчас
        segment_length: int = 2560,
        overlap_percentage: float = 60.0,
        window: str = 'blackmanharris',
        fft_length: int = None,              # None → равно segment_length
        power_units: str = 'dBW',            # 'Watts', 'dBW', 'dBm'
        return_main_power: bool = False,
        return_adjacent_powers: bool = False,
    ):
        # Валидация
        if sample_rate <= 0:
            raise ValueError("sample_rate must be > 0")
        self.fs = float(sample_rate)

        self.fc0 = float(main_channel_frequency)
        if main_measurement_bandwidth <= 0:
            raise ValueError("main_measurement_bandwidth must be > 0")
        self.bw0 = float(main_measurement_bandwidth)

        self.offsets = np.atleast_1d(adjacent_channel_offset).astype(float)
        # Если не задано, берём те же bw для всех смещений
        if adjacent_measurement_bandwidth is None:
            self.bw_adj = np.full_like(self.offsets, self.bw0)
        else:
            self.bw_adj = np.atleast_1d(adjacent_measurement_bandwidth).astype(float)
            if self.bw_adj.shape not in ((len(self.offsets),), ()):
                raise ValueError("adjacent_measurement_bandwidth must be scalar or same length as offsets")

        # Фильтр
        self.filter_source = measurement_filter_source
        if self.filter_source not in ('None', 'Property'):
            raise ValueError("measurement_filter_source must be 'None' or 'Property'")
        self.fir = np.atleast_1d(measurement_filter).astype(float)

        if spectral_estimation.lower() != 'welch':
            raise NotImplementedError("Only 'welch' spectral_estimation is supported")
        self.method = 'welch'

        if segment_length <= 0 or not isinstance(segment_length, int):
            raise ValueError("segment_length must be positive integer")
        self.nperseg = segment_length

        if not (0 <= overlap_percentage < 100):
            raise ValueError("overlap_percentage must be in [0,100)")
        self.noverlap = int(self.nperseg * overlap_percentage / 100)

        self.window = window
        self.nfft = fft_length or self.nperseg

        if power_units not in ('Watts', 'dBW', 'dBm'):
            raise ValueError("power_units must be one of 'Watts', 'dBW', 'dBm'")
        self.power_units = power_units

        self.return_main = return_main_power
        self.return_adj = return_adjacent_powers

    def __call__(self, signal):
        """
        signal: 1D array-like of complex samples (numpy or torch)
        returns: ACPR (and, опционально, main power, adjacent powers)
        """
        # 1) Приведение к numpy-комплексному вектору
        if isinstance(signal, torch.Tensor):
            signal = signal.detach().cpu().numpy()
        signal = np.asarray(signal)
        if signal.ndim != 1 or not np.iscomplexobj(signal):
            raise ValueError("signal must be 1D complex array")

        # 2) Опциональный FIR-фильтр
        if self.filter_source == 'Property':
            signal = lfilter(self.fir, [1.0], signal)

        # 3) PSD оценка via Welch (двухсторонний)
        win = get_window(self.window, self.nperseg)
        freqs, psd = welch(
            signal,
            fs=self.fs,
            window=win,
            nperseg=self.nperseg,
            noverlap=self.noverlap,
            nfft=self.nfft,
            return_onesided=False,
            scaling='density'
        )
        # shift zero-freq to center
        psd = np.fft.fftshift(psd)
        freqs = np.fft.fftshift(freqs)

        # 4) Вычисляем мощности в каждой полосе
        df = freqs[1] - freqs[0]
        # Основная полоса
        low0 = self.fc0 - self.bw0/2
        high0 = self.fc0 + self.bw0/2
        mask0 = (freqs >= low0) & (freqs <= high0)
        P0 = np.sum(psd[mask0]) * df

        # Соседние
        Pacpr = []
        Padj = []
        for offset, bw in zip(self.offsets, self.bw_adj):
            low = self.fc0 + offset - bw/2
            high = self.fc0 + offset + bw/2
            m = (freqs >= low) & (freqs <= high)
            P = np.sum(psd[m]) * df
            Pacpr.append(P)
            Padj.append(P)

        # 5) Отношение ACPR
        # Формула: (MainBW / AdjBW) * (Padj / P0)
        ACPR_vals = (self.bw0 / self.bw_adj) * (np.array(Pacpr) / P0)

        # 6) Преобразование единиц
        if self.power_units == 'Watts':
            acpr_out = ACPR_vals
            main_p = P0
            adj_p = np.array(Padj)
        else:
            # в дБ
            acpr_out = 10 * np.log10(ACPR_vals)
            main_p = 10 * np.log10(P0)
            if self.power_units == 'dBm':
                main_p += 30
            adj_p = 10 * np.log10(Padj)
            if self.power_units == 'dBm':
                adj_p = adj_p + 30

        # collcect output
        out = [acpr_out]
        if self.return_main:
            out.append(main_p)
        if self.return_adj:
            out.append(adj_p)

        return tuple(out) if len(out) > 1 else acpr_out


def calculate_acpr(eng, input_data, fs, bw_main_ch, nperseg):

    sub_ch = 300e6

    acpr = ACPR(
        sample_rate=fs,
        main_measurement_bandwidth=bw_main_ch,
        adjacent_channel_offset=np.array([-sub_ch, sub_ch]),
        segment_length=nperseg,
        overlap_percentage=60,
        window='blackmanharris',
        fft_length=nperseg,
        power_units='dBW',
        return_main_power=True,
        return_adjacent_powers=True
    )

    acpr_vals, main_pw, adj_pw = acpr(input_data)

    return acpr_vals


def add_complex_noise(signal, snr, fs, bw):
    N = len(signal)
    snr_ln = 10 ** (snr/10)
    signal_var = compute_signal_power(signal)
    noise_var = torch.mean(signal_var / snr_ln * (fs/bw) * 0.5)
    noise = torch.sqrt(noise_var) * (torch.randn(N, dtype=signal.dtype) + 1j * torch.randn(N, dtype=signal.dtype))
    noise_signal = signal + noise
    return noise_signal


def noise_realizations(num_realizations, eng, signal, y_target, snr, fs, bw, nperseg):
    nmse_values, acpr_left_values, acpr_right_values = [], [], []
    for _ in range(num_realizations):
        y_noise = add_complex_noise(signal, snr, fs, bw)
        nmse_values.append(compute_nmse(y_noise, y_target))
        acpr_left, acpr_right = calculate_acpr(eng, y_noise, fs, bw, nperseg)
        acpr_left_values.append(acpr_left)
        acpr_right_values.append(acpr_right)
    return map(lambda x: sum(x) / num_realizations, (nmse_values, acpr_left_values, acpr_right_values))
