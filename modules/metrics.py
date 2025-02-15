import torch
import numpy as np
from scipy.signal import welch
import matlab
import matlab.engine
from modules.utils import to_torch_tensor


def compute_mse(prediction, ground_truth):
    prediction, ground_truth = map(to_torch_tensor, (prediction, ground_truth))
    mse = ((ground_truth.real - prediction.real) ** 2 + (ground_truth.imag - prediction.imag) ** 2).mean()
    return mse


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
    freqs, spectrum = welch(input_data, fs=fs, nperseg=nperseg)
    freqs = np.fft.fftshift(np.fft.fftfreq(len(spectrum), d=1/fs))
    spectrum = np.fft.fftshift(spectrum)
    return freqs, spectrum


def calculate_acpr(eng, input_data, fs, bw_main_ch, nperseg):
    signal_numpy = input_data.numpy()
    signal_matlab = matlab.double(np.column_stack((signal_numpy.real, signal_numpy.imag)).tolist())
    acpr_left, acpr_right = eng.compute_acpr(signal_matlab, fs, bw_main_ch, nperseg, nargout=2)
    acpr_left, acpr_right = map(np.float32, (acpr_left, acpr_right))
    return acpr_left, acpr_right


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
