from scipy.signal import welch
import numpy as np
import torch

def power_spectrum(input_data, fs, nperseg):
    if isinstance(input_data, torch.Tensor):
        input_data = input_data.detach().cpu().numpy()
    freqs, spectrum = welch(input_data, fs=fs, nperseg=nperseg)
    freqs = np.fft.fftshift(np.fft.fftfreq(len(spectrum), d=1/fs))
    spectrum = np.fft.fftshift(spectrum)
    return freqs, spectrum

def calculate_acpr(input_data, fs, bw_main_ch, bw_sub_ch, n_sub_ch, nperseg):
    if isinstance(input_data, torch.Tensor):
        input_data = input_data.detach().cpu().numpy()
    freqs, spectrum = power_spectrum(input_data, fs=fs, nperseg=nperseg)
    main_band_start = -bw_main_ch / 2
    main_band_end = bw_main_ch / 2
    delta_f = bw_sub_ch
    main_band_mask = (freqs >= main_band_start) & (freqs <= main_band_end)
    main_band_power = np.sum(np.abs(spectrum[main_band_mask])**2)
    sub_band_power = 0
    for i in range(1, n_sub_ch + 1):
        sub_band_mask_left = (freqs >= main_band_start - i * delta_f) & (freqs < main_band_start - (i - 1) * delta_f)
        sub_band_power += np.sum(np.abs(spectrum[sub_band_mask_left])**2)

        sub_band_mask_right = (freqs > main_band_end + (i - 1) * delta_f) & (freqs <= main_band_end + i * delta_f)
        sub_band_power += np.sum(np.abs(spectrum[sub_band_mask_right])**2)
    acpr = 10 * np.log10(main_band_power / (sub_band_power + 1e-10))
    return acpr