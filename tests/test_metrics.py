import pytest
import torch
import numpy as np
from modules.metrics import (
    compute_mse, compute_nmse,
    calculate_am_am, calculate_am_pm,
    get_amplitude, calculate_gain,
    compute_signal_power, power_spectrum,
    calculate_acpr, add_complex_noise,
    noise_realizations, ACPR
)


def test_mse_zero_when_equal():
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    y = x.clone()
    assert compute_mse(x, y).item() == pytest.approx(0.0, abs=1e-8)

def test_mse_manual_calc():
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    y = torch.tensor([[2.0, 2.0], [3.0, 6.0]])
    expected = torch.mean((x - y) ** 2).item()
    assert compute_mse(x, y).item() == pytest.approx(expected, abs=1e-8)

def test_mse_with_complex_input():
    x = torch.tensor([1+2j, 3+4j])
    y = torch.tensor([1+2j, 3+4j])
    assert compute_mse(x, y).item() == pytest.approx(0.0, abs=1e-8)

def test_mse_shape_mismatch():
    x = torch.randn(3, 2)
    y = torch.randn(4, 2)
    with pytest.raises(ValueError, match="Формы входов не совпадают"):
        compute_mse(x, y)

def test_mse_last_dim_not_2():
    x = torch.randn(3, 3)
    y = torch.randn(3, 3)
    with pytest.raises(ValueError, match="Ожидается последний размер"):
        compute_mse(x, y)

# --- NMSE ---

def test_nmse_zero_when_equal():
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    y = x.clone()
    assert compute_nmse(x, y).item() == pytest.approx(-torch.inf)

def test_nmse_manual_calc():
    x = torch.tensor([[1.0, 2.0], [3.0, 5.0]])
    y = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    mse = torch.mean((x - y) ** 2)
    energy = torch.mean((y ** 2).sum(dim=-1))
    expected = 10*torch.log10(mse / energy)
    assert compute_nmse(x, y).item() == pytest.approx(expected.item(), abs=1e-8)

def test_nmse_with_complex_input():
    x = torch.tensor([1+1j, 2+2j])
    y = torch.tensor([1+1j, 2+2j])
    assert compute_nmse(x, y).item() == pytest.approx(-torch.inf)

def test_nmse_shape_mismatch():
    x = torch.randn(5, 2)
    y = torch.randn(6, 2)
    with pytest.raises(ValueError):
        compute_nmse(x, y)

def test_nmse_last_dim_not_2():
    x = torch.randn(5, 3)
    y = torch.randn(5, 3)
    with pytest.raises(ValueError):
        compute_nmse(x, y)

def test_nmse_zero_energy():
    x = torch.randn(5, 2)
    y = torch.zeros(5, 2)
    with pytest.raises(ZeroDivisionError, match="Energy of the ground truth is zero"):
        compute_nmse(x, y)


def test_calculate_am_am_normalization():
    x = torch.tensor([1+1j, 2+2j, 3+3j], dtype=torch.complex64)
    y = 2 * x
    in_amp, out_amp = calculate_am_am(x, y)

    # Все значения нормированы от 0 до 1
    assert torch.all(in_amp <= 1)
    assert torch.all(out_amp <= 1)
    assert in_amp.shape == out_amp.shape


def test_calculate_am_pm_phase_difference():
    x = torch.tensor([1+0j, 1+0j], dtype=torch.complex64)
    y = torch.tensor([1+0j, 0+1j], dtype=torch.complex64)
    in_amp, phase_diff = calculate_am_pm(x, y)

    assert in_amp.shape == phase_diff.shape
    # Первое значение ~0°, второе ~90°
    assert phase_diff[0].item() == pytest.approx(0, abs=1e-5)
    assert phase_diff[1].item() == pytest.approx(90, abs=5)


def test_get_amplitude_correctness():
    data = torch.tensor([3+4j, 1+1j], dtype=torch.complex64)
    amp = get_amplitude(data)
    assert torch.allclose(amp, torch.tensor([5.0, np.sqrt(2)], dtype=torch.float32))


def test_calculate_gain_complex():
    x = torch.tensor([1+1j, 2+2j], dtype=torch.complex64)
    y = 2 * x
    gain = calculate_gain(x, y)
    assert gain.item() == pytest.approx(2.0, rel=1e-3)


def test_compute_signal_power():
    x = torch.tensor([1+1j, 1+1j], dtype=torch.complex64)
    power = compute_signal_power(x)
    # Мощность = (|1+1j|^2 + |1+1j|^2)/2 = (2+2)/2 = 2
    assert power.item() == pytest.approx(2.0, abs=1e-6)


def test_power_spectrum_returns_freqs_and_spectrum():
    fs = 1000
    n = 256
    t = np.arange(n) / fs
    signal = np.sin(2 * np.pi * 50 * t)
    freqs, spectrum = power_spectrum(signal, fs, nperseg=64)
    assert len(freqs) == len(spectrum)
    assert np.allclose(freqs, np.fft.fftshift(np.fft.fftfreq(len(spectrum), 1/fs)))


def test_add_complex_noise_real_format():
    x = torch.randn(1000, 2)
    noisy = add_complex_noise(x, snr=20, fs=1e6, bw=200e3)
    assert noisy.shape == x.shape
    assert not torch.equal(noisy, x)


def test_add_complex_noise_complex_format():
    x = torch.randn(1000, dtype=torch.complex64)
    noisy = add_complex_noise(x, snr=20, fs=1e6, bw=200e3)
    assert noisy.shape == x.shape
    assert not torch.equal(noisy, x)


def test_acpr_class_and_call():
    fs = 1e6
    sig = np.exp(1j * 2 * np.pi * 1e3 * np.arange(10000)/fs)

    acpr = ACPR(sample_rate=fs)
    val = acpr(sig)
    assert isinstance(val, (float, np.ndarray))


def test_acpr_invalid_signal_raises():
    acpr = ACPR(sample_rate=1e6)
    with pytest.raises(ValueError):
        acpr(np.array([1, 2, 3]))


def test_calculate_acpr_wrapper():
    fs = 1e6
    sig = np.exp(1j * 2 * np.pi * 1e3 * np.arange(10000)/fs)
    acpr = ACPR(sample_rate=fs, return_main_power=True, return_adjacent_powers=True)

    val = calculate_acpr(sig, acpr)
    assert isinstance(val, (float, np.ndarray))
