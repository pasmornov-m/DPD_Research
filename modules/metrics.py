import torch
import numpy as np
from modules.utils import to_torch_tensor

def compute_mse(prediction, ground_truth):
    prediction = to_torch_tensor(prediction)
    ground_truth = to_torch_tensor(ground_truth)
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