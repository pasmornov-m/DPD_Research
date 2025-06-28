import torch
from torch import nn
import numpy as np
from scipy.signal import welch, get_window, lfilter
from modules.utils import to_torch_tensor, iq_to_complex, NoiseModel, CascadeModel
from modules import data_loader
from typing import Dict


def compute_mse(x, y):
    if torch.is_complex(x):
        x = torch.view_as_real(x)
    if torch.is_complex(y):
        y = torch.view_as_real(y)

    if x.shape != y.shape:
        raise ValueError(f"Формы входов не совпадают: x {x.shape}, y {y.shape}")
    if x.shape[-1] != 2:
        raise ValueError("Ожидается последний размер = 2 (I, Q)")

    mse = nn.MSELoss()
    return mse(x, y)


def compute_nmse(prediction, ground_truth):
    if torch.is_complex(prediction):
        prediction = torch.view_as_real(prediction)
    if torch.is_complex(ground_truth):
        ground_truth = torch.view_as_real(ground_truth)

    if prediction.shape != ground_truth.shape:
        raise ValueError(f"Формы входов не совпадают: prediction {prediction.shape}, ground_truth {ground_truth.shape}")
    if prediction.shape[-1] != 2:
        raise ValueError("Ожидается последний размер = 2 (I, Q)")

    mse = compute_mse(prediction, ground_truth)

    energy = (ground_truth ** 2).sum(dim=-1).mean()

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
    assert input_data.shape == output_data.shape, "input_data and output_data must have the same shape"
    if not input_data.is_complex():
        input_data = iq_to_complex(input_data)
    if not output_data.is_complex():
        output_data = iq_to_complex(output_data)
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


# def add_complex_noise(signal, snr, fs, bw):
#     # print(f"add_noise signal.shape: {signal.shape}")
#     N = len(signal)
#     snr_ln = 10 ** (snr/10)
#     signal_var = compute_signal_power(signal)
#     noise_var = torch.mean(signal_var / snr_ln * (fs/bw) * 0.5)
#     noise = torch.sqrt(noise_var) * (torch.randn(N, dtype=signal.dtype) + 1j * torch.randn(N, dtype=signal.dtype))
#     noise_signal = signal + noise
#     return noise_signal


def add_complex_noise(signal, snr, fs, bw):
    """
    Добавляет комплексный аддитивный шум к сигналу (в любом представлении).
    Поддерживает:
    - Комплексные тензоры: shape [...], dtype=torch.cfloat
    - Ре/им тензоры: shape [..., 2], dtype=torch.float32

    Возвращает сигнал в том же формате, что и входной.
    """
    snr_ln = 10 ** (snr / 10)

    if signal.dtype == torch.float32 and signal.shape[-1] == 2:
        is_real_im = True
        signal_complex = signal[..., 0] + 1j * signal[..., 1]
    elif signal.dtype.is_complex:
        is_real_im = False
        signal_complex = signal
    else:
        raise ValueError("Unsupported signal format")

    power_signal = torch.mean(torch.abs(signal_complex) ** 2, dim=-1, keepdim=True)
    noise_power = power_signal / snr_ln * (fs / bw) * 0.5

    noise = torch.sqrt(noise_power) * (torch.randn_like(signal_complex) + 1j * torch.randn_like(signal_complex))
    noisy = signal_complex + noise

    if is_real_im:
        return torch.stack((noisy.real, noisy.imag), dim=-1)
    else:
        return noisy


# def noise_realizations(num_realizations, signal, y_target, snr, fs, bw, acpr_meter):
#     nmse_values, acpr_left_values, acpr_right_values = [], [], []
#     for _ in range(num_realizations):
#         y_noise = add_complex_noise(signal, snr, fs, bw)
#         nmse_values.append(compute_nmse(y_noise, y_target))
#         acpr_left, acpr_right = calculate_acpr(y_noise, acpr_meter)
#         acpr_left_values.append(acpr_left)
#         acpr_right_values.append(acpr_right)
#     return map(lambda x: sum(x) / num_realizations, (nmse_values, acpr_left_values, acpr_right_values))


def noise_realizations(num_realizations, model, x, y_target, acpr_meter):
    from modules import learning
    
    nmse_values, acpr_left_values, acpr_right_values = [], [], []
    for _ in range(num_realizations):
        y_noise = learning.net_inference(net=model, x=x)
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


# def gmp_snr_metrics(arch_name: str,
#                 gmp_params: Dict,
#                 data_dict: Dict,
#                 snr_params: Dict
#                 ):
#     from modules import learning
#     from modules.gmp_model import GMP
    
#     x_train = data_dict["x_train"]
#     y_train_target = data_dict["y_train_target"]
#     x_val = data_dict["x_val"]
#     y_val_target = data_dict["y_val_target"]
    
#     snr_range = snr_params["snr_range"]
#     num_realizations = snr_params["num_realizations"]
#     fs = snr_params["fs"]
#     bw_main_ch = snr_params["bw_main_ch"]
#     epochs = snr_params["epochs"]
#     lr = snr_params["learning_rate"]
#     acpr_meter = snr_params["acpr_meter"]
#     pa_model = snr_params["pa_model"]
#     gain = snr_params["gain"]

#     results = {
#         "snr_range": snr_range,
#         "nmse": [],
#         "acpr_left": [],
#         "acpr_right": [],
#     }

#     if arch_name == "ILC":
#         results.update({
#             "nmse_uk": [],
#             "acpr_left_uk": [],
#             "acpr_right_uk": []
#         })

#     for snr in snr_range:
#         print(f"Current SNR: {snr}")
#         if arch_name == "DLA":
#             dpd_model = GMP(**gmp_params)

#             learning.optimize_dla(x_train, y_train_target, dpd_model, pa_model, epochs, lr, add_noise=True, snr=snr, fs=fs, bw=bw_main_ch)
#             y_dla_pa = pa_model.forward(dpd_model.forward(x_val)).detach()

#             nmse, acpr_left, acpr_right = noise_realizations(num_realizations, y_dla_pa, y_val_target, snr, fs, bw_main_ch, acpr_meter)
#             results["nmse"].append(nmse)
#             results["acpr_left"].append(acpr_left)
#             results["acpr_right"].append(acpr_right)

#         if arch_name == "ILA":
#             dpd_model = GMP(**gmp_params)
#             learning.optimize_ila(dpd_model, x_train, y_train_target, gain, epochs, lr, pa_model, add_noise=True, snr=snr, fs=fs, bw=bw_main_ch)
#             y_ila_pa = pa_model.forward(dpd_model.forward(x_val)).detach()

#             nmse, acpr_left, acpr_right = noise_realizations(num_realizations, y_ila_pa, y_val_target, snr, fs, bw_main_ch, acpr_meter)
#             results["nmse"].append(nmse)
#             results["acpr_left"].append(acpr_left)
#             results["acpr_right"].append(acpr_right)

#         if arch_name == "ILC":
#             dpd_model = GMP(**gmp_params)
#             uk = learning.ilc_signal(x_train, y_train_target, pa_model, 1000, 0.001, add_noise=True, snr=snr, fs=fs, bw=bw_main_ch)
            
#             uk_pa = pa_model.forward(uk).detach()
#             nmse_uk, acpr_left_uk, acpr_right_uk = noise_realizations(num_realizations, uk_pa, y_train_target, snr, fs, bw_main_ch, acpr_meter)
#             results["nmse_uk"].append(nmse_uk)
#             results["acpr_left_uk"].append(acpr_left_uk)
#             results["acpr_right_uk"].append(acpr_right_uk)

#             dpd_model.optimize_weights(x_train, uk, epochs, lr)
#             y_ilc_pa = pa_model.forward(dpd_model.forward(x_val)).detach()
            
#             nmse, acpr_left, acpr_right = noise_realizations(num_realizations, y_ilc_pa, y_val_target, snr, fs, bw_main_ch, acpr_meter)
#             results["nmse"].append(nmse)
#             results["acpr_left"].append(acpr_left)
#             results["acpr_right"].append(acpr_right)

#     return results



# def snr_metrics(arch_name: str,
#                    base_model_cls: nn.Module,
#                    model_params: Dict,
#                    data_dict: Dict,
#                    snr_params: Dict
#                    ):
#     from modules import learning
#     from modules.gmp_model import GMP
#     from modules.nn_model import GRU

#     x_train = data_dict["train_input"]
#     y_train = data_dict["train_output"]
#     x_val = data_dict["val_input"]
#     y_val = data_dict["val_output"]
#     y_train_target = data_dict['y_train_target']
#     y_val_target = data_dict['y_val_target']
        
#     snr_range = snr_params["snr_range"]
#     num_realizations = snr_params["num_realizations"]
#     fs = snr_params["fs"]
#     bw_main_ch = snr_params["bw_main_ch"]
#     epochs = snr_params["epochs"]
#     lr = snr_params["learning_rate"]
#     acpr_meter = snr_params["acpr_meter"]
#     pa_model = snr_params["pa_model"]
#     gain = snr_params["gain"]


#     results = {
#         "snr_range": snr_range,
#         "nmse": [],
#         "acpr_left": [],
#         "acpr_right": [],
#     }

#     if arch_name == "ILC":
#         results.update({
#             "nmse_uk": [],
#             "acpr_left_uk": [],
#             "acpr_right_uk": []
#         })
    
#     if base_model_cls is GMP:
#         dla_train_loader = [(x_train, y_train_target)]
#         dla_val_loader = [(x_val, y_val_target)]
#         ila_train_loader = [(x_train, x_train)]
#         ila_val_loader = [(x_val, x_val)]
#         grad_clip_val = 0
#         u_k_lr=0.001
#         u_k_epochs=1000
#     else:
#         frame_length = 64
#         batch_size = 64
#         batch_size_eval = 1024
#         dla_train_loader, dla_val_loader = data_loader.build_dataloaders(data_dict=data_dict, 
#                                                                         frame_length=frame_length, 
#                                                                         batch_size=batch_size, 
#                                                                         batch_size_eval=batch_size_eval, 
#                                                                         arch="dla")
#         ila_train_loader, ila_val_loader = data_loader.build_dataloaders(data_dict=data_dict, 
#                                                                         frame_length=frame_length, 
#                                                                         batch_size=batch_size, 
#                                                                         batch_size_eval=batch_size_eval, 
#                                                                         arch="ila")
#         grad_clip_val = 1.0
#         u_k_lr = 0.1
#         u_k_epochs = 200

#     criterion = compute_mse
#     metric_criterion = compute_nmse


#     for snr in snr_range:
#         print(f"Current SNR: {snr}")
#         if arch_name == "DLA":
#             dpd_model = base_model_cls(**model_params)
            
#             noise_gen_model = NoiseModel(snr=snr, fs=fs, bw=bw_main_ch)
#             casc_pa_noise = CascadeModel(model_1=pa_model, model_2=noise_gen_model, cascade_type="dla")
#             casc_dla = CascadeModel(model_1=dpd_model, 
#                                           model_2=casc_pa_noise, 
#                                           cascade_type="dla")
#             optimizer = torch.optim.Adam(casc_dla.parameters(), lr=lr)

#             learning.train(net=casc_dla, 
#                         criterion=criterion, 
#                         optimizer=optimizer, 
#                         train_loader=dla_train_loader, 
#                         val_loader=dla_val_loader, 
#                         grad_clip_val=grad_clip_val, 
#                         n_epochs=epochs, 
#                         metric_criterion=metric_criterion)

#             nmse, acpr_left, acpr_right = noise_realizations(num_realizations, model=casc_dla, x=x_val, y_target=y_val_target, acpr_meter=acpr_meter)
#             results["nmse"].append(nmse)
#             results["acpr_left"].append(acpr_left)
#             results["acpr_right"].append(acpr_right)

#         if arch_name == "ILA":
#             dpd_model = base_model_cls(**model_params)
            
#             noise_gen_model = NoiseModel(snr=snr, fs=fs, bw=bw_main_ch)
#             casc_pa_noise = CascadeModel(model_1=pa_model, model_2=noise_gen_model, cascade_type="dla")
#             casc_ila_train = CascadeModel(model_1=dpd_model, model_2=casc_pa_noise, gain=gain, cascade_type="ila")
#             casc_ila_eval = CascadeModel(model_1=dpd_model, model_2=casc_pa_noise, cascade_type="dla")
#             optimizer = torch.optim.Adam(casc_ila_train.parameters(), lr=lr)

#             learning.train(net=casc_ila_train, 
#                         criterion=criterion, 
#                         optimizer=optimizer, 
#                         train_loader=ila_train_loader, 
#                         val_loader=ila_val_loader, 
#                         grad_clip_val=grad_clip_val, 
#                         n_epochs=epochs, 
#                         metric_criterion=metric_criterion)

#             nmse, acpr_left, acpr_right = noise_realizations(num_realizations, model=casc_ila_eval, x=x_val, y_target=y_val_target, acpr_meter=acpr_meter)
#             results["nmse"].append(nmse)
#             results["acpr_left"].append(acpr_left)
#             results["acpr_right"].append(acpr_right)

#         if arch_name == "ILC":
#             noise_gen_model = NoiseModel(snr=snr, fs=fs, bw=bw_main_ch)
#             casc_pa_noise = CascadeModel(model_1=pa_model, model_2=noise_gen_model, cascade_type="dla")
            
#             u_k_train = learning.ilc_signal(x_train, y_train_target, casc_pa_noise, epochs=u_k_epochs, learning_rate=u_k_lr)
#             u_k_val = learning.ilc_signal(x_val, y_val_target, casc_pa_noise, epochs=u_k_epochs, learning_rate=u_k_lr)
#             u_k_pa = casc_pa_noise.forward(u_k_val).detach()

#             data_dict["ilc_train_output"] = u_k_train
#             data_dict["ilc_val_output"] = u_k_val
            
#             if base_model_cls is GMP:
#                 ilc_train_loader = [(x_train, u_k_train)]
#                 ilc_val_loader = [(x_val, u_k_val)]
#             else:
#                 ilc_train_loader, ilc_val_loader = data_loader.build_dataloaders(data_dict=data_dict, 
#                                                                                 frame_length=frame_length, 
#                                                                                 batch_size=batch_size, 
#                                                                                 batch_size_eval=batch_size_eval, 
#                                                                                 arch="ilc")
#             nmse_uk, acpr_left_uk, acpr_right_uk = noise_realizations(num_realizations, model=casc_pa_noise, x=u_k_val, y_target=y_val_target, acpr_meter=acpr_meter)
#             results["nmse_uk"].append(nmse_uk)
#             results["acpr_left_uk"].append(acpr_left_uk)
#             results["acpr_right_uk"].append(acpr_right_uk)

#             dpd_model = base_model_cls(**model_params)
#             optimizer = torch.optim.Adam(dpd_model.parameters(), lr=lr, weight_decay=1e-7)
#             learning.train(net=dpd_model, 
#                         criterion=criterion, 
#                         optimizer=optimizer, 
#                         train_loader=ilc_train_loader, 
#                         val_loader=ilc_val_loader, 
#                         grad_clip_val=grad_clip_val, 
#                         n_epochs=epochs, 
#                         metric_criterion=metric_criterion)

#             casc_ilc_eval = CascadeModel(model_1=dpd_model, model_2=casc_pa_noise, cascade_type="dla")
#             nmse, acpr_left, acpr_right = noise_realizations(num_realizations, model=casc_ilc_eval, x=x_val, y_target=y_val_target, acpr_meter=acpr_meter)
#             results["nmse"].append(nmse)
#             results["acpr_left"].append(acpr_left)
#             results["acpr_right"].append(acpr_right)

#     return results


class SNRMetricsRunner:
    def __init__(self,
                 base_model_cls: nn.Module,
                 model_params: Dict,
                 data_dict: Dict,
                 snr_params: Dict):
        
        self.base_model_cls = base_model_cls
        self.model_params = model_params
        self.data_dict = data_dict
        self.snr_params = snr_params

        self.x_train = data_dict["train_input"]
        self.y_train = data_dict["train_output"]
        self.x_val = data_dict["val_input"]
        self.y_val = data_dict["val_output"]
        self.y_train_target = data_dict['y_train_target']
        self.y_val_target = data_dict['y_val_target']
        self.u_k_train = None
        self.u_k_val = None
            
        self.snr_range = snr_params["snr_range"]
        self.num_realizations = snr_params["num_realizations"]
        self.fs = snr_params["fs"]
        self.bw_main_ch = snr_params["bw_main_ch"]
        self.epochs = snr_params["epochs"]
        self.lr = snr_params["learning_rate"]
        self.acpr_meter = snr_params["acpr_meter"]
        self.pa_model = snr_params["pa_model"]
        self.gain = snr_params["gain"]
        
        self.results = {
            "dla": {"nmse": [], "acpr_left": [], "acpr_right": []},
            "ila": {"nmse": [], "acpr_left": [], "acpr_right": []},
            "ilc": {"nmse": [], "acpr_left": [], "acpr_right": []},
            "uk":  {"nmse": [], "acpr_left": [], "acpr_right": []},
            "snr_range": self.snr_range,
            "base_model_cls": self.base_model_cls
        }
        
        self.criterion = compute_mse
        self.metric_criterion = compute_nmse
        
        self.frame_length = None
        self.batch_size = None
        self.batch_size_eval = None
        self.grad_clip_val = None
        self.u_k_lr = None
        self.u_k_epochs = None
        
        self.dla_train_loader = None
        self.dla_val_loader = None
        self.ila_train_loader = None
        self.ila_val_loader = None
        self.ilc_train_loader = None
        self.ilc_val_loader = None

        self._prepare_params()
        self._prepare_loaders()
    
    def _prepare_params(self):
        if self.base_model_cls.__name__ == "GMP":
            self.grad_clip_val = 0
            self.u_k_lr=0.001
            self.u_k_epochs=1000
        else:
            self.frame_length = 16
            self.batch_size = 256
            self.batch_size_eval = 2048
            self.grad_clip_val = 1.0
            self.u_k_lr = 0.1
            self.u_k_epochs = 200
    
    def _prepare_loaders(self):
        if self.base_model_cls.__name__ == "GMP":
            self.dla_train_loader = [(self.x_train, self.y_train_target)]
            self.dla_val_loader = [(self.x_val, self.y_val_target)]
            self.ila_train_loader = [(self.x_train, self.x_train)]
            self.ila_val_loader = [(self.x_val, self.x_val)]
        else:
            self.dla_train_loader, self.dla_val_loader = data_loader.build_dataloaders(data_dict=self.data_dict, 
                                                                            frame_length=self.frame_length, 
                                                                            batch_size=self.batch_size, 
                                                                            batch_size_eval=self.batch_size_eval, 
                                                                            arch="dla")
            self.ila_train_loader, self.ila_val_loader = data_loader.build_dataloaders(data_dict=self.data_dict, 
                                                                            frame_length=self.frame_length, 
                                                                            batch_size=self.batch_size, 
                                                                            batch_size_eval=self.batch_size_eval, 
                                                                            arch="ila")
    def _create_pa_noise_cascade(self, snr):
        noise_gen_model = NoiseModel(snr=snr, fs=self.fs, bw=self.bw_main_ch)
        casc_pa_noise = CascadeModel(model_1=self.pa_model, model_2=noise_gen_model, cascade_type="dla")
        return casc_pa_noise
    
    def _dla_arch_with_noise(self):
        from modules import learning
        
        for snr in self.snr_range:
            print(f"SNR: {snr}")
            dpd_model = self.base_model_cls(**self.model_params)
            casc_pa_noise = self._create_pa_noise_cascade(snr)
            casc_dla = CascadeModel(model_1=dpd_model, 
                                            model_2=casc_pa_noise, 
                                            cascade_type="dla")
            optimizer = torch.optim.Adam(casc_dla.parameters(), lr=self.lr)

            learning.train(net=casc_dla, 
                        criterion=self.criterion, 
                        optimizer=optimizer, 
                        train_loader=self.dla_train_loader, 
                        val_loader=self.dla_val_loader, 
                        grad_clip_val=self.grad_clip_val, 
                        n_epochs=self.epochs, 
                        metric_criterion=self.metric_criterion)

            nmse, acpr_left, acpr_right = noise_realizations(self.num_realizations, 
                                                            model=casc_dla, 
                                                            x=self.x_val, 
                                                            y_target=self.y_val_target, 
                                                            acpr_meter=self.acpr_meter)
            self.results["dla"]["nmse"].append(nmse)
            self.results["dla"]["acpr_left"].append(acpr_left)
            self.results["dla"]["acpr_right"].append(acpr_right)

        
    def _ila_arch_with_noise(self):
        from modules import learning
        
        for snr in self.snr_range:
            print(f"SNR: {snr}")
            dpd_model = self.base_model_cls(**self.model_params)
            casc_pa_noise = self._create_pa_noise_cascade(snr)
            casc_ila_train = CascadeModel(model_1=casc_pa_noise, model_2=dpd_model, gain=self.gain, cascade_type="ila")
            casc_ila_eval = CascadeModel(model_1=dpd_model, model_2=casc_pa_noise, cascade_type="dla")
            optimizer = torch.optim.Adam(casc_ila_train.parameters(), lr=self.lr)

            learning.train(net=casc_ila_train, 
                        criterion=self.criterion, 
                        optimizer=optimizer, 
                        train_loader=self.ila_train_loader, 
                        val_loader=self.ila_val_loader, 
                        grad_clip_val=self.grad_clip_val, 
                        n_epochs=self.epochs, 
                        metric_criterion=self.metric_criterion)

            nmse, acpr_left, acpr_right = noise_realizations(self.num_realizations, 
                                                            model=casc_ila_eval, 
                                                            x=self.x_val, 
                                                            y_target=self.y_val_target, 
                                                            acpr_meter=self.acpr_meter)
            self.results["ila"]["nmse"].append(nmse)
            self.results["ila"]["acpr_left"].append(acpr_left)
            self.results["ila"]["acpr_right"].append(acpr_right)

    
    def _ilc_arch_with_noise(self):
        from modules import learning
        
        for snr in self.snr_range:
            print(f"SNR: {snr}")
            casc_pa_noise = self._create_pa_noise_cascade(snr)
            
            self.u_k_train = learning.ilc_signal(self.x_train, self.y_train_target, casc_pa_noise, epochs=self.u_k_epochs, learning_rate=self.u_k_lr)
            # self.u_k_val = learning.ilc_signal(self.x_val, self.y_val_target, casc_pa_noise, epochs=self.u_k_epochs, learning_rate=self.u_k_lr)

            self.data_dict["ilc_train_output"] = self.u_k_train
            # self.data_dict["ilc_val_output"] = self.u_k_val
            
            if self.base_model_cls.__name__ == "GMP":
                self.ilc_train_loader = [(self.x_train, self.u_k_train)]
                # self.ilc_val_loader = [(self.x_val, self.u_k_val)]
                self.ilc_val_loader = self.ilc_train_loader
            else:
                self.ilc_train_loader, self.ilc_val_loader = data_loader.build_dataloaders(data_dict=self.data_dict, 
                                                                                frame_length=self.frame_length, 
                                                                                batch_size=self.batch_size, 
                                                                                batch_size_eval=self.batch_size_eval, 
                                                                                arch="ilc")

            nmse_uk, acpr_left_uk, acpr_right_uk = noise_realizations(self.num_realizations, 
                                                                    model=casc_pa_noise, 
                                                                    x=self.u_k_train, 
                                                                    y_target=self.y_train_target, 
                                                                    acpr_meter=self.acpr_meter)
            self.results["uk"]["nmse"].append(nmse_uk)
            self.results["uk"]["acpr_left"].append(acpr_left_uk)
            self.results["uk"]["acpr_right"].append(acpr_right_uk)

            dpd_model = self.base_model_cls(**self.model_params)
            optimizer = torch.optim.Adam(dpd_model.parameters(), lr=self.lr)
            learning.train(net=dpd_model, 
                        criterion=self.criterion, 
                        optimizer=optimizer, 
                        train_loader=self.ilc_train_loader, 
                        val_loader=self.ilc_val_loader, 
                        grad_clip_val=self.grad_clip_val, 
                        n_epochs=self.epochs, 
                        metric_criterion=self.metric_criterion)

            casc_ilc_eval = CascadeModel(model_1=dpd_model, model_2=casc_pa_noise, cascade_type="dla")
            nmse, acpr_left, acpr_right = noise_realizations(self.num_realizations, 
                                                            model=casc_ilc_eval, 
                                                            x=self.x_val, 
                                                            y_target=self.y_val_target, 
                                                            acpr_meter=self.acpr_meter)
            self.results["ilc"]["nmse"].append(nmse)
            self.results["ilc"]["acpr_left"].append(acpr_left)
            self.results["ilc"]["acpr_right"].append(acpr_right)

    
    def run(self, arch_name=None):
        if arch_name == "DLA":
            self._dla_arch_with_noise()
        elif arch_name == "ILA":
            self._ila_arch_with_noise()
        elif arch_name == "ILC":
            self._ilc_arch_with_noise()