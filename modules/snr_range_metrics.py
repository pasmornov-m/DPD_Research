from modules import learning
from modules import metrics
from modules.gmp_model import GMP
from typing import Dict


def snr_metrics(arch_name: str,
                gmp_params: Dict,
                data_params: Dict,
                snr_params: Dict
                ):

    x_train = data_params["x_train"]
    y_train_target = data_params["y_train_target"]
    x_val = data_params["x_val"]
    y_val_target = data_params["y_val_target"]
    
    snr_range = snr_params["snr_range"]
    num_realizations = snr_params["num_realizations"]
    fs = snr_params["fs"]
    bw_main_ch = snr_params["bw_main_ch"]
    epochs = snr_params["epochs"]
    lr = snr_params["learning_rate"]
    acpr_meter = snr_params["acpr_meter"]
    pa_model = snr_params["pa_model"]
    gain = snr_params["gain"]

    results = {
        "snr_range": snr_range,
        "nmse": [],
        "acpr_left": [],
        "acpr_right": [],
    }

    if arch_name == "ILC":
        results.update({
            "nmse_uk": [],
            "acpr_left_uk": [],
            "acpr_right_uk": []
        })

    for snr in snr_range:
        print(f"Current SNR: {snr}")
        if arch_name == "DLA":
            dpd_model = GMP(**gmp_params)

            learning.optimize_dla_grad(x_train, y_train_target, dpd_model, pa_model, epochs, lr, add_noise=True, snr=snr, fs=fs, bw=bw_main_ch)
            y_dla_pa = pa_model.forward(dpd_model.forward(x_val)).detach()

            nmse, acpr_left, acpr_right = metrics.noise_realizations(num_realizations, y_dla_pa, y_val_target, snr, fs, bw_main_ch, acpr_meter)
            results["nmse"].append(nmse)
            results["acpr_left"].append(acpr_left)
            results["acpr_right"].append(acpr_right)

        if arch_name == "ILA":
            dpd_model = GMP(**gmp_params)
            learning.optimize_ila_grad(dpd_model, x_train, y_train_target, gain, epochs, lr, pa_model, add_noise=True, snr=snr, fs=fs, bw=bw_main_ch)
            y_ila_pa = pa_model.forward(dpd_model.forward(x_val)).detach()

            nmse, acpr_left, acpr_right = metrics.noise_realizations(num_realizations, y_ila_pa, y_val_target, snr, fs, bw_main_ch, acpr_meter)
            results["nmse"].append(nmse)
            results["acpr_left"].append(acpr_left)
            results["acpr_right"].append(acpr_right)

        if arch_name == "ILC":
            dpd_model = GMP(**gmp_params)
            uk = learning.ilc_signal_grad(x_train, y_train_target, pa_model, 1000, 0.001, add_noise=True, snr=snr, fs=fs, bw=bw_main_ch)
            
            uk_pa = pa_model.forward(uk).detach()
            nmse_uk, acpr_left_uk, acpr_right_uk = metrics.noise_realizations(num_realizations, uk_pa, y_train_target, snr, fs, bw_main_ch, acpr_meter)
            results["nmse_uk"].append(nmse_uk)
            results["acpr_left_uk"].append(acpr_left_uk)
            results["acpr_right_uk"].append(acpr_right_uk)

            dpd_model.optimize_coefficients_grad(x_train, uk, epochs, lr)
            y_ilc_pa = pa_model.forward(dpd_model.forward(x_val)).detach()
            
            nmse, acpr_left, acpr_right = metrics.noise_realizations(num_realizations, y_ilc_pa, y_val_target, snr, fs, bw_main_ch, acpr_meter)
            results["nmse"].append(nmse)
            results["acpr_left"].append(acpr_left)
            results["acpr_right"].append(acpr_right)

    return results