import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import numpy as np
from typing import List, Tuple


def plot_signal_spectra(
    freqs, 
    spectra: List[Tuple[np.ndarray, str, str]], 
    title="Спектр сигнала", 
    xlabel="Частота (МГц)", 
    ylabel= "Мощность (дБ)",
    fontsize = 12,
    figsize=(12, 6)
    ):
    
    plt.figure(figsize=figsize)

    for spectrum, color, label, marker in spectra:
        plt.plot(freqs / 1e6, 10 * np.log10(np.abs(spectrum)), color=color, label=label, marker=marker)

    plt.title(title, fontsize=fontsize)
    plt.xlabel(xlabel, fontsize=fontsize)
    plt.ylabel(ylabel, fontsize=fontsize)
    plt.legend()
    plt.grid()
    plt.show()

def plot_am_am_pm(
    am_am_data: List[Tuple[np.ndarray, np.ndarray, str, str]],
    am_pm_data: List[Tuple[np.ndarray, np.ndarray, str, str]],
    figsize=(12, 6),
    titles=("AM/AM", "AM/PM"),
    xlabels=("Амплитуда на входе", "Амплитуда на входе"),
    ylabels=("Амплитуда на выходе", "Фазовый сдвиг на выходе (градусы)")
    ):

    plt.figure(figsize=figsize)

    # AM/AM Plot
    plt.subplot(1, 2, 1)
    for x, y, color, label in am_am_data:
        plt.scatter(x, y, color=color, s=10, label=label)
    plt.title(titles[0])
    plt.xlabel(xlabels[0])
    plt.ylabel(ylabels[0])
    plt.grid(True)
    plt.legend()

    # AM/PM Plot
    plt.subplot(1, 2, 2)
    for x, y, color, label in am_pm_data:
        plt.scatter(x, y, color=color, s=10, label=label)
    plt.title(titles[1])
    plt.xlabel(xlabels[1])
    plt.ylabel(ylabels[1])
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()



def plot_weight_heatmaps(
    model,
    only_weights=True,
    vmin=None,
    vmax=None,
    cmap="coolwarm",
):
    """
    Plot heatmaps of model weight tensors.

    Args:
        model (nn.Module): PyTorch model
        only_weights (bool): plot only parameters containing 'weight'
        vmin, vmax: color scale limits (None -> auto per tensor)
        cmap (str): matplotlib colormap
    """

    for module_name, module in model.named_modules():
        for param_name, param in module.named_parameters(recurse=False):
            if only_weights and "weight" not in param_name:
                continue

            data = param.detach().cpu().numpy()

            if data.ndim == 0:
                continue

            if data.ndim == 1:
                data = data.reshape(1, -1)

            mask = np.abs(data) > 0
            data_vis = np.where(mask, data, np.nan)            
            if data_vis.ndim == 3 and data_vis.shape[-1] == 1:
                data_vis = data_vis[..., 0]
            
            cmap = mpl.colormaps.get_cmap(cmap)
            cmap.set_bad(color="magenta")
                        
            plt.figure(figsize=(max(2, data_vis.shape[1]*0.6), max(2, data_vis.shape[0]*0.2)))
            sns.heatmap(
                data_vis,
                annot=True,
                fmt=".1f",
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                cbar_kws={
                'shrink': 0.5,
                'aspect': 20,
                'pad': 0.05
                }
            )
            plt.title(f"{module_name or '<root>'}.{param_name}")
            plt.xlabel("in")
            plt.ylabel("out")
            plt.tight_layout()
            plt.show()