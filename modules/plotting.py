import matplotlib.pyplot as plt
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

    for spectrum, color, label in spectra:
        plt.plot(freqs / 1e6, 10 * np.log10(np.abs(spectrum)), color=color, label=label)

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