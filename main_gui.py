import tkinter as tk
from tkinter import ttk, messagebox
import enum
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import matlab.engine
from modules import data_loader, metrics, learning
from modules.gmp_model import GMP
from modules.gmp_narx_model import GMP_NARX
from modules.moe_ar_gmp_model import MoE_GMP_AR

# ----------------------------
# 1. Описание состояний FSM
# ----------------------------
class FSMState(enum.Enum):
    INIT = 0

    ESTIMATE_GAIN = 1

    SELECT_PA_MODEL_TYPE = 2
    CONFIGURE_PA_MODEL = 3
    LOAD_OR_TRAIN_PA_MODEL = 4
    PLOT_PA_SPECTRUM = 5

    SELECT_DPD_MODEL_TYPE = 6
    CONFIGURE_DPD_MODEL = 7
    SELECT_DPD_ARCHS = 8

    LOAD_OR_TRAIN_DPD_ARCH = 9
    STORE_DPD_RESULTS = 10
    PLOT_DPD_SPECTRUM = 11

    SELECT_NOISE_RANGE = 12
    EVALUATE_NOISE_LOOP = 13
    PLOT_NOISE_RESULTS = 14

    DONE = 15
    ERROR = 99

# ----------------------------
# 2. Класс FSM
# ----------------------------
class AmplifierDPDFSM:
    def __init__(self, gui, data_path):
        """
        gui: ссылка на объект FSMGUI, чтобы из FSM читать значения полей
        data_path: путь к данным
        """
        self.gui = gui
        self.state = FSMState.INIT
        self.data_path = data_path

        # MATLAB engine и данные
        self.eng = None
        self.data = None
        self.config = None
        self.fs = None
        self.bw_main_ch = None
        self.bw_sub_ch = None
        self.n_sub_ch = None
        self.nperseg = None

        # Тренировочные/валидационные сигналы
        self.x_train = None
        self.y_train = None
        self.x_val = None
        self.y_val = None

        # Усиление и целевые сигналы
        self.gain = None
        self.y_train_target = None
        self.y_val_target = None

        # PA
        self.pa_model_type = None
        self.pa_params = {}
        self.pa_model = None
        self.y_gmp_pa = None
        self.nmse_pa = None

        # DPD
        self.dpd_model_type = None
        self.dpd_params = {}
        self.selected_archs = []
        self.dpd_models = {}       # словарь: dpd_models["DLA"] = экземпляр модели
        self.results_no_noise = {} # {"DLA": {"nmse":…, "acpr":(l,r)}, …}

        # Шумовые результаты
        self.noise_range = []
        self.num_realizations = None
        self.current_snr_index = 0
        # словарь шумовых результатов: noise_results["DLA"] = {"nmse":[…], "acpr_left":[…], "acpr_right":[…]}
        self.noise_results = {}

        # Вспомогательные индексы
        self.current_arch_index = 0

    def run(self):
        """
        Основной цикл FSM: вызываем методы, пока не дойдём до DONE или ERROR
        """
        try:
            while True:
                if self.state == FSMState.INIT:
                    self._init()
                elif self.state == FSMState.ESTIMATE_GAIN:
                    self._estimate_gain()
                elif self.state == FSMState.SELECT_PA_MODEL_TYPE:
                    self._select_pa_model_type()
                elif self.state == FSMState.CONFIGURE_PA_MODEL:
                    self._configure_pa_model()
                elif self.state == FSMState.LOAD_OR_TRAIN_PA_MODEL:
                    self._load_or_train_pa_model()
                elif self.state == FSMState.PLOT_PA_SPECTRUM:
                    self._plot_pa_spectrum()
                elif self.state == FSMState.SELECT_DPD_MODEL_TYPE:
                    self._select_dpd_model_type()
                elif self.state == FSMState.CONFIGURE_DPD_MODEL:
                    self._configure_dpd_model()
                elif self.state == FSMState.SELECT_DPD_ARCHS:
                    self._select_dpd_archs()
                elif self.state == FSMState.LOAD_OR_TRAIN_DPD_ARCH:
                    self._load_or_train_dpd_arch()
                elif self.state == FSMState.STORE_DPD_RESULTS:
                    self._store_dpd_results()
                elif self.state == FSMState.PLOT_DPD_SPECTRUM:
                    self._plot_dpd_spectrum()
                elif self.state == FSMState.SELECT_NOISE_RANGE:
                    self._select_noise_range()
                elif self.state == FSMState.EVALUATE_NOISE_LOOP:
                    self._evaluate_noise_loop()
                elif self.state == FSMState.PLOT_NOISE_RESULTS:
                    self._plot_noise_results()
                elif self.state == FSMState.DONE:
                    self._done()
                    break
                else:
                    raise RuntimeError(f"Неизвестное состояние: {self.state}")
        except Exception as e:
            print("Произошла ошибка в FSM:", e)
            self.state = FSMState.ERROR
            messagebox.showerror("FSM Error", str(e))

    # ----------------------------
    # Реализация каждого состояния
    # ----------------------------

    def _init(self):
        """INIT → ESTIMATE_GAIN."""
        print(">> STATE: INIT — запуск MATLAB и загрузка данных …")
        self.eng = matlab.engine.start_matlab()

        # Загрузим данные
        self.data = data_loader.load_data(self.data_path)
        self.config = self.data["config"]
        self.fs = self.config["input_signal_fs"]
        self.bw_main_ch = self.config["bw_main_ch"]
        self.bw_sub_ch = self.config["bw_sub_ch"]
        self.n_sub_ch = self.config["n_sub_ch"]
        self.nperseg = float(self.config["nperseg"])

        self.snr = None

        self.x_train = self.data["train_input"]
        self.y_train = self.data["train_output"]
        self.x_val = self.data["val_input"]
        self.y_val = self.data["val_output"]

        # Переход
        self.state = FSMState.ESTIMATE_GAIN
        self.gui.update_status()

    def _estimate_gain(self):
        """ESTIMATE_GAIN → SELECT_PA_MODEL_TYPE."""
        print(">> STATE: ESTIMATE_GAIN — вычисляем gain …")
        self.gain = metrics.calculate_gain_complex(self.x_train, self.y_train)
        print(f"   • Gain = {self.gain:.2f}")
        self.y_train_target = self.gain * self.x_train
        self.y_val_target = self.gain * self.x_val

        self.state = FSMState.SELECT_PA_MODEL_TYPE
        self.gui.update_status()

    def _select_pa_model_type(self):
        """SELECT_PA_MODEL_TYPE → CONFIGURE_PA_MODEL."""
        print(">> STATE: SELECT_PA_MODEL_TYPE — выбираем модель PA …")
        # Читаем из GUI
        chosen = self.gui.pa_model_choice.get()
        if chosen not in ["GMP", "GMP_NARX", "MoE_GMP_AR"]:
            messagebox.showwarning("Выбор PA", "Пожалуйста, выберите модель PA.")
            return
        self.pa_model_type = chosen

        self.state = FSMState.CONFIGURE_PA_MODEL
        self.gui.update_status()

    def _configure_pa_model(self):
        """CONFIGURE_PA_MODEL → LOAD_OR_TRAIN_PA_MODEL."""
        print(">> STATE: CONFIGURE_PA_MODEL — вводим параметры PA …")
        # Читаем из GUI
        try:
            Ka = int(self.gui.pa_Ka_entry.get())
            La = int(self.gui.pa_La_entry.get())
            Kb = int(self.gui.pa_Kb_entry.get())
            Lb = int(self.gui.pa_Lb_entry.get())
            Mb = int(self.gui.pa_Mb_entry.get())
            Kc = int(self.gui.pa_Kc_entry.get())
            Lc = int(self.gui.pa_Lc_entry.get())
            Mc = int(self.gui.pa_Mc_entry.get())
            epochs = int(self.gui.pa_epochs_entry.get())
            lr = float(self.gui.pa_lr_entry.get())
        except ValueError:
            messagebox.showwarning("Параметры PA", "Пожалуйста, корректно заполните все поля PA.")
            return

        self.pa_params = {
            "Ka": Ka, "La": La, "Kb": Kb, "Lb": Lb,
            "Mb": Mb, "Kc": Kc, "Lc": Lc, "Mc": Mc,
            "epochs": epochs, "lr": lr
        }

        self.state = FSMState.LOAD_OR_TRAIN_PA_MODEL
        self.gui.update_status()

    def _load_or_train_pa_model(self):
        """LOAD_OR_TRAIN_PA_MODEL → PLOT_PA_SPECTRUM."""
        print(">> STATE: LOAD_OR_TRAIN_PA_MODEL — загружаем или обучаем PA …")
        p = self.pa_params

        # Выбираем класс PA
        if self.pa_model_type == "GMP":
            self.pa_model = GMP(p["Ka"], p["La"], p["Kb"], p["Lb"], p["Mb"], p["Kc"], p["Lc"], p["Mc"], model_type="pa_grad")
        elif self.pa_model_type == "GMP_NARX":
            self.pa_model = GMP_NARX(p["Ka"], p["La"], p["Kb"], p["Lb"], p["Mb"], p["Kc"], p["Lc"], p["Mc"], model_type="pa_grad")
        else:  # "MoE_GMP_AR"
            self.pa_model = MoE_GMP_AR(p["Ka"], p["La"], p["Kb"], p["Lb"], p["Mb"], p["Kc"], p["Lc"], p["Mc"], model_type="pa_grad")

        # Попытка загрузить
        if not self.pa_model.load_coefficients():
            print("   • Нет старых весов PA, обучаем с нуля …")
            self.pa_model.optimize_coefficients_grad(self.x_train, self.y_train,
                                                     epochs=p["epochs"], learning_rate=p["lr"])
            self.pa_model.save_coefficients()
        else:
            print("   • Загружены старые веса PA, делаем дообучение …")
            self.pa_model.optimize_coefficients_grad(self.x_train, self.y_train,
                                                     epochs=int(p["epochs"]), learning_rate=p["lr"])
            self.pa_model.save_coefficients()

        # Прогоняем PA на валидации
        self.y_gmp_pa = self.pa_model.forward(self.x_val).detach()
        self.nmse_pa = metrics.compute_nmse(self.y_gmp_pa, self.y_val)
        print(f"   • NMSE_PA = {self.nmse_pa:.2f} dB")

        self.state = FSMState.PLOT_PA_SPECTRUM
        self.gui.update_status()

    def _plot_pa_spectrum(self):
        """PLOT_PA_SPECTRUM → SELECT_DPD_MODEL_TYPE."""
        print(">> STATE: PLOT_PA_SPECTRUM — строим спектры PA …")
        freqs, spec_in = metrics.power_spectrum(self.y_val, self.fs, self.nperseg)
        _, spec_pa = metrics.power_spectrum(self.y_gmp_pa, self.fs, self.nperseg)

        plt.figure(figsize=(8,4))
        plt.plot(freqs/1e6, 10*np.log10(np.abs(spec_in)), 'grey', label='y_val (input)')
        plt.plot(freqs/1e6, 10*np.log10(np.abs(spec_pa)), 'gold', label='y_PA (output)')
        plt.xlabel("Частота (МГц)")
        plt.ylabel("ППМ (дБ)")
        plt.title(f"PA Spectrum (NMSE={self.nmse_pa:.2f} dB)")
        plt.legend()
        plt.grid()
        plt.show()

        self.state = FSMState.SELECT_DPD_MODEL_TYPE
        self.gui.update_status()

    def _select_dpd_model_type(self):
        """SELECT_DPD_MODEL_TYPE → CONFIGURE_DPD_MODEL."""
        print(">> STATE: SELECT_DPD_MODEL_TYPE — выбираем модель DPD …")
        chosen = self.gui.dpd_model_choice.get()
        if chosen not in ["GMP", "GMP_NARX", "MoE_GMP_AR"]:
            messagebox.showwarning("Выбор DPD", "Пожалуйста, выберите модель DPD.")
            return
        self.dpd_model_type = chosen

        self.state = FSMState.CONFIGURE_DPD_MODEL
        self.gui.update_status()

    def _configure_dpd_model(self):
        """CONFIGURE_DPD_MODEL → SELECT_DPD_ARCHS."""
        print(">> STATE: CONFIGURE_DPD_MODEL — вводим параметры DPD …")
        try:
            Ka = int(self.gui.dpd_Ka_entry.get())
            La = int(self.gui.dpd_La_entry.get())
            Kb = int(self.gui.dpd_Kb_entry.get())
            Lb = int(self.gui.dpd_Lb_entry.get())
            Mb = int(self.gui.dpd_Mb_entry.get())
            Kc = int(self.gui.dpd_Kc_entry.get())
            Lc = int(self.gui.dpd_Lc_entry.get())
            Mc = int(self.gui.dpd_Mc_entry.get())
            epochs = int(self.gui.dpd_epochs_entry.get())
            lr = float(self.gui.dpd_lr_entry.get())
        except ValueError:
            messagebox.showwarning("Параметры DPD", "Пожалуйста, корректно заполните все поля DPD.")
            return

        self.dpd_params = {
            "Ka": Ka, "La": La, "Kb": Kb, "Lb": Lb,
            "Mb": Mb, "Kc": Kc, "Lc": Lc, "Mc": Mc,
            "epochs": epochs, "lr": lr
        }

        self.state = FSMState.SELECT_DPD_ARCHS
        self.gui.update_status()

    def _select_dpd_archs(self):
        """SELECT_DPD_ARCHS → LOAD_OR_TRAIN_DPD_ARCH."""
        print(">> STATE: SELECT_DPD_ARCHS — выбираем DLA, ILA, ILC …")
        # Читаем из GUI (Checkbuttons)
        archs = []
        if self.gui.var_DLA.get():
            archs.append("DLA")
        if self.gui.var_ILA.get():
            archs.append("ILA")
        if self.gui.var_ILC.get():
            archs.append("ILC")
        if not archs:
            messagebox.showwarning("Выбор архитектур", "Пожалуйста, выберите хотя бы одну архитектуру DPD.")
            return

        self.selected_archs = archs
        # Инициализируем структуру для результатов без шума
        self.results_no_noise = {arch: {} for arch in self.selected_archs}
        # Инициализируем шумовые результаты
        self.noise_results = {arch: {"nmse": [], "acpr_left": [], "acpr_right": []} for arch in self.selected_archs}

        self.current_arch_index = 0
        self.state = FSMState.LOAD_OR_TRAIN_DPD_ARCH
        self.gui.update_status()

    def _load_or_train_dpd_arch(self):
        """
        LOAD_OR_TRAIN_DPD_ARCH (для каждой архитектуры) → STORE_DPD_RESULTS.
        """
        arch = self.selected_archs[self.current_arch_index]
        print(f">> STATE: LOAD_OR_TRAIN_DPD_ARCH — обрабатываем {arch} …")
        p = self.dpd_params

        # Выбор класса DPD
        if self.dpd_model_type == "GMP":
            dpd = GMP(p["Ka"], p["La"], p["Kb"], p["Lb"], p["Mb"], p["Kc"], p["Lc"], p["Mc"], model_type=f"dpd_{arch.lower()}_grad")
        elif self.dpd_model_type == "GMP_NARX":
            dpd = GMP_NARX(p["Ka"], p["La"], p["Kb"], p["Lb"], p["Mb"], p["Kc"], p["Lc"], p["Mc"], model_type=f"dpd_{arch.lower()}_grad")
        else:  # "MoE_GMP_AR"
            dpd = MoE_GMP_AR(p["Ka"], p["La"], p["Kb"], p["Lb"], p["Mb"], p["Kc"], p["Lc"], p["Mc"], model_type=f"dpd_{arch.lower()}_grad")

        need_train = True

        if self.snr is None:
            # Без шума — пробуем загрузить
            if dpd.load_coefficients():
                print(f"   • {arch}: загружены старые веса, обучение не требуется.")
                need_train = False
            else:
                print(f"   • {arch}: нет старых весов, обучение новой модели.")

        else:
            print(f"   • {arch}: уровень шума SNR={self.snr}, обучение новой модели (без сохранения).")

        if need_train:
            if arch == "DLA":
                learning.optimize_dla_grad(self.x_train, self.y_train_target, dpd, self.pa_model,
                                        epochs=p["epochs"], learning_rate=p["lr"])
            elif arch == "ILA":
                learning.optimize_ila_grad(dpd, self.x_train, self.y_train, self.gain,
                                        epochs=p["epochs"], learning_rate=p["lr"])
            else:  # ILC
                u_k = learning.ilc_signal_grad(self.x_train, self.y_train_target, self.pa_model,
                                            max_iterations=500, learning_rate=0.001)
                dpd.optimize_coefficients_grad(self.x_train, u_k, epochs=p["epochs"], learning_rate=p["lr"])

            if self.snr is None:
                dpd.save_coefficients()

        # Сохраняем обученную модель в словарь
        self.dpd_models[arch] = dpd

        # Вычисляем метрики
        y_dpd = dpd.forward(self.x_val).detach()
        y_lin = self.pa_model.forward(y_dpd).detach()
        nmse_dpd = metrics.compute_nmse(y_lin, self.y_val_target)
        acpr_l, acpr_r = metrics.calculate_acpr(self.eng, y_lin, self.fs, self.bw_main_ch, self.nperseg)
        
        self.results_no_noise[arch] = {
            "nmse": nmse_dpd,
            "acpr": (acpr_l, acpr_r)
        }
        print(f"   • {arch}: NMSE={nmse_dpd:.2f}, ACPR=({acpr_l:.2f}, {acpr_r:.2f})")

        # Переход к следующей архитектуре
        self.current_arch_index += 1
        if self.current_arch_index < len(self.selected_archs):
            self.state = FSMState.LOAD_OR_TRAIN_DPD_ARCH
        else:
            self.state = FSMState.STORE_DPD_RESULTS
        self.gui.update_status()


    def _store_dpd_results(self):
        """STORE_DPD_RESULTS → PLOT_DPD_SPECTRUM."""
        print(">> STATE: STORE_DPD_RESULTS — результаты DPD без шума …")
        for arch, res in self.results_no_noise.items():
            print(f"   • {arch}: NMSE={res['nmse']:.2f}, ACPR=({res['acpr'][0]:.2f}, {res['acpr'][1]:.2f})")
        self.state = FSMState.PLOT_DPD_SPECTRUM
        self.gui.update_status()

    def _plot_dpd_spectrum(self):
        """PLOT_DPD_SPECTRUM → SELECT_NOISE_RANGE."""
        print(">> STATE: PLOT_DPD_SPECTRUM — строим спектры DPD …")
        freqs, spec_in = metrics.power_spectrum(self.x_val, self.fs, self.nperseg)
        _, spec_pa = metrics.power_spectrum(self.y_gmp_pa, self.fs, self.nperseg)

        plt.figure(figsize=(10,6))
        plt.plot(freqs/1e6, 10*np.log10(np.abs(spec_in)), 'grey', label='x_val (input)')
        plt.plot(freqs/1e6, 10*np.log10(np.abs(spec_pa)), 'gold', label='PA output')

        for arch in self.selected_archs:
            dpd = self.dpd_models[arch]

            y_dpd = dpd.forward(self.x_val).detach()
            y_lin = self.pa_model.forward(y_dpd).detach()

            _, spec_lin = metrics.power_spectrum(y_lin, self.fs, self.nperseg)
            plt.plot(freqs/1e6, 10*np.log10(np.abs(spec_lin)), label=f"{arch}_DPD")

        plt.xlabel("Частота (МГц)")
        plt.ylabel("ППМ (дБ)")
        plt.title("Спектры PA и PA+DPD (без шума)")
        plt.legend()
        plt.grid()
        plt.show()

        self.state = FSMState.SELECT_NOISE_RANGE
        self.gui.update_status()

    def _select_noise_range(self):
        """SELECT_NOISE_RANGE → EVALUATE_NOISE_LOOP."""
        print(">> STATE: SELECT_NOISE_RANGE — ввод диапазона SNR …")
        text = self.gui.noise_range_entry.get().strip()
        if not text:
            messagebox.showwarning("Диапазон SNR", "Пожалуйста, введите диапазон SNR (через запятую).")
            return
        try:
            # Преобразуем: "20,25,30" → [20,25,30]
            arr = [int(s.strip()) for s in text.split(",") if s.strip()]
        except ValueError:
            messagebox.showwarning("Диапазон SNR", "Неправильный формат диапазона SNR.")
            return
        if not arr:
            messagebox.showwarning("Диапазон SNR", "Введите хотя бы одно значение SNR.")
            return

        self.noise_range = arr
        self.num_realizations = int(self.gui.noise_realizations_entry.get() or "10")
        self.current_snr_index = 0

        # Инициализация шумовых результатов
        self.noise_results = {arch: {"nmse": [], "acpr_left": [], "acpr_right": []} for arch in self.selected_archs}

        self.state = FSMState.EVALUATE_NOISE_LOOP
        self.gui.update_status()

    def _evaluate_noise_loop(self):
        """EVALUATE_NOISE_LOOP → PLOT_NOISE_RESULTS (после всех SNR)."""
        if self.current_snr_index < len(self.noise_range):
            snr = self.noise_range[self.current_snr_index]
            print(f">> STATE: EVALUATE_NOISE_LOOP — SNR={snr} dB …")
            for arch in self.selected_archs:
                nmse_vals = []
                acpr_left_vals = []
                acpr_right_vals = []

                dpd = self.dpd_models[arch]
                for _ in range(self.num_realizations):
                    if arch == "DLA":
                        Ka = La = Kb = Lb = Mb = Kc = Lc = Mc = 3
                        dpd_noisy = type(dpd)(Ka, La, Kb, Lb, Mb, Kc, Lc, Mc, model_type=f"dpd_{arch.lower()}")
                        learning.optimize_dla_grad(
                            self.x_train, self.y_train_target, dpd_noisy, self.pa_model,
                            epochs=self.dpd_params["epochs"], learning_rate=self.dpd_params["lr"],
                            add_noise=True, snr=snr, fs=self.fs, bw=self.bw_main_ch
                        )
                        y_pa = self.pa_model.forward(dpd_noisy.forward(self.x_val)).detach()
                        nm_cur, left, right = metrics.noise_realizations(
                            self.num_realizations, self.eng, y_pa, self.y_val_target,
                            snr, self.fs, self.bw_main_ch, self.nperseg
                        )

                    elif arch == "ILA":
                        Ka = La = Kb = Lb = Mb = Kc = Lc = Mc = 3
                        dpd_noisy = type(dpd)(Ka, La, Kb, Lb, Mb, Kc, Lc, Mc, model_type=f"dpd_{arch.lower()}")
                        learning.optimize_ila_grad(
                            dpd_noisy, self.x_train, self.y_train, self.gain,
                            epochs=self.dpd_params["epochs"], learning_rate=self.dpd_params["lr"],
                            pa_model=self.pa_model, add_noise=True, snr=snr,
                            fs=self.fs, bw=self.bw_main_ch
                        )
                        y_pa = self.pa_model.forward(dpd_noisy.forward(self.x_val)).detach()
                        nm_cur = metrics.compute_nmse(y_pa, self.y_val_target)
                        left, right = metrics.calculate_acpr(self.eng, y_pa, self.fs, self.bw_main_ch, self.nperseg)

                    elif arch == "ILC":
                        Ka = La = Kb = Lb = Mb = Kc = Lc = Mc = 3
                        dpd_noisy = type(dpd)(Ka, La, Kb, Lb, Mb, Kc, Lc, Mc, model_type=f"dpd_{arch.lower()}")
                        u_k_noisy = learning.ilc_signal_grad(
                            self.x_train, self.y_train_target, self.pa_model,
                            max_iterations=500, learning_rate=0.001,
                            add_noise=True, snr=snr,
                            fs=self.fs, bw_main_ch=self.bw_main_ch
                        )
                        u_k_pa = self.pa_model.forward(u_k_noisy).detach()
                        nm_cur, left, right = metrics.noise_realizations(
                            self.num_realizations, self.eng, u_k_pa, self.y_val_target,
                            snr, self.fs, self.bw_main_ch, self.nperseg
                        )
                        dpd_noisy.optimize_coefficients_grad(
                            self.x_train, u_k_noisy, 
                            epochs=self.dpd_params["epochs"], 
                            learning_rate=self.dpd_params["lr"])
                        y_pa = self.pa_model.forward(dpd_noisy.forward(self.x_val)).detach()
                        nm_cur, left, right = metrics.noise_realizations(
                            self.num_realizations, self.eng, y_pa, self.y_val_target,
                            snr, self.fs, self.bw_main_ch, self.nperseg
                        )


                    nmse_vals.append(nm_cur)
                    acpr_left_vals.append(left)
                    acpr_right_vals.append(right)

                nmse_mean = float(np.mean(nmse_vals))
                acpr_left_mean = float(np.mean(acpr_left_vals))
                acpr_right_mean = float(np.mean(acpr_right_vals))

                self.noise_results[arch]["nmse"].append(nmse_mean)
                self.noise_results[arch]["acpr_left"].append(acpr_left_mean)
                self.noise_results[arch]["acpr_right"].append(acpr_right_mean)

                print(f"   • [noise={snr}dB, arch={arch}] NMSE={nmse_mean:.2f}, "
                      f"ACPR_L={acpr_left_mean:.2f}, ACPR_R={acpr_right_mean:.2f}")

            # Следующий SNR
            self.current_snr_index += 1
        else:
            self.state = FSMState.PLOT_NOISE_RESULTS
        self.gui.update_status()

    def _plot_noise_results(self):
        """PLOT_NOISE_RESULTS → DONE."""
        print(">> STATE: PLOT_NOISE_RESULTS — строим графики шума …")
        snr_vals = self.noise_range

        # NMSE vs SNR
        plt.figure(figsize=(8,4))
        for arch in self.selected_archs:
            plt.plot(snr_vals, self.noise_results[arch]["nmse"], marker='o', label=arch)
        plt.xlabel("SNR (дБ)")
        plt.ylabel("NMSE (дБ)")
        plt.title("NMSE vs SNR для DPD-архитектур")
        plt.legend()
        plt.grid()
        plt.show()

        # ACPR Left / Right vs SNR
        plt.figure(figsize=(12,5))
        plt.subplot(1,2,1)
        for arch in self.selected_archs:
            plt.plot(snr_vals, self.noise_results[arch]["acpr_left"], marker='o', label=arch)
        plt.xlabel("SNR (дБ)"); plt.ylabel("ACPR Left (дБ)")
        plt.title("ACPR Left vs SNR")
        plt.legend(); plt.grid()

        plt.subplot(1,2,2)
        for arch in self.selected_archs:
            plt.plot(snr_vals, self.noise_results[arch]["acpr_right"], marker='o', label=arch)
        plt.xlabel("SNR (дБ)"); plt.ylabel("ACPR Right (дБ)")
        plt.title("ACPR Right vs SNR")
        plt.legend(); plt.grid()

        plt.tight_layout()
        plt.show()

        self.state = FSMState.DONE
        self.gui.update_status()

    def _done(self):
        """DONE: закрытие MATLAB и финальный вывод."""
        print(">> STATE: DONE — завершаем работу FSM.")
        if self.eng is not None:
            self.eng.quit()
        messagebox.showinfo("Finished", "Все этапы FSM выполнены успешно.")
        # В качестве «финального» состояния можно что-то ещё сделать в GUI (отключить кнопки, и т.п.)


# ----------------------------
# 3. GUI на tkinter (наследуемся от tk.Tk)
# ----------------------------
class FSMGUI(tk.Tk):
    def __init__(self, data_path):
        super().__init__()
        self.title("FSM: Усилитель и DPD")
        self.geometry("800x900")

        # Ссылка на FSM (будем создавать после ввода параметров)
        self.fsm = AmplifierDPDFSM(self, data_path)

        # --- Статус FSM ---
        self.status_label = ttk.Label(self, text="Текущее состояние: INIT", font=("Arial", 12))
        self.status_label.pack(pady=10)

        # 1) Параметры модели PA
        frame_pa = ttk.LabelFrame(self, text="1. Модель PA и параметры", padding=10)
        frame_pa.pack(fill="x", padx=10, pady=5)

        ttk.Label(frame_pa, text="Выберите модель PA:").grid(row=0, column=0, sticky="w")
        self.pa_model_choice = ttk.Combobox(frame_pa, values=["GMP", "GMP_NARX", "MoE_GMP_AR"], state="readonly")
        self.pa_model_choice.grid(row=0, column=1, padx=5, pady=2)

        # Порядки PA: Ka, La, Kb, Lb, Mb, Kc, Lc, Mc
        ttk.Label(frame_pa, text="Ka:").grid(row=1, column=0, sticky="e")
        self.pa_Ka_entry = ttk.Entry(frame_pa, width=5)
        self.pa_Ka_entry.grid(row=1, column=1, sticky="w", padx=2)
        ttk.Label(frame_pa, text="La:").grid(row=1, column=2, sticky="e")
        self.pa_La_entry = ttk.Entry(frame_pa, width=5)
        self.pa_La_entry.grid(row=1, column=3, sticky="w", padx=2)

        ttk.Label(frame_pa, text="Kb:").grid(row=1, column=4, sticky="e")
        self.pa_Kb_entry = ttk.Entry(frame_pa, width=5)
        self.pa_Kb_entry.grid(row=1, column=5, sticky="w", padx=2)
        ttk.Label(frame_pa, text="Lb:").grid(row=1, column=6, sticky="e")
        self.pa_Lb_entry = ttk.Entry(frame_pa, width=5)
        self.pa_Lb_entry.grid(row=1, column=7, sticky="w", padx=2)

        ttk.Label(frame_pa, text="Mb:").grid(row=2, column=0, sticky="e")
        self.pa_Mb_entry = ttk.Entry(frame_pa, width=5)
        self.pa_Mb_entry.grid(row=2, column=1, sticky="w", padx=2)
        ttk.Label(frame_pa, text="Kc:").grid(row=2, column=2, sticky="e")
        self.pa_Kc_entry = ttk.Entry(frame_pa, width=5)
        self.pa_Kc_entry.grid(row=2, column=3, sticky="w", padx=2)

        ttk.Label(frame_pa, text="Lc:").grid(row=2, column=4, sticky="e")
        self.pa_Lc_entry = ttk.Entry(frame_pa, width=5)
        self.pa_Lc_entry.grid(row=2, column=5, sticky="w", padx=2)
        ttk.Label(frame_pa, text="Mc:").grid(row=2, column=6, sticky="e")
        self.pa_Mc_entry = ttk.Entry(frame_pa, width=5)
        self.pa_Mc_entry.grid(row=2, column=7, sticky="w", padx=2)

        ttk.Label(frame_pa, text="Эпохи PA:").grid(row=3, column=0, sticky="e", pady=4)
        self.pa_epochs_entry = ttk.Entry(frame_pa, width=7)
        self.pa_epochs_entry.grid(row=3, column=1, sticky="w", padx=2)
        ttk.Label(frame_pa, text="Скорость обучения PA:").grid(row=3, column=2, sticky="e", pady=4)
        self.pa_lr_entry = ttk.Entry(frame_pa, width=7)
        self.pa_lr_entry.grid(row=3, column=3, sticky="w", padx=2)

        # 2) Параметры модели DPD
        frame_dpd = ttk.LabelFrame(self, text="2. Модель DPD и параметры", padding=10)
        frame_dpd.pack(fill="x", padx=10, pady=5)

        ttk.Label(frame_dpd, text="Выберите модель DPD:").grid(row=0, column=0, sticky="w")
        self.dpd_model_choice = ttk.Combobox(frame_dpd, values=["GMP", "GMP_NARX", "MoE_GMP_AR"], state="readonly")
        self.dpd_model_choice.grid(row=0, column=1, padx=5, pady=2)

        # Порядки DPD: Ka, La, Kb, Lb, Mb, Kc, Lc, Mc
        ttk.Label(frame_dpd, text="Ka:").grid(row=1, column=0, sticky="e")
        self.dpd_Ka_entry = ttk.Entry(frame_dpd, width=5)
        self.dpd_Ka_entry.grid(row=1, column=1, sticky="w", padx=2)
        ttk.Label(frame_dpd, text="La:").grid(row=1, column=2, sticky="e")
        self.dpd_La_entry = ttk.Entry(frame_dpd, width=5)
        self.dpd_La_entry.grid(row=1, column=3, sticky="w", padx=2)

        ttk.Label(frame_dpd, text="Kb:").grid(row=1, column=4, sticky="e")
        self.dpd_Kb_entry = ttk.Entry(frame_dpd, width=5)
        self.dpd_Kb_entry.grid(row=1, column=5, sticky="w", padx=2)
        ttk.Label(frame_dpd, text="Lb:").grid(row=1, column=6, sticky="e")
        self.dpd_Lb_entry = ttk.Entry(frame_dpd, width=5)
        self.dpd_Lb_entry.grid(row=1, column=7, sticky="w", padx=2)

        ttk.Label(frame_dpd, text="Mb:").grid(row=2, column=0, sticky="e")
        self.dpd_Mb_entry = ttk.Entry(frame_dpd, width=5)
        self.dpd_Mb_entry.grid(row=2, column=1, sticky="w", padx=2)
        ttk.Label(frame_dpd, text="Kc:").grid(row=2, column=2, sticky="e")
        self.dpd_Kc_entry = ttk.Entry(frame_dpd, width=5)
        self.dpd_Kc_entry.grid(row=2, column=3, sticky="w", padx=2)

        ttk.Label(frame_dpd, text="Lc:").grid(row=2, column=4, sticky="e")
        self.dpd_Lc_entry = ttk.Entry(frame_dpd, width=5)
        self.dpd_Lc_entry.grid(row=2, column=5, sticky="w", padx=2)
        ttk.Label(frame_dpd, text="Mc:").grid(row=2, column=6, sticky="e")
        self.dpd_Mc_entry = ttk.Entry(frame_dpd, width=5)
        self.dpd_Mc_entry.grid(row=2, column=7, sticky="w", padx=2)

        ttk.Label(frame_dpd, text="Эпохи DPD:").grid(row=3, column=0, sticky="e", pady=4)
        self.dpd_epochs_entry = ttk.Entry(frame_dpd, width=7)
        self.dpd_epochs_entry.grid(row=3, column=1, sticky="w", padx=2)
        ttk.Label(frame_dpd, text="Скорость обучения DPD:").grid(row=3, column=2, sticky="e", pady=4)
        self.dpd_lr_entry = ttk.Entry(frame_dpd, width=7)
        self.dpd_lr_entry.grid(row=3, column=3, sticky="w", padx=2)

        # 3) Выбор архитектур DPD
        frame_arch = ttk.LabelFrame(self, text="3. Выбор DPD-архитектур", padding=10)
        frame_arch.pack(fill="x", padx=10, pady=5)

        self.var_DLA = tk.BooleanVar(value=True)
        self.chk_DLA = ttk.Checkbutton(frame_arch, text="DLA", variable=self.var_DLA)
        self.chk_DLA.grid(row=0, column=0, padx=5, pady=2)

        self.var_ILA = tk.BooleanVar(value=True)
        self.chk_ILA = ttk.Checkbutton(frame_arch, text="ILA", variable=self.var_ILA)
        self.chk_ILA.grid(row=0, column=1, padx=5, pady=2)

        self.var_ILC = tk.BooleanVar(value=True)
        self.chk_ILC = ttk.Checkbutton(frame_arch, text="ILC", variable=self.var_ILC)
        self.chk_ILC.grid(row=0, column=2, padx=5, pady=2)

        # 4) Шумовые параметры
        frame_noise = ttk.LabelFrame(self, text="4. Параметры шума", padding=10)
        frame_noise.pack(fill="x", padx=10, pady=5)

        ttk.Label(frame_noise, text="Диапазон SNR (через запятую):").grid(row=0, column=0, sticky="w")
        self.noise_range_entry = ttk.Entry(frame_noise, width=30)
        self.noise_range_entry.grid(row=0, column=1, padx=5, pady=2)
        self.noise_range_entry.insert(0, "20,25,30,35,40")

        ttk.Label(frame_noise, text="Кол-во реализаций:").grid(row=1, column=0, sticky="w", pady=4)
        self.noise_realizations_entry = ttk.Entry(frame_noise, width=7)
        self.noise_realizations_entry.grid(row=1, column=1, sticky="w", padx=5)

        # 5) Кнопки управления FSM
        frame_buttons = ttk.Frame(self, padding=10)
        frame_buttons.pack(fill="x", padx=10, pady=10)

        self.start_button = ttk.Button(frame_buttons, text="Запустить FSM", command=self.start_fsm)
        self.start_button.grid(row=0, column=0, padx=5)

        self.next_state_button = ttk.Button(frame_buttons, text="Следующий этап", command=self.next_state)
        self.next_state_button.grid(row=0, column=1, padx=5)

        self.reset_button = ttk.Button(frame_buttons, text="Сброс", command=self.reset_fsm)
        self.reset_button.grid(row=0, column=2, padx=5)

        # 6) Область для графиков
        self.canvas_frame = ttk.Frame(self)
        self.canvas_frame.pack(fill="both", expand=True, padx=10, pady=10)

    def update_status(self):
        """Обновить метку состояния FSM."""
        name = self.state_to_name(self.fsm.state)
        self.status_label.config(text=f"Текущее состояние: {name}")

    def start_fsm(self):
        """Считываем все настройки из GUI, создаём FSM и запускаем его."""
        # Проверяем, что все обязательные поля заполнены
        # (для простоты примера не будем делать детальную валидацию)
        data_path = "DPA_200MHz"  # Задайте путь к вашим данным здесь

        # Считываем параметры FSM из GUI
        snr_text = self.noise_range_entry.get().strip()
        try:
            snr_range = [int(s.strip()) for s in snr_text.split(",") if s.strip()]
        except ValueError:
            messagebox.showwarning("SNR", "Неправильный формат диапазона SNR.")
            return

        try:
            num_realizations = int(self.noise_realizations_entry.get())
        except ValueError:
            messagebox.showwarning("Количество реализаций", "Введите целое число.")
            return

        pa_model_type = self.pa_model_choice.get()
        dpd_model_type = self.dpd_model_choice.get()
        if pa_model_type not in ["GMP", "GMP_NARX", "MoE_GMP_AR"] or dpd_model_type not in ["GMP", "GMP_NARX", "MoE_GMP_AR"]:
            messagebox.showwarning("Модели", "Выберите модели PA и DPD.")
            return

        try:
            pa_epochs = int(self.pa_epochs_entry.get())
            pa_lr = float(self.pa_lr_entry.get())
            dpd_epochs = int(self.dpd_epochs_entry.get())
            dpd_lr = float(self.dpd_lr_entry.get())
        except ValueError:
            messagebox.showwarning("Эпохи/LR", "Введите корректные числовые значения эпох и скоростей обучения.")
            return

        # Передаём параметры в FSM
        self.fsm.snr_range = snr_range
        self.fsm.num_realizations = num_realizations
        self.fsm.pa_model_type = pa_model_type
        self.fsm.dpd_model_type = dpd_model_type
        self.fsm.pa_epochs = pa_epochs
        self.fsm.dpd_epochs = dpd_epochs
        self.fsm.pa_lr = pa_lr
        self.fsm.dpd_lr = dpd_lr

        # Запускаем FSM в отдельном потоке или напрямую (для простоты – напрямую)
        self.fsm.run()
        self.update_status()

    def next_state(self):
        """Перевод FSM в следующее состояние вручную."""
        try:
            # Увеличиваем state на 1, если это валидная следующая граница
            next_state_val = self.fsm.state.value + 1
            possible = {s.value for s in FSMState}
            if next_state_val in possible:
                self.fsm.state = FSMState(next_state_val)
                self.update_status()
            else:
                messagebox.showinfo("FSM", "Достигнут конец списка состояний или неверный переход.")
        except Exception as e:
            messagebox.showerror("FSM Transition Error", str(e))
    
    def reset_fsm(self):
        # Завершаем MATLAB-движок, если он запущен
        if self.fsm.matlab_eng is not None:
            self.fsm.matlab_eng.quit()

        # Создаём новый экземпляр FSM с теми же параметрами
        self.fsm = AmplifierDPDFSM(
            data_path=self.fsm.data_path,
            snr_range=self.fsm.snr_range,
            num_realizations=self.fsm.num_realizations,
            pa_model_type=self.pa_model_choice.get() or "GMP",
            dpd_model_type=self.dpd_model_choice.get() or "GMP",
            pa_epochs=int(self.pa_epochs_entry.get() or 10),
            dpd_epochs=int(self.dpd_epochs_entry.get() or 10),
            pa_lr=float(self.pa_lr_entry.get() or 0.01),
            dpd_lr=float(self.dpd_lr_entry.get() or 0.01)
        )

        # Обновляем статус
        self.status_label.config(text="Текущее состояние: INIT")
        messagebox.showinfo("Сброс", "FSM была успешно сброшена.")

    def state_to_name(self, state):
        """Перевод состояния в строку для GUI."""
        mapping = {
            FSMState.INIT: "INIT",
            FSMState.ESTIMATE_GAIN: "ESTIMATE_GAIN",
            FSMState.SELECT_PA_MODEL_TYPE: "SELECT_PA_MODEL_TYPE",
            FSMState.CONFIGURE_PA_MODEL: "CONFIGURE_PA_MODEL",
            FSMState.LOAD_OR_TRAIN_PA_MODEL: "LOAD_OR_TRAIN_PA_MODEL",
            FSMState.PLOT_PA_SPECTRUM: "PLOT_PA_SPECTRUM",
            FSMState.SELECT_DPD_MODEL_TYPE: "SELECT_DPD_MODEL_TYPE",
            FSMState.CONFIGURE_DPD_MODEL: "CONFIGURE_DPD_MODEL",
            FSMState.SELECT_DPD_ARCHS: "SELECT_DPD_ARCHS",
            FSMState.LOAD_OR_TRAIN_DPD_ARCH: "LOAD_OR_TRAIN_DPD_ARCH",
            FSMState.STORE_DPD_RESULTS: "STORE_DPD_RESULTS",
            FSMState.PLOT_DPD_SPECTRUM: "PLOT_DPD_SPECTRUM",
            FSMState.SELECT_NOISE_RANGE: "SELECT_NOISE_RANGE",
            FSMState.EVALUATE_NOISE_LOOP: "EVALUATE_NOISE_LOOP",
            FSMState.PLOT_NOISE_RESULTS: "PLOT_NOISE_RESULTS",
            FSMState.DONE: "DONE",
            FSMState.ERROR: "ERROR"
        }
        return mapping.get(state, "Неизвестно")

    def plot_results(self):
        """Пример отдельно для каких-то итоговых графиков (опционально)."""
        # В данном каркасе все графики строятся внутри FSM, здесь можно ничего не делать
        pass


# ----------------------------
# 4. Запуск приложения
# ----------------------------
if __name__ == "__main__":
    app = FSMGUI(data_path="DPA_200MHz")  # data_path указывается тут
    app.mainloop()
