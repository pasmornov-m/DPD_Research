import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import enum
from modules import data_loader, metrics, learning, plotting, utils
from modules.gmp_model import GMP
from modules.gmp_narx_model import GMP_NARX
from modules.moe_ar_gmp_model import MoE_GMP_AR
import matlab.engine

# --- FSM State Machine ---
class FSMState(enum.Enum):
    INIT = 0
    LOAD_DATA = 1
    ESTIMATE_GAIN = 2
    TRAIN_PA_MODEL = 3
    EVALUATE_PA_SPECTRUM = 4
    DPD_ARCHITECTURE_SELECT = 5
    TRAIN_DPD_DLA = 6
    TRAIN_DPD_ILA = 7
    TRAIN_DPD_ILC = 8
    STORE_DPD_RESULTS = 9
    METRICS_SUMMARY = 10
    EVALUATE_NOISE_LOOP = 11
    NOISE_ITER_DLA = 12
    NOISE_ITER_ILA = 13
    NOISE_ITER_ILC = 14
    NOISE_ITER_UK = 15
    PLOT_NOISE_RESULTS = 16
    DONE = 17
    ERROR = 99

# --- FSM logic for the algorithm ---
class AmplifierDPDFSM:
    def __init__(self, data_path, snr_range, num_realizations, pa_model_type, dpd_model_type, pa_epochs, dpd_epochs, pa_lr, dpd_lr):
        self.state = FSMState.INIT
        self.data_path = data_path
        self.snr_range = snr_range
        self.num_realizations = num_realizations
        
        self.matlab_eng = None
        self.data = None
        self.config = None
        self.fs = None
        self.bw_main_ch = None
        self.bw_sub_ch = None
        self.n_sub_ch = None
        self.nperseg = None
        
        self.x_train = None
        self.y_train = None

        self.x_val = None
        self.y_val = None
        
        self.gain = None
        self.x_target = None; self.y_target = None
        
        # PA-модель
        self.pa_model = None
        self.y_gmp_pa = None
        self.nmse_pa = None
        
        # DPD результаты (без шума)
        self.dpd_names = ["DLA", "ILA", "ILC"]
        self.current_dpd_index = 0
        self.results_no_noise = {name: {} for name in self.dpd_names}
        
        # Результаты в условиях шума
        self.noise_results = {
            "DLA": {"nmse": [], "acpr_left": [], "acpr_right": []},
            "ILA": {"nmse": [], "acpr_left": [], "acpr_right": []},
            "ILC": {"nmse": [], "acpr_left": [], "acpr_right": []},
            "UK":  {"nmse": [], "acpr_left": [], "acpr_right": []}
        }
        self.current_snr_index = 0

        # Параметры обучения и модели
        self.pa_model_type = pa_model_type
        self.dpd_model_type = dpd_model_type
        self.pa_epochs = pa_epochs
        self.dpd_epochs = dpd_epochs
        self.pa_lr = pa_lr
        self.dpd_lr = dpd_lr

    def run(self):
        try:
            while True:
                if self.state == FSMState.INIT:
                    self._init()
                elif self.state == FSMState.LOAD_DATA:
                    self._load_data()
                elif self.state == FSMState.ESTIMATE_GAIN:
                    self._estimate_gain()
                elif self.state == FSMState.TRAIN_PA_MODEL:
                    self._train_pa_model()
                elif self.state == FSMState.EVALUATE_PA_SPECTRUM:
                    self._evaluate_pa_spectrum()
                elif self.state == FSMState.DPD_ARCHITECTURE_SELECT:
                    self._select_next_dpd()
                elif self.state == FSMState.TRAIN_DPD_DLA:
                    self._train_dpd_dla()
                elif self.state == FSMState.TRAIN_DPD_ILA:
                    self._train_dpd_ila()
                elif self.state == FSMState.TRAIN_DPD_ILC:
                    self._train_dpd_ilc()
                elif self.state == FSMState.STORE_DPD_RESULTS:
                    self._store_dpd_results()
                elif self.state == FSMState.METRICS_SUMMARY:
                    self._metrics_summary()
                elif self.state == FSMState.EVALUATE_NOISE_LOOP:
                    self._evaluate_noise_loop()
                elif self.state == FSMState.NOISE_ITER_DLA:
                    self._noise_iter("DLA")
                elif self.state == FSMState.NOISE_ITER_ILA:
                    self._noise_iter("ILA")
                elif self.state == FSMState.NOISE_ITER_ILC:
                    self._noise_iter("ILC")
                elif self.state == FSMState.NOISE_ITER_UK:
                    self._noise_iter("UK")
                elif self.state == FSMState.PLOT_NOISE_RESULTS:
                    self._plot_noise_results()
                elif self.state == FSMState.DONE:
                    self._done()
                    break
                else:
                    raise RuntimeError(f"Неизвестное состояние: {self.state}")
        except Exception as e:
            print("Произошла ошибка:", e)
            self.state = FSMState.ERROR
    
    # Реализация каждого состояния (пример)
    def _init(self):
        print("STATE: INIT — запуск MATLAB-движка …")
        self.matlab_eng = matlab.engine.start_matlab()
        self.state = FSMState.LOAD_DATA

    def _load_data(self):
        print("STATE: LOAD_DATA — загружаем данные …")
        self.data = data_loader.load_data(self.data_path)
        self.config = self.data["config"]
        self.fs = self.config["input_signal_fs"]
        self.bw_main_ch = self.config["bw_main_ch"]
        self.bw_sub_ch = self.config["bw_sub_ch"]
        self.n_sub_ch = self.config["n_sub_ch"]
        self.nperseg = float(self.config["nperseg"])
        
        self.x_train = self.data["train_input"]
        self.y_train = self.data["train_output"]
        self.x_val = self.data["val_input"]
        self.y_val = self.data["val_output"]
        
        self.state = FSMState.ESTIMATE_GAIN

    def _estimate_gain(self):
        print("STATE: ESTIMATE_GAIN — оцениваем усиление PA …")
        self.gain = metrics.calculate_gain_complex(self.x_train, self.y_train)
        self.x_target = self.gain * self.x_train
        self.y_target = self.gain * self.x_val
        self.state = FSMState.TRAIN_PA_MODEL

    def _train_pa_model(self):
        print(f"STATE: TRAIN_PA_MODEL — обучение модели {self.pa_model_type} …")
        
        if self.pa_model_type == "GMP":
            self.pa_model = GMP(5, 5, 5, 5, 5, 5, 5, 5, model_type="pa_grad")
        elif self.pa_model_type == "GMP_NARX":
            self.pa_model = GMP_NARX(5, 5, 5, 5, 5, 5, 5, 5, model_type="pa_grad")
        elif self.pa_model_type == "MoE_GMP_AR":
            self.pa_model = MoE_GMP_AR(5, 5, 5, 5, 5, 5, 5, 5, model_type="pa_grad")
        else:
            raise ValueError(f"Неподдерживаемая модель PA: {self.pa_model_type}")

        self.pa_model.optimize_coefficients_grad(self.x_train, self.y_train, epochs=self.pa_epochs, learning_rate=self.pa_lr)
        self.y_gmp_pa = self.pa_model.forward(self.x_val).detach()
        self.nmse_pa = metrics.compute_nmse(self.y_gmp_pa, self.y_val)
        self.state = FSMState.EVALUATE_PA_SPECTRUM

    def _select_next_dpd(self):
        if self.current_dpd_index < len(self.dpd_names):
            name = self.dpd_names[self.current_dpd_index]
            if name == "DLA": 
                self.state = FSMState.TRAIN_DPD_DLA
            elif name == "ILA": 
                self.state = FSMState.TRAIN_DPD_ILA
            elif name == "ILC": 
                self.state = FSMState.TRAIN_DPD_ILC
        else:
            self.state = FSMState.METRICS_SUMMARY

    def _train_dpd_dla(self):
        print(f"STATE: TRAIN_DPD_DLA — обучаем модель DPD {self.dpd_model_type} …")
        if self.dpd_model_type == "GMP":
            dpd = GMP(2, 2, 2, 2, 2, 2, 2, 2, model_type="dpd_dla_grad")
        elif self.dpd_model_type == "GMP_NARX":
            dpd = GMP_NARX(2, 2, 2, 2, 2, 2, 2, 2, model_type="dpd_dla_grad")
        elif self.dpd_model_type == "MoE_GMP_AR":
            dpd = MoE_GMP_AR(2, 2, 2, 2, 2, 2, 2, 2, model_type="dpd_dla_grad")
        else:
            raise ValueError(f"Неподдерживаемая модель DPD: {self.dpd_model_type}")

        learning.optimize_dla_grad(self.x_train, self.x_target, dpd, self.pa_model, epochs=self.dpd_epochs, learning_rate=self.dpd_lr)
        self.dpd_temp_result = {"name": "DLA", "nmse": metrics.compute_nmse(dpd.forward(self.x_val), self.y_target)}
        self.state = FSMState.STORE_DPD_RESULTS

    def _store_dpd_results(self):
        name = self.dpd_temp_result["name"]
        self.results_no_noise[name] = dict(self.dpd_temp_result)
        self.current_dpd_index += 1
        self.state = FSMState.DPD_ARCHITECTURE_SELECT

    def _metrics_summary(self):
        names = self.dpd_names
        nmse_vals = [self.results_no_noise[n]["nmse"] for n in names]
        plt.figure(figsize=(6,4))
        plt.bar(names, nmse_vals, color=['blue','orange','green'])
        plt.ylabel("NMSE (dB)")
        plt.title("Сравнение NMSE без шума")
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        plt.show()
        self.state = FSMState.EVALUATE_NOISE_LOOP

    def _evaluate_noise_loop(self):
        if self.current_snr_index < len(self.snr_range):
            snr = self.snr_range[self.current_snr_index]
            self.state = FSMState.NOISE_ITER_DLA
        else:
            self.state = FSMState.PLOT_NOISE_RESULTS

    def _noise_iter(self, mode):
        snr = self.snr_range[self.current_snr_index]
        nmse_vals = []  # Список для хранения значений NMSE для каждой итерации
        acpr_left_vals = []  # Список для хранения значений ACPR для левого канала
        acpr_right_vals = []  # Список для хранения значений ACPR для правого канала
        
        # Цикл по количеству реализаций
        for _ in range(self.num_realizations):
            if mode == "DLA":
                Ka = La = Kb = Lb = Mb = Kc = Lc = Mc = 3
                dpd = GMP(Ka, La, Kb, Lb, Mb, Kc, Lc, Mc, model_type="dpd_dla")
                learning.optimize_dla_grad(self.x_in, self.x_target, dpd,
                                        self.pa_model, epochs=500, lr=0.01,
                                        use_noise=True, snr=snr,
                                        fs=self.fs, bw_main_ch=self.bw_main_ch)
                y_pa = self.pa_model.forward(dpd.forward(self.y_in)).detach()
                
                # Получаем NMSE, ACPR левого и правого канала
                nmse_cur, left, right = metrics.noise_realizations(
                    self.num_realizations, self.eng, y_pa, self.y_target,
                    snr, self.fs, self.bw_main_ch, self.nperseg
                )
            
            elif mode == "ILA":
                Ka = La = Kb = Lb = Mb = Kc = Lc = Mc = 3
                dpd = GMP(Ka, La, Kb, Lb, Mb, Kc, Lc, Mc, model_type="dpd_ila")
                learning.optimize_ila_grad(dpd, self.x_in, self.x_target,
                                        self.gain, epochs=500, lr=0.01,
                                        pa_model=self.pa_model,
                                        use_noise=True, snr=snr,
                                        fs=self.fs, bw_main_ch=self.bw_main_ch)
                y_pa = self.pa_model.forward(dpd.forward(self.y_in)).detach()
                
                # Получаем NMSE, ACPR левого и правого канала
                nmse_cur = metrics.compute_nmse(y_pa, self.y_target)
                left, right = metrics.calculate_acpr(
                    self.eng, y_pa, self.fs, self.bw_main_ch, self.nperseg
                )
            
            elif mode == "ILC":
                # Генерация оптимального сигнала u_k с шумом
                u_k = learning.ilc_signal_grad(self.x_in, self.x_target,
                                            self.pa_model, 500, 0.001,
                                            use_noise=True, snr=snr,
                                            fs=self.fs, bw_main_ch=self.bw_main_ch)
                u_k_pa = self.pa_model.forward(u_k).detach()
                
                # Получаем NMSE, ACPR левого и правого канала
                nmse_cur, left, right = metrics.noise_realizations(
                    self.num_realizations, self.eng, u_k_pa, self.x_target,
                    snr, self.fs, self.bw_main_ch, self.nperseg
                )
            
            elif mode == "UK":
                u_k = learning.ilc_signal_grad(self.x_in, self.x_target,
                                            self.pa_model, 500, 0.001,
                                            use_noise=True, snr=snr,
                                            fs=self.fs, bw_main_ch=self.bw_main_ch)
                u_k_pa = self.pa_model.forward(u_k).detach()
                
                # Получаем NMSE, ACPR левого и правого канала
                nmse_cur, left, right = metrics.noise_realizations(
                    self.num_realizations, self.eng, u_k_pa, self.x_target,
                    snr, self.fs, self.bw_main_ch, self.nperseg
                )
            
            else:
                raise RuntimeError("Неподдерживаемый режим NOISE_ITER: " + mode)
            
            # Добавляем текущие значения в списки
            nmse_vals.append(nmse_cur)
            acpr_left_vals.append(left)
            acpr_right_vals.append(right)
        
        # Расчёт среднего значения NMSE
        nmse_mean = sum(nmse_vals) / len(nmse_vals) if nmse_vals else 0.0
        acpr_left_mean = sum(acpr_left_vals) / len(acpr_left_vals) if acpr_left_vals else 0.0
        acpr_right_mean = sum(acpr_right_vals) / len(acpr_right_vals) if acpr_right_vals else 0.0
        
        # Добавляем усреднённые результаты в список результатов
        self.noise_results[mode]["nmse"].append(nmse_mean)
        self.noise_results[mode]["acpr_left"].append(acpr_left_mean)
        self.noise_results[mode]["acpr_right"].append(acpr_right_mean)
        
        # Переходим к следующему этапу
        if mode == "DLA":
            self.state = FSMState.NOISE_ITER_ILA
        elif mode == "ILA":
            self.state = FSMState.NOISE_ITER_ILC
        elif mode == "ILC":
            self.state = FSMState.NOISE_ITER_UK
        elif mode == "UK":
            # Закончили цикл для текущего SNR → переходим к следующему SNR
            self.current_snr_index += 1
            self.state = FSMState.EVALUATE_NOISE_LOOP
        else:
            self.state = FSMState.ERROR
    
    def _plot_noise_results(self):
        snr_vals = self.snr_range
        plt.figure(figsize=(9,5))
        plt.plot(snr_vals, self.noise_results["DLA"]["nmse"], linestyle='-', marker='o',   label="DLA")
        plt.xlabel("SNR (дБ)"); plt.ylabel("NMSE (дБ)")
        plt.title("NMSE vs SNR для разных корректорах")
        plt.legend(); plt.grid(); plt.show()
        self.state = FSMState.DONE

    def _done(self):
        print("STATE: DONE — завершение работы!")
        if self.matlab_eng is not None:
            self.matlab_eng.quit()

# --- GUI --- 
class FSMGUI(tk.Tk):
    def __init__(self, fsm):
        super().__init__()
        self.title("FSM для усилителя и корректора")
        self.geometry("1000x1000")
        self.fsm = fsm

        # Текущий статус FSM
        self.status_label = ttk.Label(self, text="Текущее состояние: INIT", width=30)
        self.status_label.pack(pady=10)

        # Параметры модели PA
        self.pa_model_label = ttk.Label(self, text="Модель PA:")
        self.pa_model_label.pack(pady=5)
        self.pa_model_choice = ttk.Combobox(self, values=["GMP", "GMP_NARX", "MoE_GMP_AR"])
        self.pa_model_choice.pack(pady=5)

        # Параметры модели DPD
        self.dpd_model_label = ttk.Label(self, text="Модель DPD:")
        self.dpd_model_label.pack(pady=5)
        self.dpd_model_choice = ttk.Combobox(self, values=["GMP", "GMP_NARX", "MoE_GMP_AR"])
        self.dpd_model_choice.pack(pady=5)

        # Параметры эпох и скорости обучения
        self.pa_epochs_label = ttk.Label(self, text="Эпохи PA:")
        self.pa_epochs_label.pack(pady=5)
        self.pa_epochs_entry = ttk.Entry(self)
        self.pa_epochs_entry.pack(pady=5)
        self.pa_lr_label = ttk.Label(self, text="Скорость обучения PA:")
        self.pa_lr_label.pack(pady=5)
        self.pa_lr_entry = ttk.Entry(self)
        self.pa_lr_entry.pack(pady=5)

        self.dpd_epochs_label = ttk.Label(self, text="Эпохи DPD:")
        self.dpd_epochs_label.pack(pady=5)
        self.dpd_epochs_entry = ttk.Entry(self)
        self.dpd_epochs_entry.pack(pady=5)
        self.dpd_lr_label = ttk.Label(self, text="Скорость обучения DPD:")
        self.dpd_lr_label.pack(pady=5)
        self.dpd_lr_entry = ttk.Entry(self)
        self.dpd_lr_entry.pack(pady=5)

        # Кнопки для управления FSM
        self.start_button = ttk.Button(self, text="Запустить FSM", command=self.start_fsm)
        self.start_button.pack(pady=5)

        self.next_state_button = ttk.Button(self, text="Следующий этап", command=self.next_state)
        self.next_state_button.pack(pady=5)

        # Вывод графиков
        self.plot_button = ttk.Button(self, text="Показать графики", command=self.plot_results)
        self.plot_button.pack(pady=5)

    def update_status(self):
        """Обновить текущий статус FSM в интерфейсе"""
        state_name = self.state_to_name(self.fsm.state)
        self.status_label.config(text=f"Текущее состояние: {state_name}")

    def start_fsm(self):
        """Запуск FSM"""
        # Получаем параметры от пользователя
        data_path = "path_to_data"  # Здесь можно добавить поле для ввода пути
        snr_range = [20, 22, 24, 26]
        num_realizations = 10
        pa_model_type = self.pa_model_choice.get()
        dpd_model_type = self.dpd_model_choice.get()
        pa_epochs = int(self.pa_epochs_entry.get())
        dpd_epochs = int(self.dpd_epochs_entry.get())
        pa_lr = float(self.pa_lr_entry.get())
        dpd_lr = float(self.dpd_lr_entry.get())

        # Создаём FSM с выбранными параметрами
        fsm = AmplifierDPDFSM(data_path, snr_range, num_realizations, pa_model_type, dpd_model_type, pa_epochs, dpd_epochs, pa_lr, dpd_lr)
        fsm.run()
        self.update_status()

    def next_state(self):
        """Перевести FSM в следующее состояние"""
        self.fsm.state += 1
        self.update_status()

    def state_to_name(self, state):
        """Преобразование номера состояния в строковое имя"""
        state_dict = {
            FSMState.INIT: "INIT",
            FSMState.LOAD_DATA: "LOAD_DATA",
            FSMState.ESTIMATE_GAIN: "ESTIMATE_GAIN",
            FSMState.TRAIN_PA_MODEL: "TRAIN_PA_MODEL",
            FSMState.DPD_ARCHITECTURE_SELECT: "DPD_ARCHITECTURE_SELECT",
            FSMState.TRAIN_DPD_DLA: "TRAIN_DPD_DLA",
            FSMState.TRAIN_DPD_ILA: "TRAIN_DPD_ILA",
            FSMState.TRAIN_DPD_ILC: "TRAIN_DPD_ILC",
            FSMState.STORE_DPD_RESULTS: "STORE_DPD_RESULTS",
            FSMState.METRICS_SUMMARY: "METRICS_SUMMARY",
            FSMState.EVALUATE_NOISE_LOOP: "EVALUATE_NOISE_LOOP",
            FSMState.NOISE_ITER_DLA: "NOISE_ITER_DLA",
            FSMState.NOISE_ITER_ILA: "NOISE_ITER_ILA",
            FSMState.NOISE_ITER_ILC: "NOISE_ITER_ILC",
            FSMState.NOISE_ITER_UK: "NOISE_ITER_UK",
            FSMState.PLOT_NOISE_RESULTS: "PLOT_NOISE_RESULTS",
            FSMState.DONE: "DONE",
            FSMState.ERROR: "ERROR",
        }
        return state_dict.get(state, "Неизвестное состояние")

    def plot_results(self):
        """Отобразить графики результатов работы FSM"""
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(self.fsm.snr_range, self.fsm.results['DLA'], label="DLA", marker='o')
        ax.plot(self.fsm.snr_range, self.fsm.results['ILA'], label="ILA", marker='s')
        ax.plot(self.fsm.snr_range, self.fsm.results['ILC'], label="ILC", marker='d')
        ax.set_xlabel("SNR (дБ)")
        ax.set_ylabel("Результаты")
        ax.legend()
        ax.grid(True)

        # Отображаем график в Tkinter
        canvas = FigureCanvasTkAgg(fig, master=self)
        canvas.draw()
        canvas.get_tk_widget().pack(pady=10)

# --- Основная часть программы ---
if __name__ == "__main__":
    gui = FSMGUI(None)  # Передаем None, так как fsm будет передан после настройки
    gui.mainloop()
