import enum
import numpy as np
import matplotlib.pyplot as plt
from modules.params import ModelParams
from modules import data_loader, metrics, learning
from modules.gmp_model import GMP

class FSM_State(enum.Enum):
    INIT = enum.auto()
    ESTIMATE_GAIN = enum.auto()
    CONFIGURE_PA = enum.auto()
    TRAIN_PA = enum.auto()
    PLOT_PA = enum.auto()
    CONFIGURE_DPD = enum.auto()
    TRAIN_DPD = enum.auto()
    PLOT_DPD = enum.auto()
    SELECT_NOISE_RANGE = enum.auto()
    EVALUATE_NOISE = enum.auto()
    PLOT_NOISE = enum.auto()
    DONE = enum.auto()
    ERROR = enum.auto()

class PA_DPD_FSM:
    _HANDLERS = {
        FSM_State.INIT: "_init",
        FSM_State.ESTIMATE_GAIN: "_estimate_gain",
        FSM_State.CONFIGURE_PA: "_configure_pa",
        FSM_State.TRAIN_PA: "_train_pa",
        FSM_State.PLOT_PA: "_plot_pa",
        FSM_State.CONFIGURE_DPD: "_configure_dpd",
        FSM_State.TRAIN_DPD: "_train_dpd",
        FSM_State.PLOT_DPD: "_plot_dpd",
        FSM_State.SELECT_NOISE_RANGE: "_select_noise_range",
        FSM_State.EVALUATE_NOISE: "_evaluate_noise",
        FSM_State.PLOT_NOISE: "_plot_noise",
    }

    def __init__(self, data_path: str):
        self.data_path = data_path
        self.state = FSM_State.INIT

        # данные
        self.data = None
        self.config = None
        self.fs = None
        self.bw_main_ch = None
        self.bw_sub_ch = None
        self.n_sub_ch = None
        self.sub_ch = None
        self.nperseg = None

        # ACPR meter
        self.acpr_meter = None

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
        self.pa_params: ModelParams = None
        self.pa_model = None
        self.nmse_pa = None

        # DPD
        self.dpd_model_type = None
        self.dpd_params: ModelParams = None
        self.dpd_archs = []
        # self.selected_archs = []
        self.dpd_models = {} # словарь: dpd_models["DLA"] = экземпляр модели
        
        # результаты
        self.results_no_noise = {} # {"DLA": {"nmse":…, "acpr":(l,r)}, …}
        self.noise_range = []
        self.num_realizations = None
        self.current_snr_index = 0
        # словарь шумовых результатов: noise_results["DLA"] = {"nmse":[…], "acpr_left":[…], "acpr_right":[…]}
        self.noise_results = {}

        # Вспомогательные индексы
        self.current_arch_index = 0

    # --- интерфейс от GUI ---
    def set_pa_params(self, params: ModelParams):
        self.pa_params = params

    def set_dpd_params(self, params: ModelParams):
        self.dpd_params = params

    def set_noise_range(self, snr_range: list[int], num_realizations: int = 1):
        start, end, step = snr_range
        self.noise_range = [snr for snr in range(start, end, step)]
        self.num_realizations = num_realizations

    # --- главный цикл ---
    def run(self):
        try:
            while self.state not in (FSM_State.DONE, FSM_State.ERROR):
                handler = getattr(self, self._HANDLERS[self.state], None)
                if handler is None:
                    raise RuntimeError(f"No handler for state {self.state}")
                handler()
        except Exception as e:
            self.state = FSM_State.ERROR
            print("FSM error:", e)

    # ----------------------------
    #  Реализация состояний
    # ----------------------------

    def _init(self):
        """INIT → ESTIMATE_GAIN."""
        print(">> STATE: INIT — загрузка данных …")
        
        self.data = data_loader.load_data(self.data_path)
        config = self.data["config"]
        self.fs = config["input_signal_fs"]
        self.bw_main_ch = config["bw_main_ch"]
        self.bw_sub_ch = config["bw_sub_ch"]
        self.n_sub_ch = config["n_sub_ch"]
        self.sub_ch = config["sub_ch"]
        self.nperseg = config["nperseg"]

        self.x_train = self.data["train_input"]
        self.y_train = self.data["train_output"]
        self.x_val = self.data["val_input"]
        self.y_val = self.data["val_output"]

        self.dpd_archs = ["dla","ila", "ilc"]
        self.results_no_noise = {arch: {} for arch in self.dpd_archs}
        self.results_no_noise["uk"] = {}
        self.noise_results = {arch: {"nmse":[], "acpr_left":[], "acpr_right":[]} for arch in self.dpd_archs}
        self.noise_results["uk"] = {"nmse":[], "acpr_left":[], "acpr_right":[]}

        self.acpr_meter = metrics.ACPR(
            sample_rate=self.fs,
            main_measurement_bandwidth=self.bw_main_ch,
            adjacent_channel_offset=[-self.sub_ch, self.sub_ch],
            segment_length=self.nperseg,
            overlap_percentage=60,
            window='blackmanharris',
            fft_length=self.nperseg,
            power_units='dBW',
            return_main_power=True,
            return_adjacent_powers=True
        )

        self.state = FSM_State.ESTIMATE_GAIN

    def _estimate_gain(self):
        """ESTIMATE_GAIN → SELECT_PA_MODEL_TYPE."""
        print(">> STATE: ESTIMATE_GAIN — вычисляем gain …")
        self.gain = metrics.calculate_gain_complex(self.x_train, self.y_train)
        print(f"    Gain = {self.gain:.2f}")
        self.y_train_target = self.gain * self.x_train
        self.y_val_target = self.gain * self.x_val

        self.state = FSM_State.CONFIGURE_PA

    def _configure_pa(self):
        assert self.pa_params, "PA params not set"
        p = self.pa_params

        cls = {"GMP": GMP}[p.model_type]
        model = cls(p.Ka,p.La,p.Kb,p.Lb,p.Mb,p.Kc,p.Lc,p.Mc, model_type="pa_grad")

        self.pa_model = model

        if not model.load_coefficients():
            self.state = FSM_State.TRAIN_PA
            return

        y_pa = model.forward(self.x_val).detach()
        self.nmse_pa = metrics.compute_nmse(y_pa, self.y_val)
        self.state = FSM_State.PLOT_PA

    def _train_pa(self):
        p = self.pa_params

        self.pa_model.optimize_coefficients_grad(self.x_train, self.y_train, epochs=p.epochs, learning_rate=p.lr)
        self.pa_model.save_coefficients()

        y_pa = self.pa_model.forward(self.x_val).detach()
        self.nmse_pa = metrics.compute_nmse(y_pa, self.y_val)

        self.state = FSM_State.PLOT_PA

    def _plot_pa(self):
        y_pa = self.pa_model.forward(self.x_val).detach()
        freqs, spectrum_out = metrics.power_spectrum(self.y_val, self.fs, self.nperseg)
        _, spectrum_pa   = metrics.power_spectrum(y_pa, self.fs, self.nperseg)

        plt.figure(figsize=(8,4))
        plt.plot(freqs/1e6,10*np.log10(np.abs(spectrum_out)), label="out")
        plt.plot(freqs/1e6,10*np.log10(np.abs(spectrum_pa)), label=f"PA (NMSE {self.nmse_pa:.1f} dB)")
        plt.legend()
        plt.grid()
        plt.show()

        self.state = FSM_State.CONFIGURE_DPD

    def _configure_dpd(self):
        assert self.dpd_params, "DPD params not set"
        p = self.dpd_params
        
        if self.current_arch_index == 0:
            self.dpd_models = {}
            self.results_no_noise = {}

        # Проходим по архитектурам начиная с current_arch_index
        for i in range(self.current_arch_index, len(self.dpd_archs)):
            arch = self.dpd_archs[i]
            # создаём модель нужного класса
            cls = {"GMP": GMP}[p.model_type]
            model = cls(p.Ka,p.La,p.Kb,p.Lb,p.Mb,p.Kc,p.Lc,p.Mc,
                        model_type=f"dpd_{arch.lower()}_grad")

            # если нет сохранённого — нужно обучить
            if not model.load_coefficients():
                # запоминаем, какую архитектуру тренить
                self.current_arch = arch
                self.current_arch_index = i
                self.dpd_models[arch] = model # сохраним для TRAIN_DPD
                self.state = FSM_State.TRAIN_DPD
                return # прерываем конфигурирование

            # иначе сразу считаем метрики и сохраняем
            y_dpd = model.forward(self.x_val).detach()
            y_lin = self.pa_model.forward(y_dpd).detach()
            nmse = metrics.compute_nmse(y_lin, self.gain * self.x_val)
            acpr_l, acpr_r = metrics.calculate_acpr(y_lin, self.acpr_meter)
            spectrum = metrics.power_spectrum(y_lin, self.fs, self.nperseg)

            self.dpd_models[arch] = model

            self.results_no_noise[arch] = {
                "nmse": nmse,
                "acpr": (acpr_l, acpr_r),
                "spectrum": spectrum
            }

        u_k = learning.ilc_signal_grad(
            self.x_train, self.y_train_target, self.pa_model,
            epochs=500, learning_rate=0.001
        )
        u_k_pa = self.pa_model.forward(u_k).detach()
        self.results_no_noise["uk"] = {
            "nmse": metrics.compute_nmse(u_k_pa, self.y_train_target),
            "acpr": metrics.calculate_acpr(u_k_pa, self.acpr_meter),
            "spectrum": metrics.power_spectrum(u_k_pa, self.fs, self.nperseg)
        }

        # если все architectures обработаны — идём к PLOT_DPD
        self.state = FSM_State.PLOT_DPD

    def _train_dpd(self):
        arch = self.current_arch
        p = self.dpd_params
        model = self.dpd_models[arch]

        # запускаем обучение для этой архитектуры
        if arch == "DLA":
            learning.optimize_dla_grad(
                self.x_train, self.y_train_target,
                model, self.pa_model,
                epochs=p.epochs, learning_rate=p.lr
            )
            model.save_coefficients()
        elif arch == "ILA":
            learning.optimize_ila_grad(
                model, self.x_train, self.y_train, self.gain,
                epochs=p.epochs, learning_rate=p.lr
            )
            model.save_coefficients()
        elif arch == "ILC":
            u_k = learning.ilc_signal_grad(
                self.x_train, self.y_train_target, self.pa_model,
                epochs=500, learning_rate=0.001
            )
            uk_pa = self.pa_model.forward(u_k).detach()
            self.results_no_noise["uk"] = {
                "nmse": metrics.compute_nmse(uk_pa, self.y_train_target),
                "acpr": metrics.calculate_acpr(uk_pa, self.acpr_meter),
                "spectrum": metrics.power_spectrum(uk_pa, self.fs, self.nperseg)
            }
            model.optimize_coefficients_grad(
                self.x_train, u_k,
                epochs=p.epochs, learning_rate=p.lr
            )
            model.save_coefficients()

        y_dpd = model.forward(self.x_val).detach()
        y_lin = self.pa_model.forward(y_dpd).detach()
        nmse = metrics.compute_nmse(y_lin, self.y_val_target)
        acpr_l, acpr_r = metrics.calculate_acpr(y_lin, self.acpr_meter)
        spectrum = metrics.power_spectrum(y_lin, self.fs, self.nperseg)

        self.results_no_noise[arch] = {
            "nmse": nmse,
            "acpr": (acpr_l, acpr_r),
            "spectrum": spectrum
        }

        # переходим к следующей архитектуре
        self.current_arch_index += 1
        self.state = FSM_State.CONFIGURE_DPD

    def _plot_dpd(self):
        # Спектры DPD
        freqs, spectrum_in = metrics.power_spectrum(self.x_val, self.fs, self.nperseg)
        plt.figure(figsize=(8,4))
        plt.plot(freqs/1e6, 10*np.log10(np.abs(spectrum_in)), label="in")

        # PA
        y_pa = self.pa_model.forward(self.x_val).detach()
        _, spec_pa = metrics.power_spectrum(y_pa, self.fs, self.nperseg)
        plt.plot(freqs/1e6, 10*np.log10(np.abs(spec_pa)), label="PA")

        # DPD archs
        for arch in self.dpd_archs:
            f, s = self.results_no_noise[arch]["spectrum"]
            plt.plot(f/1e6, 10*np.log10(np.abs(s)), label=arch.upper())

        # UK сигнал
        fuk, suk = self.results_no_noise["uk"]["spectrum"]
        plt.plot(fuk/1e6, 10*np.log10(np.abs(suk)), "--", label="uk")

        plt.legend(); plt.grid(); plt.show()
        self.state = FSM_State.SELECT_NOISE_RANGE
    
    def _select_noise_range(self):
        """SELECT_NOISE_RANGE → EVALUATE_NOISE."""
        # Предполагаем, что GUI уже вызвал set_noise_range()
        assert self.noise_range, "Noise range not set!"
        assert self.num_realizations is not None, "num_realizations not set!"

        # Инициализируем структуру для накопления результатов
        self.noise_results = {
            arch: {"nmse": [], "acpr_left": [], "acpr_right": []}
            for arch in self.dpd_models
        }
        self.noise_results["uk"] = {"nmse": [], "acpr_left": [], "acpr_right": []}

        self.current_snr_index = 0
        self.state = FSM_State.EVALUATE_NOISE

    def _evaluate_noise(self):
        """EVALUATE_NOISE → PLOT_NOISE."""
        # Бежим по всем SNR
        for snr in self.noise_range:
            print(f">> STATE: EVALUATE_NOISE — SNR = {snr} dB")

            u_k_noisy = learning.ilc_signal_grad(
                self.x_train, self.y_train_target, self.pa_model,
                epochs=500, learning_rate=0.001,
                add_noise=True, snr=snr, fs=self.fs, bw=self.bw_main_ch
            )
            u_k_pa = self.pa_model.forward(u_k_noisy).detach()
            nm_u_k, l_u_k, r_u_k = metrics.noise_realizations(
                self.num_realizations, u_k_pa, self.y_train_target,
                snr, self.fs, self.bw_main_ch, self.acpr_meter
            )
            self.noise_results["uk"]["nmse"].append(nm_u_k)
            self.noise_results["uk"]["acpr_left"].append(l_u_k)
            self.noise_results["uk"]["acpr_right"].append(r_u_k)

            print(f"   • [u_k @ {snr}dB] NMSE={self.noise_results['uk']['nmse'][-1]:.2f}, "
                      f"ACPR_L={self.noise_results['uk']['acpr_left'][-1]:.2f}, "
                      f"ACPR_R={self.noise_results['uk']['acpr_right'][-1]:.2f}")

            for arch, base_model in self.dpd_models.items():
                # создаём чистую копию модели той же архитектуры
                cls = type(base_model)
                params = (
                    base_model.Ka, base_model.La,
                    base_model.Kb, base_model.Lb,
                    base_model.Mb,
                    base_model.Kc, base_model.Lc, base_model.Mc
                )
                dpd_noisy = cls(*params, model_type=base_model.model_type)

                # выбираем стратегию обучения с шумом
                if arch.lower() == "dla":
                    learning.optimize_dla_grad(
                        self.x_train, self.y_train_target,
                        dpd_noisy, self.pa_model,
                        epochs=self.dpd_params.epochs,
                        learning_rate=self.dpd_params.lr,
                        add_noise=True, snr=snr,
                        fs=self.fs, bw=self.bw_main_ch
                    )
                    y_out = self.pa_model.forward(dpd_noisy.forward(self.x_val)).detach()
                    nm, l, r = metrics.noise_realizations(
                        self.num_realizations, y_out, self.y_val_target,
                        snr, self.fs, self.bw_main_ch, self.acpr_meter
                    )

                elif arch.lower() == "ila":
                    learning.optimize_ila_grad(
                        dpd_noisy, self.x_train, self.y_train, self.gain,
                        epochs=self.dpd_params.epochs,
                        learning_rate=self.dpd_params.lr,
                        pa_model=self.pa_model,
                        add_noise=True, snr=snr,
                        fs=self.fs, bw=self.bw_main_ch
                    )
                    y_out = self.pa_model.forward(dpd_noisy.forward(self.x_val)).detach()
                    nm, l, r = metrics.noise_realizations(
                        self.num_realizations, y_out, self.y_val_target,
                        snr, self.fs, self.bw_main_ch, self.acpr_meter
                    )

                elif arch.lower() == "ilc":
                    dpd_noisy.optimize_coefficients_grad(
                        self.x_train, u_k_noisy, 
                        epochs=self.dpd_params.epochs, learning_rate=self.dpd_params.lr
                    )
                    y_out = self.pa_model.forward(dpd_noisy.forward(self.x_val)).detach()
                    nm, l, r = metrics.noise_realizations(
                        self.num_realizations, y_out, self.y_val_target,
                        snr, self.fs, self.bw_main_ch, self.acpr_meter
                    )

                self.noise_results[arch]["nmse"].append(nm)
                self.noise_results[arch]["acpr_left"].append(l)
                self.noise_results[arch]["acpr_right"].append(r)

                print(f"   • [{arch} @ {snr}dB] NMSE={self.noise_results[arch]['nmse'][-1]:.2f}, "
                      f"ACPR_L={self.noise_results[arch]['acpr_left'][-1]:.2f}, "
                      f"ACPR_R={self.noise_results[arch]['acpr_right'][-1]:.2f}")

        # После всех SNR
        self.state = FSM_State.PLOT_NOISE

    def _plot_noise(self):
        """PLOT_NOISE → DONE."""
        snr_vals = self.noise_range

        # 1) NMSE vs SNR
        plt.figure(figsize=(8,4))
        for arch, res in self.noise_results.items():
            plt.plot(snr_vals, res["nmse"], marker='o', label=arch)
        plt.xlabel("SNR (дБ)")
        plt.ylabel("NMSE (дБ)")
        plt.title("NMSE vs SNR")
        plt.grid()
        plt.legend()
        plt.show()

        # 2) ACPR Left & Right vs SNR
        plt.figure(figsize=(12,5))
        plt.subplot(1,2,1)
        for arch, res in self.noise_results.items():
            plt.plot(snr_vals, res["acpr_left"], marker='o', label=arch)
        plt.xlabel("SNR (дБ)")
        plt.ylabel("ACPR Left (дБ)")
        plt.grid()
        plt.legend()

        plt.subplot(1,2,2)
        for arch, res in self.noise_results.items():
            plt.plot(snr_vals, res["acpr_right"], marker='o', label=arch)
        plt.xlabel("SNR (дБ)")
        plt.ylabel("ACPR Right (дБ)")
        plt.grid(); plt.legend()

        plt.tight_layout()
        plt.show()

        self.state = FSM_State.DONE