import enum
import logging
import threading
import numpy as np
import matplotlib.pyplot as plt
from modules import data_loader, metrics, learning, params
from modules.gmp_model import GMP


DPI = 100
SPECTRUM_TITLE = "Спектральная плотность мощнсоти (дБ)"
FREQ_TITLE = "Частота (МГц)"
U_K_EPOCHS = 1000
U_K_LR = 0.001

logger = logging.getLogger("PA_DPD_FSM")
logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)


class FSM_State(enum.Enum):
    INIT = enum.auto()
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

    def __init__(self, data_path: str, gui=None):
        self.data_path = data_path
        self.gui = gui
        self.state = FSM_State.INIT

        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._pause_event.set()

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

        # data_params для snr_metrics
        self.data_params = None

        # PA
        self.pa_params: params.ModelParams = None
        self.pa_model = None
        self.nmse_pa = None

        # DPD
        self.dpd_params: params.ModelParams = None
        self.dpd_archs = ["DLA", "ILA", "ILC"]
        self.current_arch = None
        self.current_arch_index = 0
        self.dpd_models = {}
        self.u_k = None

        # результаты
        self.results_no_noise = {}
        self.noise_range = []
        self.num_realizations = None
        self.current_snr_index = 0
        self.noise_results = {}

        logger.info("PA_DPD_FSM initialized with data_path=%s", data_path)

    # --- интерфейс от GUI ---
    def set_pa_params(self, params: params.ModelParams):
        self.pa_params = params
        logger.info("PA params set: %s", params)

    def set_dpd_params(self, params: params.ModelParams):
        self.dpd_params = params
        logger.info("DPD params set: %s", params)

    def set_noise_range(self, snr_range: list[int], num_realizations: int = 1):
        start, end, step = snr_range
        self.noise_range = [snr for snr in range(start, end, step)]
        self.num_realizations = num_realizations
        logger.info("Noise range set: %s, realizations=%d", self.noise_range, num_realizations)
    
    def pause(self):
        """Приостановить выполнение FSM."""
        self._pause_event.clear()
        logging.info("FSM paused")

    def resume(self):
        """Возобновить, если был на паузе."""
        self._pause_event.set()
        logging.info("FSM resumed")

    def stop(self):
        """
        Остановить FSM: прерывает текущее выполнение. 
        После этого run() выйдет из цикла. 
        """
        self._stop_event.set()
        # если на паузе, разблокируем, чтобы выйти
        self._pause_event.set()
        logging.info("FSM stop requested")

    def reset(self):
        """
        Сбросить FSM к начальному состоянию. 
        Вызывать после stop(), когда FSM-поток завершился.
        """
        # Сброс состояния и индексов
        self.state = FSM_State.INIT
        self.current_arch_index = 0
        self.current_arch = None
        self.results_no_noise.clear()
        self.noise_results.clear()
        self.u_k = None
        self._stop_event.clear()
        self._pause_event.set()
        logging.info("FSM reset to INIT")
    
    def _compute_and_store_results(self, input_signal, target_signal, key, store_dict, models_dict=None, model=None):
        if not model:
            y = input_signal
        else:
            y = model.forward(input_signal).detach()
        y_lin = self.pa_model.forward(y).detach()
        nmse = metrics.compute_nmse(y_lin, target_signal)
        acpr = metrics.calculate_acpr(y_lin, self.acpr_meter)
        spectrum = metrics.power_spectrum(y_lin, self.fs, self.nperseg)
        store_dict[key] = {"nmse": nmse, "acpr": acpr, "spectrum": spectrum}
        if models_dict is not None:
            models_dict[key] = model


    def run(self):
        logger.info("FSM run started")
        try:
            while not self._stop_event.is_set() and self.state not in (FSM_State.DONE, FSM_State.ERROR):
                self._pause_event.wait()
                if self._stop_event.is_set():
                    break

                handler_name = self._HANDLERS.get(self.state)
                if handler_name is None:
                    raise RuntimeError(f"No handler for state {self.state}")
                handler = getattr(self, handler_name)
                logging.info(f"Entering state: {self.state.name}")
                handler()
                if self.gui:
                    self.gui.after(0, self.gui.refresh_status)
                if self._stop_event.is_set():
                    break
            if self._stop_event.is_set():
                logging.info("FSM stopped by request")
            elif self.state == FSM_State.DONE:
                logging.info("FSM completed successfully")
            elif self.state == FSM_State.ERROR:
                logging.error("FSM entered ERROR state")

        except Exception as e:
            self.state = FSM_State.ERROR
            logging.exception("Exception in FSM.run:")
        finally:
            if self.gui:
                self.gui.after(0, self.gui.on_fsm_finished)


    def _init(self):
        logger.info("STATE INIT: загрузка данных")

        if self._stop_event.is_set():
            return

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

        self.gain = metrics.calculate_gain_complex(self.x_train, self.y_train)
        self.y_train_target = self.gain * self.x_train
        self.y_val_target = self.gain * self.x_val
        logger.info("   Gain = %.3f", self.gain)

        self.data_params = params.make_data_params(
                                                    self.x_train, self.y_train_target,
                                                    self.x_val, self.y_val_target
                                                )

        self.current_arch_index = 0
        self.dpd_models.clear()
        self.results_no_noise.clear()
        self.noise_results.clear()
        self.u_k = None

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

        logger.info("   Data loaded: fs=%s, bw_main_ch=%s", self.fs, self.bw_main_ch)
        self.state = FSM_State.CONFIGURE_PA

    def _configure_pa(self):
        assert self.pa_params, "PA params not set"
        logger.info("STATE CONFIGURE_PA: параметры PA %s", self.pa_params)
        if self._stop_event.is_set():
            return
        
        pa_config = params.make_gmp_params(
            Ka=self.pa_params.Ka, 
            La=self.pa_params.La, 
            Kb=self.pa_params.Kb, 
            Lb=self.pa_params.Lb,
            Mb=self.pa_params.Mb, 
            Kc=self.pa_params.Kc, 
            Lc=self.pa_params.Lc, 
            Mc=self.pa_params.Mc,
            model_type="pa_grad"
        )

        self.pa_model = GMP(**pa_config)

        if not self.pa_model.load_weights():
            logger.info("   Нет сохранённых весов PA: перейдем к TRAIN_PA")
            self.state = FSM_State.TRAIN_PA
        else:
            y_pa = self.pa_model.forward(self.x_val).detach()
            self.nmse_pa = metrics.compute_nmse(y_pa, self.y_val)
            logger.info("   Загружены веса PA, NMSE_PA=%.3f dB", self.nmse_pa)
            self.state = FSM_State.PLOT_PA

    def _train_pa(self):
        logger.info("STATE TRAIN_PA: обучение PA, epochs=%d, lr=%.5f", self.pa_params.epochs, self.pa_params.lr)

        if self._stop_event.is_set():
            return
        
        self.pa_model.optimize_weights(self.x_train, self.y_train, epochs=self.pa_params.epochs, learning_rate=self.pa_params.lr)
        self.pa_model.save_weights()

        y_pa = self.pa_model.forward(self.x_val).detach()
        self.nmse_pa = metrics.compute_nmse(y_pa, self.y_val)
        logger.info("   После обучения PA, NMSE_PA=%.3f dB", self.nmse_pa)
        self.state = FSM_State.PLOT_PA

    def _plot_pa(self):
        logger.info("STATE PLOT_PA: строим спектр PA")
        if self._stop_event.is_set():
            return
        self._pause_event.wait()

        y_pa = self.pa_model.forward(self.x_val).detach()
        freqs, spectrum_out = metrics.power_spectrum(self.y_val, self.fs, self.nperseg)
        _, spectrum_pa = metrics.power_spectrum(y_pa, self.fs, self.nperseg)

        fig = plt.Figure(figsize=(6,3), dpi=DPI)
        ax = fig.add_subplot(111)
        ax.plot(freqs/1e6, 10*np.log10(np.abs(spectrum_out)), label="out")
        ax.plot(freqs/1e6, 10*np.log10(np.abs(spectrum_pa)), label=f"PA (NMSE {self.nmse_pa:.1f} dB)")
        ax.set_xlabel(FREQ_TITLE)
        ax.set_ylabel(SPECTRUM_TITLE)
        ax.legend()
        ax.grid()

        if self.gui:
            logger.debug("  Отображаем спектр PA в GUI")
            self.gui.after(0, lambda f=fig: self.gui.display_pa_figure(f))
        else:
            logger.debug("  Отображаем спектр PA через plt.show()")
            plt.figure(figsize=(8,4))
            plt.plot(freqs/1e6,10*np.log10(np.abs(spectrum_out)), label="out")
            plt.plot(freqs/1e6,10*np.log10(np.abs(spectrum_pa)), label=f"PA (NMSE {self.nmse_pa:.1f} dB)")
            plt.legend()
            plt.grid()
            plt.show()

        self.state = FSM_State.CONFIGURE_DPD

    def _configure_dpd(self):
        if self._stop_event.is_set():
            return
        
        assert self.dpd_params, "DPD params not set"
        logger.info("STATE CONFIGURE_DPD: параметры DPD %s", self.dpd_params)
        
        if self.current_arch_index == 0:
            self.dpd_models.clear()
            self.results_no_noise.clear()
            logger.debug("  Инициализация списка DPD моделей и результатов")

        # Проходим по архитектурам начиная с current_arch_index
        for i in range(self.current_arch_index, len(self.dpd_archs)):
            arch = self.dpd_archs[i]
            arch_lower = arch.lower()
            logger.info("   CONFIGURE_DPD: проверка архитектуры %s", arch)

            model_type = f"dpd_{arch_lower}_grad"
            dpd_config = params.make_gmp_params(
                                                Ka=self.dpd_params.Ka, 
                                                La=self.dpd_params.La, 
                                                Kb=self.dpd_params.Kb, 
                                                Lb=self.dpd_params.Lb,
                                                Mb=self.dpd_params.Mb, 
                                                Kc=self.dpd_params.Kc, 
                                                Lc=self.dpd_params.Lc, 
                                                Mc=self.dpd_params.Mc,
                                                model_type=model_type
                                                )
            model = GMP(**dpd_config)

            if not model.load_weights():
                logger.info("   Нет сохранённых весов для %s: перейдем к TRAIN_DPD", arch)
                self.current_arch = arch
                self.current_arch_index = i
                self.dpd_models[arch] = model
                self.state = FSM_State.TRAIN_DPD
                return

            self._compute_and_store_results(self.x_val, self.y_val_target, arch, self.results_no_noise, self.dpd_models, model)

            logger.info("   Архитектура %s: загружены веса, NMSE=%.3f, ACPR=(%.3f, %.3f)", 
                        arch, self.results_no_noise[arch]["nmse"], 
                        self.results_no_noise[arch]["acpr"][0], self.results_no_noise[arch]["acpr"][1])

        if self._stop_event.is_set():
            return
        self._pause_event.wait()

        logger.info("   CONFIGURE_DPD: вычисление u_k для ILC на чистых данных")
        self.u_k = learning.ilc_signal(
            self.x_train, self.y_train_target, self.pa_model,
            epochs=U_K_EPOCHS, learning_rate=U_K_LR
        )
        self._compute_and_store_results(self.u_k, self.y_train_target, "uk", self.results_no_noise)

        self.state = FSM_State.PLOT_DPD

    def _train_dpd(self):
        arch = self.current_arch
        model = self.dpd_models[arch]
        if model is None:
            logger.error("No model for arch %s in TRAIN_DPD", arch)
            self.state = FSM_State.ERROR
            return

        if self._stop_event.is_set():
            return
        self._pause_event.wait()

        logger.info("STATE TRAIN_DPD: обучение архитектуры %s", arch)

        if arch == "DLA":
            learning.optimize_dla(self.x_train, self.y_train_target, model, self.pa_model, 
                                       epochs=self.dpd_params.epochs, learning_rate=self.dpd_params.lr)
        elif arch == "ILA":
            learning.optimize_ila(model, self.x_train, self.y_train, self.gain, 
                                       epochs=self.dpd_params.epochs, learning_rate=self.dpd_params.lr)
        elif arch == "ILC":
            model.optimize_weights(self.x_train, self.u_k,
                                             epochs=self.dpd_params.epochs, learning_rate=self.dpd_params.lr)
        else:
            logger.warning("Неизвестная архитектура %s в TRAIN_DPD", arch)
        
        model.save_weights()
        logger.info("   %s обучена и сохранена", arch)

        if self._stop_event.is_set():
            return
        self._pause_event.wait()

        self._compute_and_store_results(self.x_val, self.y_val_target, arch, self.results_no_noise, models_dict=None, model=model)

        self.current_arch_index += 1
        self.state = FSM_State.CONFIGURE_DPD

    def _plot_dpd(self):
        logger.info("STATE PLOT_DPD: строим спектр PA+DPD и ILC u_k")
        if self._stop_event.is_set():
            return
        self._pause_event.wait()

        freqs, spectrum_in = metrics.power_spectrum(self.x_val, self.fs, self.nperseg)

        fig = plt.Figure(figsize=(6,3), dpi=DPI)
        ax = fig.add_subplot(111)
        ax.plot(freqs/1e6, 10*np.log10(np.abs(spectrum_in)), label="in")

        y_pa = self.pa_model.forward(self.x_val).detach()
        _, spec_pa = metrics.power_spectrum(y_pa, self.fs, self.nperseg)
        ax.plot(freqs/1e6, 10*np.log10(np.abs(spec_pa)), label="PA")

        for arch in self.dpd_archs:
            res = self.results_no_noise.get(arch)
            if res and "spectrum" in res:
                f, s = res["spectrum"]
                ax.plot(f/1e6, 10*np.log10(np.abs(s)), label=arch.upper())

        if "uk" in self.results_no_noise:
            f_uk, s_uk = self.results_no_noise["uk"]["spectrum"]
            ax.plot(f_uk/1e6, 10*np.log10(np.abs(s_uk)), "--", label="UK")
        ax.set_xlabel("Частота (МГц)")
        ax.set_ylabel(SPECTRUM_TITLE)
        ax.legend()
        ax.grid()

        if self.gui:
            logger.debug("  Отображаем спектры DPD в GUI")
            self.gui.after(0, lambda f=fig: self.gui.display_dpd_figure(f))
        else:
            logger.debug("  Отображаем спектры DPD через plt.show()")
            plt.figure()
            plt.show()

        self.state = FSM_State.SELECT_NOISE_RANGE
    
    def _select_noise_range(self):
        logger.info("STATE SELECT_NOISE_RANGE: инициализация snr вычислений")
        if self._stop_event.is_set():
            return

        assert self.noise_range, "Noise range not set!"
        assert self.num_realizations is not None, "num_realizations not set!"

        self.noise_results.clear()
        self.state = FSM_State.EVALUATE_NOISE

    def _evaluate_noise(self):
        logger.info("STATE EVALUATE_NOISE: запуск по SNR")
        if self._stop_event.is_set():
            return
        self._pause_event.wait()

        snr_params = params.make_snr_params(
            snr_range=self.noise_range,
            num_realizations=self.num_realizations,
            fs=self.fs,
            bw_main_ch=self.bw_main_ch,
            epochs=self.dpd_params.epochs,
            learning_rate=self.dpd_params.lr,
            acpr_meter=self.acpr_meter,
            pa_model=self.pa_model,
            gain=self.gain
        )

        gmp_base_config = params.make_gmp_params(
            Ka=self.dpd_params.Ka, La=self.dpd_params.La,
            Kb=self.dpd_params.Kb, Lb=self.dpd_params.Lb,
            Mb=self.dpd_params.Mb, Kc=self.dpd_params.Kc,
            Lc=self.dpd_params.Lc, Mc=self.dpd_params.Mc,
            model_type=""
        )

        for arch in self.dpd_archs:
            if self._stop_event.is_set():
                break
            self._pause_event.wait()

            logger.info(f"   Оцениваем архитектуру {arch} через snr_metrics")
            results = metrics.snr_metrics(
                arch_name=arch,
                gmp_params=gmp_base_config,
                data_params=self.data_params,
                snr_params=snr_params,
            )

            self.noise_results[arch] = {
                "nmse": results["nmse"],
                "acpr_left": results["acpr_left"],
                "acpr_right": results["acpr_right"],
            }

            if arch == "ILC":
                self.noise_results["uk"] = {
                    "nmse": results["nmse_uk"],
                    "acpr_left": results["acpr_left_uk"],
                    "acpr_right": results["acpr_right_uk"],
                }

        self.state = FSM_State.PLOT_NOISE

    def _plot_noise(self):
        logger.info("STATE PLOT_NOISE: строим графики snr результатов")
        if self._stop_event.is_set():
            return
        self._pause_event.wait()

        snr_vals = self.noise_range

        # 1) NMSE vs SNR
        fig1 = plt.Figure(figsize=(5,3), dpi=DPI)
        ax1 = fig1.add_subplot(111)
        for arch, res in self.noise_results.items():
            ax1.plot(snr_vals, res["nmse"], marker='o', label=arch.upper())
        ax1.set_xlabel("SNR (дБ)")
        ax1.set_ylabel("NMSE (дБ)")
        ax1.set_title("NMSE vs SNR")
        ax1.grid()
        ax1.legend()

        # 2) ACPR Left & Right vs SNR
        fig2 = plt.Figure(figsize=(5,3), dpi=DPI)
        ax2 = fig2.add_subplot(111)
        for arch, res in self.noise_results.items():
            ax2.plot(snr_vals, res["acpr_left"], marker='o', label=arch.upper())
        ax2.set_xlabel("SNR (дБ)")
        ax2.set_ylabel("ACPR Left (дБ)")
        ax2.grid()
        ax2.legend()

        fig3 = plt.Figure(figsize=(5,3), dpi=DPI)
        ax3 = fig3.add_subplot(111)
        for arch, res in self.noise_results.items():
            ax3.plot(snr_vals, res["acpr_right"], marker='o', label=arch.upper())
        ax3.set_xlabel("SNR (дБ)")
        ax3.set_ylabel("ACPR Right (дБ)")
        ax3.grid()
        ax3.legend()

        if self.gui:
            logger.debug("  Отображаем snr графики в GUI")
            self.gui.after(0, lambda: self.gui.display_noise_figures(fig1, fig2, fig3))
        else:
            logger.debug("  Отображаем шумовые графики через plt.show()")
            plt.figure()
            plt.show()

        self.state = FSM_State.DONE
    
    def on_fsm_finished(self):
        logging.info("GUI: FSM finished; updating buttons")
        # предполагается, что GUI имеет start_btn, pause_btn, resume_btn, stop_btn
        try:
            self.start_btn.config(state="normal")
            self.pause_btn.config(state="disabled")
            self.resume_btn.config(state="disabled")
            self.stop_btn.config(state="disabled")
        except Exception as e:
            print(e)
            pass
