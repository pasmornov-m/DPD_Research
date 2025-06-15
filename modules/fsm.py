import enum
import logging
import threading
import numpy as np
import matplotlib.pyplot as plt
from modules.params import ModelParams
from modules import data_loader, metrics, learning
from modules.gmp_model import GMP


DPI = 100
SPECTRUM_TITLE = "Спектральная плотность мощнсоти (дБ)"
FREQ_TITLE = "Частота (МГц)"

logger = logging.getLogger("PA_DPD_FSM")
logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)


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

        # PA
        self.pa_model_type = None
        self.pa_params: ModelParams = None
        self.pa_model = None
        self.nmse_pa = None

        # DPD
        self.dpd_model_type = None
        self.dpd_params: ModelParams = None
        self.dpd_archs = []
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

        logger.info("PA_DPD_FSM initialized with data_path=%s", data_path)

    # --- интерфейс от GUI ---
    def set_pa_params(self, params: ModelParams):
        self.pa_params = params
        logger.info("PA params set: %s", params)

    def set_dpd_params(self, params: ModelParams):
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
        # Очистить события:
        self._stop_event.clear()
        self._pause_event.set()
        logging.info("FSM reset to INIT")

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

        logger.info("   Data loaded: fs=%s, bw_main_ch=%s", self.fs, self.bw_main_ch)
        self.state = FSM_State.ESTIMATE_GAIN

    def _estimate_gain(self):
        logger.info("STATE ESTIMATE_GAIN")
        if self._stop_event.is_set():
            return
        self.gain = metrics.calculate_gain_complex(self.x_train, self.y_train)
        logger.info(f"  Gain = {self.gain:.2f}")
        self.y_train_target = self.gain * self.x_train
        self.y_val_target = self.gain * self.x_val

        self.state = FSM_State.CONFIGURE_PA

    def _configure_pa(self):
        assert self.pa_params, "PA params not set"
        p = self.pa_params
        logger.info("STATE CONFIGURE_PA: параметры PA %s", p)
        if self._stop_event.is_set():
            return

        cls = {"GMP": GMP}[p.model_type]
        model = cls(p.Ka,p.La,p.Kb,p.Lb,p.Mb,p.Kc,p.Lc,p.Mc, model_type="pa_grad")

        self.pa_model = model

        if not model.load_coefficients():
            logger.info("   Нет сохранённых весов PA: перейдем к TRAIN_PA")
            self.state = FSM_State.TRAIN_PA
            return

        y_pa = model.forward(self.x_val).detach()
        self.nmse_pa = metrics.compute_nmse(y_pa, self.y_val)
        logger.info("   Загружены веса PA, NMSE_PA=%.3f dB", self.nmse_pa)
        self.state = FSM_State.PLOT_PA

    def _train_pa(self):
        p = self.pa_params
        logger.info("STATE TRAIN_PA: обучение PA, epochs=%d, lr=%.5f", p.epochs, p.lr)

        if self._stop_event.is_set():
            return
        
        self.pa_model.optimize_coefficients_grad(self.x_train, self.y_train, epochs=p.epochs, learning_rate=p.lr)
        self.pa_model.save_coefficients()

        y_pa = self.pa_model.forward(self.x_val).detach()
        self.nmse_pa = metrics.compute_nmse(y_pa, self.y_val)
        logger.info("   После обучения PA, NMSE_PA=%.3f dB", self.nmse_pa)
        self.state = FSM_State.PLOT_PA

    def _plot_pa(self):
        logger.info("STATE PLOT_PA: строим спектр PA")
        if self._stop_event.is_set():
            return
        
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

        # отображаем в GUI
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
        p = self.dpd_params
        logger.info("STATE CONFIGURE_DPD: параметры DPD %s", p)
        
        if self.current_arch_index == 0:
            self.dpd_models = {}
            self.results_no_noise = {}
            logger.debug("  Инициализация списка DPD моделей и результатов")

        # Проходим по архитектурам начиная с current_arch_index
        for i in range(self.current_arch_index, len(self.dpd_archs)):
            arch = self.dpd_archs[i]
            logger.info("   CONFIGURE_DPD: проверка архитектуры %s", arch)
            # создаём модель нужного класса
            cls = {"GMP": GMP}[p.model_type]
            model = cls(p.Ka,p.La,p.Kb,p.Lb,p.Mb,p.Kc,p.Lc,p.Mc,
                        model_type=f"dpd_{arch.lower()}_grad")

            # если нет сохранённого — нужно обучить
            if not model.load_coefficients():
                # запоминаем, какую архитектуру тренить
                logger.info("   Нет сохранённых весов для %s: перейдем к TRAIN_DPD", arch)
                self.current_arch = arch
                self.current_arch_index = i
                self.dpd_models[arch] = model # сохраним для TRAIN_DPD
                self.state = FSM_State.TRAIN_DPD
                return # прерываем конфигурирование

            # иначе сразу считаем метрики и сохраняем
            y_dpd = model.forward(self.x_val).detach()
            y_lin = self.pa_model.forward(y_dpd).detach()
            nmse = metrics.compute_nmse(y_lin, self.y_val_target)
            acpr_l, acpr_r = metrics.calculate_acpr(y_lin, self.acpr_meter)
            spectrum = metrics.power_spectrum(y_lin, self.fs, self.nperseg)
            logger.info("   Архитектура %s: загружены веса, NMSE=%.3f, ACPR=(%.3f, %.3f)", arch, nmse, acpr_l, acpr_r)

            self.dpd_models[arch] = model

            self.results_no_noise[arch] = {
                "nmse": nmse,
                "acpr": (acpr_l, acpr_r),
                "spectrum": spectrum
            }

        if self._stop_event.is_set():
            return
        self._pause_event.wait()

        logger.info("   CONFIGURE_DPD: вычисление u_k для ILC на чистых данных")
        u_k = learning.ilc_signal_grad(
            self.x_train, self.y_train_target, self.pa_model,
            epochs=500, learning_rate=0.001
        )
        u_k_pa = self.pa_model.forward(u_k).detach()
        nmse_uk = metrics.compute_nmse(u_k_pa, self.y_train_target)
        acpr_l_uk, acpr_r_uk = metrics.calculate_acpr(u_k_pa, self.acpr_meter)
        spectrum_uk = metrics.power_spectrum(u_k_pa, self.fs, self.nperseg)
        logger.info("   ILC u_k: NMSE=%.3f, ACPR=(%.3f, %.3f)", nmse_uk, acpr_l_uk, acpr_r_uk)
        self.results_no_noise["uk"] = {
            "nmse": nmse_uk,
            "acpr": (acpr_l_uk, acpr_r_uk),
            "spectrum": spectrum_uk
        }

        # если все architectures обработаны — идём к PLOT_DPD
        self.state = FSM_State.PLOT_DPD

    def _train_dpd(self):
        arch = self.current_arch
        p = self.dpd_params
        model = self.dpd_models[arch]

        if self._stop_event.is_set():
            return
        self._pause_event.wait()

        logger.info("STATE TRAIN_DPD: обучение архитектуры %s", arch)

        # запускаем обучение для этой архитектуры
        if arch == "dla":
            learning.optimize_dla_grad(
                self.x_train, self.y_train_target,
                model, self.pa_model,
                epochs=p.epochs, learning_rate=p.lr
            )
            model.save_coefficients()
            logger.info("   DLA обучена и сохранена")
        elif arch == "ila":
            learning.optimize_ila_grad(
                model, self.x_train, self.y_train, self.gain,
                epochs=p.epochs, learning_rate=p.lr
            )
            model.save_coefficients()
            logger.info("   ILA обучена и сохранена")
        elif arch == "ilc":
            u_k = learning.ilc_signal_grad(
                self.x_train, self.y_train_target, self.pa_model,
                epochs=500, learning_rate=0.001
            )
            u_k_pa = self.pa_model.forward(u_k).detach()
            nmse_uk = metrics.compute_nmse(u_k_pa, self.y_train_target)
            acpr_l_uk, acpr_r_uk = metrics.calculate_acpr(u_k_pa, self.acpr_meter)
            spectrum_uk = metrics.power_spectrum(u_k_pa, self.fs, self.nperseg)
            logger.info("   ILC u_k до обучения: NMSE=%.3f, ACPR=(%.3f, %.3f)", nmse_uk, acpr_l_uk, acpr_r_uk)
            # Сохраним метрики u_k
            self.results_no_noise["uk"] = {
                "nmse": nmse_uk,
                "acpr": (acpr_l_uk, acpr_r_uk),
                "spectrum": spectrum_uk
            }
            model.optimize_coefficients_grad(
                self.x_train, u_k,
                epochs=p.epochs, learning_rate=p.lr
            )
            model.save_coefficients()
            logger.info("   ILC DPD обучена и сохранена")

        if self._stop_event.is_set():
            return
        self._pause_event.wait()

        y_dpd = model.forward(self.x_val).detach()
        y_lin = self.pa_model.forward(y_dpd).detach()
        nmse = metrics.compute_nmse(y_lin, self.y_val_target)
        acpr_l, acpr_r = metrics.calculate_acpr(y_lin, self.acpr_meter)
        spectrum = metrics.power_spectrum(y_lin, self.fs, self.nperseg)
        logger.info("   После обучения %s: NMSE=%.3f, ACPR=(%.3f, %.3f)", arch, nmse, acpr_l, acpr_r)

        self.results_no_noise[arch] = {
            "nmse": nmse,
            "acpr": (acpr_l, acpr_r),
            "spectrum": spectrum
        }

        # переходим к следующей архитектуре
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

        # PA
        y_pa = self.pa_model.forward(self.x_val).detach()
        _, spec_pa = metrics.power_spectrum(y_pa, self.fs, self.nperseg)
        ax.plot(freqs/1e6, 10*np.log10(np.abs(spec_pa)), label="PA")

        # DPD archs
        for arch in self.dpd_archs:
            res = self.results_no_noise.get(arch)
            if res and "spectrum" in res:
                f, s = res["spectrum"]
                ax.plot(f/1e6, 10*np.log10(np.abs(s)), label=arch.upper())

        # UK сигнал
        if "uk" in self.results_no_noise:
            f_uk, s_uk = self.results_no_noise["uk"]["spectrum"]
            ax.plot(f_uk/1e6, 10*np.log10(np.abs(s_uk)), "--", label="UK")
        ax.set_xlabel("Частота (МГц)")
        ax.set_ylabel(SPECTRUM_TITLE)
        ax.legend()
        ax.grid()

        # вывод на GUI
        if self.gui:
            logger.debug("  Отображаем спектры DPD в GUI")
            # self.gui.display_dpd_figure(fig)
            self.gui.after(0, lambda f=fig: self.gui.display_dpd_figure(f))
            # self.gui.update()
        else:
            logger.debug("  Отображаем спектры DPD через plt.show()")
            plt.figure()
            plt.show()
        self.state = FSM_State.SELECT_NOISE_RANGE
    
    def _select_noise_range(self):
        logger.info("STATE SELECT_NOISE_RANGE: инициализация шумовых вычислений")
        if self._stop_event.is_set():
            return
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
        logger.info("STATE EVALUATE_NOISE: запуск по шумовым SNR")
        for snr in self.noise_range:
            if self._stop_event.is_set():
                break
            self._pause_event.wait()
            logger.info("   Обработка SNR=%d dB", snr)

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

            logger.info("   u_k @%ddB: NMSE=%.3f, ACPR=(%.3f, %.3f)", snr, nm_u_k, l_u_k, r_u_k)

            for arch, base_model in self.dpd_models.items():
                if self._stop_event.is_set():
                    break
                self._pause_event.wait()
                logger.info("   Обработка архитектуры %s при SNR=%d", arch, snr)
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
                
                else:
                    logger.warning("    Неизвестная архитектура %s, пропускаем", arch)
                    continue

                if self._stop_event.is_set():
                    break

                self.noise_results[arch]["nmse"].append(nm)
                self.noise_results[arch]["acpr_left"].append(l)
                self.noise_results[arch]["acpr_right"].append(r)
                logger.info("   %s @%ddB: NMSE=%.3f, ACPR=(%.3f, %.3f)", arch, snr, nm, l, r)

        # После всех SNR
        self.state = FSM_State.PLOT_NOISE

    def _plot_noise(self):
        logger.info("STATE PLOT_NOISE: строим графики шумовых результатов")
        if self._stop_event.is_set():
            return
        
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
            logger.debug("  Отображаем шумовые графики в GUI")
            # self.gui.display_noise_figures(fig1, fig2, fig3)
            self.gui.after(0, lambda: self.gui.display_noise_figures(fig1, fig2, fig3))
            # self.gui.update()
        else:
            logger.debug("  Отображаем шумовые графики через plt.show()")
            plt.figure()
            plt.show()

        self.state = FSM_State.DONE
    
    def on_fsm_finished(self):
        logging.info("GUI: FSM finished; updating buttons")
        # Включаем Start, отключаем Pause/Resume/Stop
        self.start_btn.config(state="normal")
        self.pause_btn.config(state="disabled")
        self.resume_btn.config(state="disabled")
        self.stop_btn.config(state="disabled")
