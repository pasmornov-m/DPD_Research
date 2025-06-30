import enum
import logging
import threading
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from modules import data_loader, metrics
from modules.gmp_model import GMP
from modules.pipelines import SimplePipeline, SnrPipeline


DPI = 100
SPECTRUM_TITLE = "Спектральная плотность мощнсоти (дБ)"
FREQ_TITLE = "Частота (МГц)"


logger = logging.getLogger("PA_DPD_FSM")
logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)


class FSM_State(enum.Enum):
    INIT = enum.auto()
    RUN_PA = enum.auto()
    PLOT_PA = enum.auto()
    RUN_DPD = enum.auto()
    PLOT_DPD = enum.auto()
    SNR_SETUP = enum.auto()
    EVAL_SNR = enum.auto()
    PLOT_SNR = enum.auto()
    DONE = enum.auto()
    ERROR = enum.auto()

class PA_DPD_FSM:
    _HANDLERS = {
        FSM_State.INIT: "_init",
        FSM_State.RUN_PA: "_run_pa",
        FSM_State.PLOT_PA: "_plot_pa",
        FSM_State.RUN_DPD: "_run_dpd",
        FSM_State.PLOT_DPD: "_plot_dpd",
        FSM_State.SNR_SETUP: "_snr_setup",
        FSM_State.EVAL_SNR: "_eval_snr",
        FSM_State.PLOT_SNR: "_plot_snr",
    }

    def __init__(self, data_path: str, gui=None):
        self.data_path = data_path
        self.gui = gui
        self.state = FSM_State.INIT

        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._pause_event.set()
        
        self.data_dict = None
        self.train_props = None
        self.base_model = None
        self.acpr_meter = None
        self.pa_model = None

        logger.info("PA_DPD_FSM initialized with data_path=%s", data_path)

    # --- интерфейс от GUI ---
    def set_gmp_params(self, params):
        self.gmp_params = int(params)
        logger.info("GMP params set: %s", self.gmp_params)
    
    def set_train_props(self, lr, epochs):
        self.lr = float(lr)
        self.epochs = int(epochs)

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
        self.state = FSM_State.INIT
        self.current_arch_index = 0
        self.current_arch = None
        self.u_k = None
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

        self.data_dict = data_loader.load_data(self.data_path)
        
        self.config = self.data_dict["config"]
        self.fs = self.config["input_signal_fs"]
        self.bw_main_ch = self.config["bw_main_ch"]
        self.sub_ch = self.config["sub_ch"]
        self.nperseg = self.config["nperseg"]
        self.x_val = self.data_dict["train_output"]
        self.y_val = self.data_dict["val_output"]

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

        self.gmp_train_props = {
                "gmp_degree": self.gmp_params,
                "lr": self.lr,
                "epochs": self.epochs,
                "acpr_meter": self.acpr_meter
            }
        
        self.gmp_pipeline = SimplePipeline(data_dict=self.data_dict, 
                              train_props=self.gmp_train_props, 
                              base_model=GMP)

        logger.info("   Data loaded: fs=%s, bw_main_ch=%s", self.fs, self.bw_main_ch)
        self.state = FSM_State.RUN_PA

    def _run_pa(self):
        if self._stop_event.is_set():
            return
        self._pause_event.wait()

        self.gmp_pipeline.run_pa()
        self.pa_model = self.gmp_pipeline.get_pa_model()
        
        self.state = FSM_State.PLOT_PA

    def _plot_pa(self):
        logger.info("STATE PLOT_PA: строим спектр PA")
        if self._stop_event.is_set():
            return
        self._pause_event.wait()

        y_pa = self.gmp_pipeline.get_results()["pa"]["y_val_pa_model"]
        freqs, spectrum_out = metrics.power_spectrum(self.y_val, self.fs, self.nperseg)
        _, spectrum_pa = metrics.power_spectrum(y_pa, self.fs, self.nperseg)

        fig = Figure(figsize=(6,3), dpi=DPI)
        ax = fig.add_subplot(111)
        ax.plot(freqs/1e6, 10*np.log10(np.abs(spectrum_out)), label="out")
        ax.plot(freqs/1e6, 10*np.log10(np.abs(spectrum_pa)), label=f"PA")
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
            plt.plot(freqs/1e6,10*np.log10(np.abs(spectrum_pa)), label=f"PA")
            plt.legend()
            plt.grid()
            plt.show()

        self.state = FSM_State.RUN_DPD

    def _run_dpd(self):
        if self._stop_event.is_set():
            return
        
        self.gmp_pipeline.run_dla()
        self.gmp_pipeline.run_ila()
        self.gmp_pipeline.run_ilc()
        
        self.state = FSM_State.PLOT_DPD

    def _plot_dpd(self):
        logger.info("STATE PLOT_DPD: строим спектр PA+DPD и ILC u_k")
        if self._stop_event.is_set():
            return
        self._pause_event.wait()
        
        gmp_results = self.gmp_pipeline.get_results()
        gmp_y_val_pa = gmp_results["pa"]["y_val_pa_model"]
        gmp_y_val_dla = gmp_results["dla"]["y_val_dla"]
        gmp_y_val_ila = gmp_results["ila"]["y_val_ila"]
        gmp_y_val_ilc = gmp_results["ilc"]["y_val_ilc"]
        gmp_u_k_pa = gmp_results["uk"]["u_k_pa"]
        
        freqs, spectrum_y_in = metrics.power_spectrum(self.x_val, self.fs, self.nperseg)
        _, spectrum_gmp_y_val_pa = metrics.power_spectrum(gmp_y_val_pa, self.fs, self.nperseg)
        _, spectrum_gmp_y_val_dla = metrics.power_spectrum(gmp_y_val_dla, self.fs, self.nperseg)
        _, spectrum_gmp_y_val_ila = metrics.power_spectrum(gmp_y_val_ila, self.fs, self.nperseg)
        _, spectrum_gmp_y_val_ilc = metrics.power_spectrum(gmp_y_val_ilc, self.fs, self.nperseg)
        _, spectrum_gmp_u_k_pa = metrics.power_spectrum(gmp_u_k_pa, self.fs, self.nperseg)

        fig = Figure(figsize=(6,3), dpi=DPI)
        ax = fig.add_subplot(111)
        ax.plot(freqs / 1e6, 10 * np.log10(np.abs(spectrum_y_in)), color='grey', label='Входной сигнал')
        ax.plot(freqs / 1e6, 10 * np.log10(np.abs(spectrum_gmp_y_val_pa)), color='gold', label='Выходной сигнал')
        ax.plot(freqs / 1e6, 10 * np.log10(np.abs(spectrum_gmp_u_k_pa)), color='red', label='Оптимальный сигнал ILC')
        ax.plot(freqs / 1e6, 10 * np.log10(np.abs(spectrum_gmp_y_val_dla)), color='darkorange', label='Сигнал с предыскажением DLA')
        ax.plot(freqs / 1e6, 10 * np.log10(np.abs(spectrum_gmp_y_val_ila)), color='limegreen', label='Сигнал с предыскажением ILA')
        ax.plot(freqs / 1e6, 10 * np.log10(np.abs(spectrum_gmp_y_val_ilc)), color='lightcoral', label='Сигнал с предыскажением ILC')

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

        self.state = FSM_State.SNR_SETUP
    
    def _snr_setup(self):
        logger.info("STATE SNR_SETUP: инициализация snr вычислений")
        if self._stop_event.is_set():
            return

        assert self.noise_range, "Noise range not set!"
        assert self.num_realizations is not None, "num_realizations not set!"

        self.snr_params = {
            "snr_range": self.noise_range,
            "num_realizations": self.num_realizations,
            "fs": self.fs,
            "bw_main_ch": self.bw_main_ch,
            "epochs": self.epochs,
            "learning_rate": self.lr,
            "acpr_meter": self.acpr_meter,
            "pa_model": self.pa_model,
            "gain": self.gmp_pipeline.gain
        }
        
        gmp_params = {"gmp_degree": self.gmp_params}
        
        self.snr_metrics_runner = SnrPipeline(base_model=GMP, 
                                              input_model_params=gmp_params,
                                              data_dict=self.data_dict, 
                                              snr_params=self.snr_params)
        self.state = FSM_State.EVAL_SNR

    def _eval_snr(self):
        logger.info("STATE EVAL_SNR: запуск по SNR")
        if self._stop_event.is_set():
            return
        self._pause_event.wait()

        self.snr_metrics_runner.run(arch_name="DLA")
        self.snr_metrics_runner.run(arch_name="ILA")
        self.snr_metrics_runner.run(arch_name="ILC")

        self.state = FSM_State.PLOT_SNR

    def _plot_snr(self):
        logger.info("STATE PLOT_SNR: строим графики snr результатов")
        if self._stop_event.is_set():
            return
        self._pause_event.wait()

        snr_range = self.noise_range
        results = self.snr_metrics_runner.get_results()
        nmse_dla_list = results["dla"]["nmse"]
        acpr_left_dla_list = results["dla"]["acpr_left"]
        acpr_right_dla_list = results["dla"]["acpr_right"]

        nmse_ila_list = results["ila"]["nmse"]
        acpr_left_ila_list = results["ila"]["acpr_left"]
        acpr_right_ila_list = results["ila"]["acpr_right"]

        nmse_ilc_list = results["ilc"]["nmse"]
        acpr_left_ilc_list = results["ilc"]["acpr_left"]
        acpr_right_ilc_list = results["ilc"]["acpr_right"]

        nmse_uk_list = results["uk"]["nmse"]
        acpr_left_uk_list = results["uk"]["acpr_left"]
        acpr_right_uk_list = results["uk"]["acpr_right"]

        # 1) NMSE vs SNR
        fig1 = Figure(figsize=(5,3), dpi=DPI)
        ax1 = fig1.add_subplot(111)
        markersize = 8
        ax1.plot(snr_range, nmse_dla_list,color='blue', linestyle='-', label="Сигнал с предыскажением DLA",
                marker='o', markersize=markersize)
        ax1.plot(snr_range, nmse_ila_list, color='orange', linestyle='--', label="Сигнал с предыскажением ILA",
                marker='s', markersize=markersize)
        ax1.plot(snr_range, nmse_ilc_list, color='green', linestyle='-.', label="Сигнал с предыскажением ILC",
                marker='d', markersize=markersize)
        ax1.plot(snr_range, nmse_uk_list, color='red', linestyle='-.', label="Оптимальный сигнал ILC",
                marker='*', markersize=markersize)
        ax1.set_xlabel("SNR (дБ)")
        ax1.set_ylabel("NMSE (дБ)")
        ax1.set_title("NMSE vs SNR")
        ax1.grid()
        ax1.legend()

        # 2) ACPR Left & Right vs SNR
        fig2 = Figure(figsize=(5,3), dpi=DPI)
        ax2 = fig2.add_subplot(111)
        markersize=8
        ax2.plot(snr_range, acpr_left_dla_list, color='blue', linestyle='-', label="Сигнал с предыскажением DLA", 
         marker='o', markersize=markersize)
        ax2.plot(snr_range, acpr_left_ila_list, color='orange', linestyle='--', label="Сигнал с предыскажением ILA", 
                marker='s', markersize=markersize)
        ax2.plot(snr_range, acpr_left_ilc_list, color='green', linestyle='-.', label="Сигнал с предыскажением ILC", 
                marker='d', markersize=markersize)
        ax2.plot(snr_range, acpr_left_uk_list, color='red', linestyle='-.', label="Оптимальный сигнал ILC", 
                marker='*', markersize=markersize)
        ax2.set_xlabel("SNR (дБ)")
        ax2.set_ylabel("ACPR Left (дБ)")
        ax2.grid()
        ax2.legend()

        fig3 = Figure(figsize=(5,3), dpi=DPI)
        ax3 = fig3.add_subplot(111)
        ax3.plot(snr_range, acpr_right_dla_list, color='blue', linestyle='-', label="Сигнал с предыскажением DLA", 
         marker='o', markersize=markersize)
        ax3.plot(snr_range, acpr_right_ila_list, color='orange', linestyle='--', label="Сигнал с предыскажением ILA", 
                marker='s', markersize=markersize)
        ax3.plot(snr_range, acpr_right_ilc_list, color='green', linestyle='-.', label="Сигнал с предыскажением ILC", 
                marker='d', markersize=markersize)
        ax3.plot(snr_range, acpr_right_uk_list, color='red', linestyle='-.', label="Оптимальный сигнал ILC", 
                marker='*', markersize=markersize)
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
