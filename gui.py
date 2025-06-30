import threading
import os
import logging
import tkinter as tk
from tkinter import ttk, messagebox
from tkinter.scrolledtext import ScrolledText
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from modules.fsm_for_gui import PA_DPD_FSM


DATA_PATH = "./DPA_200MHz"


class TextHandler(logging.Handler):
    def __init__(self, text_widget):
        super().__init__()
        self.text_widget = text_widget
        fmt = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
        self.setFormatter(fmt)

    def emit(self, record):
        msg = self.format(record) + '\n'
        # Вставляем в текстовый виджет из главного потока
        def append():
            # Ограничение длины, чтобы не разрастался бесконечно
            self.text_widget.configure(state='normal')
            self.text_widget.insert(tk.END, msg)
            self.text_widget.configure(state='disabled')
            # Прокрутка вниз
            self.text_widget.yview(tk.END)
        # Гарантируем вызов в GUI-потоке
        self.text_widget.after(0, append)

class FSMGUI(tk.Tk):
    def __init__(self, data_path):
        super().__init__()
        self.title("GMP DPD")
        self.geometry("1000x1000")
        self.fsm = PA_DPD_FSM(data_path, gui=self)
        
        # --- Общие параметры GMP модели и обучения ---
        common_frame = ttk.LabelFrame(self, text="Model & Training Params", padding=10)
        common_frame.grid(row=0, column=0, padx=10, pady=10, sticky="ew")

        # GMP Degree
        ttk.Label(common_frame, text="GMP Degree").grid(row=0, column=0, sticky="e")
        self.gmp_degree = ttk.Entry(common_frame, width=7)
        self.gmp_degree.grid(row=0, column=1, sticky="w")
        self.gmp_degree.insert(0, 3)

        # Epochs
        ttk.Label(common_frame, text="Epochs").grid(row=1, column=0, sticky="e")
        self.train_epochs = ttk.Entry(common_frame, width=7)
        self.train_epochs.grid(row=1, column=1, sticky="w")
        self.train_epochs.insert(0, 1000)

        # Learning Rate
        ttk.Label(common_frame, text="Learning Rate").grid(row=2, column=0, sticky="e")
        self.train_lr = ttk.Entry(common_frame, width=7)
        self.train_lr.grid(row=2, column=1, sticky="w")
        self.train_lr.insert(0, 0.01)

        # --- SNR диапазон и кол-во реализаций ---
        noise_frame = ttk.LabelFrame(self, text="Noise Settings", padding=10)
        noise_frame.grid(row=2, column=0, padx=10, pady=10, sticky="ew")

        ttk.Label(noise_frame, text="SNRs (start, end, step)").grid(row=0, column=0, sticky="e")
        self.noise = ttk.Entry(noise_frame, width=20)
        self.noise.grid(row=0, column=1, sticky="w")
        self.noise.insert(0, "20, 61, 10")

        ttk.Label(noise_frame, text="Realizations").grid(row=1, column=0, sticky="e")
        self.noise_real = ttk.Entry(noise_frame, width=7)
        self.noise_real.grid(row=1, column=1, sticky="w")
        self.noise_real.insert(0, "1")

        # --- Статус и кнопка Старт ---
        # Создаём фрейм для кнопок
        button_frame = ttk.LabelFrame(self, text="Buttons", padding=10)
        button_frame.grid(row=3, column=0, padx=10, pady=10, sticky="ew")
        button_frame.columnconfigure(0, weight=1)
        button_frame.columnconfigure(1, weight=1)
        # Размещаем статус (можно тоже внутри фрейма или снаружи, по вкусу)
        self.status_label = ttk.Label(button_frame, text="State: INIT")
        self.status_label.grid(row=2, column=0, columnspan=2, pady=(10, 0))
        # Кнопки внутри button_frame:
        self.start_btn = ttk.Button(button_frame, text="Start", command=self.on_start)
        self.start_btn.grid(row=0, column=0, padx=5, pady=5, sticky="e")
        self.pause_btn = ttk.Button(button_frame, text="Pause", command=self.on_pause, state="disabled")
        self.pause_btn.grid(row=0, column=1, padx=5, pady=5, sticky="w")
        self.resume_btn = ttk.Button(button_frame, text="Resume", command=self.on_resume, state="disabled")
        self.resume_btn.grid(row=1, column=0, padx=5, pady=5, sticky="e")
        self.stop_btn = ttk.Button(button_frame, text="Stop", command=self.on_stop, state="disabled")
        self.stop_btn.grid(row=1, column=1, padx=5, pady=5, sticky="w")

        # Область для вкладок с графиками:
        self.plot_frame = ttk.Frame(self)
        self.plot_frame.grid(row=0, column=1, rowspan=10, sticky="nsew", padx=10, pady=10)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(9, weight=1)

        # Notebook с тремя вкладками
        self.notebook = ttk.Notebook(self.plot_frame)
        self.tab_pa = ttk.Frame(self.notebook)
        self.tab_dpd = ttk.Frame(self.notebook)
        self.tab_nmse = ttk.Frame(self.notebook)
        self.tab_acpr_l = ttk.Frame(self.notebook)
        self.tab_acpr_r = ttk.Frame(self.notebook)
        self.tab_files = ttk.Frame(self.notebook)
        self.tab_log = ttk.Frame(self.notebook)
        
        self.notebook.add(self.tab_pa, text="PA Spectrum")
        self.notebook.add(self.tab_dpd, text="DPD Spectrum")
        self.notebook.add(self.tab_nmse, text="NMSE vs SNR")
        self.notebook.add(self.tab_acpr_l, text="ACPR Left vs SNR")
        self.notebook.add(self.tab_acpr_r, text="ACPR Right vs SNR")
        self.notebook.add(self.tab_files, text="Model Params Files")
        self.notebook.add(self.tab_log, text="Logs")
        self.notebook.pack(fill="both", expand=True)

        # Настроить виджет ScrolledText для логов
        self.log_widget = ScrolledText(self.tab_log, state='disabled', wrap='word', height=10)
        self.log_widget.pack(fill="both", expand=True)

        # Настроить logging: добавить TextHandler
        text_handler = TextHandler(self.log_widget)
        text_handler.setLevel(logging.DEBUG)
        # Можно установить уровень корневого логгера или конкретного
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)
        root_logger.addHandler(text_handler)
        # Опционально: вывод в консоль тоже оставить
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', '%H:%M:%S'))
        console_handler.setLevel(logging.INFO)
        root_logger.addHandler(console_handler)

        files_frame = ttk.Frame(self.tab_files)
        files_frame.pack(fill="both", expand=True, padx=5, pady=5)
        # Кнопка Refresh
        btn_refresh = ttk.Button(files_frame, text="Обновить список", command=self.update_file_list)
        btn_refresh.pack(anchor="nw", pady=5)
        # Listbox для файлов
        self.files_listbox = tk.Listbox(files_frame, height=15)
        self.files_listbox.pack(fill="both", expand=True)
        self.update_file_list()
        self.after(5000, self.periodic_update_file_list)

        # Периодически обновлять статус из FSM
        self.after(200, self.refresh_status)
    
    def update_file_list(self):
        """Считывает файлы в ./model_params и отображает их в Listbox."""
        folder = "./model_params"
        try:
            items = os.listdir(folder)
        except Exception as e:
            items = []
            logging.error(f"Не удалось открыть папку '{folder}': {e}")
        # Очищаем и вставляем
        self.files_listbox.delete(0, tk.END)
        if not items:
            self.files_listbox.insert(tk.END, "(пусто или недоступно)")
        else:
            for fname in sorted(items):
                self.files_listbox.insert(tk.END, fname)

    def periodic_update_file_list(self):
        """Опционально: автоматически обновлять каждые N мс."""
        self.update_file_list()
        # повтор через 5 секунд
        self.after(5000, self.periodic_update_file_list)
    
    def display_pa_figure(self, fig):
        """Вставить Figure в вкладку PA (tab_pa)."""
        # очистить содержимое tab_pa
        for w in self.tab_pa.winfo_children():
            w.destroy()
        canvas = FigureCanvasTkAgg(fig, master=self.tab_pa)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    def display_dpd_figure(self, fig):
        """Вставить Figure в вкладку DPD (tab_dpd)."""
        for w in self.tab_dpd.winfo_children():
            w.destroy()
        canvas = FigureCanvasTkAgg(fig, master=self.tab_dpd)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    def display_noise_figures(self, fig_nmse, fig_acpr_l, fig_acpr_r):
        """
        Вставить три Figure в три разных вкладки:
          - fig_nmse → self.tab_nmse
          - fig_acpr_l → self.tab_acpr_l
          - fig_acpr_r → self.tab_acpr_r
        """
        # NMSE vs SNR
        for w in self.tab_nmse.winfo_children():
            w.destroy()
        canvas1 = FigureCanvasTkAgg(fig_nmse, master=self.tab_nmse)
        canvas1.draw()
        canvas1.get_tk_widget().pack(fill="both", expand=True)

        # ACPR Left vs SNR
        for w in self.tab_acpr_l.winfo_children():
            w.destroy()
        canvas2 = FigureCanvasTkAgg(fig_acpr_l, master=self.tab_acpr_l)
        canvas2.draw()
        canvas2.get_tk_widget().pack(fill="both", expand=True)

        # ACPR Right vs SNR
        for w in self.tab_acpr_r.winfo_children():
            w.destroy()
        canvas3 = FigureCanvasTkAgg(fig_acpr_r, master=self.tab_acpr_r)
        canvas3.draw()
        canvas3.get_tk_widget().pack(fill="both", expand=True)

    def refresh_status(self):
        """Обновляем метку состояния из FSM."""
        st = self.fsm.state.name
        self.status_label.config(text=f"State: {st}")
        # запускаем снова через 200ms
        self.after(200, self.refresh_status)

    def on_start(self):
        """Считать всё из полей, сконфигурировать FSM и запустить его."""
        try:
            self.fsm.reset()
            self.fsm.set_gmp_params(int(self.gmp_degree.get()))

            self.fsm.set_train_props(float(self.train_lr.get()), int(self.train_epochs.get()))

            snr_range = [int(s.strip()) for s in self.noise.get().split(",") if s.strip()]
            num_real = int(self.noise_real.get())
            self.fsm.set_noise_range(snr_range, num_real)

            if hasattr(self.fsm, 'reset_events'):
                self.fsm.reset_events()

            # Запустить FSM в фоне
            threading.Thread(target=self.fsm.run, daemon=True).start()
            # self.fsm.run()

            self.start_btn.config(state="disabled")
            self.pause_btn.config(state="normal")
            self.resume_btn.config(state="disabled")
            self.stop_btn.config(state="normal")

        except ValueError:
            messagebox.showwarning("Input error", "Проверьте корректность введённых параметров.")
    
    def on_pause(self):
        # выставляем паузу в FSM
        self.fsm.pause()
        logging.info("FSM paused by user")
        # Обновляем кнопки: Disable Pause, Enable Resume
        self.pause_btn.config(state="disabled")
        self.resume_btn.config(state="normal")
        # Stop остаётся активна, Start остаётся disabled

    def on_resume(self):
        self.fsm.resume()
        logging.info("FSM resumed by user")
        # Обновляем кнопки: Enable Pause, Disable Resume
        self.pause_btn.config(state="normal")
        self.resume_btn.config(state="disabled")
        # Stop остаётся активна

    def on_stop(self):
        # выставляем stop
        self.fsm.stop()
        logging.info("FSM stop requested by user")
        # Обновляем кнопки: Disable Pause/Resume/Stop, Enable Start
        self.pause_btn.config(state="disabled")
        self.resume_btn.config(state="disabled")
        self.stop_btn.config(state="disabled")
        self.start_btn.config(state="disabled")
        # Можно обновить статус_label:
        self.status_label.config(text="State: STOPPED")
    
    def on_fsm_finished(self):
        """Вызывается из FSM, когда он завершился или был остановлен."""
        # Сброс кнопок: включить Start, отключить Pause/Resume/Stop
        try:
            self.start_btn.config(state="normal")
            self.pause_btn.config(state="disabled")
            self.resume_btn.config(state="disabled")
            self.stop_btn.config(state="disabled")
        except Exception:
            pass
        self.status_label.config(text="State: DONE")

        if hasattr(self.fsm, "reset"):
            self.fsm.reset()
        logging.info("GUI: FSM finished, buttons reset")

if __name__ == "__main__":
    app = FSMGUI(data_path=DATA_PATH)
    app.mainloop()
