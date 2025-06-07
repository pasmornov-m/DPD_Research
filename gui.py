import threading
import tkinter as tk
from tkinter import ttk, messagebox

from modules.params import ModelParams
from modules.fsm import PA_DPD_FSM

class FSMGUI(tk.Tk):
    def __init__(self, data_path):
        super().__init__()
        self.title("PA & DPD FSM")
        self.geometry("1000x1000")
        self.fsm = PA_DPD_FSM(data_path)

        # --- PA-параметры ---
        pa_frame = ttk.LabelFrame(self, text="PA Params", padding=10)
        pa_frame.grid(row=0, column=0, padx=10, pady=10, sticky="ew")
        self.pa_fields = {}
        for i, name in enumerate(["Ka","La","Kb","Lb","Mb","Kc","Lc","Mc"]):
            ttk.Label(pa_frame, text=name).grid(row=i, column=0, sticky="e")
            e = ttk.Entry(pa_frame, width=5)
            e.grid(row=i, column=1, sticky="w")
            e.insert(0, 2)
            self.pa_fields[name] = e

        ttk.Label(pa_frame, text="Epochs").grid(row=8, column=0, sticky="e")
        self.pa_epoch = ttk.Entry(pa_frame, width=7)
        self.pa_epoch.grid(row=8, column=1, sticky="w")
        self.pa_epoch.insert(0, 100)

        ttk.Label(pa_frame, text="LR").grid(row=9, column=0, sticky="e")
        self.pa_lr = ttk.Entry(pa_frame, width=7)
        self.pa_lr.grid(row=9, column=1, sticky="w")
        self.pa_lr.insert(0, 0.1)

        # --- DPD-параметры ---
        dpd_frame = ttk.LabelFrame(self, text="DPD Params", padding=10)
        dpd_frame.grid(row=1, column=0, padx=10, pady=10, sticky="ew")
        self.dpd_fields = {}
        for i, name in enumerate(["Ka","La","Kb","Lb","Mb","Kc","Lc","Mc"]):
            ttk.Label(dpd_frame, text=name).grid(row=i, column=0, sticky="e")
            e = ttk.Entry(dpd_frame, width=5)
            e.grid(row=i, column=1, sticky="w")
            e.insert(0, 2)
            self.dpd_fields[name] = e


        ttk.Label(dpd_frame, text="Epochs").grid(row=8, column=0, sticky="e")
        self.dpd_epoch = ttk.Entry(dpd_frame, width=7)
        self.dpd_epoch.grid(row=8, column=1, sticky="w")
        self.dpd_epoch.insert(0, 100)

        ttk.Label(dpd_frame, text="LR").grid(row=9, column=0, sticky="e")
        self.dpd_lr = ttk.Entry(dpd_frame, width=7)
        self.dpd_lr.grid(row=9, column=1, sticky="w")
        self.dpd_lr.insert(0, 0.1)

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
        self.status_label = ttk.Label(self, text="State: INIT")
        self.status_label.grid(row=3, column=0, pady=5)

        start_btn = ttk.Button(self, text="Start FSM", command=self.on_start)
        start_btn.grid(row=4, column=0, pady=10)

        # Периодически обновлять статус из FSM
        self.after(200, self.refresh_status)

    def refresh_status(self):
        """Обновляем метку состояния из FSM."""
        st = self.fsm.state.name
        self.status_label.config(text=f"State: {st}")
        # запускаем снова через 200ms
        self.after(200, self.refresh_status)

    def on_start(self):
        """Считать всё из полей, сконфигурировать FSM и запустить его."""
        try:
            # PA params
            pa = ModelParams(
                model_type="GMP",
                Ka=int(self.pa_fields["Ka"].get()),
                La=int(self.pa_fields["La"].get()),
                Kb=int(self.pa_fields["Kb"].get()),
                Lb=int(self.pa_fields["Lb"].get()),
                Mb=int(self.pa_fields["Mb"].get()),
                Kc=int(self.pa_fields["Kc"].get()),
                Lc=int(self.pa_fields["Lc"].get()),
                Mc=int(self.pa_fields["Mc"].get()),
                epochs=int(self.pa_epoch.get()),
                lr=float(self.pa_lr.get())
            )
            self.fsm.set_pa_params(pa)

            # DPD params
            dpd = ModelParams(
                model_type="GMP",
                Ka=int(self.dpd_fields["Ka"].get()),
                La=int(self.dpd_fields["La"].get()),
                Kb=int(self.dpd_fields["Kb"].get()),
                Lb=int(self.dpd_fields["Lb"].get()),
                Mb=int(self.dpd_fields["Mb"].get()),
                Kc=int(self.dpd_fields["Kc"].get()),
                Lc=int(self.dpd_fields["Lc"].get()),
                Mc=int(self.dpd_fields["Mc"].get()),
                epochs=int(self.dpd_epoch.get()),
                lr=float(self.dpd_lr.get())
            )
            self.fsm.set_dpd_params(dpd)

            # Noise range
            snr_range = [int(s.strip()) for s in self.noise.get().split(",") if s.strip()]
            num_real = int(self.noise_real.get())
            self.fsm.set_noise_range(snr_range, num_real)

            # Запустить FSM в фоне
            # threading.Thread(target=self.fsm.run, daemon=True).start()
            self.fsm.run()

        except ValueError:
            messagebox.showwarning("Input error", "Проверьте корректность введённых параметров.")

if __name__ == "__main__":
    app = FSMGUI(data_path="DPA_200MHz")
    app.mainloop()
