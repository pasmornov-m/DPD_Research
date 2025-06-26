from modules import metrics, learning, params
from modules.gmp_model import GMP


class GMPipeline:
    def __init__(self, data_dict, train_props):
        self.data = data_dict
        cfg = data_dict["config"]
        self.fs = cfg["input_signal_fs"]
        self.bw_main = cfg["bw_main_ch"]
        self.nperseg = cfg["nperseg"]

        self.x_train = data_dict["train_input"]
        self.y_train = data_dict["train_output"]
        self.x_val = data_dict["val_input"]
        self.y_val = data_dict["val_output"]

        self.gain = metrics.calculate_gain_complex(self.x_train, self.y_train)
        print(f"[PA] Calculated gain: {self.gain:.4f}")
        self.y_train_target = self.gain * self.x_train
        self.y_val_target = self.gain * self.x_val

        self.pa_deg = train_props["pa_degree"]
        self.dpd_deg = train_props["dpd_degree"]
        self.lr = train_props["lr"]
        self.epochs = train_props["epochs"]
        
        self.uk_epochs = 1000
        self.uk_lr = 0.001

        self.results = {}

    def _make_model(self, degree, mtype):
        cfg = params.make_gmp_params(
            Ka=degree, La=degree,
            Kb=degree, Lb=degree, Mb=degree,
            Kc=degree, Lc=degree, Mc=degree,
            model_type=mtype
        )
        return GMP(**cfg)

    def run_pa(self):
        model = self._make_model(self.pa_deg, "pa_grad")
        if not model.load_weights():
            model.optimize_weights(self.x_train, self.y_train, epochs=self.epochs, learning_rate=self.lr)
            model.save_weights()
        y_pa = model(self.x_val).detach()
        self.results["y_gmp_pa"] = y_pa
        self.results["nmse_pa"] = metrics.compute_nmse(y_pa, self.y_val)
        self.pa_model = model
        print(f"[PA] NMSE: {self.results['nmse_pa']:.4f}")

    def run_dla(self):
        model = self._make_model(self.dpd_deg, "dpd_dla_grad")
        if not model.load_weights():
            learning.optimize_dla(
                self.x_train, self.y_train_target,
                model, self.pa_model,
                epochs=self.epochs, learning_rate=self.lr
            )
            model.save_weights()
        y_dla = model(self.x_val).detach()
        y_lin = self.pa_model(y_dla).detach()
        self.results["y_dpd_dla_grad"] = y_dla
        self.results["y_linearized_dla_grad"] = y_lin
        self.results["nmse_dla"] = metrics.compute_nmse(y_lin, self.y_val_target)
        print(f"[DLA] NMSE after PA: {self.results['nmse_dla']:.4f}")

    def run_ila(self, epochs=1_000, lr=1e-2):
        model = self._make_model(self.dpd_deg, "dpd_ila_grad")
        if not model.load_weights():
            learning.optimize_ila(
                model, self.x_train, self.y_train,
                self.gain, epochs=epochs, learning_rate=lr,
                pa_model=self.pa_model
            )
            model.save_weights()
        y_ila = model(self.x_val).detach()
        y_lin = self.pa_model(y_ila).detach()
        self.results["y_dpd_ila_grad"] = y_ila
        self.results["y_linearized_ila_grad"] = y_lin
        self.results["nmse_ila"] = metrics.compute_nmse(y_lin, self.y_val_target)
        print(f"[ILA] NMSE after PA: {self.results['nmse_ila']:.4f}")

    def run_ilc(self):
        u = learning.ilc_signal(
            self.x_train, self.y_train_target,
            self.pa_model, epochs=self.uk_epochs, learning_rate=self.uk_lr
        )
        self.results["u_k"] = u
        u_pa = self.pa_model(u).detach()
        self.results["u_k_pa"] = u_pa

        model = self._make_model(self.dpd_deg, "dpd_ilc_grad")
        if not model.load_weights():
            model.optimize_weights(self.x_train, u, epochs=self.epochs, learning_rate=self.lr)
            model.save_weights()

        y_ilc_dpd = model(self.x_val).detach()
        y_lin = self.pa_model(y_ilc_dpd).detach()

        self.results["y_dpd_ilc_grad"] = y_ilc_dpd
        self.results["y_linearized_ilc_grad"] = y_lin
        self.results["nmse_ilc"] = metrics.compute_nmse(y_lin, self.y_val_target)
        print(f"[ILC] NMSE after PA: {self.results['nmse_ilc']:.4f}")

    def run(self):
        """Запустить весь конвейер последовательно."""
        self.run_pa()
        self.run_dla()
        self.run_ila()
        self.run_ilc()
        return self.results
