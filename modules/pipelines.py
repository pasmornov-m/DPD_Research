from modules import metrics, learning, params, data_loader, utils
from modules.gmp_model import GMP
from modules.nn_model import GRU
import torch


class SimplePipeline():
    def __init__(self, data_dict, train_props, base_model_cls):
        
        self.base_model_cls = base_model_cls
        self.pa_model = None
        
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
        
        if self.base_model_cls.__name__ == "GMP":
            self.model_params = params.make_gmp_params(**train_props["gmp_degree"])
        else:
            self.model_params = [train_props["hidden_size"], train_props["num_layers"]]
        self.lr = train_props["lr"]
        self.epochs = train_props["epochs"]
        self.acpr_meter = train_props["acpr_meter"]

        self.results = {}
        
        self.criterion = metrics.compute_mse
        self.metric_criterion = metrics.compute_nmse
        
        self.frame_length = None
        self.batch_size = None
        self.batch_size_eval = None
        self.grad_clip_val = None
        self.u_k_lr = None
        self.u_k_epochs = None
        
        self.dla_train_loader = None
        self.dla_val_loader = None
        self.ila_train_loader = None
        self.ila_val_loader = None
        self.ilc_train_loader = None
        self.ilc_val_loader = None
        
        self._prepare_params()
        self._prepare_loaders()
    
    def _prepare_params(self):
        if self.base_model_cls.__name__ == "GMP":
            self.grad_clip_val = 0
            self.u_k_lr=0.001
            self.u_k_epochs=1000
        else:
            self.frame_length = 16
            self.batch_size = 256
            self.batch_size_eval = 2048
            self.grad_clip_val = 1.0
            self.u_k_lr = 0.1
            self.u_k_epochs = 200
    
    def _prepare_loaders(self):
        if self.base_model_cls.__name__ == "GMP":
            self.dla_train_loader = [(self.x_train, self.y_train_target)]
            self.dla_val_loader = [(self.x_val, self.y_val_target)]
            self.ila_train_loader = [(self.x_train, self.x_train)]
            self.ila_val_loader = [(self.x_val, self.x_val)]
        else:
            self.dla_train_loader, self.dla_val_loader = data_loader.build_dataloaders(data_dict=self.data_dict, 
                                                                            frame_length=self.frame_length, 
                                                                            batch_size=self.batch_size, 
                                                                            batch_size_eval=self.batch_size_eval, 
                                                                            arch="dla")
            self.ila_train_loader, self.ila_val_loader = data_loader.build_dataloaders(data_dict=self.data_dict, 
                                                                            frame_length=self.frame_length, 
                                                                            batch_size=self.batch_size, 
                                                                            batch_size_eval=self.batch_size_eval, 
                                                                            arch="ila")
    def run_pa(self):
        self.pa_model = self.base_model_cls(**self.model_params, model_name="pa")

        optimizer = torch.optim.Adam(self.pa_model.parameters(), lr=self.lr)

        if not self.pa_model.load_weights():
            learning.train(net=self.pa_model, 
                    criterion=self.criterion, 
                    optimizer=optimizer, 
                    train_loader=self.pa_train_loader, 
                    val_loader=self.pa_val_loader, 
                    grad_clip_val=1.0, 
                    n_epochs=self.epochs, 
                    metric_criterion=self.metric_criterion)
            self.pa_model.save_weights()
        utils.freeze_pa_model(self.pa_model)

        y_val_pa_model = learning.net_inference(net=self.pa_model, x=self.x_val)
        pa_model_nmse = metrics.compute_nmse(y_val_pa_model, self.y_val)
        pa_model_acpr = metrics.calculate_acpr(y_val_pa_model, self.acpr_meter)
        print(f"[PA] NMSE: {pa_model_nmse:.2f}, ACPR: {pa_model_acpr}")
        self.results["pa"] = {
            "nmse": pa_model_nmse,
            "acpr": pa_model_acpr,
            "y_val_pa_model": y_val_pa_model
        }
    
    def run_dla(self):
        
        dpd_model = self.base_model_cls(**self.model_params)
        is_load = dpd_model.load_weights()
        casc_dla = utils.CascadeModel(model_1=dpd_model, 
                                        model_2=self.pa_model, 
                                        cascade_type="dla")
        optimizer = torch.optim.Adam(casc_dla.parameters(), lr=self.lr)

        if not is_load:
            learning.train(net=casc_dla, 
                        criterion=self.criterion, 
                        optimizer=optimizer, 
                        train_loader=self.dla_train_loader, 
                        val_loader=self.dla_val_loader, 
                        grad_clip_val=self.grad_clip_val, 
                        n_epochs=self.epochs, 
                        metric_criterion=self.metric_criterion)
            dpd_model.save_weights()
        

        y_val_dla = learning.net_inference(net=casc_dla, x=self.x_val)
        dla_nmse = metrics.compute_nmse(y_val_dla, self.y_val_target)
        dla_acpr = metrics.calculate_acpr(y_val_dla, self.acpr_meter)
        print(f"[DLA] NMSE: {dla_nmse:.2f}, ACPR: {dla_acpr}")

        self.results["dla"] = {
            "nmse": dla_nmse,
            "acpr": dla_acpr,
            "y_val_dla": y_val_dla
        }

    def run_ila(self):
        dpd_model = self.base_model_cls(**self.model_params)
        is_load = dpd_model.load_weights()
        casc_ila_train = utils.CascadeModel(model_1=self.pa_model, model_2=dpd_model, gain=self.gain, cascade_type="ila")
        casc_ila_eval = utils.CascadeModel(model_1=dpd_model, model_2=self.pa_model)
        optimizer = torch.optim.Adam(casc_ila_train.parameters(), lr=self.lr)

        if not is_load:
            learning.train(net=casc_ila_train, 
                        criterion=self.criterion, 
                        optimizer=optimizer, 
                        train_loader=self.ila_train_loader, 
                        val_loader=self.ila_val_loader, 
                        grad_clip_val=self.grad_clip_val, 
                        n_epochs=self.epochs, 
                        metric_criterion=self.metric_criterion)
            dpd_model.save_weights()
        
        y_val_ila = learning.net_inference(net=casc_ila_eval, x=self.x_val)
        ila_nmse = metrics.compute_nmse(y_val_ila, self.y_val_target)
        ila_acpr = metrics.calculate_acpr(y_val_ila, self.acpr_meter)
        print(f"[ILA] NMSE: {ila_nmse:.2f}, ACPR: {ila_acpr}")

        self.results["ila"] = {
            "nmse": ila_nmse,
            "acpr": ila_acpr,
            "y_val_ila": y_val_ila
        }
        
    def run_ilc(self):
        self.u_k_train = learning.ilc_signal(self.x_train, self.y_train_target, self.pa_model, epochs=self.u_k_epochs, learning_rate=self.u_k_lr)
        self.u_k_pa = self.pa_model.forward(self.u_k_train).detach()

        # self.data_dict["ilc_train_output"] = self.u_k_train
        ilc_nmse_uk = metrics.compute_nmse(self.u_k_pa, self.y_train_target)
        ilc_acpr_uk = metrics.calculate_acpr(self.u_k_pa, self.acpr_meter)
        print(f"[UK] NMSE: {ilc_nmse_uk:.2f}, ACPR: {ilc_acpr_uk}")

        self.results["uk"] = {
            "nmse": ilc_nmse_uk,
            "acpr": ilc_acpr_uk,
            "u_k_train": self.u_k_train
        }
        
        if self.base_model_cls.__name__ == "GMP":
            self.ilc_train_loader = [(self.x_train, self.u_k_train)]
            self.ilc_val_loader = self.ilc_train_loader
        else:
            self.ilc_train_loader, self.ilc_val_loader = data_loader.build_dataloaders(data_dict=self.data_dict, 
                                                                            frame_length=self.frame_length, 
                                                                            batch_size=self.batch_size, 
                                                                            batch_size_eval=self.batch_size_eval, 
                                                                            arch="ilc")

        dpd_model = self.base_model_cls(**self.model_params)
        is_load = dpd_model.load_weights()
        optimizer = torch.optim.Adam(dpd_model.parameters(), lr=self.lr)
        if not is_load:
            learning.train(net=dpd_model, 
                        criterion=self.criterion, 
                        optimizer=optimizer, 
                        train_loader=self.ilc_train_loader, 
                        val_loader=self.ilc_val_loader, 
                        grad_clip_val=self.grad_clip_val, 
                        n_epochs=self.epochs, 
                        metric_criterion=self.metric_criterion)
            dpd_model.save_weights()

        casc_ilc_eval = utils.CascadeModel(model_1=dpd_model, model_2=self.pa_model)
        y_val_ilc = learning.net_inference(net=casc_ilc_eval, x=self.x_val)
        ilc_nmse = metrics.compute_nmse(y_val_ilc, self.y_val_target)
        ilc_acpr = metrics.calculate_acpr(y_val_ilc, self.acpr_meter)
        print(f"[ILC] NMSE: {ilc_nmse:.2f}, ACPR: {ilc_acpr}")

        self.results["ilc"] = {
            "nmse": ilc_nmse,
            "acpr": ilc_acpr,
            "y_val_ilc": y_val_ilc
        }