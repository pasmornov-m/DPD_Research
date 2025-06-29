from modules import metrics, learning, params, data_loader, utils
from modules.gmp_model import GMP
from modules.nn_model import GRU
import torch
from torch import nn
from typing import Dict
import time
from datetime import timedelta


class SimplePipeline():
    def __init__(self, data_dict, train_props, base_model):
        
        self.base_model = base_model
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
        print(f"[PA] Calculated gain: {self.gain:.2f}")
        self.y_train_target = self.gain * self.x_train
        self.y_val_target = self.gain * self.x_val
        
        if self.base_model.__name__ == "GMP":
            self.model_params = params.make_gmp_params(Ka=train_props["gmp_degree"],
                                                       La=train_props["gmp_degree"],
                                                       Kb=train_props["gmp_degree"],
                                                       Lb=train_props["gmp_degree"],
                                                       Mb=train_props["gmp_degree"],
                                                       Kc=train_props["gmp_degree"],
                                                       Lc=train_props["gmp_degree"],
                                                       Mc=train_props["gmp_degree"])
        else:
            self.model_params = {"hidden_size": train_props["hidden_size"], 
                                 "num_layers": train_props["num_layers"]}
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
        
        self.pa_train_loader = None
        self.pa_val_loader = None
        self.dla_train_loader = None
        self.dla_val_loader = None
        self.ila_train_loader = None
        self.ila_val_loader = None
        self.ilc_train_loader = None
        self.ilc_val_loader = None
        
        self._prepare_params()
        self._prepare_loaders()
    
    def _prepare_params(self):
        if self.base_model.__name__ == "GMP":
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
        if self.base_model.__name__ == "GMP":
            self.pa_train_loader = [(self.x_train, self.y_train)]
            self.pa_val_loader = [(self.x_val, self.y_val)]
            self.dla_train_loader = [(self.x_train, self.y_train_target)]
            self.dla_val_loader = [(self.x_val, self.y_val_target)]
            self.ila_train_loader = [(self.x_train, self.x_train)]
            self.ila_val_loader = [(self.x_val, self.x_val)]
        else:
            self.pa_train_loader, self.pa_val_loader = data_loader.build_dataloaders(data_dict=self.data, 
                                                                           frame_length=self.frame_length, 
                                                                           batch_size=self.batch_size, 
                                                                           batch_size_eval=self.batch_size_eval)
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
        print("Run PA")
        self.pa_model = self.base_model(**self.model_params, model_name="pa")

        optimizer = torch.optim.Adam(self.pa_model.parameters(), lr=self.lr)

        time_train = 0
        if not self.pa_model.load_weights():
            start = time.time()
            learning.train(net=self.pa_model, 
                    criterion=self.criterion, 
                    optimizer=optimizer, 
                    train_loader=self.pa_train_loader, 
                    val_loader=self.pa_val_loader, 
                    grad_clip_val=1.0, 
                    n_epochs=self.epochs, 
                    metric_criterion=self.metric_criterion)
            elapsed = time.time() - start
            time_train = timedelta(seconds=round(elapsed))
            self.pa_model.save_weights()
        utils.freeze_pa_model(self.pa_model)

        y_val_pa_model = learning.net_inference(net=self.pa_model, x=self.x_val)
        pa_model_nmse = metrics.compute_nmse(y_val_pa_model, self.y_val)
        pa_model_acpr = metrics.calculate_acpr(y_val_pa_model, self.acpr_meter)
        print(f"[PA] NMSE: {pa_model_nmse:.2f}, ACPR: {pa_model_acpr}")
        self.results["pa"] = {
            "nmse": pa_model_nmse.item(),
            "acpr": pa_model_acpr,
            "y_val_pa_model": y_val_pa_model,
            "time_train": time_train
        }
    
    def run_dla(self):
        print("Run DLA")
        dpd_model = self.base_model(**self.model_params, model_name="dla")
        is_load = dpd_model.load_weights()
        casc_dla = utils.CascadeModel(model_1=dpd_model, 
                                        model_2=self.pa_model, 
                                        cascade_type="dla")
        optimizer = torch.optim.Adam(casc_dla.parameters(), lr=self.lr)

        time_train = 0
        if not is_load:
            start = time.time()
            train_time = learning.train(net=casc_dla, 
                        criterion=self.criterion, 
                        optimizer=optimizer, 
                        train_loader=self.dla_train_loader, 
                        val_loader=self.dla_val_loader, 
                        grad_clip_val=self.grad_clip_val, 
                        n_epochs=self.epochs, 
                        metric_criterion=self.metric_criterion)
            elapsed = time.time() - start
            time_train = timedelta(seconds=round(elapsed))
            dpd_model.save_weights()
        
        y_val_dla = learning.net_inference(net=casc_dla, x=self.x_val)
        dla_nmse = metrics.compute_nmse(y_val_dla, self.y_val_target)
        dla_acpr = metrics.calculate_acpr(y_val_dla, self.acpr_meter)
        print(f"[DLA] NMSE: {dla_nmse:.2f}, ACPR: {dla_acpr}")

        self.results["dla"] = {
            "nmse": dla_nmse.item(),
            "acpr": dla_acpr,
            "y_val_dla": y_val_dla,
            "time_train": time_train
        }

    def run_ila(self):
        print("Run ILA")
        dpd_model = self.base_model(**self.model_params, model_name="ila")
        is_load = dpd_model.load_weights()
        casc_ila_train = utils.CascadeModel(model_1=self.pa_model, model_2=dpd_model, gain=self.gain, cascade_type="ila")
        casc_ila_eval = utils.CascadeModel(model_1=dpd_model, model_2=self.pa_model)
        optimizer = torch.optim.Adam(casc_ila_train.parameters(), lr=self.lr)

        time_train = 0
        if not is_load:
            start = time.time()
            learning.train(net=casc_ila_train, 
                        criterion=self.criterion, 
                        optimizer=optimizer, 
                        train_loader=self.ila_train_loader, 
                        val_loader=self.ila_val_loader, 
                        grad_clip_val=self.grad_clip_val, 
                        n_epochs=self.epochs, 
                        metric_criterion=self.metric_criterion)
            elapsed = time.time() - start
            time_train = timedelta(seconds=round(elapsed))
            dpd_model.save_weights()
        
        y_val_ila = learning.net_inference(net=casc_ila_eval, x=self.x_val)
        ila_nmse = metrics.compute_nmse(y_val_ila, self.y_val_target)
        ila_acpr = metrics.calculate_acpr(y_val_ila, self.acpr_meter)
        print(f"[ILA] NMSE: {ila_nmse:.2f}, ACPR: {ila_acpr}")

        self.results["ila"] = {
            "nmse": ila_nmse.item(),
            "acpr": ila_acpr,
            "y_val_ila": y_val_ila,
            "time_train": time_train
        }
        
    def run_ilc(self):
        print("Run ILC")
        time_train = 0
        start = time.time()
        
        self.u_k_train = learning.ilc_signal(self.x_train, self.y_train_target, self.pa_model, epochs=self.u_k_epochs, learning_rate=self.u_k_lr)
        self.u_k_pa = self.pa_model.forward(self.u_k_train).detach()
        
        elapsed = time.time() - start
        time_train = timedelta(seconds=round(elapsed))

        # self.data_dict["ilc_train_output"] = self.u_k_train
        ilc_nmse_uk = metrics.compute_nmse(self.u_k_pa, self.y_train_target)
        ilc_acpr_uk = metrics.calculate_acpr(self.u_k_pa, self.acpr_meter)
        print(f"[UK] NMSE: {ilc_nmse_uk:.2f}, ACPR: {ilc_acpr_uk}")

        self.results["uk"] = {
            "nmse": ilc_nmse_uk.item(),
            "acpr": ilc_acpr_uk,
            "u_k_pa": self.u_k_pa,
            "u_k_train": self.u_k_train,
            "time_train": time_train
        }
        
        if self.base_model.__name__ == "GMP":
            self.ilc_train_loader = [(self.x_train, self.u_k_train)]
            self.ilc_val_loader = self.ilc_train_loader
        else:
            self.ilc_train_loader, self.ilc_val_loader = data_loader.build_dataloaders(data_dict=self.data_dict, 
                                                                            frame_length=self.frame_length, 
                                                                            batch_size=self.batch_size, 
                                                                            batch_size_eval=self.batch_size_eval, 
                                                                            arch="ilc")

        dpd_model = self.base_model(**self.model_params, model_name="ilc")
        is_load = dpd_model.load_weights()
        optimizer = torch.optim.Adam(dpd_model.parameters(), lr=self.lr)
        
        time_train = 0
        if not is_load:
            start = time.time()
            learning.train(net=dpd_model, 
                        criterion=self.criterion, 
                        optimizer=optimizer, 
                        train_loader=self.ilc_train_loader, 
                        val_loader=self.ilc_val_loader, 
                        grad_clip_val=self.grad_clip_val, 
                        n_epochs=self.epochs, 
                        metric_criterion=self.metric_criterion)
            elapsed = time.time() - start
            time_train = timedelta(seconds=round(elapsed))
            dpd_model.save_weights()

        casc_ilc_eval = utils.CascadeModel(model_1=dpd_model, model_2=self.pa_model)
        y_val_ilc = learning.net_inference(net=casc_ilc_eval, x=self.x_val)
        ilc_nmse = metrics.compute_nmse(y_val_ilc, self.y_val_target)
        ilc_acpr = metrics.calculate_acpr(y_val_ilc, self.acpr_meter)
        print(f"[ILC] NMSE: {ilc_nmse:.2f}, ACPR: {ilc_acpr}")

        self.results["ilc"] = {
            "nmse": ilc_nmse.item(),
            "acpr": ilc_acpr,
            "y_val_ilc": y_val_ilc,
            "time_train": time_train
        }
    
    def run(self):
        self.run_pa()
        self.run_dla()
        self.run_ila()
        self.run_ilc()
        return self.results
    
    def get_results(self):
        return self.results
    
    def get_pa_model(self):
        return self.pa_model


class SnrPipeline:
    def __init__(self,
                 base_model: nn.Module,
                 model_params: Dict,
                 data_dict: Dict,
                 snr_params: Dict):
        
        self.base_model = base_model
        self.model_params = model_params
        self.data_dict = data_dict
        self.snr_params = snr_params

        self.x_train = data_dict["train_input"]
        self.y_train = data_dict["train_output"]
        self.x_val = data_dict["val_input"]
        self.y_val = data_dict["val_output"]
        self.y_train_target = data_dict['y_train_target']
        self.y_val_target = data_dict['y_val_target']
        self.u_k_train = None
        self.u_k_val = None
            
        self.snr_range = snr_params["snr_range"]
        self.num_realizations = snr_params["num_realizations"]
        self.fs = snr_params["fs"]
        self.bw_main_ch = snr_params["bw_main_ch"]
        self.epochs = snr_params["epochs"]
        self.lr = snr_params["learning_rate"]
        self.acpr_meter = snr_params["acpr_meter"]
        self.pa_model = snr_params["pa_model"]
        self.gain = snr_params["gain"]
        
        self.results = {
            "dla": {"nmse": [], "acpr_left": [], "acpr_right": []},
            "ila": {"nmse": [], "acpr_left": [], "acpr_right": []},
            "ilc": {"nmse": [], "acpr_left": [], "acpr_right": []},
            "uk":  {"nmse": [], "acpr_left": [], "acpr_right": []},
            "snr_range": self.snr_range,
            "base_model": self.base_model
        }
        
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
        if self.base_model.__name__ == "GMP":
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
        if self.base_model.__name__ == "GMP":
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
    def _create_pa_noise_cascade(self, snr):
        noise_gen_model = utils.NoiseModel(snr=snr, fs=self.fs, bw=self.bw_main_ch)
        casc_pa_noise = utils.CascadeModel(model_1=self.pa_model, model_2=noise_gen_model, cascade_type="dla")
        return casc_pa_noise
    
    def _dla_arch_with_noise(self):
        
        for snr in self.snr_range:
            print(f"SNR: {snr}")
            dpd_model = self.base_model(**self.model_params)
            casc_pa_noise = self._create_pa_noise_cascade(snr)
            casc_dla = utils.CascadeModel(model_1=dpd_model, 
                                            model_2=casc_pa_noise, 
                                            cascade_type="dla")
            optimizer = torch.optim.Adam(casc_dla.parameters(), lr=self.lr)

            learning.train(net=casc_dla, 
                        criterion=self.criterion, 
                        optimizer=optimizer, 
                        train_loader=self.dla_train_loader, 
                        val_loader=self.dla_val_loader, 
                        grad_clip_val=self.grad_clip_val, 
                        n_epochs=self.epochs, 
                        metric_criterion=self.metric_criterion)

            nmse, acpr_left, acpr_right = metrics.noise_realizations(self.num_realizations, 
                                                            model=casc_dla, 
                                                            x=self.x_val, 
                                                            y_target=self.y_val_target, 
                                                            acpr_meter=self.acpr_meter)
            self.results["dla"]["nmse"].append(nmse.item())
            self.results["dla"]["acpr_left"].append(acpr_left)
            self.results["dla"]["acpr_right"].append(acpr_right)

        
    def _ila_arch_with_noise(self):
        
        for snr in self.snr_range:
            print(f"SNR: {snr}")
            dpd_model = self.base_model(**self.model_params)
            casc_pa_noise = self._create_pa_noise_cascade(snr)
            casc_ila_train = utils.CascadeModel(model_1=casc_pa_noise, model_2=dpd_model, gain=self.gain, cascade_type="ila")
            casc_ila_eval = utils.CascadeModel(model_1=dpd_model, model_2=casc_pa_noise, cascade_type="dla")
            optimizer = torch.optim.Adam(casc_ila_train.parameters(), lr=self.lr)

            learning.train(net=casc_ila_train, 
                        criterion=self.criterion, 
                        optimizer=optimizer, 
                        train_loader=self.ila_train_loader, 
                        val_loader=self.ila_val_loader, 
                        grad_clip_val=self.grad_clip_val, 
                        n_epochs=self.epochs, 
                        metric_criterion=self.metric_criterion)

            nmse, acpr_left, acpr_right = metrics.noise_realizations(self.num_realizations, 
                                                            model=casc_ila_eval, 
                                                            x=self.x_val, 
                                                            y_target=self.y_val_target, 
                                                            acpr_meter=self.acpr_meter)
            self.results["ila"]["nmse"].append(nmse.item())
            self.results["ila"]["acpr_left"].append(acpr_left)
            self.results["ila"]["acpr_right"].append(acpr_right)

    
    def _ilc_arch_with_noise(self):
        
        for snr in self.snr_range:
            print(f"SNR: {snr}")
            casc_pa_noise = self._create_pa_noise_cascade(snr)
            
            self.u_k_train = learning.ilc_signal(self.x_train, self.y_train_target, casc_pa_noise, epochs=self.u_k_epochs, learning_rate=self.u_k_lr)
            # self.u_k_val = learning.ilc_signal(self.x_val, self.y_val_target, casc_pa_noise, epochs=self.u_k_epochs, learning_rate=self.u_k_lr)

            self.data_dict["ilc_train_output"] = self.u_k_train
            # self.data_dict["ilc_val_output"] = self.u_k_val
            
            if self.base_model.__name__ == "GMP":
                self.ilc_train_loader = [(self.x_train, self.u_k_train)]
                # self.ilc_val_loader = [(self.x_val, self.u_k_val)]
                self.ilc_val_loader = self.ilc_train_loader
            else:
                self.ilc_train_loader, self.ilc_val_loader = data_loader.build_dataloaders(data_dict=self.data_dict, 
                                                                                frame_length=self.frame_length, 
                                                                                batch_size=self.batch_size, 
                                                                                batch_size_eval=self.batch_size_eval, 
                                                                                arch="ilc")

            nmse_uk, acpr_left_uk, acpr_right_uk = metrics.noise_realizations(self.num_realizations, 
                                                                    model=casc_pa_noise, 
                                                                    x=self.u_k_train, 
                                                                    y_target=self.y_train_target, 
                                                                    acpr_meter=self.acpr_meter)
            self.results["uk"]["nmse"].append(nmse_uk.item())
            self.results["uk"]["acpr_left"].append(acpr_left_uk)
            self.results["uk"]["acpr_right"].append(acpr_right_uk)

            dpd_model = self.base_model(**self.model_params)
            optimizer = torch.optim.Adam(dpd_model.parameters(), lr=self.lr)
            learning.train(net=dpd_model, 
                        criterion=self.criterion, 
                        optimizer=optimizer, 
                        train_loader=self.ilc_train_loader, 
                        val_loader=self.ilc_val_loader, 
                        grad_clip_val=self.grad_clip_val, 
                        n_epochs=self.epochs, 
                        metric_criterion=self.metric_criterion)

            casc_ilc_eval = utils.CascadeModel(model_1=dpd_model, model_2=casc_pa_noise, cascade_type="dla")
            nmse, acpr_left, acpr_right = metrics.noise_realizations(self.num_realizations, 
                                                            model=casc_ilc_eval, 
                                                            x=self.x_val, 
                                                            y_target=self.y_val_target, 
                                                            acpr_meter=self.acpr_meter)
            self.results["ilc"]["nmse"].append(nmse.item())
            self.results["ilc"]["acpr_left"].append(acpr_left)
            self.results["ilc"]["acpr_right"].append(acpr_right)

    
    def run(self, arch_name=None):
        if arch_name == "DLA":
            self._dla_arch_with_noise()
        elif arch_name == "ILA":
            self._ila_arch_with_noise()
        elif arch_name == "ILC":
            self._ilc_arch_with_noise()
    
    def get_results(self):
        return self.results