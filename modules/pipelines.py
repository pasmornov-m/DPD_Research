from typing import Dict, Type, Callable
import time
from datetime import timedelta
import copy
import torch
import numpy as np
from modules import metrics, learning, params, data_loader, utils, prunning
from modules.config import FRAME_LENGTH, BATCH_SIZE, BATCH_SIZE_EVAL, GRAD_CLIP_VAL, U_K_LR, U_K_EPOCHS
from models.gmp import GMP
from models.lstm import LSTM
from models.gru import GRU
from models.rtdtnn import RTDTNN
from models.leann import LEANN
from models.tcn import TCN


class SimplePipeline:
    def __init__(self, 
                 container: data_loader.DataContainer, 
                 train_props: Dict, 
                 base_model: Type[torch.nn.Module],
                 pa_model: Type[torch.nn.Module] = None):
        
        self.base_model = base_model
        self.container = container
        
        self.pa_model = pa_model
        self.dla_model = None
        self.ila_model = None
        self.ilc_model = None

        self.model_params = {}
        self.train_props = train_props
        self.lr = self.train_props["lr"]
        self.epochs = self.train_props["epochs"]
        self.acpr_meter = self.train_props["acpr_meter"]
        self.features_extractor = params.EXTRACTORS_REGISTRY[base_model]
        self.dataloader = params.DATALOADERS_REGISTRY[base_model]
        
        self.results = {
            "pa": {},
            "dla": {},
            "ila": {},
            "ilc": {}
        }
        
        self.criterion = metrics.compute_mse
        self.metric_criterion = metrics.compute_nmse
        
        self.frame_length = FRAME_LENGTH
        self.batch_size = BATCH_SIZE
        self.batch_size_eval = BATCH_SIZE_EVAL
        self.grad_clip_val = GRAD_CLIP_VAL
        self.u_k_lr = U_K_LR
        self.u_k_epochs = U_K_EPOCHS
        
        self._prepare_model_params()
        self.loader_props = params.DATALOADERS_PROPS[self.base_model](self.features_extractor, self.train_props)
        
        self.count_params = self.base_model(**self.model_params).count_params()
    
    def _prepare_model_params(self):
        try:
            builder = params.MODEL_REGISTRY[self.base_model]
        except KeyError:
            raise ValueError(f"Unsupported base_model: {self.base_model}")

        self.model_params = builder(self.train_props)
    
    def prepare_loaders(self, arch):
        train_loader, val_loader, input_norm, target_norm = self.dataloader(self.container,
                                                                            arch=arch,
                                                                            **self.loader_props)
        return train_loader, val_loader, input_norm, target_norm
    
    def _build_optimizer(self, model):
        return torch.optim.AdamW(model.parameters(), lr=self.lr)
    
    def _build_scheduler(self, optimizer):
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
        
    def run_pa(self):
        print("Run PA")
        self.pa_model = self.base_model(**self.model_params, model_name="pa")
        
        count_params = self.pa_model.count_params()
        count_flops = self.pa_model.count_flops()
        count_macs = self.pa_model.count_macs()
        print(f"count_params: {count_params}")
        print(f"count_flops: {count_flops}")
        print(f"count_macs: {count_macs}")
        
        is_load = self.pa_model.load_weights()
        optimizer = self._build_optimizer(self.pa_model)
        scheduler = self._build_scheduler(optimizer)
        
        train_loader, val_loader, input_norm, target_norm = self.prepare_loaders(arch='pa')

        time_train = 0
        if not is_load:
            start = time.time()
            learning.train(net=self.pa_model, 
                    criterion=self.criterion, 
                    optimizer=optimizer, 
                    train_loader=train_loader, 
                    val_loader=val_loader, 
                    grad_clip_val=self.grad_clip_val, 
                    n_epochs=self.epochs, 
                    metric_criterion=self.metric_criterion,
                    scheduler=scheduler)
            elapsed = time.time() - start
            time_train = timedelta(seconds=round(elapsed))
            self.pa_model.save_weights()
        utils.freeze_model(self.pa_model)
        
        y_val_pa_model = learning.net_inference(net=self.pa_model, x=self.container.val_input)
        
        pa_model_nmse = metrics.compute_nmse(y_val_pa_model, self.container.val_output)
        pa_model_acpr = metrics.calculate_acpr(y_val_pa_model, self.acpr_meter)
        print(f"[PA] NMSE: {pa_model_nmse:.2f}, ACPR: {pa_model_acpr[0]:.2f} {pa_model_acpr[1]:.2f}")
        self.results["pa"] = {
            "nmse": pa_model_nmse.item(),
            "acpr": pa_model_acpr,
            "y_val_pa_model": y_val_pa_model,
            "time_train": time_train,
            "count_params": count_params,
            "count_macs": count_macs,
            "count_flops": count_flops,
        }
    
    def run_dla(self):
        print("Run DLA")
        self.dla_model = self.base_model(**self.model_params, model_name="dla")
        
        count_params = self.dla_model.count_params()
        count_macs = self.dla_model.count_macs()
        count_flops = self.dla_model.count_flops()
        print(f"count_params: {count_params}")
        print(f"count_macs: {count_macs}")
        print(f"count_flops: {count_flops}")
        
        is_load = self.dla_model.load_weights()
        casc_dla = utils.CascadeModel(model_1=self.dla_model, 
                                        model_2=self.pa_model)
        optimizer = self._build_optimizer(self.dla_model)
        scheduler = self._build_scheduler(optimizer)
        
        train_loader, val_loader, input_norm, target_norm = self.prepare_loaders(arch='dla')

        time_train = 0
        if not is_load:
            start = time.time()
            learning.train(net=casc_dla, 
                        criterion=self.criterion, 
                        optimizer=optimizer, 
                        train_loader=train_loader, 
                        val_loader=val_loader, 
                        grad_clip_val=self.grad_clip_val, 
                        n_epochs=self.epochs, 
                        metric_criterion=self.metric_criterion,
                        scheduler=scheduler)
            elapsed = time.time() - start
            time_train = timedelta(seconds=round(elapsed))
            self.dla_model.save_weights()
        
        use_normalize = (input_norm is not None) and (target_norm is not None)

        if use_normalize:
            x_val = input_norm.transform(self.container.val_input)
        else:
            x_val = self.container.val_input
        
        if issubclass(self.base_model, (RTDTNN, LEANN)):
            x_val = self.features_extractor(self.container.val_input_orig, M=self.train_props["M"], P=self.train_props.get("P"))
        else:
            x_val = self.container.val_input

        if issubclass(self.base_model, (RTDTNN, LEANN)):
            x_val = input_norm.transform(self.container.val_input_orig)
            x_val = self.features_extractor(x_val, M=self.train_props["M"], P=self.train_props.get("P"))
        
        if use_normalize:
            casc_dla_eval = utils.CascadeModel(model_1=self.dla_model, model_2=self.pa_model, normalizer=target_norm)
        else:
            casc_dla_eval = utils.CascadeModel(model_1=self.dla_model, model_2=self.pa_model)
            
        y_val_dla = learning.net_inference(net=casc_dla_eval, x=x_val)
        dla_nmse = metrics.compute_nmse(y_val_dla, self.container.val_output_target)
        dla_acpr = metrics.calculate_acpr(y_val_dla, self.acpr_meter)
        print(f"[DLA] NMSE: {dla_nmse:.2f}, ACPR: {dla_acpr[0]:.2f} {dla_acpr[1]:.2f}, count_params: {count_params}")

        self.results["dla"] = {
            "nmse": dla_nmse.item(),
            "acpr": dla_acpr,
            "y_val_dla": y_val_dla,
            "time_train": time_train,
            "count_params": count_params,
            "count_macs": count_macs,
            "count_flops": count_flops,
        }

    def run_ila(self):
        print("Run ILA")
        self.ila_model = self.base_model(**self.model_params, model_name="ila")
        
        count_params = self.ila_model.count_params()
        count_macs = self.ila_model.count_macs()
        count_flops = self.ila_model.count_flops()
        print(f"count_params: {count_params}")
        print(f"count_macs: {count_macs}")
        print(f"count_flops: {count_flops}")
        
        is_load = self.ila_model.load_weights()
        casc_ila_train = utils.CascadeModel(model_1=self.pa_model, model_2=self.ila_model, gain=self.container.gain, cascade_type="ila")
        optimizer = self._build_optimizer(self.ila_model)
        scheduler = self._build_scheduler(optimizer)
        
        train_loader, val_loader, input_norm, target_norm = self.prepare_loaders(arch='ila')

        time_train = 0
        if not is_load:
            start = time.time()
            learning.train(net=casc_ila_train, 
                        criterion=self.criterion, 
                        optimizer=optimizer, 
                        train_loader=train_loader, 
                        val_loader=val_loader, 
                        grad_clip_val=self.grad_clip_val, 
                        n_epochs=self.epochs, 
                        metric_criterion=self.metric_criterion,
                        scheduler=scheduler)
            elapsed = time.time() - start
            time_train = timedelta(seconds=round(elapsed))
            self.ila_model.save_weights()
                
        use_normalize = (input_norm is not None) and (target_norm is not None)

        if use_normalize:
            x_val = input_norm.transform(self.container.val_input)
        else:
            x_val = self.container.val_input
        
        if issubclass(self.base_model, (RTDTNN, LEANN)):
            x_val = self.features_extractor(self.container.val_input_orig, M=self.train_props["M"], P=self.train_props.get("P"))
        else:
            x_val = self.container.val_input

        if issubclass(self.base_model, (RTDTNN, LEANN)):
            x_val = input_norm.transform(self.container.val_input_orig)
            x_val = self.features_extractor(x_val, M=self.train_props["M"], P=self.train_props.get("P"))
        
        if use_normalize:
            casc_ila_eval = utils.CascadeModel(model_1=self.ila_model, model_2=self.pa_model, normalizer=target_norm)
        else:
            casc_ila_eval = utils.CascadeModel(model_1=self.ila_model, model_2=self.pa_model)
    
        y_val_ila = learning.net_inference(net=casc_ila_eval, x=x_val)
        ila_nmse = metrics.compute_nmse(y_val_ila, self.container.val_output_target)
        ila_acpr = metrics.calculate_acpr(y_val_ila, self.acpr_meter)
        print(f"[ILA] NMSE: {ila_nmse:.2f}, ACPR: {ila_acpr[0]:.2f} {ila_acpr[1]:.2f}, count_params: {count_params}")

        self.results["ila"] = {
            "nmse": ila_nmse.item(),
            "acpr": ila_acpr,
            "y_val_ila": y_val_ila,
            "time_train": time_train,
            "count_params": count_params,
            "count_macs": count_macs,
            "count_flops": count_flops,
        }
    
    def evaluate_ilc_signal(self):
        time_train = 0
        start = time.time()
        
        if self.container.ilc_train_output is None:
            self.container.ilc_train_output = learning.ilc_signal(self.container.train_input, 
                                                                  self.container.train_output_target, 
                                                                  self.pa_model, 
                                                                  epochs=self.u_k_epochs, 
                                                                  learning_rate=self.u_k_lr)
        if self.container.ilc_val_output is None:
            self.container.ilc_val_output = learning.ilc_signal(self.container.val_input, 
                                                                self.container.val_output_target, 
                                                                self.pa_model, 
                                                                epochs=self.u_k_epochs, 
                                                                learning_rate=self.u_k_lr)
            
        self.u_k_pa = self.pa_model.forward(self.container.ilc_train_output).detach()
        
        elapsed = time.time() - start
        time_train = timedelta(seconds=round(elapsed))

        ilc_nmse_uk = metrics.compute_nmse(self.u_k_pa, self.container.train_output_target)
        ilc_acpr_uk = metrics.calculate_acpr(self.u_k_pa, self.acpr_meter)
        print(f"[UK] NMSE: {ilc_nmse_uk:.2f}, ACPR: {ilc_acpr_uk[0]:.2f} {ilc_acpr_uk[1]:.2f}")

        self.results["uk"] = {
            "nmse": ilc_nmse_uk.item(),
            "acpr": ilc_acpr_uk,
            "u_k_pa": self.u_k_pa,
            "u_k_train": self.container.ilc_train_output,
            "time_train": time_train
        }
        
    def run_ilc(self, load_weights: bool = True):
        print("Run ILC")

        self.ilc_model = self.base_model(**self.model_params, model_name="ilc")
        
        count_params = self.ilc_model.count_params()
        count_macs = self.ilc_model.count_macs()
        count_flops = self.ilc_model.count_flops()
        print(f"count_params: {count_params}")
        print(f"count_macs: {count_macs}")
        print(f"count_flops: {count_flops}")
        
        if load_weights:
            is_load = self.ilc_model.load_weights()
        else:
            is_load = False
        
        optimizer = self._build_optimizer(self.ilc_model)
        scheduler = self._build_scheduler(optimizer)
        
        time_train = 0
        if not is_load:
            self.evaluate_ilc_signal()
            
        train_loader, val_loader, input_norm, target_norm = self.prepare_loaders(arch='ilc')
        
        if not is_load:
            start = time.time()
            learning.train(net=self.ilc_model, 
                        criterion=self.criterion, 
                        optimizer=optimizer, 
                        train_loader=train_loader, 
                        val_loader=val_loader, 
                        grad_clip_val=self.grad_clip_val, 
                        n_epochs=self.epochs, 
                        metric_criterion=self.metric_criterion,
                        scheduler=scheduler)
            elapsed = time.time() - start
            time_train = timedelta(seconds=round(elapsed))
            self.ilc_model.save_weights()
        
        x_val = input_norm.transform(self.container.val_input)
            
        if issubclass(self.base_model, (RTDTNN, LEANN)):
            x_val = input_norm.transform(self.container.val_input_orig)
            x_val = self.features_extractor(x_val, M=self.train_props["M"], P=self.train_props.get("P"))
        
        casc_ilc_eval = utils.CascadeModel(model_1=self.ilc_model, model_2=self.pa_model, normalizer=target_norm)
        
        # if issubclass(self.base_model, RTDTNN):
        #     casc_ilc_eval = utils.CascadeModel(model_1=self.ilc_model, model_2=self.pa_model)
        
        if issubclass(self.base_model, LEANN):
            y_val_ilc = learning.net_inference(net=casc_ilc_eval, x=x_val, model_type="LEANN")
        else:
            y_val_ilc = learning.net_inference(net=casc_ilc_eval, x=x_val)
        ilc_nmse = metrics.compute_nmse(y_val_ilc, self.container.val_output_target)
        ilc_acpr = metrics.calculate_acpr(y_val_ilc, self.acpr_meter)
        print(f"[ILC] NMSE: {ilc_nmse:.2f}, ACPR: {ilc_acpr[0]:.2f} {ilc_acpr[1]:.2f}, count_params: {count_params}")

        self.results["ilc"] = {
            "nmse": ilc_nmse.item(),
            "acpr": ilc_acpr,
            "y_val_ilc": y_val_ilc,
            "time_train": time_train,
            "count_params": count_params,
            "count_macs": count_macs,
            "count_flops": count_flops,
        }
    
    def run_ilc_pleann(self, load_weights: bool = True):
        print("Run ILC")

        self.ilc_model = self.base_model(**self.model_params, model_name="ilc_prune")
        
        count_params = self.ilc_model.count_params()
        count_macs = self.ilc_model.count_macs()
        count_flops = self.ilc_model.count_flops()
        print(f"count_params: {count_params}")
        print(f"count_macs: {count_macs}")
        print(f"count_flops: {count_flops}")
        
        if load_weights:
            is_load = self.ilc_model.load_weights()
        else:
            is_load = False
        
        criterion_reg = metrics.RegLoss(self.ilc_model, 
                                        original_loss=metrics.compute_mse, 
                                        lambda_reg=self.train_props['lambda_reg'])

        optimizer1 = torch.optim.AdamW(self.ilc_model.parameters(), 
                                      lr=self.train_props["lr"])
        scheduler1 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer1, 
                                                               mode='min', 
                                                               factor=0.5, 
                                                               patience=10)

        time_train = 0
        
        if not is_load:
            self.evaluate_ilc_signal()
        
        train_loader, val_loader, input_norm, target_norm = self.prepare_loaders(arch='ilc')

        if not is_load:
            start = time.time()
            learning.train(net=self.ilc_model, 
                        criterion=self.criterion, 
                        optimizer=optimizer1, 
                        train_loader=train_loader, 
                        val_loader=val_loader, 
                        grad_clip_val=self.grad_clip_val, 
                        n_epochs=self.epochs, 
                        metric_criterion=self.metric_criterion,
                        scheduler=scheduler1)
            
            optimizer2 = torch.optim.AdamW(self.ilc_model.parameters(), 
                                          lr=self.train_props["lr2"])
            scheduler2 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer2, 
                                                                   mode='min', 
                                                                   factor=0.5, 
                                                                   patience=10)
            
            learning.train(net=self.ilc_model, 
                criterion=criterion_reg, 
                optimizer=optimizer2, 
                train_loader=train_loader, 
                val_loader=val_loader, 
                n_epochs=self.train_props['epochs2'], 
                metric_criterion=self.metric_criterion,
                scheduler=scheduler2)
            
            prunning.leann_prune(self.ilc_model, 
                                 alpha=self.train_props['alpha'], 
                                 tau=self.train_props['tau'],
                                 verbose=True)
            
            elapsed = time.time() - start
            time_train = timedelta(seconds=round(elapsed))
            self.ilc_model.save_weights()
        
        use_normalize = (input_norm is not None) and (target_norm is not None)
            
        x_val = input_norm.transform(self.container.val_input_orig)
        x_val = self.features_extractor(x_val, M=self.train_props["M"])
        
        if use_normalize:
            casc_ilc_eval = utils.CascadeModel(model_1=self.ilc_model, model_2=self.pa_model, normalizer=target_norm)
        else:
            casc_ilc_eval = utils.CascadeModel(model_1=self.ilc_model, model_2=self.pa_model)

        y_val_ilc = learning.net_inference(net=casc_ilc_eval, x=x_val, model_type="LEANN")
        
        ilc_nmse = metrics.compute_nmse(y_val_ilc, self.container.val_output_target)
        ilc_acpr = metrics.calculate_acpr(y_val_ilc, self.acpr_meter)
        print(f"[ILC] NMSE: {ilc_nmse:.2f}, ACPR: {ilc_acpr[0]:.2f} {ilc_acpr[1]:.2f}, count_params: {count_params}")

        self.results["ilc_pleann"] = {
            "nmse": ilc_nmse.item(),
            "acpr": ilc_acpr,
            "y_val_ilc": y_val_ilc,
            "time_train": time_train,
            "count_params": count_params,
            "count_macs": count_macs,
            "count_flops": count_flops,
        }
    
    def evaluate_with_statistics(self, arch: str, n_runs: int = 5):
        statistics_list = []
        
        architectures_registry = {
            'pa': self.run_pa,
            'dla': self.run_dla,
            'ila': self.run_ila,
            'ilc': self.run_ilc,
            'ilc_pleann': self.run_ilc_pleann
        }
        
        if arch not in architectures_registry:
            raise ValueError(f"Unknown architecture: {arch}")
        
        arch_method = architectures_registry[arch]
        
        for i in range(n_runs):
            print(f"\n=== Run {i+1}/{n_runs} ===")
            torch.manual_seed(i)
            np.random.seed(i)
            
            if arch != "pa" and self.pa_model is None:
                raise AttributeError("pa_model not found in pipeline")

            arch_method(load_weights=False)
            stats = copy.deepcopy(self.results[arch])
            statistics_list.append(stats)
        
        return statistics_list
    
    def run(self):
        if self.pa_model is None:
            self.run_pa()
        self.run_dla()
        self.run_ila()
        self.run_ilc()
    
    def get_results(self):
        return self.results
    
    def get_pa_model(self):
        return self.pa_model
    
    def get_count_params(self) -> int:
        return self.count_params
    
    def reset_u_k_signal(self):
        self.container.reset_ilc()
    
    def _reset_model(self, arch):
        if arch == "pa":
            self.pa_model = None
        elif arch == "dla":
            self.dla_model = None
        elif arch == "ila":
            self.ila_model = None
        elif arch == "ilc":
            self.ilc_model = None
    
    def get_dpd_model(self, arch: str):
        arch = arch.lower()
        assert arch in ('dla', 'ila', 'ilc'), \
                "arch must be specified: 'dla', 'ila' or 'ilc'"
        match arch:
            case "dla":
                return self.dla_model
            case "ila":
                return self.ila_model
            case "ilc":
                return self.ilc_model
            case _:
                raise ValueError(f"Unknown arch '{arch}'")
            


class SparsityPipeline(SimplePipeline):
    def __init__(self, 
                 container: data_loader.DataContainer, 
                 train_props, 
                 base_model,
                 pa_model=None):

        super().__init__(container, train_props, base_model, pa_model)
        

    def ilc_prune_amount(self, amount_range=np.arange(0, 1, 0.1), n_runs: int = 0, pruning_method=prunning.prune.L1Unstructured):
        
        self.evaluate_ilc_signal()
        self.ilc_train_loader, self.ilc_val_loader = data_loader.build_dataloaders(container=self.container, 
                                                                            frame_length=self.frame_length, 
                                                                            batch_size=self.batch_size, 
                                                                            batch_size_eval=self.batch_size_eval, 
                                                                            arch="ilc")
        
        for amount in amount_range:
            
            nmse_list = []
            acpr_list = []
            
            for run_idx in range(n_runs):
                print(f"[Amount {amount:.2f}] Run {run_idx+1}/{n_runs}")

                dpd_model = self.base_model(**self.model_params, model_name="prune")
                optimizer = self._build_optimizer(dpd_model)
                scheduler = self._build_scheduler(optimizer)

                time_train = 0
                start = time.time()

                learning.train(net=dpd_model, 
                            criterion=self.criterion,
                            metric_criterion=self.metric_criterion,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            train_loader=self.ilc_train_loader, 
                            val_loader=self.ilc_val_loader, 
                            grad_clip_val=self.grad_clip_val, 
                            n_epochs=self.epochs)
                    

                y_val_ilc_base, ilc_base_nmse, ilc_base_acpr = self._calculate_ilc_metrics(dpd_model)
                nonzero_params_base, count_params = prunning.count_sparsity_parameters(dpd_model)
                print(f"[ILC] [Base] NMSE: {ilc_base_nmse:.2f}, ACPR: {ilc_base_acpr}, nonzero_params: {nonzero_params_base}")
                
                                
                parameters_to_prune = prunning.get_parameters_to_prune(dpd_model)
                prunning.prune.global_unstructured(parameters_to_prune,
                                          pruning_method=pruning_method,
                                          amount=amount)

                y_val_ilc_prune, ilc_prune_nmse, ilc_prune_acpr = self._calculate_ilc_metrics(dpd_model)
                nonzero_params_prune, _ = prunning.count_sparsity_parameters(dpd_model)
                prunning.log_layerwise_sparsity(dpd_model)
                print(f"[ILC] [Prune] NMSE: {ilc_prune_nmse:.2f}, ACPR: {ilc_prune_acpr}, nonzero_params: {nonzero_params_prune}")

                learning.train(net=dpd_model, 
                            criterion=self.criterion, 
                            optimizer=optimizer, 
                            train_loader=self.ilc_train_loader, 
                            val_loader=self.ilc_val_loader, 
                            grad_clip_val=self.grad_clip_val, 
                            n_epochs=self.epochs, 
                            metric_criterion=self.metric_criterion,
                            scheduler=scheduler)
                
                elapsed = time.time() - start
                time_train = timedelta(seconds=round(elapsed))
                
                prunning.remove_prunned_parameters(parameters_to_prune)
                
                y_val_ilc_finetune, ilc_finetune_nmse, ilc_finetune_acpr = self._calculate_ilc_metrics(dpd_model)
                nonzero_params_finetune, _ = prunning.count_sparsity_parameters(dpd_model)
                prunning.log_layerwise_sparsity(dpd_model)
                print(f"[ILC] [Finetune] NMSE: {ilc_finetune_nmse:.2f}, ACPR: {ilc_finetune_acpr}, nonzero_params: {nonzero_params_finetune}")

                nmse_list.append(ilc_finetune_nmse)
                acpr_list.append(ilc_finetune_acpr)

            
            self.results["ilc"][amount] = {
                "nmse_base": ilc_base_nmse.item(),
                "acpr_base": ilc_base_acpr,
                "y_val_ilc_base": y_val_ilc_base,
                "nonzero_params_base": nonzero_params_base,
                
                "nmse_prune": ilc_prune_nmse.item(),
                "acpr_prune": ilc_prune_acpr,
                "y_val_ilc_prune": y_val_ilc_prune,
                "nonzero_params_prune": nonzero_params_prune,
                
                "nmse_finetune": np.mean(nmse_list),
                "acpr_finetune": np.mean(acpr_list),
                "y_val_ilc_finetune": y_val_ilc_finetune,
                "nonzero_params_finetune": nonzero_params_finetune,
                
                "time_train": time_train,
                "count_params": count_params
            }
    
    
    def _calculate_ilc_metrics(self, dpd_model):
        casc_ilc_eval = utils.CascadeModel(model_1=dpd_model, model_2=self.pa_model)
        y_signal = learning.net_inference(net=casc_ilc_eval, x=self.container.val_input)
        nmse = metrics.compute_nmse(y_signal, self.container.val_output_target)
        acpr = metrics.calculate_acpr(y_signal, self.acpr_meter)
        return y_signal, nmse, acpr


class SnrPipeline:
    def __init__(self,
                 container: data_loader.DataContainer, 
                 train_props: Dict, 
                 base_model: Type[torch.nn.Module],
                 snr_params: Dict,
                 pa_model: Type[torch.nn.Module] = None):
        
        assert issubclass(base_model, (GMP, LSTM, GRU, TCN)), "unsupported base_model, must be (GMP, LSTM, GRU, TCN)"
        
        self.snr_range = snr_params["snr_range"]
        self.num_realizations = snr_params["num_realizations"]
        
        self.fs = container.input_signal_fs
        self.bw_main_ch = container.bw_main_ch
        self.gain = container.gain
        
        self.base_model = base_model
        self.container = container
        
        self.pa_model = pa_model
        self.dla_model = None
        self.ila_model = None
        self.ilc_model = None

        self.model_params = {}
        self.train_props = train_props
        self.lr = self.train_props["lr"]
        self.epochs = self.train_props["epochs"]
        self.acpr_meter = self.train_props["acpr_meter"]
        self.features_extractor = params.EXTRACTORS_REGISTRY[base_model]
        self.dataloader = params.DATALOADERS_REGISTRY[base_model]
        
        self.results = {
            "dla": {},
            "ila": {},
            "ilc": {},
        }
        
        self.noisy_pa = {}
        
        self.criterion = metrics.compute_mse
        self.metric_criterion = metrics.compute_nmse
        
        self.frame_length = FRAME_LENGTH
        self.batch_size = BATCH_SIZE
        self.batch_size_eval = BATCH_SIZE_EVAL
        self.grad_clip_val = GRAD_CLIP_VAL
        self.u_k_lr = U_K_LR
        self.u_k_epochs = U_K_EPOCHS
        
        self._prepare_model_params()
        self.loader_props = params.DATALOADERS_PROPS[self.base_model](self.features_extractor, self.train_props)
        
        self._create_pa_noise_models()
    
    def _prepare_model_params(self):
        try:
            builder = params.MODEL_REGISTRY[self.base_model]
        except KeyError:
            raise ValueError(f"Unsupported base_model: {self.base_model}")

        self.model_params = builder(self.train_props)
    
    def prepare_loaders(self, arch):
        train_loader, val_loader, input_norm, target_norm = self.dataloader(self.container,
                                                                            arch=arch,
                                                                            **self.loader_props)
        return train_loader, val_loader, input_norm, target_norm
    
    def _create_pa_noise_models(self):
        for snr in self.snr_range:
            self.noisy_pa[snr] = self._create_pa_noise_cascade(snr)
    
    def _build_optimizer(self, model):
        return torch.optim.AdamW(model.parameters(), lr=self.lr)
    
    def _build_scheduler(self, optimizer):
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    def _create_pa_noise_cascade(self, snr):
        assert self.pa_model is None, "pa_model must be added to the pipeline"
        noise_gen_model = utils.NoiseModel(snr=snr, fs=self.fs, bw=self.bw_main_ch)
        casc_pa_noise = utils.CascadeModel(model_1=self.pa_model, model_2=noise_gen_model)
        return casc_pa_noise
    
    def run_dla_noise(self):
        print("Run DLA")
        for snr in self.snr_range:
            print(f"SNR: {snr}")
            dpd_model = self.base_model(**self.model_params)
            count_params = self.dpd_model.count_params()
            print(f"count_params: {count_params}")
            
            casc_pa_noise = self.noisy_pa[snr]
            casc_dla = utils.CascadeModel(model_1=dpd_model, 
                                            model_2=casc_pa_noise)
            
            optimizer = self._build_optimizer(casc_dla)
            scheduler = self._build_scheduler(optimizer)
            
            train_loader, val_loader, input_norm, target_norm = self.prepare_loaders(arch='dla')

            time_train = 0
            start = time.time()
            learning.train(net=casc_dla, 
                        criterion=self.criterion, 
                        optimizer=optimizer, 
                        train_loader=train_loader, 
                        val_loader=val_loader, 
                        grad_clip_val=self.grad_clip_val, 
                        n_epochs=self.epochs, 
                        metric_criterion=self.metric_criterion,
                        scheduler=scheduler)
            elapsed = time.time() - start
            time_train = timedelta(seconds=round(elapsed))
            
            use_normalize = (input_norm is not None) and (target_norm is not None)

            if use_normalize:
                x_val = input_norm.transform(self.container.val_input)
            else:
                x_val = self.container.val_input
            
            if use_normalize:
                casc_dla_eval = utils.CascadeModel(model_1=dpd_model, model_2=casc_pa_noise, normalizer=target_norm)
            else:
                casc_dla_eval = utils.CascadeModel(model_1=dpd_model, model_2=casc_pa_noise)
            
            nmse, acpr_left, acpr_right = metrics.noise_realizations(self.num_realizations, 
                                                            model=casc_dla_eval, 
                                                            x=x_val, 
                                                            y_target=self.container.val_output_target, 
                                                            acpr_meter=self.acpr_meter)
            
            print(f"[DLA] NMSE: {nmse:.2f}, ACPR: {acpr_left:.2f} {acpr_right:.2f}, count_params: {count_params}")

            self.results["dla"] = {
                "nmse": nmse.item(),
                "acpr": [acpr_left, acpr_right],
                "time_train": time_train,
                "count_params": count_params
            }

        
    def run_ila_noise(self):
        print("Run ILA")
        for snr in self.snr_range:
            print(f"SNR: {snr}")
            dpd_model = self.base_model(**self.model_params)
            count_params = self.dpd_model.count_params()
            print(f"count_params: {count_params}")
            
            casc_pa_noise = self.noisy_pa[snr]
            casc_ila_train = utils.CascadeModel(model_1=casc_pa_noise, model_2=dpd_model, gain=self.gain, cascade_type="ila")
            
            optimizer = self._build_optimizer(casc_ila_train)
            scheduler = self._build_scheduler(optimizer)
            
            train_loader, val_loader, input_norm, target_norm = self.prepare_loaders(arch='ila')
            
            time_train = 0
            start = time.time()
            learning.train(net=casc_ila_train, 
                        criterion=self.criterion, 
                        optimizer=optimizer, 
                        train_loader=train_loader, 
                        val_loader=val_loader, 
                        grad_clip_val=self.grad_clip_val, 
                        n_epochs=self.epochs, 
                        metric_criterion=self.metric_criterion,
                        scheduler=scheduler)
            elapsed = time.time() - start
            time_train = timedelta(seconds=round(elapsed))

            use_normalize = (input_norm is not None) and (target_norm is not None)

            if use_normalize:
                x_val = input_norm.transform(self.container.val_input)
            else:
                x_val = self.container.val_input
                        
            if use_normalize:
                casc_ila_eval = utils.CascadeModel(model_1=dpd_model, model_2=casc_pa_noise, normalizer=target_norm)
            else:
                casc_ila_eval = utils.CascadeModel(model_1=dpd_model, model_2=casc_pa_noise)
        
            nmse, acpr_left, acpr_right = metrics.noise_realizations(self.num_realizations, 
                                                            model=casc_ila_eval, 
                                                            x=x_val, 
                                                            y_target=self.container.val_output_target, 
                                                            acpr_meter=self.acpr_meter)
            print(f"[ILA] NMSE: {nmse:.2f}, ACPR: {acpr_left:.2f} {acpr_right:.2f}, count_params: {count_params}")
            
            self.results["ila"] = {
            "nmse": nmse.item(),
            "acpr": [acpr_left, acpr_right],
            "time_train": time_train,
            "count_params": count_params
        }
    
    def run_ilc_noise(self):
        print("Run ILC")
        
        for snr in self.snr_range:
            print(f"SNR: {snr}")
            
            casc_pa_noise = self.noisy_pa[snr]
            
            time_train = 0
            start = time.time()
            
            self.container.ilc_train_output = learning.ilc_signal(self.container.train_input, 
                                                                  self.container.train_output_target, 
                                                                  casc_pa_noise, 
                                                                  epochs=self.u_k_epochs, 
                                                                  learning_rate=self.u_k_lr)
            self.container.ilc_val_output = learning.ilc_signal(self.container.val_input, 
                                                                self.container.val_output_target, 
                                                                casc_pa_noise, 
                                                                epochs=self.u_k_epochs, 
                                                                learning_rate=self.u_k_lr)
            elapsed = time.time() - start
            time_train = timedelta(seconds=round(elapsed))
            
            nmse_uk, acpr_left_uk, acpr_right_uk = metrics.noise_realizations(self.num_realizations, 
                                                                    model=casc_pa_noise, 
                                                                    x=self.container.ilc_train_output, 
                                                                    y_target=self.container.train_output_target, 
                                                                    acpr_meter=self.acpr_meter)
            
            self.results["uk"] = {
                "nmse": nmse_uk.item(),
                "acpr": [acpr_left_uk, acpr_right_uk],
                "time_train": time_train
            }

            dpd_model = self.base_model(**self.model_params)
            count_params = self.dpd_model.count_params()
            print(f"count_params: {count_params}")
            
            optimizer = self._build_optimizer(dpd_model)
            scheduler = self._build_scheduler(optimizer)
            
            train_loader, val_loader, input_norm, target_norm = self.prepare_loaders(arch='ilc')
            
            time_train = 0
            start = time.time()
            learning.train(net=dpd_model, 
                        criterion=self.criterion, 
                        optimizer=optimizer, 
                        train_loader=train_loader, 
                        val_loader=val_loader, 
                        grad_clip_val=self.grad_clip_val, 
                        n_epochs=self.epochs, 
                        metric_criterion=self.metric_criterion,
                        scheduler=scheduler)
            elapsed = time.time() - start
            time_train = timedelta(seconds=round(elapsed))
            
            use_normalize = (input_norm is not None) and (target_norm is not None)

            if use_normalize:
                x_val = input_norm.transform(self.container.val_input)
            else:
                x_val = self.container.val_input
            
            if use_normalize:
                casc_ilc_eval = utils.CascadeModel(model_1=dpd_model, model_2=casc_pa_noise, normalizer=target_norm)
            else:
                casc_ilc_eval = utils.CascadeModel(model_1=dpd_model, model_2=casc_pa_noise)
        
            nmse, acpr_left, acpr_right = metrics.noise_realizations(self.num_realizations, 
                                                            model=casc_ilc_eval, 
                                                            x=x_val, 
                                                            y_target=self.container.val_output_target, 
                                                            acpr_meter=self.acpr_meter)
            
            print(f"[ILC] NMSE: {nmse:.2f}, ACPR: {acpr_left:.2f} {acpr_right:.2f}, count_params: {count_params}")

            self.results["ilc"] = {
                "nmse": nmse.item(),
                "acpr": [acpr_left, acpr_right],
                "time_train": time_train,
                "count_params": count_params
            }

    def run(self):
        self.run_dla_noise()
        self.run_ila_noise()
        self.run_ilc_noise()
    
    def get_results(self):
        return self.results