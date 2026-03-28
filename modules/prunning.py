import time
from datetime import timedelta
import numpy as np
import torch
import torch.nn.utils.prune as prune
from modules import data_loader, metrics, learning, utils, pipelines


def count_sparsity_parameters(model, only_weights=True) -> tuple[int, int]:
    """
    Returns:
        tuple[int, int]: A tuple containing:
            - total_nonzero (int): Total number of nonzero parameters in the model.
            - total_params (int): Total number of parameters in the model.
    """
    
    total_nonzero = 0
    total_params = 0
    
    for module_name, module in model.named_modules():
        layer_nonzero = 0
        layer_total = 0
        layer_has_params = False

        for param_name, param in module.named_parameters(recurse=False):
            if only_weights and "weight" not in param_name:
                continue

            if param is None:
                continue

            data = param.detach()

            nonzero = (data.abs() > 0).sum().item()
            total = data.numel()

            layer_nonzero += nonzero
            layer_total += total
            layer_has_params = True            

        total_nonzero += layer_nonzero
        total_params += layer_total
    
    return total_nonzero, total_params


def log_layerwise_sparsity(model, only_weights: bool = True, eps: float = 0.0):
    """
    Prints layer-wise sparsity statistics for a pruned model.

    Args:
        model (nn.Module): pruned model (after prune.remove)
        only_weights (bool): log only parameters containing 'weight'
        eps (float): threshold to consider a value as zero
    """
    total_nonzero = 0
    total_params = model.count_params()

    print("=" * 72)
    print("Layer-wise sparsity report")
    print("=" * 72)

    for module_name, module in model.named_modules():
        layer_nonzero = 0
        layer_total = 0
        layer_has_params = False

        for param_name, param in module.named_parameters(recurse=False):
            if only_weights and "weight" not in param_name:
                continue

            if param is None:
                continue

            data = param.detach()

            nonzero = (data.abs() > 0).sum().item()
            total = data.numel()

            layer_nonzero += nonzero
            layer_total += total
            layer_has_params = True

            print(
                f"{module_name or '<root>'}.{param_name:20s} | "
                f"nonzero: {nonzero:8d} / {total:8d} "
                f"({nonzero / total:6.2%})"
            )

        if layer_has_params:
            print(
                f"{module_name or '<root>':24s} | "
                f"LAYER TOTAL: {layer_nonzero:8d} / {layer_total:8d} "
                f"({layer_nonzero / layer_total:6.2%})"
            )
            print("-" * 72)

        total_nonzero += layer_nonzero

    print("=" * 72)
    print(
        f"MODEL TOTAL: nonzero {total_nonzero} / {total_params} "
        f"({total_nonzero / total_params:6.2%})"
    )
    print("=" * 72)


def remove_prunned_parameters(parameters_to_prune):
    for module, name in parameters_to_prune:
        prune.remove(module, name)


def get_parameters_to_prune(model):
    """
    Collect all parameters suitable for unstructured pruning in RTDTNN.

    Returns:
        List of (module, name) tuples for pruning.
    """
    parameters_to_prune = []

    for module in model.modules():
        # Linear layers
        if isinstance(module, torch.nn.Linear):
            if hasattr(module, "weight") and module.weight is not None:
                parameters_to_prune.append((module, "weight"))

        # LSTM / GRU layers
        elif isinstance(module, (torch.nn.LSTM, torch.nn.GRU)):
            for name, param in module.named_parameters(recurse=False):
                if "weight" in name and param is not None:
                    parameters_to_prune.append((module, name))

        # MultiheadAttention
        elif isinstance(module, torch.nn.MultiheadAttention):
            # in_proj_weight (combined Q,K,V)
            if hasattr(module, "in_proj_weight") and module.in_proj_weight is not None:
                parameters_to_prune.append((module, "in_proj_weight"))
            # out_proj weight
            if hasattr(module, "out_proj") and hasattr(module.out_proj, "weight"):
                parameters_to_prune.append((module.out_proj, "weight"))

        # Conv1d layers
        elif isinstance(module, torch.nn.Conv1d):
            if hasattr(module, "weight") and module.weight is not None:
                parameters_to_prune.append((module, "weight"))

        # Ignore LayerNorm / bias
        elif isinstance(module, torch.nn.LayerNorm):
            continue

    return parameters_to_prune


class GlobalL1ThresholdUnstructured(prune.BasePruningMethod):
    PRUNING_TYPE = "unstructured"

    def __init__(self, threshold: float):
        if threshold < 0:
            raise ValueError("threshold must be non-negative")
        self.threshold = threshold

    def compute_mask(self, t, default_mask):
        """
        t – flattened importance_scores или weight_orig
        default_mask – предыдущая глобальная маска
        """
        new_mask = (t.abs() >= self.threshold).to(default_mask.dtype)
        return default_mask * new_mask


class SparsityPipeline(pipelines.SimplePipeline):
    def __init__(self, 
                 container: data_loader.DataContainer, 
                 train_props, 
                 base_model,
                 pa_model=None):

        super().__init__(container, train_props, base_model, pa_model)
        

    def ilc_prune_amount(self, amount_range=np.arange(0, 1, 0.1), n_runs: int = 0, pruning_method=prune.L1Unstructured):
        
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
                nonzero_params_base, count_params = count_sparsity_parameters(dpd_model)
                print(f"[ILC] [Base] NMSE: {ilc_base_nmse:.2f}, ACPR: {ilc_base_acpr}, nonzero_params: {nonzero_params_base}")
                
                                
                parameters_to_prune = get_parameters_to_prune(dpd_model)
                prune.global_unstructured(parameters_to_prune,
                                          pruning_method=pruning_method,
                                          amount=amount)

                y_val_ilc_prune, ilc_prune_nmse, ilc_prune_acpr = self._calculate_ilc_metrics(dpd_model)
                nonzero_params_prune, _ = count_sparsity_parameters(dpd_model)
                log_layerwise_sparsity(dpd_model)
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
                
                remove_prunned_parameters(parameters_to_prune)
                
                y_val_ilc_finetune, ilc_finetune_nmse, ilc_finetune_acpr = self._calculate_ilc_metrics(dpd_model)
                nonzero_params_finetune, _ = count_sparsity_parameters(dpd_model)
                log_layerwise_sparsity(dpd_model)
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
        
    
