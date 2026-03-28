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


def leann_prune(model,
                alpha=1e-3,
                tau=1e-3,
                verbose=True):
    """
    Pruning for LEANN model

    alpha — threshold for LEA weights
    tau   — threshold for FIR weights
    """

    with torch.no_grad():

        stats = {}

        # =========================
        # LEA-1 pruning
        # =========================

        w1 = model.lea_1_weight

        mask1 = torch.abs(w1) < alpha

        pruned_1 = mask1.sum().item()
        total_1 = w1.numel()

        w1[mask1] = 0.0

        stats["lea_1"] = (pruned_1, total_1)

        # =========================
        # LEA-K pruning
        # =========================

        wk = model.lea_k_weight

        maskk = torch.abs(wk) < alpha

        pruned_k = maskk.sum().item()
        total_k = wk.numel()

        wk[maskk] = 0.0

        stats["lea_k"] = (pruned_k, total_k)

        # =========================
        # FIR pruning
        # =========================

        wf = model.fir.weight

        maskf = torch.abs(wf) < tau

        pruned_f = maskf.sum().item()
        total_f = wf.numel()

        wf[maskf] = 0.0

        stats["fir"] = (pruned_f, total_f)

        # =========================
        # Reporting
        # =========================

        if verbose:

            print("\n=== LEANN PRUNING REPORT ===")

            for k, (p, t) in stats.items():

                percent = 100 * p / t

                print(
                    f"{k}: "
                    f"{p}/{t} pruned "
                    f"({percent:.2f}%)"
                )

        return stats