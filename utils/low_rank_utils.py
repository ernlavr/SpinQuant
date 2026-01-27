import random
from typing import Tuple
import os
import json

import functools
import math
import torch
import time

import torch
from tqdm import tqdm

from utils import eval_utils, monkeypatch, quant_utils, utils
from utils.hadamard_utils import (
    apply_exact_had_to_linear,
    is_pow2,
    random_hadamard_matrix,
)
from utils.utils import HadamardTransform
import matplotlib.pyplot as plt
from modules.linears import SVDLinear, LowRankLinear
OUTPUT_DIR = "output_dir/low_rank_analysis/singular_values"

def plot_series(data_series, title, xlabel, ylabel, output_name):
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    plt.figure(figsize=(10, 6))
    for label, data in data_series.items():
        plt.plot(data, label=label)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, output_name))
    plt.close()
    


def perform_svd_decomp(matrix: torch.Tensor, rank: int = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Perform SVD decomposition on the given matrix and return the low-rank approximation.

    Args:
        matrix (torch.Tensor): The input matrix to decompose.
        rank (int): The target rank for the low-rank approximation.

    Returns:
        U (torch.Tensor): Left singular vectors.
        S (torch.Tensor): Singular values.
        Vh (torch.Tensor): Right
    """
    dtype = matrix.dtype
    matrix_ = matrix.to(device="cuda", dtype=torch.float64)
    U, S, Vh = torch.linalg.svd(matrix_, full_matrices=False)
    if rank is not None:
        U = U[:, :rank]
        S = S[:rank]
        Vh = Vh[:rank, :]
        
    U = U.to(device="cpu", dtype=torch.float64)
    S = S.to(device="cpu", dtype=torch.float64)
    Vh = Vh.to(device="cpu", dtype=torch.float64)
    return U, S, Vh


def decompose_embeddings(model) -> None:
    # Rotate the embeddings.
    for W in [model.model.embed_tokens]:
        U, S, Vh = perform_svd_decomp(W.weight.data)
        plot_series({"embeddings": S.detach().cpu().numpy()}, "Singular Values for Embeddings", "Index", "Singular Value", "singular_values_embeddings.png")

def decompose_attention(layer, layer_idx) -> None:
    to_plot = {}
    # Rotate the WQ, WK and WV matrices of the self-attention layer.
    for W, name in [(layer.self_attn.q_proj, "q_proj"), (layer.self_attn.k_proj, "k_proj"), (layer.self_attn.v_proj, "v_proj")]:
        U, S, Vh = perform_svd_decomp(W.weight.data)
        to_plot[name] = S.detach().cpu().numpy()
    
    W = layer.self_attn.o_proj.weight.data
    U, S, Vh = perform_svd_decomp(W)
    to_plot["o_proj"] = S.detach().cpu().numpy()
    
    
    plot_series(to_plot, f"Layer {layer_idx}: Singular Values for Attention Output", "Index", "Singular Value", f"attention_layer_{layer_idx}.png")
    

def decompose_attention_inputs(layer, layer_idx) -> None:
    # Rotate the WQ, WK and WV matrices of the self-attention layer.
    for W, name in [(layer.self_attn.q_proj, "q_proj"), (layer.self_attn.k_proj, "k_proj"), (layer.self_attn.v_proj, "v_proj")]:
        U, S, Vh = perform_svd_decomp(W.weight.data)
        plot_series({name: S.detach().cpu().numpy()}, f"Layer {layer_idx}: Singular Values for Embeddings", "Index", "Singular Value", f"layer_{layer_idx}_{name}.png")


def decompose_attention_output(layer, layer_idx) -> None:
    # Rotate output matrix of the self-attention layer.
    W = layer.self_attn.o_proj.weight.data
    U, S, Vh = perform_svd_decomp(W)
    plot_series({"o_proj": S.detach().cpu().numpy()}, f"Layer {layer_idx}: Singular Values for Attention Output", "Index", "Singular Value", f"o_proj_layer_{layer_idx}.png")

def decompose_mlp(layer, layer_idx) -> None:
    # Rotate the MLP weights.
    mlp_weights = [
        (layer.mlp.up_proj, "up_proj"),
        (layer.mlp.gate_proj, "gate_proj"),
        (layer.mlp.down_proj, "down_proj"),
    ]
    to_plot = {}
    for W, name in mlp_weights:
        U, S, Vh = perform_svd_decomp(W.weight.data)
        to_plot[name] = S.detach().cpu().numpy()
    plot_series(to_plot, f"Layer {layer_idx}: Singular Values for MLP", "Index", "Singular Value", f"mlp_layer_{layer_idx}.png")

def decompose_mlp_input(layer, layer_idx) -> None:
    # Rotate the MLP input weights.
    mlp_inputs = [(layer.mlp.up_proj, "up_proj"), (layer.mlp.gate_proj, "gate_proj")]
    for W, name in mlp_inputs:
        U, S, Vh = perform_svd_decomp(W.weight.data)
        plot_series({name: S.detach().cpu().numpy()}, f"Layer {layer_idx}: Singular Values for MLP Input", "Index", "Singular Value", f"{name}_layer_{layer_idx}.png")

#def rotate_mlp_output(layer, R1, args):
def decompose_mlp_output(layer, layer_idx) -> None:
    # Rotate the MLP output weights and bias.
    W = layer.mlp.down_proj.weight.data
    U, S, Vh = perform_svd_decomp(W)
    plot_series({"down_proj": S.detach().cpu().numpy()}, f"Layer {layer_idx}: Singular Values for MLP Output", "Index", "Singular Value", f"down_proj_layer_{layer_idx}.png")

def decompose_head(model) -> None:
    # Rotate the head.
    W = model.lm_head.weight.data
    U, S, Vh = perform_svd_decomp(W)
    plot_series({"lm_head": S.detach().cpu().numpy()}, "Singular Values for LM Head", "Index", "Singular Value", f"singular_values_lm_head.png")

def decompose_ov_proj(layer, head_num, head_dim, R2=None):
    v_proj = layer.self_attn.v_proj
    o_proj = layer.self_attn.o_proj

    apply_exact_had_to_linear(v_proj, had_dim=head_dim, output=True, R2=R2)
    apply_exact_had_to_linear(o_proj, had_dim=head_dim, output=False, R2=R2)


@torch.no_grad()
def calib_sensitivity_ppl(model, calib_loader, args, use_cache=True):
    model_id = model.config._name_or_path
    cache_dir = '/eos/home-e/elavrino/git/SpinQuant/output_dir/low_rank_analysis/cache'
    cache_file = f"{cache_dir}/{model_id.replace('/','_')}_calib_sensitivity_ppl.pt"
    os.makedirs(cache_dir, exist_ok=True)
    if os.path.exists(cache_file) and use_cache:
        sensitivity_dict = torch.load(cache_file, map_location="cpu")
        return sensitivity_dict
    model.eval()

    full_name_dict = {module: name for name, module in model.named_modules()}
    linear_info = {}
    modules = [model]
    while len(modules) > 0:
        submodule = modules.pop()
        for name, raw_linear in submodule.named_children():
            if isinstance(raw_linear, torch.nn.Linear):
                full_name = full_name_dict[raw_linear]
                linear_info[raw_linear] = {
                    "father": submodule,
                    "name": name,
                    "full_name": full_name,
                }
            else:
                modules.append(raw_linear)

    sensitivity_dict = {}
    if False: # args.compress_kv_cache
        param_ratio_candidates = [0.1 * i for i in range(1, 20)]
    else:
        param_ratio_candidates = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    # compute baseline ppl
    # ppl, avg_time_per_token = eval_utils.evaluator(model, calib_loader, utils.DEV, args)
    sensitivity_dict["baseline"] = ppl    
    
    # input_ids = torch.cat([_["input_ids"] for _ in calib_loader], 0)
    # print(f"input_ids.shape={input_ids.shape}")
    pbar = tqdm(total=len(linear_info) * len(param_ratio_candidates))
    for raw_linear, info in linear_info.items():
        sensitivity_dict[info["full_name"]] = {}
        for param_ratio in param_ratio_candidates:
            torch.cuda.empty_cache()
            svd_linear = SVDLinear.from_linear(
                raw_linear,
                param_ratio=param_ratio,
                act_aware=False,
            )
            setattr(info["father"], info["name"], svd_linear)
            ppl, avg_time_per_token = eval_utils.evaluator(model, calib_loader, utils.DEV, args)            
            sensitivity_dict[info["full_name"]][param_ratio] = ppl
            print(f"Processed Layer: {info['full_name']}; Param Ratio: {param_ratio}; PPL: {ppl}")
            print(f"Saving sensitivity results to: {cache_file}")
            torch.save(sensitivity_dict, cache_file)
            pbar.update(1)
            
        setattr(info["father"], info["name"], raw_linear)
        
    print(f"Saved sensitivity results to {cache_file}")
    torch.save(sensitivity_dict, cache_file)
    
    return sensitivity_dict

def binary_search_truncation_rank(model, sensitivity_dict, calib_loader, args):
    """
    Binary search over truncation ranks to find optimal compression of model layers.
    
    Compresses either weight parameters or KV cache based on args.compress_kv_cache.
    Uses binary search to find compression ratios that satisfy either a perplexity
    target (ppl_target) or a parameter ratio target (ratio_target).
    
    Args:
        model: PyTorch model to compress
        sensitivity_dict: Dictionary mapping layer names to sensitivity metrics
        calib_loader: DataLoader for calibration samples
        args: Configuration object with compression parameters
        
    From: https://github.dev/hahnyuan/ASVD4LLM/blob/main/binary_search.py
    """
    # Build dictionaries for efficient module lookup    
    module_dict = {name: module for name, module in model.named_modules()}
    full_name_dict = {module: name for name, module in model.named_modules()}
    linear_info = {}
    sensitivity_dict.pop("baseline", None)
    
    # DEBUG: Set args
    args.compress_kv_cache = False
    args.ppl_target = 0
    args.param_ratio_target = 0.5
    
    # Recursively find all Linear modules in the model
    modules = [model]
    while len(modules) > 0:
        submodule = modules.pop()
        for name, raw_linear in submodule.named_children():
            if isinstance(raw_linear, torch.nn.Linear):
                full_name = full_name_dict[raw_linear]

                linear_info[raw_linear] = {
                    "father": submodule,
                    "name": name,
                    "full_name": full_name,
                }
            else:
                modules.append(raw_linear)

    # Set compression target and filtering based on mode
    if args.compress_kv_cache:
        ratio_target = args.kv_cache_ratio_target
        # Filter sensitivity dict to only KV cache projections
        sensitivity_dict = {
            k: v for k, v in sensitivity_dict.items() 
            if "k_proj" in k or "v_proj" in k
        }
        assert args.ppl_target < 0, "ppl_target is not supported when compressing kv_cache"
        default_param_ratio = 2
    else:
        ratio_target = args.param_ratio_target
        default_param_ratio = 1

    print(
        f"=== {'compress kv_cache' if args.compress_kv_cache else 'compress weight'} target: ppl={args.ppl_target}, ratio_target={ratio_target} ==="
    )

    # Build sorted list of (layer_name, param_ratio, ppl) tuples
    sensitivity_list = []
    for layername, v in sensitivity_dict.items():
        for param_ratio, ppl in v.items():
            if not args.compress_kv_cache and param_ratio >= 1:
                # we need to compress the weights, so parameter ratio should be less than 1
                continue
            sensitivity_list.append((layername, param_ratio, ppl))
    
    # Sort by perplexity (descending) - higher PPL = more sensitive layers
    sorted_sensitive_list = sorted(sensitivity_list, key=lambda x: -x[2])

    # Validate that at least one target is specified
    assert args.ppl_target > 0 or ratio_target > 0
    
    # binary search
    high = len(sorted_sensitive_list) - 1
    low = 0
    while low < high:
        mid = (low + high) // 2
        
        # Initialize all layers with default ratio, then apply compression to sensitive ones
        layers_min_ratio = {layername: default_param_ratio for layername in sensitivity_dict.keys()}
        for layername, param_ratio, ppl in sorted_sensitive_list[mid:]:
            layers_min_ratio[layername] = min(layers_min_ratio[layername], param_ratio)
        
        # Compute total and compressed parameters
        tot_params = 0
        compress_params = 0
        if args.ppl_target > 0:
            # Search based on perplexity target
            assert not args.compress_kv_cache, "ppl_target is not supported when compressing kv_cache now"
            
            # Apply SVD decomposition with current ratios
            for layername, param_ratio in layers_min_ratio.items():
                raw_linear = module_dict[layername]
                info = linear_info[raw_linear]
                svd_linear = SVDLinear.from_linear(
                    raw_linear,
                    param_ratio=param_ratio,
                    alpha=args.alpha,
                    act_aware=args.act_aware,
                    sigma_fuse=args.sigma_fuse,
                    rank_align=args.rank_align,
                )
                setattr(info["father"], info["name"], svd_linear)
                tot_params += raw_linear.weight.numel()
                compress_params += raw_linear.weight.numel() * param_ratio
            
            # Eval perplexity
            ppl = eval_utils.evaluator(model, calib_loader, utils.DEV, args)
            param_ratio = compress_params / tot_params
            msg = f"low={low} mid={mid}, high={high}, ppl={ppl}, param_ratio={param_ratio}"
            print(msg)
            
            # Adjust search range based on perplexity
            if ppl < args.ppl_target:
                high = mid
            else:
                low = mid + 1
        else:
            # Search based on parameter ratio target
            for layername, param_ratio in layers_min_ratio.items():
                raw_linear = module_dict[layername]
                tot_params += raw_linear.weight.numel()
                compress_params += raw_linear.weight.numel() * param_ratio
            now_ratio = compress_params / tot_params
            
            if args.compress_kv_cache:
                # because param ratio is the params for ALinear+BLienar, so the rank ratio is param ratio/2
                now_ratio /= 2
                
            msg = f"low={low} mid={mid}, high={high}, now_ratio={now_ratio}, params=({compress_params}/{tot_params})"
            print(msg)
            
            # Adjust search range based on parameter ratio
            if now_ratio > ratio_target:
                high = mid
            else:
                low = mid + 1

    print(f"=== Searching done, decomposing layers... ===")
    
    # Final decomposition with optimal ratios
    layers_min_ratio = {layername: default_param_ratio for layername in sensitivity_dict.keys()}
    for layername, param_ratio, ppl in sorted_sensitive_list[mid:]:
        if layers_min_ratio[layername] is None:
            layers_min_ratio[layername] = param_ratio
        else:
            layers_min_ratio[layername] = min(layers_min_ratio[layername], param_ratio)
    
    # Apply SVD decomposition to all layers
    st = time.time()
    for layername, param_ratio in tqdm(layers_min_ratio.items()):
        # set ratio
        raw_linear = module_dict[layername]
        info = linear_info[raw_linear]
        
        # Skip decomposition if using default ratio (no compression)
        if param_ratio == default_param_ratio:
            svd_linear = raw_linear
        else:
            svd_linear = SVDLinear.from_linear(
                raw_linear,
                param_ratio=param_ratio,
                alpha=args.alpha,
                act_aware=args.act_aware,
                sigma_fuse=args.sigma_fuse,
                rank_align=args.rank_align,
            )
            raw_linear.to("cpu")
        
        # Replace original linear layer with SVD version
        setattr(info["father"], info["name"], svd_linear)

    ed = time.time()
    print(f"decompose time: {ed-st}")

@torch.inference_mode()
def decompose_model(model, args):
    selective_had_layers = None
    selective_had_path = getattr(args, 'selective_had_layers_path', None)

    
    config = model.config
    num_heads = config.num_attention_heads
    model_dim = config.hidden_size
    head_dim = model_dim // num_heads

    decompose_embeddings(model)
    decompose_head(model)
    utils.cleanup_memory()
    layers = [layer for layer in model.model.layers]
    for idx, layer in enumerate(tqdm.tqdm(layers, unit="layer", desc="Decomposing")):
        decompose_attention(layers[idx], idx)
        decompose_mlp(layers[idx], idx)
        
        #decompose_mlp_output(layers[idx], R1, args)
        # decompose_mlp_output(layers[idx], idx)
        # decompose_ov_proj(layers[idx], num_heads, head_dim)