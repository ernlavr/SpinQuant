import torch
from typing import Tuple
import os
import json

import functools
import math

import torch
import tqdm

from utils import monkeypatch, quant_utils, utils
from utils.hadamard_utils import (
    apply_exact_had_to_linear,
    is_pow2,
    random_hadamard_matrix,
)
from utils.utils import HadamardTransform
import matplotlib.pyplot as plt
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

class QKRotationWrapper(torch.nn.Module):
    def __init__(self, func, config, *args, **kwargs):
        super().__init__()
        self.config = config
        num_heads = config.num_attention_heads
        model_dim = config.hidden_size
        head_dim = model_dim // num_heads
        assert is_pow2(
            head_dim
        ), f"Only power of 2 head_dim is supported for K-cache Quantization!"
        self.func = func
        self.k_quantizer = quant_utils.ActQuantizer()
        self.k_bits = 16
        # Pop the new argument to control rotation, defaulting to False
        self.apply_rotation = kwargs.pop("apply_rotation", False)
        if kwargs is not None:
            assert kwargs["k_groupsize"] in [
                -1,
                head_dim,
            ], f"Only token-wise/{head_dim}g quantization is supported for K-cache"
            self.k_bits = kwargs["k_bits"]
            self.k_groupsize = kwargs["k_groupsize"]
            self.k_sym = kwargs["k_sym"]
            self.k_clip_ratio = kwargs["k_clip_ratio"]
            self.k_quantizer.configure(
                bits=self.k_bits,
                groupsize=-1,  # we put -1 to be toke-wise quantization and handle head-wise quantization by ourself
                sym=self.k_sym,
                clip_ratio=self.k_clip_ratio,
            )

    def forward(self, *args, **kwargs):
        q, k = self.func(*args, **kwargs)

        # If no quantization or rotation is needed, return immediately
        if self.k_bits >= 16 and not self.apply_rotation:
            return q, k
        
        dtype = q.dtype
        
        # R3 rotation
        if self.apply_rotation:
            q = (HadamardTransform.apply(q.float()) / math.sqrt(q.shape[-1])).to(dtype)
            k = (HadamardTransform.apply(k.float()) / math.sqrt(k.shape[-1])).to(dtype)
            
        (bsz, num_heads, seq_len, head_dim) = k.shape

        if self.k_bits < 16:
            if self.k_groupsize == -1:  # token-wise quantization
                token_wise_k = k.transpose(1, 2).reshape(-1, num_heads * head_dim)
                self.k_quantizer.find_params(token_wise_k)
                k = (
                    self.k_quantizer(token_wise_k)
                    .reshape((bsz, seq_len, num_heads, head_dim))
                    .transpose(1, 2)
                    .to(q)
                )
            else:  # head-wise quantization
                per_head_k = k.reshape(-1, head_dim)
                self.k_quantizer.find_params(per_head_k)
                k = (
                    self.k_quantizer(per_head_k)
                    .reshape((bsz, num_heads, seq_len, head_dim))
                    .to(q)
                )

            self.k_quantizer.free()

        return q, k


def add_qk_rotation_wrapper_after_function_call_in_forward(
    module,
    function_name,
    *args,
    **kwargs,
):
    """
    This function adds a rotation wrapper after the output of a function call in forward.
    Only calls directly in the forward function are affected. calls by other functions called in forward are not affected.
    """

    attr_name = f"{function_name}_qk_rotation_wrapper"
    assert not hasattr(module, attr_name)
    wrapper = monkeypatch.add_wrapper_after_function_call_in_method(
        module,
        "forward",
        function_name,
        functools.partial(QKRotationWrapper, *args, **kwargs),
    )
    setattr(module, attr_name, wrapper)