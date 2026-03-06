# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# This code is based on QuaRot(https://github.com/spcl/QuaRot/tree/main/quarot).
# Licensed under Apache License 2.0.

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
from utils.low_rank_utils import perform_svd_decomp

def random_orthogonal_matrix(size, device):
    """
    Generate a random orthogonal matrix of the specified size.
    First, we generate a random matrix with entries from a standard distribution.
    Then, we use QR decomposition to obtain an orthogonal matrix.
    Finally, we multiply by a diagonal matrix with diag r to adjust the signs.

    Args:
    size (int): The size of the matrix (size x size).

    Returns:
    torch.Tensor: An orthogonal matrix of the specified size.
    """
    torch.cuda.empty_cache()
    random_matrix = torch.randn(size, size, dtype=torch.float64).to(device)
    q, r = torch.linalg.qr(random_matrix)
    q *= torch.sign(torch.diag(r)).unsqueeze(0)
    return q

def perturb_rotation_matrix(R_original, noise_scale, device, args):
    """
    Applies a small random rotation 'noise' to an original rotation matrix.
    
    Args:
        R_original (torch.Tensor): The base rotation matrix (size x size).
        noise_scale (float): Magnitude of the noise (standard deviation of the angle).
    
    Returns:
        torch.Tensor: Perturbed rotation matrix.
    """
    cache_dir = "output_dir/precomputed_noises"
    size = R_original.shape[0]
    
    # 1. Generate random matrix, forward declare
    A = None
    
    # check if output_dir/precomputed_noises/ contains entry for size_size_seed
    pt_file = f"{cache_dir}/{size}_{size}_seed-{args.seed}.pt"
    if os.path.exists(pt_file):
        A = torch.load(pt_file)
        print(f"INFO: Loaded precomputed random matrix from {os.path.abspath(pt_file)}")
    else:
        A = torch.randn(size, size, dtype=R_original.dtype, device=device)
        os.makedirs(cache_dir, exist_ok=True)
        torch.save(A, pt_file)
        print(f"INFO: Saved precomputed random matrix to {os.path.abspath(pt_file)}") 
    
    
    # 2. Create skew-symmetric matrix (Lie Algebra element)
    # Dividing by sqrt(2) ensures the elements have unit variance before scaling
    skew = (A - A.T) / torch.sqrt(torch.tensor(2.0)) 
    
    # 3. Apply scale
    skew *= noise_scale
    
    # 4. Exponentiate to get the rotation noise
    # matrix_exp maps the skew-symmetric matrix to SO(n)
    R_noise = torch.linalg.matrix_exp(skew)
    
    # 5. Compose with original rotation
    # Note: Order matters. R_noise @ R_original perturbs in the global frame.
    # R_original @ R_noise perturbs in the local frame. 
    # For isotropic noise, they are statistically equivalent.
    R_perturbed = R_noise @ R_original
    
    return R_perturbed

def get_orthogonal_matrix(size, mode, device="cuda"):
    if mode == "random":
        return random_orthogonal_matrix(size, device)
    elif mode == "hadamard":
        return random_hadamard_matrix(size, device)
    else:
        raise ValueError(f"Unknown mode {mode}")


def rotate_embeddings(model, R1: torch.Tensor) -> None:
    # Rotate the embeddings.
    for W in [model.model.embed_tokens]:
        dtype = W.weight.data.dtype
        W_ = W.weight.data.to(device="cuda", dtype=torch.float64)
        W.weight.data = torch.matmul(W_, R1).to(device="cpu", dtype=dtype)


def rotate_attention_inputs(layer, R1) -> None:
    # Rotate the WQ, WK and WV matrices of the self-attention layer.
    for W in [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj]:
        dtype = W.weight.dtype
        W_ = W.weight.to(device="cuda", dtype=torch.float64)
        
        u, s, vh = perform_svd_decomp(W_)
        
        W.weight.data = torch.matmul(W_, R1).to(device="cpu", dtype=dtype)


def rotate_attention_output(layer, R1) -> None:
    # Rotate output matrix of the self-attention layer.
    W = layer.self_attn.o_proj

    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(device="cuda", dtype=torch.float64)
    W.weight.data = torch.matmul(R1.T, W_).to(device="cpu", dtype=dtype)
    if W.bias is not None:
        b = W.bias.data.to(device="cuda", dtype=torch.float64)
        W.bias.data = torch.matmul(R1.T, b).to(device="cpu", dtype=dtype)


def rotate_mlp_input(layer, R1):
    # Rotate the MLP input weights.
    mlp_inputs = [layer.mlp.up_proj, layer.mlp.gate_proj]
    for W in mlp_inputs:
        dtype = W.weight.dtype
        W_ = W.weight.data.to(device="cuda", dtype=torch.float64)
        W.weight.data = torch.matmul(W_, R1).to(device="cpu", dtype=dtype)


#def rotate_mlp_output(layer, R1, args):
def rotate_mlp_output(layer, R1, layer_idx, selective_had_layers, args):
    # Rotate the MLP output weights and bias.
    W = layer.mlp.down_proj
    layer_identifier = f"Layer {layer_idx} ({W})"
    
    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(device="cuda", dtype=torch.float64)
    W.weight.data = torch.matmul(R1.T, W_).to(device="cpu", dtype=dtype)

    if selective_had_layers is not None:
        # If a list was provided, apply only if this layer is in the list
        if layer_idx in selective_had_layers:
            apply_exact_had_to_linear(
                 W, had_dim=-1, output=False
             )
            print(f"INFO: Applying inverse Hadamard to weights of {layer_identifier} (Selective Mode).")
        else:
             print(f"INFO: Skipping inverse Hadamard on weights of {layer_identifier} (Not in selective list).")
    elif args.hadamard_online:
         print(f"INFO: Applying inverse Hadamard to weights of {W} (SpinQuant_had mode).")
         apply_exact_had_to_linear(
             W, had_dim=-1, output=False
         ) # apply exact (inverse) hadamard on the weights of mlp output
    else:
         print(f"INFO: Skipping inverse Hadamard on weights of {W} (SpinQuant_no_had mode).")
    if W.bias is not None:
        b = W.bias.data.to(device="cuda", dtype=torch.float64)
        W.bias.data = torch.matmul(R1.T, b).to(device="cpu", dtype=dtype)


def rotate_head(model, R1: torch.Tensor) -> None:
    # Rotate the head.
    W = model.lm_head
    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(device="cuda", dtype=torch.float64)
    W.weight.data = torch.matmul(W_, R1).to(device="cpu", dtype=dtype)


def rotate_ov_proj(layer, head_num, head_dim, R2=None):
    v_proj = layer.self_attn.v_proj
    o_proj = layer.self_attn.o_proj

    apply_exact_had_to_linear(v_proj, had_dim=head_dim, output=True, R2=R2)
    apply_exact_had_to_linear(o_proj, had_dim=head_dim, output=False, R2=R2)


@torch.inference_mode()
def rotate_model(model, args):
    selective_had_layers = None
    selective_had_path = getattr(args, 'selective_had_layers_path', None)

    if selective_had_path:
        print(f"INFO (rotate_model): Found selective had layers path: {selective_had_path}")
        if os.path.exists(selective_had_path):
            try:
                with open(selective_had_path, 'r') as f:
                    data = json.load(f)
                    if "layers_to_rotate" in data and isinstance(data["layers_to_rotate"], list):
                        selective_had_layers = set(data["layers_to_rotate"])
                        print(f"INFO (rotate_model): Loaded {len(selective_had_layers)} indices for selective weight Had compensation.")
                    else:
                         print(f"Warning (rotate_model): JSON invalid format in {selective_had_path}. Weight Had compensation might be incorrect.")
            except Exception as e:
                print(f"Warning (rotate_model): Error loading JSON {selective_had_path}: {e}. Weight Had compensation might be incorrect.")
        else:
            print(f"Warning (rotate_model): Selective had layers file not found: {selective_had_path}. Weight Had compensation might be incorrect.")
            
    R1 = get_orthogonal_matrix(model.config.hidden_size, args.rotate_mode)
    print(f"INFO: optimized rotation path: {args.optimized_rotation_path}")
    if args.optimized_rotation_path is not None:
        R_cpk = args.optimized_rotation_path
        print(f"INFO (rotate_model): Found optimized rotation path: {R_cpk}")
        R1 = torch.load(R_cpk)["R1"].cuda().to(torch.float64)
        
        # if args.noise_scalar is not None:
        #     R1 = perturb_rotation_matrix(R1, noise_scale=args.noise_scalar, device="cuda", args=args)
        
    # add random gaussian noise to R1
    # gaussian = (torch.randn_like(R1) * R1.std() + R1.mean()) * 0.1
    
    config = model.config
    num_heads = config.num_attention_heads
    model_dim = config.hidden_size
    head_dim = model_dim // num_heads

    rotate_embeddings(model, R1)
    rotate_head(model, R1)
    utils.cleanup_memory()
    layers = [layer for layer in model.model.layers]
    for idx, layer in enumerate(tqdm.tqdm(layers, unit="layer", desc="Rotating")):
        if args.optimized_rotation_path is not None:
            key = f"model.layers.{idx}.self_attn.R2"
            R2 = torch.load(R_cpk)[key].cuda().to(torch.float64)
        else:
            R2 = get_orthogonal_matrix(head_dim, args.rotate_mode)
        # adding noise
        # if args.noise_scalar is not None:
        #     R2 = perturb_rotation_matrix(R2, noise_scale=args.noise_scalar, device="cuda", args=args)
        # end noise
        
        u, s, vh = perform_svd_decomp(model.model.embed_tokens.weight.data)
        
        
        rotate_attention_inputs(layers[idx], R1)
        rotate_attention_output(layers[idx], R1)
        rotate_mlp_input(layers[idx], R1)
        #rotate_mlp_output(layers[idx], R1, args)
        rotate_mlp_output(layers[idx], R1, idx, selective_had_layers, args)
        rotate_ov_proj(layers[idx], num_heads, head_dim, R2=R2)


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