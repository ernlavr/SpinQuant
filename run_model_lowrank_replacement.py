import os
import sys
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Tuple, Optional
import math
from utils import low_rank_utils as lru

from utils import data_utils, eval_utils, utils
from utils.process_args import process_args_ptq
from modules.linears import LowRankLinear
import utils.wandb_utils as wandb_utils
from dotenv import load_dotenv


def decompose_weight(weight: torch.Tensor, rank: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Decompose a weight matrix into low-rank factors using SVD.
    
    This provides a good initialization for training by preserving the original
    weight information as much as possible.
    
    Args:
        weight: Original weight matrix (out_features, in_features) from nn.Linear
        rank: Target rank for decomposition
    
    Returns:
        L: (in_features, rank) matrix
        R: (rank, out_features) matrix
    """
    # Transpose to get (in_features, out_features)
    W = weight.t()
    
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    
    # With full_matrices=False:
    # - U: (in_features, min(in_features, out_features))
    # - S: (min(in_features, out_features),)
    # - Vh: (min(in_features, out_features), out_features)
    
    # Truncate to rank
    S_truncated = S[:rank]
    U_truncated = U[:, :rank]  # (in_features, rank)
    Vh_truncated = Vh[:rank, :]  # (rank, out_features)
    
    # Distribute singular values: W ≈ (U @ sqrt(S)) @ (sqrt(S) @ V^T)
    # This is better for training than putting all S in one factor
    sqrt_S = torch.sqrt(S_truncated)
    L = U_truncated @ torch.diag(sqrt_S)  # (in_features, rank)
    R = torch.diag(sqrt_S) @ Vh_truncated  # (rank, out_features)
    
    return L, R


def replace_linear_with_lowrank(model: nn.Module, rank_modifier: float, 
                                target_modules: Optional[list] = None,
                                init_with_svd: bool = True):
    """
    Replace Linear layers in the model with LowRankLinear.
    
    Args:
        model: The model to compress
        rank: Target rank for low-rank decomposition
        target_modules: List of exact module names to replace.
                       If None, replaces all Linear layers.
                       Examples: 
                         - ['mlp.up_proj', 'mlp.gate_proj', 'self_attn.q_proj']
                         - ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj']
                       Use get_all_linear_module_names() to see available modules.
        init_with_svd: If True, initialize with SVD decomposition of original weights.
                      If False, use random initialization (for starting from scratch).
    """
    # Convert target_modules to a set for faster lookup
    target_set = set(target_modules) if target_modules is not None else None
    progress_bar = tqdm(range(0, len(target_modules)), desc="Replacing Linear with LowRankLinear", total=len(target_modules))
    
    def replace_fn(module: nn.Module, prefix: str = ''):
        for child_name, child in list(module.named_children()):
            full_name = f"{prefix}{child_name}" if prefix else child_name
            
            if isinstance(child, nn.Linear):
                # Check if this module should be replaced
                should_replace = (target_set is None) or (full_name in target_set)
                
                if should_replace:
                    in_feat = child.in_features
                    out_feat = child.out_features
                    has_bias = child.bias is not None
                    
                    # compute the rank based on modifier
                    rank = max(1, int(min(in_feat, out_feat) * rank_modifier))
                    
                    new_module = LowRankLinear(
                        in_features=in_feat,
                        out_features=out_feat,
                        rank=rank,
                        bias=has_bias,
                        device=child.weight.device,
                        dtype=child.weight.dtype
                    )
                    
                    # Initialize with SVD decomposition of original weights
                    if init_with_svd:
                        L, R = decompose_weight(child.weight, rank)
                        with torch.no_grad():
                            new_module.L.copy_(L)
                            new_module.R.copy_(R)
                            if has_bias:
                                new_module.bias.copy_(child.bias)
                    
                    setattr(module, child_name, new_module)
                    progress_bar.update(1)
            else:
                # Recursively apply to children
                replace_fn(child, prefix=f"{full_name}.")
    
    replace_fn(model)


def get_all_linear_module_names(model: nn.Module) -> list:
    """
    Get all Linear module names in the model.
    Useful for figuring out which modules to target.
    
    Returns:
        List of full module names that contain Linear layers
    """
    linear_names = []
    
    def find_linear(module: nn.Module, prefix: str = ''):
        for child_name, child in module.named_children():
            full_name = f"{prefix}{child_name}" if prefix else child_name
            
            if isinstance(child, nn.Linear):
                linear_names.append(full_name)
            else:
                find_linear(child, prefix=f"{full_name}.")
    
    find_linear(model)
    return sorted(linear_names)


def get_compression_stats(model: nn.Module) -> dict:
    """
    Calculate parameter count and compression ratio.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    lowrank_params = sum(p.numel() for m in model.modules() 
                        if isinstance(m, LowRankLinear) 
                        for p in m.parameters())
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'lowrank_params': lowrank_params,
    }

def perform_binary_search_truncation(model, sensitivity_dict, calib_loader, args):
    return lru.binary_search_truncation_rank(model, sensitivity_dict, calib_loader, args)
    
def test_calib_sensitivity_ppl(model, training_args, model_args, ptq_args):
    tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=model_args.input_model,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=True,
            add_eos_token=False,
            add_bos_token=False,
            token=model_args.access_token,
        )
    
    testloader = data_utils.get_wikitext2(
            seed=ptq_args.seed,
            seqlen=2048,
            tokenizer=tokenizer,
            eval_mode=True,
        )
    
    return lru.calib_sensitivity_ppl(model, testloader, ptq_args, use_cache=ptq_args.use_sensitivity_cache)
    
def process():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_args, training_args, ptq_args = process_args_ptq()
    print("Loading teacher model...")
    # teacher_model = AutoModelForCausalLM.from_pretrained(
    #     "meta-llama/Llama-2-7b-hf",
    #     torch_dtype=torch.float16,
    #     device_map="auto"
    # )
    # teacher_model.eval()
        
    rank_modifier = 0.25
    print("Loading student model (will be compressed)...")
    student_model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.1-8B",
        torch_dtype=torch.float32,
        device_map="auto",
        token=model_args.access_token,
    )
    student_model.seqlen = training_args.model_max_length
    uncompressed_stats = get_compression_stats(student_model)
    print(f"Compressing student with rank={rank_modifier}...")
    
    # tokenizer, data loader
    tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=model_args.input_model,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=True,
            add_eos_token=False,
            add_bos_token=False,
            token=model_args.access_token,
        )
    
    testloader = data_utils.get_wikitext2(
            seed=ptq_args.seed,
            seqlen=2048,
            tokenizer=tokenizer,
            eval_mode=True,
        )
    
    # test sensitivity from ASVD paper
    sensitivity = test_calib_sensitivity_ppl(student_model, training_args, model_args, ptq_args)
    
    if sensitivity is not None:
        perform_binary_search_truncation(student_model, sensitivity, testloader, ptq_args)
        
    
    # First, see what Linear modules are available
    print("\nAvailable Linear modules:")
    all_linear_modules = get_all_linear_module_names(student_model)
    for name in all_linear_modules[:10]:  # Show first 10
        print(f"  - {name}")
    if len(all_linear_modules) > 10:
        print(f"  ... and {len(all_linear_modules) - 10} more")
    
    
    # Defines which modules to replace
    layer_id = list(range(0, 32))
    target_modules = [i for i in all_linear_modules for j in layer_id if f"model.layers.{j}." in i]
    
    # Do the replacement
    replace_linear_with_lowrank(
        student_model, 
        rank_modifier=rank_modifier,
        target_modules=target_modules,
        init_with_svd=True
    )
    
    student_model.train()
    
    # Print compression stats
    compressed_stats = get_compression_stats(student_model)
    print(f"\nCompression Statistics:")
    print(f"  Total parameters original: {uncompressed_stats['total_params']:,}")
    print(f"  Total parameters decomposed: {compressed_stats['total_params']:,}")
    print(f"  Low-rank parameters: {compressed_stats['lowrank_params']:,}")
    compression_ratio = compressed_stats['total_params'] / uncompressed_stats['total_params']
    print(f"  Compression ratio: {compression_ratio:.2%}")
    
    tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=model_args.input_model,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=True,
            add_eos_token=False,
            add_bos_token=False,
            token=model_args.access_token,
        )
    
    testloader = data_utils.get_wikitext2(
            seed=ptq_args.seed,
            seqlen=2048,
            tokenizer=tokenizer,
            eval_mode=True,
        )

    ppl, avg_time_per_token = eval_utils.evaluator(student_model, testloader, utils.DEV, ptq_args)
    print(f"[SUCCESS] Layer {layer_id}: Wiki2 PPL = {ppl:.2f}, Time = {avg_time_per_token:.4f} ms/token")
    
    # # Example training setup
    # print("\nSetting up training...")
    # optimizer = torch.optim.AdamW(student_model.parameters(), lr=1e-4)
    
    # # Example batch
    # tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    # texts = ["The meaning of life is", "Artificial intelligence is"]
    # inputs = tokenizer(texts, return_tensors="pt", padding=True).to(device)
    
    # print("Running forward pass...")
    # with torch.no_grad():
    #     teacher_outputs = teacher_model(**inputs, output_hidden_states=True)
    
    # student_outputs = student_model(**inputs, output_hidden_states=True)
    
    # # Simple MSE loss on hidden states (example distillation loss)
    # loss = torch.nn.functional.mse_loss(
    #     student_outputs.hidden_states[-1],
    #     teacher_outputs.hidden_states[-1]
    # )
    
    # print(f"Loss: {loss.item():.4f}")
    
    # print("Running backward pass...")
    # optimizer.zero_grad()
    # loss.backward()
    # optimizer.step()
    
    print("Training step completed successfully!")

def main():
    # dist.init_process_group(backend="nccl", timeout=datetime.timedelta(hours=8))
    load_dotenv('.env')
    if "--wandb_sweep" in sys.argv:
        wandb_utils.start_sweep(None, process)
    elif "--wandb_run" in sys.argv:
        wandb_utils.start_run("lowrank_replacement_test", None, process)
    else:
        process()
    # dist.destroy_process_group()

# Example usage for student-teacher training
if __name__ == "__main__":
    main()
    