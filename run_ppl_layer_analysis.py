import datetime
from logging import Logger

import os
import json

import re

import torch
import torch.distributed as dist
from transformers import LlamaTokenizerFast
import transformers
from transformers import AutoModelForCausalLM
from eval_utils.main import ptq_model
from eval_utils.modeling_llama import LlamaForCausalLM
from utils import data_utils, eval_utils, utils, quant_utils
from utils.process_args import process_args_ptq
import wandb
import sys
os.environ['WANDB_API_KEY'] = '49c8e7dca82f91f9d65021c3dd71101b686c1f53'


def main():
    # --- Setup ---
    # We start by parsing arguments just like in ptq.py
    # We'll override the 'exclude_activations_layers' argument in our loop
    model_args, training_args, ptq_args = process_args_ptq()
    local_rank = utils.get_local_rank()

    print("the rank is {}".format(local_rank))
    torch.distributed.barrier()

    config = transformers.AutoConfig.from_pretrained(
        model_args.input_model, token=model_args.access_token
    )
    # Llama v3.2 specific: Spinquant is not compatiable with tie_word_embeddings, clone lm_head from embed_tokens
    process_word_embeddings = False
    if config.tie_word_embeddings:
        config.tie_word_embeddings = False
        process_word_embeddings = True
    dtype = torch.bfloat16 if training_args.bf16 else torch.float16

    # Define the output file for the analysis results
    model_name = os.path.basename(model_args.input_model)
    output_dir = "sensitivity_results"
    os.makedirs(output_dir, exist_ok=True)
    results_file = os.path.join(output_dir, f"sensitivity_analysis_single_layer_{model_name}_w{ptq_args.w_bits}_a{ptq_args.a_bits}_v{ptq_args.v_bits}.json")

    all_results = []
    # Assuming a 32-layer model like Llama-2-7b. Adjust if necessary.
    total_layers = 32

    print("="*80)
    print(f"Starting single-layer sensitivity analysis for {model_name}")
    print(f"Results will be saved to: {results_file}")
    print("="*80)

    # --- Main Analysis Loop ---
    for i in range(0, total_layers, 6):
        print(f"\n[INFO] Running analysis for layer {i} (quantizing ONLY this layer's activations)")

        # --- Dynamically set the layers to exclude ---
        # Create a list of all layer indices EXCEPT the current one
        layers_to_exclude = [j for j in range(total_layers) if j != i]
        ptq_args.exclude_activations_layers = layers_to_exclude
        
        # --- Run the Quantization and Evaluation ---
        # This logic is adapted directly from ptq.py
        
        # 1. Load the base model
        
        model = LlamaForCausalLM.from_pretrained(
            pretrained_model_name_or_path=model_args.input_model,
            config=config,
            torch_dtype=dtype,
            token=model_args.access_token,
        )
        if process_word_embeddings:
            model.lm_head.weight.data = model.model.embed_tokens.weight.data.clone()
        model.cuda()
        current_target_device = model.device

        # 2. Apply quantization settings for the current iteration
        model = ptq_model(ptq_args, model, model_args)
        model.seqlen = training_args.model_max_length

        model.to(current_target_device)

        tokenizer = LlamaTokenizerFast.from_pretrained(
            pretrained_model_name_or_path=model_args.input_model,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=True,
            add_eos_token=False,
            add_bos_token=False,
            token=model_args.access_token,
        )

        
        
        # 3. Evaluate the model to get PPL and Inference Time
        # The 'nb_eval_runs' argument controls how many times evaluation is run for timing
        testloader = data_utils.get_wikitext2(
            seed=ptq_args.seed,
            seqlen=2048,
            tokenizer=tokenizer,
            eval_mode=True,
        )

        ppl, avg_time_per_token = eval_utils.evaluator(model, testloader, utils.DEV, ptq_args)
        wandb.log({"perplexity": ppl, "layer_id": i, "seed": ptq_args.seed})
        
        dist.barrier()

        print(f"[SUCCESS] Layer {i}: Wiki2 PPL = {ppl:.2f}, Time = {avg_time_per_token:.4f} ms/token")

        # Store the result for this layer
        all_results.append({
            "layer_id": i,
            "perplexity": ppl,
            "inference_time_ms_per_token": avg_time_per_token
        })

        # --- Write the results to the file after each iteration ---
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=4)
        
        print("-" * 80)

    print("Sensitivity analysis complete.")
    print(f"Final results have been saved to: {results_file}")

def pre_main_sweep_func():
    sweep_config = {
        'method': 'grid',
        'metric': {
            'name': 'perplexity',
            'goal': 'minimize'
        },
        'parameters': {
            'seed': {
                'values': [int(42), int(13), int(0)]
            },
            'noise_scalar': {
                'values': [1, 0.5, 0.2, 0.1, 0.05, 0.01, 0]
            }
        }
    }
    
    sweep_id = wandb.sweep(sweep_config, project='spinquant-noise')
    def sweep_main():
        with wandb.init() as run:
            config = wandb.config
            # Here you can access config.noise_scalar and use it in your training
            print(f"Running training with noise_scalar: {config.noise_scalar}")
            main()
    wandb.agent(sweep_id, function=sweep_main)

def pre_main():
    dist.init_process_group(backend="nccl", timeout=datetime.timedelta(hours=8))
    
    if "--wandb_sweep" in sys.argv:
        pre_main_sweep_func()
    else:
        main()
    dist.destroy_process_group()

if __name__ == "__main__":
    pre_main()