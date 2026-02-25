# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# This code is based on QuaRot(https://github.com/spcl/QuaRot/tree/main/quarot).
# Licensed under Apache License 2.0.

import random
from typing import Any, Dict
from torch.utils.data.distributed import DistributedSampler
import datasets
import torch
import transformers
import os

class TupleDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    

def get_wikitext2(nsamples=128, seed=0, seqlen=2048, model="", tokenizer=None, eval_mode=False, bs=4):
    if tokenizer is None:
        tokenizer = transformers.AutoTokenizer.from_pretrained(model, use_fast=False)

    if eval_mode:
        testdata = datasets.load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")[
            "test"
        ]
        testenc = tokenizer("\n\n".join(testdata["text"]), return_tensors="pt")
        # Remove batch dimension
        input_ids = testenc['input_ids'].squeeze(0)  # Shape: (289007,)
        num_sequences = len(input_ids) // seqlen
        input_ids = input_ids[:num_sequences * seqlen]
        sequences = input_ids.view(num_sequences, seqlen)
        dataloader = torch.utils.data.DataLoader(
            sequences,
            batch_size=bs,
            shuffle=False
        )
        
        return dataloader
    else:
        traindata = datasets.load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")[
            "train"
        ]
        trainenc = tokenizer("\n\n".join(traindata["text"]), return_tensors="pt")
        random.seed(seed)
        trainloader = []
        for _ in range(nsamples):
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = trainenc.input_ids[:, i:j]
            tar = inp.clone()
            tar[:, :-1] = -100
            trainloader.append((inp, tar))
            
        # create a dataloader
        trainloader = torch.utils.data.DataLoader(
            TupleDataset(trainloader), 
            batch_size=bs, 
            shuffle=True,
        )
        return trainloader

def get_c4(nsamples=128, seed=0, seqlen=2048, model="", tokenizer=None, eval_mode=False):
    """
    Get a loader for the C4 dataset.
    
    Args:
        nsamples (int): Number of samples to generate.
        seed (int): Random seed for reproducibility.
        seqlen (int): The sequence length of each sample.
        model (str): The model name to load the tokenizer from.
        tokenizer: An already-initialized tokenizer.
        eval_mode (bool): If True, returns the validation set; otherwise, returns a training loader.
        
    Returns:
        If eval_mode is True, a tokenized tensor of the validation data.
        If eval_mode is False, a list of (input, target) tensor pairs for training.
    """
    # Initialize the tokenizer if not provided
    if tokenizer is None:
        tokenizer = transformers.AutoTokenizer.from_pretrained(model, use_fast=False)

    local_dataset_path = "./datasets/c4_en"  # Path where the C4 dataset is stored locally
    # --- Evaluation Mode ---
    if eval_mode:
        # Load a subset of the C4 validation set using streaming
        # valdata = datasets.load_dataset(
        #     'allenai/c4', 'en', split='validation', streaming=True
        # )
        # Load the validation set from the local disk
        valdata = datasets.load_from_disk(os.path.join(local_dataset_path, 'validation'))
        val_text = "\n\n".join(valdata["text"])
        
        # Take the first 10,000 documents for a manageable validation set
        #val_dataset_head = valdata.take(10000)
        #val_text = "\n\n".join([d['text'] for d in val_dataset_head])
        
        valenc = tokenizer(val_text, return_tensors="pt")
        return valenc
        
    # --- Training Mode ---
    else:
        # Load a subset of the C4 training set using streaming
        traindata = datasets.load_dataset(
            'allenai/c4', 'en', split='train', streaming=True
        )

        # Create a large text buffer by taking a fixed number of documents from the stream.
        # This avoids loading the entire massive dataset. 50,000 documents provide
        # a sufficiently large and diverse text corpus to sample from.
        dataset_head = traindata.take(50000) 
        text_samples = [d['text'] for d in dataset_head]
        
        # Concatenate and tokenize the text samples
        train_text = "\n\n".join(text_samples)
        trainenc = tokenizer(train_text, return_tensors="pt")

        # Generate training samples with the same logic as get_wikitext2
        random.seed(seed)
        trainloader = []
        for _ in range(nsamples):
            # Select a random starting point in the tokenized text
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            
            # Extract the input sequence
            inp = trainenc.input_ids[:, i:j]
            
            # Create the target tensor, masking all but the last token
            tar = inp.clone()
            tar[:, :-1] = -100
            
            trainloader.append((inp, tar))
            
        return trainloader



class CustomJsonDataset(torch.utils.data.IterableDataset):
    def __init__(self, dataset, tokenizer, block_size: int = 1024) -> None:
        raw_data = dataset
        self.tokenizer = tokenizer
        self.block_size = block_size
        tokenized_datasets = []
        for d in raw_data:
            tokenized_datasets.append(self.tokenize_function(d))

        grouped_dataset = self.group_texts(tokenized_datasets)
        self.input_ids = grouped_dataset["input_ids"]
        self.labels = grouped_dataset["labels"]
        self.data = [
            dict(input_ids=self.input_ids[i], labels=self.labels[i])
            for i in range(len(self.input_ids))
        ]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, i) -> Dict[str, Any]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])

    def __iter__(self):
        return iter(self.data)

    def tokenize_function(self, examples):
        return self.tokenizer(examples["text"])

    def group_texts(self, examples):
        # Concatenate all texts.
        # Initialize an empty dictionary
        concatenated_examples = {}

        # Loop through the list of dictionaries
        for d in examples:
            # Loop through the keys in each dictionary
            for key in d.keys():
                # If the key is not already a key in the dict_of_lists, create a new list
                if key not in concatenated_examples:
                    concatenated_examples[key] = []
                # Append the value to the list associated with the key in dict_of_lists
                concatenated_examples[key].extend(d[key])
        total_length = len(concatenated_examples["input_ids"])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= self.block_size:
            total_length = (total_length // self.block_size) * self.block_size
        # Split by chunks of max_len.
        result = {
            k: [
                t[i : i + self.block_size]
                for i in range(0, total_length, self.block_size)
            ]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result
