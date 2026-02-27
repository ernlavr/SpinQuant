import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from typing import Optional, Tuple, List
from dataclasses import dataclass
from tqdm import tqdm

import transformers
import wandb

from utils import data_utils, eval_utils, utils


@dataclass
class DistillationConfig:
    """Configuration for knowledge distillation"""
    temperature: float = 1.0  # Temperature for softmax
    alpha: float = 0.7  # Weight for distillation loss (vs task loss)
    batch_size: int = 8
    learning_rate: float = 1e-6
    num_epochs: int = 3
    max_seq_length: int = 2048
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class KnowledgeDistiller:
    """Knowledge Distillation trainer for language models"""

    def __init__(self, student, teacher, tokenizer, config: DistillationConfig):
        self.config = config
        self.device = torch.device(config.device)

        # Load teacher model
        self.teacher = teacher
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False

        # Load student model
        self.student = student
        self.student.train()

        # Load tokenizer
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.student.parameters(),
            lr=config.learning_rate
        )
        self.scheduler = None # will be defined later

    def distillation_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        targets: torch.Tensor,
        temperature: float = 4.0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calculate distillation loss

        Args:
            student_logits: Logits from student model
            teacher_logits: Logits from teacher model
            targets: Ground truth labels
            temperature: Temperature for softening probabilities

        Returns:
            Tuple of (total_loss, distillation_loss, task_loss)
        """
        # move logits to the same device which has more memory
        more_mem_dev = self.teacher.device if torch.cuda.get_device_properties(self.teacher.device).total_memory > torch.cuda.get_device_properties(self.student.device).total_memory else self.student.device
        
        student_logits = student_logits.to(more_mem_dev)
        teacher_logits = teacher_logits.to(more_mem_dev)
        targets = targets.to(more_mem_dev)
        
        # Distillation loss (KL divergence with temperature)
        student_probs = F.log_softmax(student_logits / temperature, dim=-1)
        teacher_probs = F.softmax(teacher_logits / temperature, dim=-1).detach()

        kl_loss = F.kl_div(
            student_probs,
            teacher_probs,
            reduction="batchmean"
        ) * (temperature ** 2)

        # Task loss (cross-entropy with ground truth)
        task_loss = F.cross_entropy(student_logits, targets)

        # Combined loss
        total_loss = self.config.alpha * kl_loss + (1 - self.config.alpha) * task_loss

        return total_loss, kl_loss, task_loss
    
    def enable_left_components(self):
        """Enable training only for left components of SVDLinear layers"""
        for name, param in self.student.named_parameters():
            param.requires_grad = False
            if "ALinear" in name:
                param.requires_grad = True
    
    def enable_right_components(self):
        """Enable training only for right components of SVDLinear layers"""
        for name, param in self.student.named_parameters():
            param.requires_grad = False
            if "BLinear" in name:
                param.requires_grad = True
                
    def perform_training_step(self, batch, use_distillation=True):
        """Perform a single training step"""
        input_ids = batch[0]
        loss_mask = None # batch[1].to(self.device)
        input_ids = input_ids.squeeze(1)
        
        # Forward pass through teacher (no grad)
        with torch.no_grad():
            teacher_outputs = self.teacher(
                input_ids=input_ids.to(self.teacher.device),
                output_hidden_states=False,
            )
            teacher_logits = teacher_outputs.logits

        # Forward pass through student
        student_outputs = self.student(
            input_ids=input_ids.to(self.student.device),
            output_hidden_states=False,
        )
        student_logits = student_outputs.logits

        # Prepare targets (shift for language modeling)
        targets = input_ids[:, 1:].contiguous()
        student_logits = student_logits[:, :-1, :].contiguous()
        teacher_logits = teacher_logits[:, :-1, :].contiguous()

        # Reshape for loss computation
        batch_size, seq_len, vocab_size = student_logits.shape
        student_logits_flat = student_logits.view(-1, vocab_size)
        teacher_logits_flat = teacher_logits.view(-1, vocab_size)
        targets_flat = targets.view(-1)

        # Calculate losses
        loss, kl_loss, task_loss = self.distillation_loss(
            student_logits_flat,
            teacher_logits_flat,
            targets_flat,
            temperature=self.config.temperature
        )
        
        # Backward pass
        self.optimizer.zero_grad()
        if use_distillation:
            loss.backward()
        else:
            task_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.student.parameters(), max_norm=1.0)
        self.optimizer.step()            
            
        # clear up memory
        del student_logits, teacher_logits, student_logits_flat, teacher_logits_flat, targets_flat
        torch.cuda.empty_cache()
        
        return loss, kl_loss, task_loss

    def train_epoch(self, train_loader: DataLoader, use_distillation=True, use_alternating_LR_training=False) -> dict:
        """Train for one epoch"""
        self.student.train()
        print(f"\nTraining for one epoch with distillation: {use_distillation} and alternating LR training: {use_alternating_LR_training} ...")
        
        # enable only SVDLinear layers to be trained
        print("Freezing all student parameters except SVDLinear layers...")
        for name, param in self.student.named_parameters():
            param.requires_grad = False
            if "ALinear" in name or "BLinear" in name:
                param.requires_grad = True
        
        total_loss = 0
        total_kl_loss = 0
        total_task_loss = 0
        num_batches = 0

        for batch_idx, batch in tqdm(enumerate(train_loader), f"Training with distillation: {use_distillation} ..."):
            
            if use_alternating_LR_training:
                self.enable_left_components()
                loss, kl_loss, task_loss = self.perform_training_step(batch, use_distillation)
                total_loss += loss.item()
                total_kl_loss += kl_loss.item()
                total_task_loss += task_loss.item()
                num_batches += 1
                
                self.enable_right_components()
                loss, kl_loss, task_loss = self.perform_training_step(batch, use_distillation)
                total_loss += loss.item()
                total_kl_loss += kl_loss.item()
                total_task_loss += task_loss.item()
                num_batches += 1
            
            else:
                # Perform training step
                loss, kl_loss, task_loss = self.perform_training_step(batch, use_distillation)
                # Track metrics
                total_loss += loss.item()
                total_kl_loss += kl_loss.item()
                total_task_loss += task_loss.item()
                num_batches += 1
            
            # Step it here to avoid stepping twice in alternating LR training
            if self.scheduler is not None:
                self.scheduler.step()

            if (batch_idx + 1) % 10 == 0:
                print(
                    f"Batch {batch_idx + 1}/{len(train_loader)} | "
                    f"Loss: {loss.item():.4f} | "
                    f"KL Loss: {kl_loss.item():.4f} | "
                    f"Task Loss: {task_loss.item():.4f} | "
                    f"LR: {self.scheduler.get_last_lr()[0]:.8f}"
                )
                
            wandb.log({
                "distillation/loss": loss.item(),
                "distillation/kl_loss": kl_loss.item(),
                "distillation/task_loss": task_loss.item(),
                "distillation/lr": self.scheduler.get_last_lr()[0],
            })
            
            
        wandb.log({
            "distillation/epoch_avg_loss": total_loss / num_batches,
            "distillation/epoch_avg_kl_loss": total_kl_loss / num_batches,
            "distillation/epoch_avg_task_loss": total_task_loss / num_batches,
        })

        return {"avg_loss": total_loss / num_batches,
            "avg_kl_loss": total_kl_loss / num_batches,
            "avg_task_loss": total_task_loss / num_batches,
        }
        
    def define_cos_scheduler_with_warmup(self, train_loader):
        """Define a cosine learning rate scheduler with warmup"""
        steps_per_epoch = len(train_loader)
        total_steps = self.config.num_epochs * steps_per_epoch

        warmup_ratio = 0.1           # 5–10% is typical
        warmup_steps = int(total_steps * warmup_ratio)
        
        scheduler = transformers.get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        return scheduler

    def train(self, train_loader, test_loader, ptq_args):
        """Train the student model using knowledge distillation"""            
        print(f"\nStarting knowledge distillation training...")
        print(f"Temperature: {self.config.temperature}")
        print(f"Alpha (KL weight): {self.config.alpha}")
        self.scheduler = self.define_cos_scheduler_with_warmup(train_loader)
        # self.run_eval(self.student, test_loader, utils.DEV, ptq_args, 0)
        
        for epoch in range(self.config.num_epochs):
            print(f"\n{'='*50}")
            print(f"Epoch {epoch + 1}/{self.config.num_epochs}")
            print(f"{'='*50}")

            metrics = self.train_epoch(train_loader, ptq_args.use_distillation, ptq_args.use_alternating_LR_training)
            
            print(f"\nEpoch {epoch + 1} Summary:")
            print(f"  Average Loss: {metrics['avg_loss']:.4f}")
            print(f"  Average KL Loss: {metrics['avg_kl_loss']:.4f}")
            print(f"  Average Task Loss: {metrics['avg_task_loss']:.4f}")
            
            self.run_eval(self.student, test_loader, utils.DEV, ptq_args, epoch + 1)
            
            if torch.cuda.device_count() > 1:
                dist.barrier()
                torch.cuda.empty_cache()
            

    def run_eval(self, model, test_loader, device, ptq_args, epoch=0):
        ppl, avg_time_per_token = eval_utils.evaluator_single_gpu_simplified(self.student, test_loader, device, ptq_args) 
        print(f"Student model PPL after Epoch {epoch}: {ppl:.2f}, Avg time per token: {avg_time_per_token*1000:.2f} ms")
        
        if wandb.run is not None:
            wandb.log({
                "epoch": epoch,
                "student_ppl": ppl
            })
    
    def save_student_model(self, save_path: str):
        """Save the distilled student model"""
        print(f"\nSaving student model to {save_path}")
        self.student.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)

    def generate(self, prompt: str, max_length: int = 100) -> str:
        """Generate text using the student model"""
        self.student.eval()
        self.student.to(self.device)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.student.generate(
                **inputs,
                max_length=max_length,
                num_beams=1,
                do_sample=False,
                temperature=0.7,
            )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)