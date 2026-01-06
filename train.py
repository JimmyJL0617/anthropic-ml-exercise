#!/usr/bin/env python3
"""
Training Script for Transformer Sentiment Classifier

This script provides complete training infrastructure including:
- Training loop with validation and early stopping
- Checkpoint saving and resumption
- TensorBoard logging
- Learning rate scheduling with warmup
- Gradient clipping and mixed precision training

Usage:
    python train.py --config config.yaml
    python train.py --config config.yaml --use-pretrained
    python train.py --config config.yaml --resume checkpoints/last.pt
    python train.py --config config.yaml --debug
"""

import os
import sys
import json
import time
import random
import argparse
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any

import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from model import create_model, create_dataloaders, get_edge_cases


# =============================================================================
# Configuration Classes
# =============================================================================

@dataclass
class TrainingConfig:
    """Training configuration."""
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    epochs: int = 20
    warmup_steps: int = 1000
    gradient_clip_norm: float = 1.0
    label_smoothing: float = 0.1
    use_amp: bool = True
    log_every_n_steps: int = 50
    save_dir: str = "checkpoints"
    log_dir: str = "logs"
    experiment_name: str = "transformer_sentiment"
    use_tensorboard: bool = True
    device: str = "cuda"
    seed: int = 42


@dataclass
class EarlyStoppingConfig:
    """Early stopping configuration."""
    patience: int = 3
    min_delta: float = 0.001
    monitor: str = "val_loss"
    mode: str = "min"


# =============================================================================
# Utility Functions
# =============================================================================

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(preferred: str = "cuda") -> torch.device:
    """Get the best available device."""
    if preferred == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    elif preferred == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Count model parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


def ensure_dir(path: str) -> Path:
    """Ensure directory exists."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


# =============================================================================
# Early Stopping
# =============================================================================

class EarlyStopping:
    """Early stopping handler."""
    
    def __init__(self, config: EarlyStoppingConfig):
        self.patience = config.patience
        self.min_delta = config.min_delta
        self.monitor = config.monitor
        self.mode = config.mode
        self.counter = 0
        self.best_score = None
        self.should_stop = False
    
    def __call__(self, metrics: Dict[str, float]) -> bool:
        current = metrics.get(self.monitor)
        if current is None:
            return False
        
        if self.best_score is None:
            self.best_score = current
            return False
        
        improved = (current < self.best_score - self.min_delta) if self.mode == "min" else (current > self.best_score + self.min_delta)
        
        if improved:
            self.best_score = current
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        
        return self.should_stop


# =============================================================================
# Checkpoint Manager
# =============================================================================

class CheckpointManager:
    """Manages model checkpoints."""
    
    def __init__(self, save_dir: str, save_top_k: int = 3, monitor: str = "val_loss", mode: str = "min"):
        self.save_dir = ensure_dir(save_dir)
        self.save_top_k = save_top_k
        self.monitor = monitor
        self.mode = mode
        self.checkpoints = []
    
    def save(self, model, optimizer, scheduler, epoch, step, metrics, config) -> str:
        checkpoint = {
            "epoch": epoch,
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "metrics": metrics,
            "config": config
        }
        
        # Save last checkpoint
        torch.save(checkpoint, self.save_dir / "last.pt")
        
        # Save numbered checkpoint
        score = metrics.get(self.monitor, 0)
        checkpoint_path = self.save_dir / f"checkpoint_epoch{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        self.checkpoints.append((checkpoint_path, score))
        self.checkpoints.sort(key=lambda x: x[1], reverse=(self.mode == "max"))
        
        # Remove old checkpoints
        while len(self.checkpoints) > self.save_top_k:
            old_path, _ = self.checkpoints.pop()
            if old_path.exists():
                old_path.unlink()
        
        # Save best
        if self.checkpoints:
            best = torch.load(self.checkpoints[0][0], weights_only=False)
            torch.save(best, self.save_dir / "best.pt")
        
        return str(checkpoint_path)
    
    def load(self, model, optimizer=None, scheduler=None, checkpoint_path=None):
        if checkpoint_path is None:
            checkpoint_path = self.save_dir / "best.pt"
        
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        
        if optimizer and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scheduler and checkpoint.get("scheduler_state_dict"):
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        return checkpoint


# =============================================================================
# Metrics Logger
# =============================================================================

class MetricsLogger:
    """Handles logging to TensorBoard and JSON."""
    
    def __init__(self, log_dir: str, experiment_name: str, use_tensorboard: bool = True, config: Optional[dict] = None):
        self.log_dir = ensure_dir(log_dir)
        self.use_tensorboard = use_tensorboard
        self.writer = None
        
        if use_tensorboard:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir=str(self.log_dir / experiment_name))
        
        self.metrics_file = self.log_dir / f"{experiment_name}_metrics.json"
        self.metrics_history = []
    
    def log(self, metrics: Dict[str, float], step: int, prefix: str = ""):
        if prefix:
            metrics = {f"{prefix}/{k}": v for k, v in metrics.items()}
        
        if self.writer:
            for name, value in metrics.items():
                self.writer.add_scalar(name, value, step)
        
        self.metrics_history.append({"step": step, **metrics})
        with open(self.metrics_file, "w") as f:
            json.dump(self.metrics_history, f, indent=2)
    
    def close(self):
        if self.writer:
            self.writer.close()


# =============================================================================
# Learning Rate Scheduler
# =============================================================================

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps: int, num_training_steps: int) -> LambdaLR:
    """Create cosine schedule with linear warmup."""
    import math
    
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    
    return LambdaLR(optimizer, lr_lambda)


# =============================================================================
# Trainer Class
# =============================================================================

class Trainer:
    """Training manager for transformer models."""
    
    def __init__(self, model, train_loader, val_loader, config: TrainingConfig, early_stopping_config: Optional[EarlyStoppingConfig] = None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        
        self.device = get_device(config.device)
        self.model = self.model.to(self.device)
        
        self.criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
        
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        num_training_steps = len(train_loader) * config.epochs
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=config.warmup_steps,
            num_training_steps=num_training_steps
        )
        
        self.scaler = GradScaler() if config.use_amp and self.device.type == "cuda" else None
        
        self.early_stopping = EarlyStopping(early_stopping_config) if early_stopping_config else None
        
        self.checkpoint_manager = CheckpointManager(save_dir=config.save_dir)
        
        self.logger = MetricsLogger(
            log_dir=config.log_dir,
            experiment_name=config.experiment_name,
            use_tensorboard=config.use_tensorboard,
            config=asdict(config)
        )
        
        self.global_step = 0
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch}", leave=False)
        
        for batch in progress_bar:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)
            
            self.optimizer.zero_grad()
            
            if self.scaler:
                with autocast():
                    outputs = self.model(input_ids, attention_mask)
                    loss = self.criterion(outputs["logits"], labels)
                
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(input_ids, attention_mask)
                loss = self.criterion(outputs["logits"], labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_norm)
                self.optimizer.step()
            
            self.scheduler.step()
            
            total_loss += loss.item()
            predictions = outputs["logits"].argmax(dim=-1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            
            self.global_step += 1
            
            if self.global_step % self.config.log_every_n_steps == 0:
                self.logger.log({
                    "loss": loss.item(),
                    "learning_rate": self.scheduler.get_last_lr()[0],
                    "accuracy": correct / total
                }, self.global_step, prefix="train")
            
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{correct/total:.4f}"})
        
        return {"loss": total_loss / len(self.train_loader), "accuracy": correct / total}
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch in tqdm(self.val_loader, desc="Validating", leave=False):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)
            
            outputs = self.model(input_ids, attention_mask)
            loss = self.criterion(outputs["logits"], labels)
            
            total_loss += loss.item()
            predictions = outputs["logits"].argmax(dim=-1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
        
        return {"loss": total_loss / len(self.val_loader), "accuracy": correct / total}
    
    def train(self) -> Dict[str, Any]:
        """Full training loop."""
        print(f"Training on {self.device}")
        print(f"Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        history = {"train": [], "val": []}
        
        for epoch in range(1, self.config.epochs + 1):
            epoch_start = time.time()
            
            train_metrics = self.train_epoch(epoch)
            history["train"].append(train_metrics)
            
            val_metrics = self.validate()
            history["val"].append(val_metrics)
            
            epoch_time = time.time() - epoch_start
            self.logger.log(train_metrics, epoch, prefix="train/epoch")
            self.logger.log(val_metrics, epoch, prefix="val/epoch")
            
            print(f"Epoch {epoch}/{self.config.epochs} - "
                  f"Train Loss: {train_metrics['loss']:.4f}, Train Acc: {train_metrics['accuracy']:.4f}, "
                  f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.4f}, "
                  f"Time: {epoch_time:.1f}s")
            
            self.checkpoint_manager.save(
                self.model, self.optimizer, self.scheduler,
                epoch, self.global_step,
                {"val_loss": val_metrics["loss"], "val_accuracy": val_metrics["accuracy"]},
                asdict(self.config)
            )
            
            if self.early_stopping and self.early_stopping({"val_loss": val_metrics["loss"]}):
                print(f"Early stopping at epoch {epoch}")
                break
        
        self.logger.close()
        return history
    
    def resume_training(self, checkpoint_path: str):
        """Resume training from checkpoint."""
        checkpoint = self.checkpoint_manager.load(
            self.model, self.optimizer, self.scheduler, checkpoint_path
        )
        self.global_step = checkpoint.get("step", 0)
        print(f"Resumed from epoch {checkpoint.get('epoch', 0)}, step {self.global_step}")


# =============================================================================
# Main Entry Point
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Train Transformer Sentiment Classifier")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to configuration file")
    parser.add_argument("--use-pretrained", action="store_true", help="Use pretrained transformer")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--experiment-name", type=str, default=None, help="Override experiment name")
    parser.add_argument("--debug", action="store_true", help="Debug mode with limited samples")
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    if args.experiment_name:
        config["logging"]["experiment_name"] = args.experiment_name
    if args.debug:
        config["data"]["max_samples"] = 1000
        config["training"]["epochs"] = 2
    
    set_seed(config.get("seed", 42))
    device = get_device(config.get("device", "cuda"))
    print(f"Using device: {device}")
    
    # Create dataloaders
    print("\nLoading data...")
    train_loader, val_loader, test_loader, tokenizer = create_dataloaders(config, seed=config.get("seed", 42))
    
    # Create model
    print("\nCreating model...")
    model = create_model(config["model"], use_pretrained=args.use_pretrained)
    
    params = count_parameters(model)
    print(f"Total parameters: {params['total']:,}")
    print(f"Trainable parameters: {params['trainable']:,}")
    
    # Create training config
    training_config = TrainingConfig(
        batch_size=config["training"]["batch_size"],
        learning_rate=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
        epochs=config["training"]["epochs"],
        warmup_steps=config["training"]["warmup_steps"],
        gradient_clip_norm=config["training"]["gradient_clip_norm"],
        label_smoothing=config["training"]["label_smoothing"],
        log_every_n_steps=config["logging"]["log_every_n_steps"],
        save_dir=config["checkpoint"]["save_dir"],
        log_dir=config["logging"]["log_dir"],
        experiment_name=config["logging"]["experiment_name"],
        use_tensorboard=config["logging"]["use_tensorboard"],
        device=str(device),
        seed=config.get("seed", 42)
    )
    
    early_stopping_config = EarlyStoppingConfig(
        patience=config["early_stopping"]["patience"],
        min_delta=config["early_stopping"]["min_delta"],
        monitor=config["early_stopping"]["monitor"],
        mode=config["early_stopping"]["mode"]
    )
    
    # Create trainer
    trainer = Trainer(model, train_loader, val_loader, training_config, early_stopping_config)
    
    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        trainer.resume_training(args.resume)
    
    # Train
    print("\nStarting training...")
    history = trainer.train()
    
    print("\nTraining complete!")
    print(f"Best checkpoint saved to: {config['checkpoint']['save_dir']}/best.pt")
    
    return history


if __name__ == "__main__":
    main()
