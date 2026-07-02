from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Optional

import torch
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

from config import TrainingConfig
from training.callbacks import EarlyStopping
from training.lr_scheduler import CosineWarmupScheduler
from training.metrics import MetricsTracker
from utils.logging import setup_logging


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        train_loader: DataLoader,
        device: Optional[str] = None,
        checkpoint_dir: str | os.PathLike[str] | None = None,
        log_dir: str | os.PathLike[str] | None = None,
        use_tensorboard: bool = False,
        use_wandb: bool = False,
        val_loader: DataLoader | None = None,
        dataset: Dataset | None = None,
    ) -> None:
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.dataset = dataset
        self.device = torch.device(device or config.device)
        self.checkpoint_dir = Path(checkpoint_dir or "checkpoints")
        self.log_dir = Path(log_dir or "logs")
        self.logger = setup_logging(str(self.log_dir))
        self.use_tensorboard = use_tensorboard
        self.use_wandb = use_wandb
        self.model.to(self.device)
        
        # Enable gradient checkpointing if configured
        if getattr(config, "use_gradient_checkpointing", False):
            self._enable_gradient_checkpointing()
        
        # Setup optimizer with parameter groups for better weight decay
        self.optimizer = self._build_optimizer()
        
        # Setup loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Setup mixed precision scaler
        self.scaler = GradScaler(
            enabled=config.use_mixed_precision and self.device.type == "cuda"
        )
        
        # Setup learning rate scheduler
        self.lr_scheduler = self._build_scheduler()
        
        # Setup early stopping
        self.early_stopping = EarlyStopping(
            patience=getattr(config, "early_stopping_patience", 3),
            min_delta=getattr(config, "early_stopping_min_delta", 0.0),
        )
        
        # Training state
        self.best_val_loss: float | None = None
        self.global_step = 0
        self.current_epoch = 0
        self.train_metrics = MetricsTracker()
        self.val_metrics = MetricsTracker()
        
        # Tracking setup
        self._tb_writer: Any | None = None
        self._wandb_run: Any | None = None
        self._setup_tracking()
        
        # Training timing
        self.start_time: float | None = None
        self.epoch_start_time: float | None = None

    def _enable_gradient_checkpointing(self) -> None:
        """Enable gradient checkpointing for memory efficiency."""
        if hasattr(self.model, 'blocks'):
            for block in self.model.blocks:
                if hasattr(block, 'enable_gradient_checkpointing'):
                    block.enable_gradient_checkpointing()
        self.logger.info("Gradient checkpointing enabled")

    def _build_optimizer(self) -> AdamW:
        """Build optimizer with parameter groups for differential learning rates."""
        # Separate parameters for weight decay
        no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
        
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.config.weight_decay,
            },
            {
                "params": [
                    p for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        
        return AdamW(
            optimizer_grouped_parameters,
            lr=self.config.learning_rate,
            betas=(0.9, 0.95),
            eps=1e-8,
        )

    def _build_scheduler(self) -> Any:
        """Build learning rate scheduler with warmup."""
        if self.config.warmup_steps > 0:
            # Use cosine warmup scheduler
            total_steps = getattr(self.config, "max_steps", 1000)
            if total_steps <= 0:
                total_steps = len(self.train_loader) * self.config.epochs
            
            base_scheduler = CosineWarmupScheduler(
                initial_lr=self.config.learning_rate,
                warmup_steps=self.config.warmup_steps,
                total_steps=total_steps,
            )
            
            return torch.optim.lr_scheduler.LambdaLR(
                self.optimizer,
                lr_lambda=base_scheduler.get_lr,
            )
        
        # No warmup, use constant or step scheduler
        return torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=1.0)

    def _setup_tracking(self) -> None:
        """Setup TensorBoard and Weights & Biases tracking."""
        if self.use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self._tb_writer = SummaryWriter(log_dir=str(self.log_dir / "tensorboard"))
                self.logger.info("TensorBoard logging enabled")
            except Exception as e:
                self.logger.warning(f"Failed to initialize TensorBoard: {e}")
                self._tb_writer = None
        
        if self.use_wandb:
            try:
                import wandb
                self._wandb_run = wandb.init(
                    project="llm-from-scratch",
                    dir=str(self.log_dir),
                    config={
                        "model": {
                            "vocab_size": getattr(self.model.config, "vocab_size", None),
                            "d_model": getattr(self.model.config, "d_model", None),
                            "n_layers": getattr(self.model.config, "n_layers", None),
                            "n_heads": getattr(self.model.config, "n_heads", None),
                        },
                        "training": {
                            "batch_size": self.config.batch_size,
                            "learning_rate": self.config.learning_rate,
                            "weight_decay": self.config.weight_decay,
                            "warmup_steps": self.config.warmup_steps,
                            "gradient_clip": self.config.gradient_clip,
                        }
                    }
                )
                self.logger.info("Weights & Biases logging enabled")
            except Exception as e:
                self.logger.warning(f"Failed to initialize Weights & Biases: {e}")
                self._wandb_run = None

    def _forward_step(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Execute a single forward pass."""
        input_ids = batch["input_ids"].to(self.device)
        targets = batch["targets"].to(self.device)
        
        with autocast(enabled=self.scaler.is_enabled()):
            logits = self.model(input_ids, use_cache=False)
            if isinstance(logits, tuple):
                logits = logits[0]
            loss = self.criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        
        return logits, loss

    def _optimizer_step(self, loss: torch.Tensor) -> None:
        """Execute optimizer step with gradient clipping and mixed precision."""
        self.optimizer.zero_grad(set_to_none=True)
        
        if self.scaler.is_enabled():
            # Mixed precision training
            self.scaler.scale(loss).backward()
            
            # Unscale gradients for clipping
            if self.config.gradient_clip > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.gradient_clip
                )
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            # Standard training
            loss.backward()
            
            # Gradient clipping
            if self.config.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.gradient_clip
                )
            
            self.optimizer.step()

    def train(self) -> None:
        """Main training loop."""
        self.model.train()
        self.start_time = time.time()
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(
            "Starting training: epochs=%d, batch_size=%d, device=%s",
            self.config.epochs,
            self.config.batch_size,
            self.device,
        )
        
        for epoch in range(self.config.epochs):
            self.current_epoch = epoch
            self.epoch_start_time = time.time()
            self.train_metrics = MetricsTracker()
            
            # Update curriculum sampler if used
            if hasattr(self.train_loader.sampler, 'set_epoch'):
                self.train_loader.sampler.set_epoch(epoch)
            
            for batch_idx, batch in enumerate(self.train_loader):
                # Handle different batch formats
                if isinstance(batch, dict):
                    batch_dict = batch
                else:
                    batch_dict = {"input_ids": batch, "targets": batch[:, 1:]}
                
                # Forward pass
                logits, loss = self._forward_step(batch_dict)
                
                # Backward pass
                self._optimizer_step(loss)
                
                # Update learning rate
                self.lr_scheduler.step()
                
                # Update metrics
                self.global_step += 1
                self.train_metrics.update(loss)
                
                # Logging
                if self.global_step % 10 == 0:
                    self._log_training_step(loss, batch_idx)
                
                # Validation
                if self.val_loader is not None and self.global_step % getattr(
                    self.config, "val_every", 100
                ) == 0:
                    val_loss = self.validate()
                    self._check_early_stopping(val_loss)
                    if self.early_stopping.stopped:
                        self.logger.info("Early stopping triggered")
                        return
                
                # Checkpointing
                if self.global_step % self.config.checkpoint_every == 0:
                    self.save_checkpoint(self.global_step)
                
                # Max steps check
                if self.global_step >= getattr(self.config, "max_steps", float("inf")):
                    self.logger.info(f"Reached max steps: {self.global_step}")
                    self.save_checkpoint("latest")
                    return
            
            # End of epoch validation
            if self.val_loader is not None:
                val_loss = self.validate()
                self._check_early_stopping(val_loss)
                if self.early_stopping.stopped:
                    self.logger.info("Early stopping triggered")
                    break
            
            # Log epoch summary
            self._log_epoch_summary()
            
            # Max steps check
            if self.global_step >= getattr(self.config, "max_steps", float("inf")):
                break
        
        # Save final checkpoint
        self.save_checkpoint("latest")
        self.logger.info("Training complete")

    def _log_training_step(self, loss: torch.Tensor, batch_idx: int) -> None:
        """Log training step metrics."""
        elapsed = time.time() - (self.epoch_start_time or time.time())
        lr = self.optimizer.param_groups[0]["lr"]
        
        self.logger.info(
            "epoch=%d/%d step=%d batch=%d loss=%.4f lr=%.6f time=%.2fs",
            self.current_epoch + 1,
            self.config.epochs,
            self.global_step,
            batch_idx + 1,
            float(loss.item()),
            lr,
            elapsed,
        )
        
        # TensorBoard logging
        if self._tb_writer is not None:
            self._tb_writer.add_scalar("train/loss", float(loss.item()), self.global_step)
            self._tb_writer.add_scalar("train/lr", lr, self.global_step)
            self._tb_writer.add_scalar("train/perplexity", self.train_metrics.perplexity, self.global_step)
        
        # Weights & Biases logging
        if self._wandb_run is not None:
            self._wandb_run.log({
                "train/loss": float(loss.item()),
                "train/lr": lr,
                "train/perplexity": self.train_metrics.perplexity,
                "step": self.global_step,
            })

    def _log_epoch_summary(self) -> None:
        """Log epoch summary metrics."""
        elapsed = time.time() - (self.start_time or time.time())
        train_loss = self.train_metrics.mean
        train_ppl = self.train_metrics.perplexity
        
        self.logger.info(
            "Epoch %d/%d complete - train_loss=%.4f train_ppl=%.2f elapsed=%.2fs",
            self.current_epoch + 1,
            self.config.epochs,
            train_loss,
            train_ppl,
            elapsed,
        )

    def _check_early_stopping(self, val_loss: float) -> None:
        """Check early stopping condition."""
        if self.best_val_loss is None or val_loss < self.best_val_loss - getattr(
            self.config, "early_stopping_min_delta", 0.0
        ):
            self.best_val_loss = val_loss
            # Save best model
            self.save_checkpoint("best")
        
        if self.early_stopping.update(val_loss):
            self.logger.info(
                "Early stopping triggered: best_val_loss=%.4f current_val_loss=%.4f",
                self.best_val_loss,
                val_loss,
            )

    def validate(self) -> float:
        """Run validation loop."""
        if self.val_loader is None:
            return float("inf")
        
        self.model.eval()
        self.val_metrics = MetricsTracker()
        total_loss = 0.0
        count = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                # Handle different batch formats
                if isinstance(batch, dict):
                    batch_dict = batch
                else:
                    batch_dict = {"input_ids": batch, "targets": batch[:, 1:]}
                
                _, loss = self._forward_step(batch_dict)
                total_loss += float(loss.item())
                self.val_metrics.update(loss)
                count += 1
        
        self.model.train()
        
        if count == 0:
            return float("inf")
        
        avg_loss = total_loss / count
        
        # Log validation metrics
        self.logger.info(
            "Validation - step=%d loss=%.4f perplexity=%.2f",
            self.global_step,
            avg_loss,
            self.val_metrics.perplexity,
        )
        
        # TensorBoard logging
        if self._tb_writer is not None:
            self._tb_writer.add_scalar("val/loss", avg_loss, self.global_step)
            self._tb_writer.add_scalar("val/perplexity", self.val_metrics.perplexity, self.global_step)
        
        # Weights & Biases logging
        if self._wandb_run is not None:
            self._wandb_run.log({
                "val/loss": avg_loss,
                "val/perplexity": self.val_metrics.perplexity,
                "step": self.global_step,
            })
        
        return avg_loss

    def save_checkpoint(self, step: int | str) -> None:
        """Save model checkpoint."""
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.lr_scheduler.state_dict(),
            "step": self.global_step,
            "epoch": self.current_epoch,
            "best_val_loss": self.best_val_loss,
            "config": self.config,
        }
        
        # Save numbered checkpoint
        if isinstance(step, int):
            path = self.checkpoint_dir / f"checkpoint_{step}.pt"
            torch.save(checkpoint, path)
        
        # Save latest checkpoint
        latest_path = self.checkpoint_dir / "checkpoint_latest.pt"
        torch.save(checkpoint, latest_path)
        
        # Save best checkpoint
        if step == "best":
            best_path = self.checkpoint_dir / "checkpoint_best.pt"
            torch.save(checkpoint, best_path)
        
        self.logger.info("Checkpoint saved: %s", step)

    def load_checkpoint(self, path: str | os.PathLike[str]) -> None:
        """Load model checkpoint and resume training."""
        path = Path(path)
        if not path.exists():
            self.logger.warning("Checkpoint not found: %s", path)
            return
        
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        # Load model state
        self.model.load_state_dict(checkpoint["model_state"])
        
        # Load optimizer state
        if "optimizer_state" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        
        # Load scheduler state
        if "scheduler_state" in checkpoint:
            self.lr_scheduler.load_state_dict(checkpoint["scheduler_state"])
        
        # Restore training state
        self.global_step = int(checkpoint.get("step", 0))
        self.current_epoch = int(checkpoint.get("epoch", 0))
        self.best_val_loss = checkpoint.get("best_val_loss")
        
        self.logger.info(
            "Resumed from checkpoint: step=%d epoch=%d best_val_loss=%s",
            self.global_step,
            self.current_epoch,
            self.best_val_loss,
        )

    def cleanup(self) -> None:
        """Cleanup resources."""
        if self._tb_writer is not None:
            self._tb_writer.close()
        
        if self._wandb_run is not None:
            self._wandb_run.finish()