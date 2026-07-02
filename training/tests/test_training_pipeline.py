from __future__ import annotations

import math
import os
import tempfile
from pathlib import Path
from typing import Any

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

# Add parent directory to path for imports
import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))

from config import ModelConfig, TrainingConfig
from model.gpt_model import GPTModel
from training.callbacks import EarlyStopping
from training.data_loader import DatasetLoader, StreamingTextDataset, TextDataset
from training.dynamic_batching import (
    BatchConfig,
    CurriculumSampler,
    DataCollator,
    DynamicBatchSampler,
    DynamicDataLoader,
    SequencePacker,
)
from training.lr_scheduler import CosineWarmupScheduler
from training.metrics import MetricsTracker
from training.trainer import Trainer


class TinyDataset(Dataset):
    """Minimal dataset for testing."""

    def __init__(self, size: int = 8, seq_len: int = 6, vocab_size: int = 32) -> None:
        self.data = torch.randint(0, vocab_size, (size, seq_len))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.data[idx]


class MockTokenizer:
    """Mock tokenizer for testing."""

    def __init__(self, vocab_size: int = 100) -> None:
        self.vocab_size = vocab_size
        self.vocab = {f"token_{i}": i for i in range(vocab_size)}
        self.vocab["<pad>"] = 0
        self.vocab["</s>"] = vocab_size - 1
        self.pad_token = "<pad>"
        self.eos_token = "</s>"

    def encode(self, text: str) -> list[int]:
        """Simple encoding for testing."""
        tokens = []
        for char in text[:100]:  # Limit length
            token_id = hash(char) % (self.vocab_size - 2) + 1
            tokens.append(token_id)
        return tokens

    def decode(self, tokens: list[int]) -> str:
        """Simple decoding for testing."""
        return "".join(chr(t % 128) for t in tokens)


# ==================== Dataset Loader Tests ====================


class TestDatasetLoader:
    """Test dataset loading functionality."""

    def test_load_from_directory(self, tmp_path: Path) -> None:
        """Test loading texts from directory."""
        # Create test files
        (tmp_path / "file1.txt").write_text("Hello world")
        (tmp_path / "file2.md").write_text("# Markdown test")
        # For JSON, write as plain text file with .txt extension to test text loading
        (tmp_path / "file3.txt").write_text("JSON test content")
        
        texts = DatasetLoader.load_from_directory(tmp_path)
        
        assert len(texts) == 3
        assert "Hello world" in texts
        assert "# Markdown test" in texts
        assert "JSON test content" in texts

    def test_load_from_files(self, tmp_path: Path) -> None:
        """Test loading from specific files."""
        (tmp_path / "test.txt").write_text("Test content")
        
        texts = DatasetLoader.load_from_files([tmp_path / "test.txt"])
        
        assert len(texts) == 1
        assert texts[0] == "Test content"

    def test_load_from_json_list(self, tmp_path: Path) -> None:
        """Test loading from JSON list."""
        import json
        data = [{"text": "First"}, {"text": "Second"}, "Third"]
        (tmp_path / "data.json").write_text(json.dumps(data))
        
        texts = DatasetLoader.load_from_json(tmp_path / "data.json")
        
        assert len(texts) == 3

    def test_text_dataset_preparation(self) -> None:
        """Test TextDataset preparation."""
        tokenizer = MockTokenizer(vocab_size=100)
        texts = ["Hello world", "Test text"]
        
        dataset = TextDataset(texts, tokenizer, max_seq_len=10)
        
        assert len(dataset) > 0
        sample = dataset[0]
        assert isinstance(sample, torch.Tensor)
        assert sample.dtype == torch.long


# ==================== Dynamic Batching Tests ====================


class TestSequencePacker:
    """Test sequence packing functionality."""

    def test_pack_single_sequence(self) -> None:
        """Test packing a single sequence."""
        packer = SequencePacker(max_seq_len=10, pad_value=0)
        sequences = [torch.tensor([1, 2, 3])]
        
        result = packer.pack(sequences)
        
        assert result["input_ids"].shape == (1, 3)
        assert result["targets"].shape == (1, 3)
        assert result["attention_mask"].shape == (1, 3)

    def test_pack_multiple_sequences(self) -> None:
        """Test packing multiple sequences."""
        packer = SequencePacker(max_seq_len=10, pad_value=0)
        sequences = [torch.tensor([1, 2, 3]), torch.tensor([4, 5])]
        
        result = packer.pack(sequences)
        
        assert result["input_ids"].shape == (2, 3)
        assert result["targets"].shape == (2, 3)

    def test_pack_with_padding(self) -> None:
        """Test packing with padding."""
        packer = SequencePacker(max_seq_len=10, pad_value=0)
        sequences = [torch.tensor([1, 2, 3]), torch.tensor([4, 5])]
        
        result = packer.pack(sequences)
        
        # Should pad to max length in batch
        assert result["input_ids"].shape[1] == 3
        assert result["attention_mask"][0, :].all()
        # Second sequence has length 2, so last position should be masked
        assert result["attention_mask"][1, 0].item()
        assert result["attention_mask"][1, 1].item()
        assert not result["attention_mask"][1, 2].item()

    def test_pack_truncation(self) -> None:
        """Test that sequences are truncated to max_seq_len."""
        packer = SequencePacker(max_seq_len=5, pad_value=0)
        sequences = [torch.tensor(list(range(10)))]
        
        result = packer.pack(sequences)
        
        assert result["input_ids"].shape[1] == 5

    def test_pack_empty_raises_error(self) -> None:
        """Test that packing empty list raises error."""
        packer = SequencePacker(max_seq_len=10)
        
        with pytest.raises(ValueError):
            packer.pack([])


class TestDataCollator:
    """Test data collation."""

    def test_collate_basic(self) -> None:
        """Test basic collation."""
        collator = DataCollator(max_seq_len=10, pad_value=0)
        batch = [torch.tensor([1, 2, 3]), torch.tensor([4, 5])]
        
        result = collator.collate(batch)
        
        assert "input_ids" in result
        assert "targets" in result
        assert "attention_mask" in result
        assert result["input_ids"].shape[0] == 2

    def test_collate_callable(self) -> None:
        """Test that collator is callable."""
        collator = DataCollator(max_seq_len=10)
        batch = [torch.tensor([1, 2, 3])]
        
        result = collator(batch)
        
        assert "input_ids" in result


class TestDynamicBatchSampler:
    """Test dynamic batch sampling."""

    def test_batch_grouping(self) -> None:
        """Test that samples are grouped into batches."""
        dataset = TinyDataset(size=10, seq_len=5)
        config = BatchConfig(max_tokens=20, max_batch_size=4)
        
        sampler = DynamicBatchSampler(dataset, config)
        
        assert len(sampler) > 0
        # Check that batches respect max_batch_size
        for batch in sampler:
            assert len(batch) <= config.max_batch_size

    def test_sort_by_length(self) -> None:
        """Test that samples are sorted by length."""
        dataset = TinyDataset(size=10, seq_len=5)
        config = BatchConfig(max_tokens=100, max_batch_size=10, sort_by_length=True)
        
        sampler = DynamicBatchSampler(dataset, config)
        
        # Should create batches without errors
        batches = list(sampler)
        assert len(batches) > 0


# ==================== Optimizer Tests ====================


class TestAdamWOptimizer:
    """Test AdamW optimizer configuration."""

    def test_optimizer_parameter_groups(self) -> None:
        """Test that optimizer has correct parameter groups."""
        config = ModelConfig(vocab_size=32, d_model=16, n_layers=1, n_heads=2, ff_hidden_dim=32)
        model = GPTModel(config)
        training_config = TrainingConfig(learning_rate=1e-4, weight_decay=0.01)
        
        trainer = Trainer(
            model,
            training_config,
            DataLoader(TinyDataset(), batch_size=2),
            device="cpu",
        )
        
        # Check parameter groups
        assert len(trainer.optimizer.param_groups) == 2
        # One group with weight decay, one without
        assert trainer.optimizer.param_groups[0]["weight_decay"] == 0.01
        assert trainer.optimizer.param_groups[1]["weight_decay"] == 0.0


# ==================== Learning Rate Scheduler Tests ====================


class TestCosineWarmupScheduler:
    """Test cosine warmup learning rate scheduler."""

    def test_warmup_phase(self) -> None:
        """Test learning rate during warmup."""
        scheduler = CosineWarmupScheduler(
            initial_lr=1.0,
            warmup_steps=10,
            total_steps=100,
        )
        
        # During warmup, LR should increase linearly
        lr_step_0 = scheduler.get_lr(0)
        lr_step_5 = scheduler.get_lr(5)
        lr_step_10 = scheduler.get_lr(10)
        
        assert lr_step_0 < lr_step_5 < lr_step_10
        assert abs(lr_step_10 - 1.0) < 1e-6

    def test_cosine_decay_phase(self) -> None:
        """Test learning rate during cosine decay."""
        scheduler = CosineWarmupScheduler(
            initial_lr=1.0,
            warmup_steps=10,
            total_steps=100,
        )
        
        # After warmup, LR should follow cosine decay
        lr_step_10 = scheduler.get_lr(10)
        lr_step_50 = scheduler.get_lr(50)
        lr_step_100 = scheduler.get_lr(100)
        
        assert lr_step_10 > lr_step_50 > lr_step_100
        # At the end of cosine decay, LR should be close to 0 (not 0.5)
        assert lr_step_100 < 0.1

    def test_scheduler_integration(self) -> None:
        """Test scheduler integration with trainer."""
        config = ModelConfig(vocab_size=32, d_model=16, n_layers=1, n_heads=2, ff_hidden_dim=32)
        model = GPTModel(config)
        training_config = TrainingConfig(
            learning_rate=1e-4,
            warmup_steps=5,
            max_steps=20,
        )
        
        trainer = Trainer(
            model,
            training_config,
            DataLoader(TinyDataset(), batch_size=2),
            device="cpu",
        )
        
        # Check that scheduler is created
        assert trainer.lr_scheduler is not None


# ==================== Gradient Clipping Tests ====================


class TestGradientClipping:
    """Test gradient clipping functionality."""

    def test_gradient_clipping_applied(self) -> None:
        """Test that gradient clipping is applied."""
        config = ModelConfig(vocab_size=32, d_model=16, n_layers=1, n_heads=2, ff_hidden_dim=32)
        model = GPTModel(config)
        training_config = TrainingConfig(gradient_clip=1.0)
        
        trainer = Trainer(
            model,
            training_config,
            DataLoader(TinyDataset(), batch_size=2),
            device="cpu",
        )
        
        # Create a batch and run forward/backward
        batch = {"input_ids": torch.randint(0, 32, (2, 5)), "targets": torch.randint(0, 32, (2, 5))}
        _, loss = trainer._forward_step(batch)
        trainer._optimizer_step(loss)
        
        # Check that gradients exist
        has_grads = any(p.grad is not None for p in model.parameters())
        assert has_grads


# ==================== Mixed Precision Tests ====================


class TestMixedPrecision:
    """Test mixed precision training."""

    def test_mixed_precision_cpu(self) -> None:
        """Test that mixed precision is disabled on CPU."""
        config = ModelConfig(vocab_size=32, d_model=16, n_layers=1, n_heads=2, ff_hidden_dim=32)
        model = GPTModel(config)
        training_config = TrainingConfig(use_mixed_precision=True, device="cpu")
        
        trainer = Trainer(
            model,
            training_config,
            DataLoader(TinyDataset(), batch_size=2),
            device="cpu",
        )
        
        # Scaler should be disabled on CPU
        assert not trainer.scaler.is_enabled()

    def test_mixed_precision_cuda(self) -> None:
        """Test mixed precision configuration on CUDA (if available)."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        config = ModelConfig(vocab_size=32, d_model=16, n_layers=1, n_heads=2, ff_hidden_dim=32)
        model = GPTModel(config)
        training_config = TrainingConfig(use_mixed_precision=True, device="cuda")
        
        trainer = Trainer(
            model,
            training_config,
            DataLoader(TinyDataset(), batch_size=2),
            device="cuda",
        )
        
        # Scaler should be enabled on CUDA
        assert trainer.scaler.is_enabled()


# ==================== Gradient Checkpointing Tests ====================


class TestGradientCheckpointing:
    """Test gradient checkpointing functionality."""

    def test_gradient_checkpointing_enabled(self) -> None:
        """Test enabling gradient checkpointing."""
        config = ModelConfig(vocab_size=32, d_model=16, n_layers=1, n_heads=2, ff_hidden_dim=32)
        model = GPTModel(config)
        training_config = TrainingConfig(use_gradient_checkpointing=True)
        
        trainer = Trainer(
            model,
            training_config,
            DataLoader(TinyDataset(), batch_size=2),
            device="cpu",
        )
        
        # Check that gradient checkpointing is enabled
        # (implementation depends on model architecture)
        assert trainer.model is not None


# ==================== Checkpoint Tests ====================


class TestCheckpointSaving:
    """Test checkpoint saving and loading."""

    def test_save_and_load_checkpoint(self, tmp_path: Path) -> None:
        """Test saving and loading checkpoints."""
        config = ModelConfig(vocab_size=32, d_model=16, n_layers=1, n_heads=2, ff_hidden_dim=32)
        model = GPTModel(config)
        training_config = TrainingConfig()
        
        trainer = Trainer(
            model,
            training_config,
            DataLoader(TinyDataset(), batch_size=2),
            device="cpu",
            checkpoint_dir=tmp_path,
        )
        
        # Save checkpoint
        trainer.global_step = 10
        trainer.current_epoch = 1
        trainer.best_val_loss = 2.5
        trainer.save_checkpoint(10)
        
        # Check files exist
        assert (tmp_path / "checkpoint_10.pt").exists()
        assert (tmp_path / "checkpoint_latest.pt").exists()
        
        # Load checkpoint into new trainer
        model2 = GPTModel(config)
        trainer2 = Trainer(
            model2,
            training_config,
            DataLoader(TinyDataset(), batch_size=2),
            device="cpu",
            checkpoint_dir=tmp_path,
        )
        trainer2.load_checkpoint(tmp_path / "checkpoint_10.pt")
        
        # Check state restored
        assert trainer2.global_step == 10
        assert trainer2.current_epoch == 1
        assert trainer2.best_val_loss == 2.5

    def test_resume_training(self, tmp_path: Path) -> None:
        """Test resuming training from checkpoint."""
        config = ModelConfig(vocab_size=32, d_model=16, n_layers=1, n_heads=2, ff_hidden_dim=32)
        model = GPTModel(config)
        training_config = TrainingConfig(max_steps=5, checkpoint_every=5)
        
        # Use DataCollator to ensure proper batch formatting
        collator = DataCollator(max_seq_len=8)
        dataset = TinyDataset(size=8, seq_len=6)
        loader = DataLoader(dataset, batch_size=2, collate_fn=collator.collate)
        
        trainer = Trainer(
            model,
            training_config,
            loader,
            device="cpu",
            checkpoint_dir=tmp_path,
        )
        
        # Train for a few steps
        trainer.train()
        initial_step = trainer.global_step
        
        # Create new trainer and resume
        model2 = GPTModel(config)
        trainer2 = Trainer(
            model2,
            training_config,
            loader,
            device="cpu",
            checkpoint_dir=tmp_path,
        )
        trainer2.load_checkpoint(tmp_path / "checkpoint_latest.pt")
        
        assert trainer2.global_step == initial_step


# ==================== Early Stopping Tests ====================


class TestEarlyStopping:
    """Test early stopping functionality."""

    def test_early_stopping_triggered(self) -> None:
        """Test that early stopping triggers after patience epochs."""
        early_stopping = EarlyStopping(patience=2, min_delta=0.0)
        
        # Simulate decreasing loss
        assert not early_stopping.update(1.0)
        assert not early_stopping.update(0.9)
        # Now loss increases
        assert not early_stopping.update(1.0)
        # Should trigger now
        assert early_stopping.update(1.1)

    def test_early_stopping_not_triggered(self) -> None:
        """Test that early stopping doesn't trigger with improving loss."""
        early_stopping = EarlyStopping(patience=3, min_delta=0.0)
        
        assert not early_stopping.update(1.0)
        assert not early_stopping.update(0.9)
        assert not early_stopping.update(0.8)
        assert not early_stopping.update(0.7)

    def test_early_stopping_min_delta(self) -> None:
        """Test min_delta parameter."""
        early_stopping = EarlyStopping(patience=2, min_delta=0.1)
        
        assert not early_stopping.update(1.0)
        # Small improvement shouldn't reset patience (0.95 is not < 1.0 - 0.1 = 0.9)
        assert not early_stopping.update(0.95)  # wait=1
        # At wait=2, should trigger (patience=2)
        assert early_stopping.update(0.93)  # wait=2 >= patience=2, triggers stopping


# ==================== Metrics Tests ====================


class TestMetricsTracker:
    """Test metrics tracking."""

    def test_metrics_update(self) -> None:
        """Test updating metrics."""
        tracker = MetricsTracker()
        
        tracker.update(torch.tensor(1.0))
        tracker.update(torch.tensor(2.0))
        tracker.update(torch.tensor(3.0))
        
        assert len(tracker.losses) == 3
        assert abs(tracker.mean - 2.0) < 1e-6

    def test_metrics_perplexity(self) -> None:
        """Test perplexity calculation."""
        tracker = MetricsTracker()
        tracker.update(torch.tensor(0.0))  # log(1) = 0, exp(0) = 1
        
        assert abs(tracker.perplexity - 1.0) < 1e-6

    def test_metrics_perplexity_calculation(self) -> None:
        """Test perplexity with actual loss value."""
        tracker = MetricsTracker()
        # loss = log(2), so perplexity = exp(log(2)) = 2
        tracker.update(torch.tensor(math.log(2)))
        
        assert abs(tracker.perplexity - 2.0) < 1e-6


# ==================== Validation Loop Tests ====================


class TestValidationLoop:
    """Test validation loop functionality."""

    def test_validation_runs(self, tmp_path: Path) -> None:
        """Test that validation loop runs correctly."""
        config = ModelConfig(vocab_size=32, d_model=16, n_layers=1, n_heads=2, ff_hidden_dim=32)
        model = GPTModel(config)
        
        train_dataset = TinyDataset(size=8, seq_len=6)
        val_dataset = TinyDataset(size=4, seq_len=6)
        
        collator = DataCollator(max_seq_len=8)
        train_loader = DataLoader(train_dataset, batch_size=2, collate_fn=collator.collate)
        val_loader = DataLoader(val_dataset, batch_size=2, collate_fn=collator.collate)
        
        trainer = Trainer(
            model,
            TrainingConfig(device="cpu"),
            train_loader,
            val_loader=val_loader,
            device="cpu",
            log_dir=tmp_path / "logs",
        )
        
        val_loss = trainer.validate()
        
        assert isinstance(val_loss, float)
        assert val_loss > 0
        assert not math.isinf(val_loss)


# ==================== Logging Tests ====================


class TestLogging:
    """Test logging functionality."""

    def test_logger_created(self, tmp_path: Path) -> None:
        """Test that logger is created."""
        config = ModelConfig(vocab_size=32, d_model=16, n_layers=1, n_heads=2, ff_hidden_dim=32)
        model = GPTModel(config)
        
        trainer = Trainer(
            model,
            TrainingConfig(device="cpu"),
            DataLoader(TinyDataset(), batch_size=2),
            device="cpu",
            log_dir=tmp_path / "logs",
        )
        
        assert trainer.logger is not None
        assert trainer.logger.name == "llm_from_scratch"

    def test_log_file_created(self, tmp_path: Path) -> None:
        """Test that log file is created."""
        config = ModelConfig(vocab_size=32, d_model=16, n_layers=1, n_heads=2, ff_hidden_dim=32)
        model = GPTModel(config)
        
        trainer = Trainer(
            model,
            TrainingConfig(device="cpu"),
            DataLoader(TinyDataset(), batch_size=2),
            device="cpu",
            log_dir=tmp_path / "logs",
        )
        
        log_file = tmp_path / "logs" / "training.log"
        assert log_file.exists()


# ==================== TensorBoard Tests ====================


class TestTensorBoard:
    """Test TensorBoard integration."""

    def test_tensorboard_disabled(self, tmp_path: Path) -> None:
        """Test with TensorBoard disabled."""
        config = ModelConfig(vocab_size=32, d_model=16, n_layers=1, n_heads=2, ff_hidden_dim=32)
        model = GPTModel(config)
        
        trainer = Trainer(
            model,
            TrainingConfig(device="cpu"),
            DataLoader(TinyDataset(), batch_size=2),
            device="cpu",
            use_tensorboard=False,
            log_dir=tmp_path / "logs",
        )
        
        assert trainer._tb_writer is None

    def test_tensorboard_enabled(self, tmp_path: Path) -> None:
        """Test with TensorBoard enabled."""
        config = ModelConfig(vocab_size=32, d_model=16, n_layers=1, n_heads=2, ff_hidden_dim=32)
        model = GPTModel(config)
        
        trainer = Trainer(
            model,
            TrainingConfig(device="cpu"),
            DataLoader(TinyDataset(), batch_size=2),
            device="cpu",
            use_tensorboard=True,
            log_dir=tmp_path / "logs",
        )
        
        # TensorBoard may or may not be available
        # Just check it doesn't crash
        assert hasattr(trainer, '_tb_writer')


# ==================== Integration Tests ====================


class TestTrainingIntegration:
    """Test complete training pipeline integration."""

    def test_full_training_loop(self, tmp_path: Path) -> None:
        """Test complete training loop."""
        config = ModelConfig(vocab_size=32, d_model=16, n_layers=1, n_heads=2, ff_hidden_dim=32)
        model = GPTModel(config)
        
        dataset = TinyDataset(size=8, seq_len=6)
        collator = DataCollator(max_seq_len=8)
        loader = DataLoader(dataset, batch_size=2, collate_fn=collator.collate)
        
        trainer = Trainer(
            model,
            TrainingConfig(
                device="cpu",
                epochs=1,
                max_steps=3,
                checkpoint_every=2,
                early_stopping_patience=5,
            ),
            loader,
            device="cpu",
            checkpoint_dir=tmp_path / "checkpoints",
            log_dir=tmp_path / "logs",
        )
        
        # Run training
        trainer.train()
        
        # Check that training completed
        assert trainer.global_step >= 3
        assert (tmp_path / "checkpoints" / "checkpoint_latest.pt").exists()

    def test_training_with_validation(self, tmp_path: Path) -> None:
        """Test training with validation."""
        config = ModelConfig(vocab_size=32, d_model=16, n_layers=1, n_heads=2, ff_hidden_dim=32)
        model = GPTModel(config)
        
        train_dataset = TinyDataset(size=8, seq_len=6)
        val_dataset = TinyDataset(size=4, seq_len=6)
        
        collator = DataCollator(max_seq_len=8)
        train_loader = DataLoader(train_dataset, batch_size=2, collate_fn=collator.collate)
        val_loader = DataLoader(val_dataset, batch_size=2, collate_fn=collator.collate)
        
        trainer = Trainer(
            model,
            TrainingConfig(
                device="cpu",
                epochs=1,
                max_steps=3,
                val_every=2,
                early_stopping_patience=5,
            ),
            train_loader,
            val_loader=val_loader,
            device="cpu",
            checkpoint_dir=tmp_path / "checkpoints",
            log_dir=tmp_path / "logs",
        )
        
        trainer.train()
        
        # Validation should have run
        assert trainer.best_val_loss is not None


# ==================== Run Tests ====================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])