# Production Training Pipeline

## Overview

This document describes the complete production training pipeline implemented for the GPT model from scratch. The pipeline includes all essential components for training large language models in production.

## Components Implemented

### 1. Dataset Loader (`training/data_loader.py`)

**Features:**
- **TextDataset**: Standard dataset with tokenization and chunking
- **StreamingTextDataset**: Memory-efficient streaming for large corpora
- **DatasetLoader**: Utility class for loading from multiple sources
  - `load_from_directory()`: Load all text files from a directory
  - `load_from_files()`: Load from specific file paths
  - `load_from_json()`: Load from JSON files with text extraction

**Key Features:**
- Supports multiple file formats (.txt, .md, .json, .csv)
- Overlapping chunks with configurable stride
- EOS token handling
- Minimum length filtering

### 2. Dynamic Batching (`training/dynamic_batching.py`)

**Features:**
- **BatchConfig**: Configuration for batching parameters
- **DynamicBatchSampler**: Groups samples by length to minimize padding
- **DynamicDataLoader**: DataLoader with dynamic batching support
- **SequencePacker**: Efficiently packs sequences with minimal padding
- **DataCollator**: Collates batches with proper formatting
- **CurriculumSampler**: Curriculum learning support (starts with shorter sequences)

**Key Features:**
- Sorts by length for efficient batching
- Respects max_tokens and max_batch_size constraints
- Automatic padding and attention mask generation
- Curriculum learning for progressive training

### 3. Data Collator (`training/dynamic_batching.py`)

**Features:**
- Packs variable-length sequences into batches
- Creates input_ids, targets, and attention_mask
- Handles truncation to max_seq_len
- Proper padding with configurable pad_value

### 4. AdamW Optimizer (`training/trainer.py`)

**Features:**
- Parameter groups for differential weight decay
- No weight decay for bias and LayerNorm parameters
- Configurable betas (0.9, 0.95) and epsilon (1e-8)
- Proper weight decay handling

**Implementation:**
```python
no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
optimizer_grouped_parameters = [
    {"params": [p for n, p in model.named_parameters() 
                if not any(nd in n for nd in no_decay)], 
     "weight_decay": weight_decay},
    {"params": [p for n, p in model.named_parameters() 
                if any(nd in n for nd in no_decay)], 
     "weight_decay": 0.0},
]
```

### 5. Learning Rate Scheduler (`training/lr_scheduler.py`)

**Features:**
- **CosineWarmupScheduler**: Cosine annealing with linear warmup
- Configurable warmup steps and total steps
- Smooth transition from warmup to decay
- Integrated with PyTorch LambdaLR

**Schedule:**
- Linear warmup for first `warmup_steps`
- Cosine decay for remaining steps
- Ends at ~0.5 * initial_lr

### 6. Warmup Scheduler

**Implementation:**
- Integrated into CosineWarmupScheduler
- Linear increase from 0 to initial_lr during warmup
- Prevents early training instability

### 7. Gradient Clipping (`training/trainer.py`)

**Features:**
- Configurable gradient clipping threshold
- Applied after unscaling for mixed precision
- Uses `torch.nn.utils.clip_grad_norm_`
- Prevents exploding gradients

### 8. Mixed Precision Training (`training/trainer.py`)

**Features:**
- Automatic mixed precision with GradScaler
- Only enabled on CUDA devices
- Proper gradient unscaling before clipping
- Compatible with gradient checkpointing

**Implementation:**
```python
self.scaler = GradScaler(
    enabled=config.use_mixed_precision and self.device.type == "cuda"
)
```

### 9. Gradient Checkpointing (`training/trainer.py`)

**Features:**
- Memory-efficient training
- Trades compute for memory
- Configurable via `use_gradient_checkpointing`
- Integrated with model architecture

### 10. Checkpoint Saving (`training/trainer.py`)

**Features:**
- Saves model state, optimizer state, scheduler state
- Multiple checkpoint types:
  - Numbered checkpoints (checkpoint_100.pt)
  - Latest checkpoint (checkpoint_latest.pt)
  - Best checkpoint (checkpoint_best.pt)
- Includes training state (step, epoch, best_val_loss)

### 11. Resume Training (`training/trainer.py`)

**Features:**
- Loads model, optimizer, and scheduler states
- Restores training state (global_step, epoch, best_val_loss)
- Continues training from exact point of interruption
- Validates checkpoint existence before loading

### 12. TensorBoard Integration (`training/trainer.py`)

**Features:**
- Logs training loss, learning rate, perplexity
- Logs validation loss and perplexity
- Automatic directory creation
- Graceful fallback if TensorBoard not installed

**Metrics Logged:**
- train/loss
- train/lr
- train/perplexity
- val/loss
- val/perplexity

### 13. Weights & Biases Integration (`training/trainer.py`)

**Features:**
- Optional W&B logging
- Logs same metrics as TensorBoard
- Configures project and hyperparameters
- Graceful fallback if W&B not installed

### 14. Early Stopping (`training/callbacks.py`)

**Features:**
- Configurable patience
- Min delta for meaningful improvement
- Tracks best score
- Prevents overfitting

**Implementation:**
```python
class EarlyStopping:
    def __init__(self, patience: int = 3, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.wait = 0
        self.stopped = False
```

### 15. Validation Loop (`training/trainer.py`)

**Features:**
- Runs at configurable intervals (val_every)
- Computes average validation loss
- Tracks validation perplexity
- Updates best model checkpoint
- Logs metrics to all tracking systems

### 16. Metrics (`training/metrics.py`)

**Features:**
- **MetricsTracker**: Tracks loss values
- Computes mean loss
- Calculates perplexity (exp(mean_loss))
- Memory-efficient list-based tracking

### 17. Logging (`utils/logging.py`)

**Features:**
- Dual logging (file and console)
- Timestamped log entries
- Configurable log levels
- Automatic directory creation
- UTF-8 encoding for international characters

## Configuration

### TrainingConfig

```python
@dataclass
class TrainingConfig:
    batch_size: int = 8
    max_steps: int = 200
    epochs: int = 5
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    gradient_clip: float = 1.0
    warmup_steps: int = 20
    device: str = "cpu"
    use_mixed_precision: bool = False
    use_gradient_checkpointing: bool = False
    checkpoint_every: int = 50
    val_every: int = 100
    early_stopping_patience: int = 3
    early_stopping_min_delta: float = 0.0
```

## Usage

### Basic Training

```python
from config import ModelConfig, TokenizerConfig, TrainingConfig
from model.gpt_model import GPTModel
from tokenizer.base import BPETokenizer
from training.data_loader import TextDataset
from training.dynamic_batching import DataCollator
from training.trainer import Trainer
from torch.utils.data import DataLoader

# Configuration
model_config = ModelConfig(...)
training_config = TrainingConfig(...)

# Data preparation
tokenizer = BPETokenizer(vocab_size=2000)
dataset = TextDataset(texts, tokenizer, max_seq_len=256)
collator = DataCollator(max_seq_len=256)
loader = DataLoader(dataset, batch_size=8, collate_fn=collator.collate)

# Model and trainer
model = GPTModel(model_config)
trainer = Trainer(
    model=model,
    config=training_config,
    train_loader=loader,
    val_loader=val_loader,
    use_tensorboard=True,
    use_wandb=False,
)

# Training
trainer.train()

# Cleanup
trainer.cleanup()
```

### Advanced Features

#### Mixed Precision Training
```python
training_config = TrainingConfig(
    use_mixed_precision=True,
    device="cuda"
)
```

#### Gradient Checkpointing
```python
training_config = TrainingConfig(
    use_gradient_checkpointing=True
)
```

#### Resume from Checkpoint
```python
trainer = Trainer(...)
trainer.load_checkpoint("checkpoints/checkpoint_latest.pt")
trainer.train()  # Resumes from checkpoint
```

#### Early Stopping
```python
training_config = TrainingConfig(
    early_stopping_patience=3,
    early_stopping_min_delta=0.01
)
```

## Testing

Comprehensive test suite with 36 tests covering:
- Dataset loading and preparation
- Dynamic batching and sequence packing
- Optimizer configuration
- Learning rate scheduling
- Gradient clipping
- Mixed precision
- Gradient checkpointing
- Checkpoint saving/loading
- Early stopping
- Metrics tracking
- Validation loop
- Logging (file and TensorBoard)
- Full integration tests

Run tests:
```bash
python -m pytest training/tests/test_training_pipeline.py -v
```

## Test Results

```
35 passed, 1 skipped
- All core functionality tested
- Integration tests verify end-to-end pipeline
- Edge cases covered
```

## Training Output Example

```
2026-07-01 21:06:58,009 - INFO - Configuration loaded
2026-07-01 21:06:58,009 - INFO - Device: cpu
2026-07-01 21:06:58,012 - INFO - Tokenizer trained and saved
2026-07-01 21:06:58,013 - INFO - Dataset prepared with 21 samples
2026-07-01 21:06:58,034 - INFO - Model initialized: 0M parameters
2026-07-01 21:06:59,442 - INFO - Starting training: epochs=5, batch_size=8, device=cpu
2026-07-01 21:07:00,984 - INFO - Validation - step=3 loss=1.0556 perplexity=2.87
2026-07-01 21:07:01,073 - INFO - Checkpoint saved: best
2026-07-01 21:07:01,073 - INFO - Epoch 1/5 complete - train_loss=1.0562 train_ppl=2.88
...
2026-07-01 21:07:09,012 - INFO - Training complete
2026-07-01 21:07:09,195 - INFO - Pipeline verification complete!
```

## Key Design Decisions

1. **Modular Design**: Each component is independent and reusable
2. **Production-Ready**: Comprehensive error handling and logging
3. **Memory Efficient**: Streaming datasets, gradient checkpointing, mixed precision
4. **Flexible**: Configurable components that work together seamlessly
5. **Well-Tested**: 36 comprehensive tests ensuring reliability
6. **Integrated**: All components work with the GPT model out of the box

## Dependencies

- PyTorch
- TensorBoard (optional)
- Weights & Biases (optional)
- pytest (for testing)

## Future Enhancements

- Distributed training support
- More sophisticated learning rate schedules
- Gradient accumulation
- Data parallelism
- Model parallelism for large models
- Advanced sampling strategies
- More comprehensive metrics (BLEU, ROUGE, etc.)