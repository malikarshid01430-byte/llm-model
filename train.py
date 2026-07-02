from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

from config import ModelConfig, TokenizerConfig, TrainingConfig
from model.gpt_model import GPTModel
from tokenizer.base import BPETokenizer
from training.data_loader import DatasetLoader, StreamingTextDataset, TextDataset
from training.dynamic_batching import BatchConfig, DataCollator, DynamicDataLoader
from training.trainer import Trainer
from utils.logging import setup_logging
from utils.normalization import normalize_text


def main() -> None:
    """Main training function with full production pipeline."""
    logger = setup_logging("logs")
    
    # Configuration
    tokenizer_config = TokenizerConfig(vocab_size=2000)
    model_config = ModelConfig(
        vocab_size=2000,
        d_model=128,
        n_layers=4,
        n_heads=4,
        ff_hidden_dim=512,
        max_seq_len=256,
    )
    training_config = TrainingConfig(
        batch_size=8,
        epochs=5,
        max_steps=500,
        learning_rate=3e-4,
        weight_decay=0.01,
        gradient_clip=1.0,
        warmup_steps=50,
        device="cuda" if torch.cuda.is_available() else "cpu",
        use_mixed_precision=torch.cuda.is_available(),
        use_gradient_checkpointing=False,
        checkpoint_every=100,
        val_every=50,
        early_stopping_patience=3,
        early_stopping_min_delta=0.0,
    )
    
    logger.info("Configuration loaded")
    logger.info("Device: %s", training_config.device)
    
    # Load dataset
    data_path = Path("data")
    if data_path.exists() and any(data_path.iterdir()):
        logger.info("Loading dataset from %s", data_path)
        texts = DatasetLoader.load_from_directory(data_path)
    else:
        logger.info("Using sample dataset")
        texts = [
            "This is a sample training corpus for an educational GPT style model built from scratch. " * 10,
            "The model implements a transformer architecture with multi-head attention. " * 10,
            "Training involves forward passes, backward passes, and optimization steps. " * 10,
            "Gradient clipping prevents exploding gradients during training. " * 10,
            "Learning rate warmup helps stabilize training in the early stages. " * 10,
        ]
    
    if not texts:
        logger.warning("No texts found, using default sample")
        texts = ["This is a sample training corpus."]
    
    logger.info("Loaded %d text samples", len(texts))
    
    # Initialize tokenizer
    tokenizer = BPETokenizer(vocab_size=tokenizer_config.vocab_size)
    tokenizer.fit(texts)
    tokenizer.save("tokenizer/vocab.json")
    logger.info("Tokenizer trained and saved")
    
    # Update model config with actual vocab size
    actual_vocab_size = max(tokenizer.vocab.values()) + 1
    model_config.vocab_size = actual_vocab_size
    
    # Create dataset
    full_dataset = TextDataset(
        texts=texts,
        tokenizer=tokenizer,
        max_seq_len=model_config.max_seq_len,
        stride=model_config.max_seq_len // 2,
    )
    
    logger.info("Dataset prepared with %d samples", len(full_dataset))
    
    # Split into train and validation
    val_size = max(1, int(len(full_dataset) * 0.1))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    logger.info("Train size: %d, Validation size: %d", train_size, val_size)
    
    # Create data collator
    pad_id = tokenizer.vocab.get(tokenizer.pad_token, 0)
    collator = DataCollator(max_seq_len=model_config.max_seq_len, pad_value=pad_id)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config.batch_size,
        shuffle=True,
        collate_fn=collator.collate,
        num_workers=0,
        pin_memory=training_config.device == "cuda",
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=training_config.batch_size,
        shuffle=False,
        collate_fn=collator.collate,
        num_workers=0,
        pin_memory=training_config.device == "cuda",
    )
    
    # Initialize model
    model = GPTModel(model_config)
    logger.info(
        "Model initialized: %dM parameters",
        sum(p.numel() for p in model.parameters()) / 1e6,
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        config=training_config,
        train_loader=train_loader,
        val_loader=val_loader,
        device=training_config.device,
        checkpoint_dir="checkpoints",
        log_dir="logs",
        use_tensorboard=True,
        use_wandb=False,
    )
    
    # Train model
    logger.info("Starting training...")
    try:
        trainer.train()
        logger.info("Training completed successfully")
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error("Training failed: %s", e, exc_info=True)
        raise
    finally:
        trainer.cleanup()
    
    # Test generation
    logger.info("Testing generation...")
    model.eval()
    test_prompt = "This is"
    tokens = tokenizer.encode(test_prompt)
    input_tensor = torch.tensor([tokens], dtype=torch.long, device=training_config.device)
    
    with torch.no_grad():
        generated = model.generate(
            input_tensor,
            max_new_tokens=20,
            temperature=0.8,
            top_k=50,
        )
    
    generated_text = tokenizer.decode(generated[0].cpu().tolist())
    logger.info("Generated text: %s", generated_text)
    logger.info("Pipeline verification complete!")


if __name__ == "__main__":
    import torch
    main()