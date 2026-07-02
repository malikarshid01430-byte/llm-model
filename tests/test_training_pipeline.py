import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

sys.path.append(str(Path(__file__).resolve().parents[1]))

from config import ModelConfig, TrainingConfig
from model.gpt_model import GPTModel
from training.data import DataCollator, SequencePacker
from training.trainer import Trainer


class TinyDataset(Dataset):
    def __init__(self, size: int = 8, seq_len: int = 6) -> None:
        self.data = torch.randint(0, 32, (size, seq_len))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.data[idx]


def test_sequence_packer_and_collator_work() -> None:
    sequences = [torch.tensor([1, 2, 3]), torch.tensor([4, 5])]
    packer = SequencePacker(max_seq_len=4)
    packed = packer.pack(sequences)
    assert packed.input_ids.shape[0] == len(sequences)
    assert packed.input_ids.shape[1] <= 4

    collator = DataCollator(max_seq_len=4)
    batch = collator.collate(sequences)
    assert batch["input_ids"].shape[0] == len(sequences)
    assert batch["targets"].shape == batch["input_ids"].shape


def test_trainer_runs_and_saves_checkpoint(tmp_path: Path) -> None:
    config = ModelConfig(
        vocab_size=32,
        d_model=16,
        n_layers=1,
        n_heads=2,
        ff_hidden_dim=32,
        max_seq_len=8,
    )
    model = GPTModel(config)
    dataset = TinyDataset(size=8, seq_len=6)
    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=DataCollator(max_seq_len=8).collate,
    )
    trainer = Trainer(
        model,
        TrainingConfig(
            batch_size=2,
            epochs=1,
            max_steps=2,
            checkpoint_every=1,
            device="cpu",
            gradient_clip=1.0,
            early_stopping_patience=1,
        ),
        loader,
        device="cpu",
        checkpoint_dir=tmp_path / "checkpoints",
        log_dir=tmp_path / "logs",
        use_tensorboard=False,
        use_wandb=False,
    )

    trainer.train()

    assert (tmp_path / "checkpoints" / "checkpoint_latest.pt").exists()
    assert trainer.global_step == 2
