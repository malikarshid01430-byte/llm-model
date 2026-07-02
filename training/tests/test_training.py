from training.dpo_trainer import DPOTrainer
from training.lr_scheduler import CosineWarmupScheduler
from training.ppo_trainer import PPOTrainer


def test_cosine_scheduler_values() -> None:
    scheduler = CosineWarmupScheduler(initial_lr=0.1, warmup_steps=2, total_steps=10)
    assert scheduler.get_lr(0) > 0


def test_ppo_trainer_tracks_rewards() -> None:
    trainer = PPOTrainer()
    trainer.update(1.0)
    assert trainer.get_average_reward() == 1.0


def test_dpo_trainer_records_examples() -> None:
    trainer = DPOTrainer()
    trainer.add_example("chosen", "rejected")
    assert trainer.train() == 1
