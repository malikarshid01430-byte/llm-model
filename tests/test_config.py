from config import ModelConfig, TrainingConfig


def test_config_defaults() -> None:
    model_config = ModelConfig()
    training_config = TrainingConfig()
    assert model_config.d_model > 0
    assert training_config.batch_size > 0
