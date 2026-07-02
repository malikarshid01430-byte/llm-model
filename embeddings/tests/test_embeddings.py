import torch

from embeddings.rotary_embedding import RotaryEmbedding


def test_rotary_embedding_shape() -> None:
    embedding = RotaryEmbedding(8)
    x = torch.randn(2, 4, 8)
    output = embedding(x)
    assert output.shape == (4, 4, 2)
