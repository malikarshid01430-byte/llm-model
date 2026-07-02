import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from tokenizer.base import BPETokenizer


def test_tokenizer_round_trip() -> None:
    tokenizer = BPETokenizer(vocab_size=64)
    tokenizer.fit(["hello world", "hello there"])
    ids = tokenizer.encode("hello world")
    decoded = tokenizer.decode(ids)
    assert isinstance(ids, list)
    assert decoded
