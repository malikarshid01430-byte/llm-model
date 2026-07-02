import json
from pathlib import Path

from tokenizer.subword.bpe_tokenizer import BPETokenizer


def test_bpe_tokenizer_fits_and_round_trips(tmp_path: Path) -> None:
    tokenizer = BPETokenizer(vocab_size=50)
    texts = ["hello world", "hello there", "world peace", "machine learning"]
    tokenizer.fit(texts)

    encoded = tokenizer.encode("hello world")
    assert encoded
    assert tokenizer.decode(encoded) == "hello world"

    save_path = tmp_path / "test_bpe_tokenizer.json"
    tokenizer.save(save_path)
    loaded = BPETokenizer(vocab_size=50)
    loaded.load(save_path)
    assert loaded.decode(loaded.encode("hello world")) == "hello world"

    payload = json.loads(save_path.read_text(encoding="utf-8"))
    assert "vocab" in payload
