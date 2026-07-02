from tokenizer.character_tokenizer import CharacterTokenizer
from tokenizer.sentencepiece_tokenizer import SentencePieceStyleTokenizer
from tokenizer.word_tokenizer import WordTokenizer


def test_character_tokenizer_round_trip() -> None:
    tokenizer = CharacterTokenizer()
    tokenizer.fit(["hello"])
    ids = tokenizer.encode("hello")
    assert ids
    assert tokenizer.decode(ids) == "hello"


def test_word_tokenizer_round_trip() -> None:
    tokenizer = WordTokenizer()
    tokenizer.fit(["hello world"])
    ids = tokenizer.encode("hello world")
    assert ids
    assert tokenizer.decode(ids) == "hello world"


def test_sentencepiece_style_tokenizer_round_trip() -> None:
    tokenizer = SentencePieceStyleTokenizer()
    tokenizer.fit(["hello world"])
    ids = tokenizer.encode("hello world")
    assert ids
    assert tokenizer.decode(ids) == "hello world"
