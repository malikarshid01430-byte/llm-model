from __future__ import annotations

from typing import List

from tokenizer.character_tokenizer import CharacterTokenizer
from tokenizer.sentencepiece_tokenizer import SentencePieceStyleTokenizer
from tokenizer.word_tokenizer import WordTokenizer


class TokenizerTrainer:
    """Utility that trains several tokenizer variants for comparison."""

    def train_character_tokenizer(self, texts: List[str]) -> CharacterTokenizer:
        tokenizer = CharacterTokenizer()
        tokenizer.fit(texts)
        return tokenizer

    def train_word_tokenizer(self, texts: List[str]) -> WordTokenizer:
        tokenizer = WordTokenizer()
        tokenizer.fit(texts)
        return tokenizer

    def train_sentencepiece_style_tokenizer(
        self, texts: List[str]
    ) -> SentencePieceStyleTokenizer:
        tokenizer = SentencePieceStyleTokenizer()
        tokenizer.fit(texts)
        return tokenizer
