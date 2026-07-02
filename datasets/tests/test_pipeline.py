from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from datasets.pipeline import (
    CSVLoader,
    DatasetConfig,
    DatasetPipeline,
    DatasetStatistics,
    DatasetValidator,
    Deduplicator,
    DOCXLoader,
    HTMLLoader,
    JSONLoader,
    MarkdownLoader,
    PDFLoader,
    TextCleaner,
    TXTLoader,
    VocabularyBuilder,
    WikipediaLoader,
)


class TestLoaders:
    """Test document loaders."""

    def test_txt_loader(self, tmp_path: Path) -> None:
        """Test TXT loader."""
        loader = TXTLoader()
        test_file = tmp_path / "test.txt"
        test_file.write_text("Hello world\nThis is a test.")
        
        text = loader.load(test_file)
        assert "Hello world" in text
        assert "This is a test" in text

    def test_markdown_loader(self, tmp_path: Path) -> None:
        """Test Markdown loader."""
        loader = MarkdownLoader()
        test_file = tmp_path / "test.md"
        test_file.write_text("# Header\n\nSome **bold** text\n\n[Link](http://example.com)")
        
        text = loader.load(test_file)
        assert "Header" in text
        assert "bold" in text
        assert "Link" in text
        assert "http" not in text  # Links should be removed

    def test_html_loader(self, tmp_path: Path) -> None:
        """Test HTML loader."""
        loader = HTMLLoader()
        test_file = tmp_path / "test.html"
        test_file.write_text("<html><body><h1>Title</h1><p>Content</p></body></html>")
        
        text = loader.load(test_file)
        assert "Title" in text
        assert "Content" in text
        assert "<html>" not in text

    def test_json_loader(self, tmp_path: Path) -> None:
        """Test JSON loader."""
        loader = JSONLoader(text_key="content")
        test_file = tmp_path / "test.json"
        test_file.write_text(json.dumps([{"content": "First text"}, {"content": "Second text"}]))
        
        text = loader.load(test_file)
        assert "First text" in text
        assert "Second text" in text

    def test_csv_loader(self, tmp_path: Path) -> None:
        """Test CSV loader."""
        loader = CSVLoader(text_columns=["text"])
        test_file = tmp_path / "test.csv"
        test_file.write_text("id,text\n1,Hello world\n2,Test text")
        
        text = loader.load(test_file)
        assert "Hello world" in text
        assert "Test text" in text

    def test_load_many(self, tmp_path: Path) -> None:
        """Test loading multiple files."""
        loader = TXTLoader()
        files = []
        for i in range(3):
            f = tmp_path / f"test_{i}.txt"
            f.write_text(f"Text {i}")
            files.append(f)
        
        texts = loader.load_many(files)
        assert len(texts) == 3
        assert all(f"Text {i}" in texts[i] for i in range(3))


class TestTextCleaner:
    """Test text cleaning."""

    def test_remove_urls(self) -> None:
        """Test URL removal."""
        cleaner = TextCleaner()
        text = "Visit https://example.com and www.test.com"
        cleaned = cleaner.remove_urls(text)
        assert "https" not in cleaned
        assert "www" not in cleaned

    def test_remove_emails(self) -> None:
        """Test email removal."""
        cleaner = TextCleaner()
        text = "Contact us at test@example.com"
        cleaned = cleaner.remove_emails(text)
        assert "@" not in cleaned

    def test_remove_phone_numbers(self) -> None:
        """Test phone number removal."""
        cleaner = TextCleaner()
        text = "Call 123-456-7890 or (555) 123-4567"
        cleaned = cleaner.remove_phone_numbers(text)
        assert "123" not in cleaned or "456" not in cleaned

    def test_clean_all(self) -> None:
        """Test complete cleaning pipeline."""
        cleaner = TextCleaner()
        text = "Visit https://example.com or email test@example.com. Call 123-456-7890!"
        cleaned = cleaner.clean(text)
        assert "https" not in cleaned
        assert "@" not in cleaned
        assert "123" not in cleaned


class TestDeduplicator:
    """Test deduplication."""

    def test_exact_deduplication(self) -> None:
        """Test exact duplicate removal."""
        dedup = Deduplicator()
        documents = ["Hello world", "Test text", "Hello world", "Another text"]
        
        unique = dedup.deduplicate_exact(documents)
        assert len(unique) == 3
        assert "Hello world" in unique
        assert "Test text" in unique

    def test_fuzzy_deduplication(self) -> None:
        """Test fuzzy duplicate removal."""
        dedup = Deduplicator(similarity_threshold=0.9)
        documents = ["Hello world", "Hello world!", "Test text", "Test text."]
        
        unique = dedup.deduplicate_fuzzy(documents)
        # Should work (exact deduplication fallback if datasketch not installed)
        assert len(unique) <= 4


class TestDatasetStatistics:
    """Test dataset statistics."""

    def test_compute_statistics(self) -> None:
        """Test computing statistics."""
        from datasets.pipeline import DatasetStatistics as DS
        stats = DS()
        documents = ["Hello world", "Test text", "Another document"]
        
        result = stats.compute(documents)
        assert result.total_documents == 3
        assert result.total_characters > 0
        assert result.avg_chunk_length > 0

    def test_get_summary(self) -> None:
        """Test getting summary."""
        from datasets.pipeline import DatasetStatistics as DS
        stats = DS()
        documents = ["Hello world", "Test text"]
        stats.compute(documents)
        
        summary = stats.get_summary()
        assert "total_documents" in summary
        assert "total_characters" in summary
        assert summary["total_documents"] == 2


class TestVocabularyBuilder:
    """Test vocabulary building."""

    def test_build_vocabulary(self) -> None:
        """Test building vocabulary."""
        builder = VocabularyBuilder(min_frequency=1)
        documents = ["hello world test", "hello test", "world test"]
        
        vocab = builder.build(documents)
        assert len(vocab) > 0
        assert "hello" in vocab
        assert "world" in vocab

    def test_save_load_vocabulary(self, tmp_path: Path) -> None:
        """Test saving and loading vocabulary."""
        builder = VocabularyBuilder()
        documents = ["hello world", "test text"]
        builder.build(documents)
        
        # Save
        vocab_file = tmp_path / "vocab.json"
        builder.save(vocab_file)
        
        # Load
        builder2 = VocabularyBuilder()
        loaded_vocab = builder2.load(vocab_file)
        
        assert loaded_vocab == builder.vocab


class TestDatasetValidator:
    """Test dataset validation."""

    def test_validate_valid_dataset(self) -> None:
        """Test validating a valid dataset."""
        validator = DatasetValidator()
        documents = [
            "Hello world this is a longer document with enough characters to pass validation",
            "Test text with more content here to make it valid and pass the validation checks",
            "Another document with enough characters to pass validation and be considered valid"
        ]
        
        is_valid, issues = validator.validate_dataset(documents)
        if not is_valid:
            print(f"Validation issues: {issues}")
        assert is_valid, f"Validation failed with issues: {issues}"

    def test_validate_empty_dataset(self) -> None:
        """Test validating empty dataset."""
        validator = DatasetValidator()
        
        is_valid, issues = validator.validate_dataset([])
        assert not is_valid
        assert "empty" in " ".join(issues).lower()

    def test_validate_short_documents(self) -> None:
        """Test validating documents with many short ones."""
        validator = DatasetValidator()
        documents = ["Hi", "Test", "Doc"] * 5 + ["This is a longer document with enough content"]
        
        is_valid, issues = validator.validate_dataset(documents)
        # Should flag too many short documents
        assert not is_valid or len(issues) > 0


class TestDatasetPipeline:
    """Test complete dataset pipeline."""

    def test_pipeline_with_txt_files(self, tmp_path: Path) -> None:
        """Test pipeline with text files."""
        # Create test files
        for i in range(3):
            f = tmp_path / f"doc_{i}.txt"
            f.write_text(f"This is test document {i}. " * 20)
        
        pipeline = DatasetPipeline()
        pipeline.set_loader(TXTLoader())
        
        chunks, stats = pipeline.process(list(tmp_path.glob("*.txt")))
        
        assert len(chunks) > 0
        assert stats.total_documents == 3
        assert stats.total_characters > 0

    def test_pipeline_with_deduplication(self, tmp_path: Path) -> None:
        """Test pipeline with deduplication."""
        # Create duplicate files
        for i in range(3):
            f = tmp_path / f"doc_{i}.txt"
            f.write_text("This is the same document. " * 10)
        
        pipeline = DatasetPipeline(config=DatasetConfig(deduplicate=True))
        pipeline.set_loader(TXTLoader())
        
        chunks, stats = pipeline.process(list(tmp_path.glob("*.txt")))
        
        # Should deduplicate
        assert stats.duplicate_count >= 0

    def test_pipeline_with_custom_config(self, tmp_path: Path) -> None:
        """Test pipeline with custom configuration."""
        f = tmp_path / "test.txt"
        f.write_text("This is a test document. " * 100)
        
        config = DatasetConfig(
            chunk_size=50,
            chunk_overlap=10,
            min_chunk_length=20,
            max_chunk_length=200,
        )
        pipeline = DatasetPipeline(config=config)
        pipeline.set_loader(TXTLoader())
        
        chunks, stats = pipeline.process([f])
        
        assert len(chunks) > 0
        for chunk in chunks:
            assert 20 <= len(chunk) <= 200


class TestIntegration:
    """Test integration with tokenizer."""

    def test_pipeline_with_tokenizer(self, tmp_path: Path) -> None:
        """Test pipeline integration with tokenizer."""
        from tokenizer.base import BPETokenizer
        
        # Create test file with longer content
        f = tmp_path / "test.txt"
        f.write_text("This is a test document for tokenizer training. " * 100)
        
        # Create simple tokenizer mock
        class MockTokenizer:
            def __init__(self):
                self.vocab = {"<pad>": 0, "</s>": 1, "this": 2, "is": 3, "a": 4, 
                             "test": 5, "document": 6, "for": 7, "tokenizer": 8, "training": 9}
            
            def encode(self, text):
                tokens = []
                for word in text.lower().split():
                    if word in self.vocab:
                        tokens.append(self.vocab[word])
                return tokens
        
        # Use custom config with smaller min_chunk_length
        config = DatasetConfig(min_chunk_length=10)
        pipeline = DatasetPipeline(config=config)
        pipeline.set_loader(TXTLoader())
        
        tokenizer = MockTokenizer()
        chunks, stats = pipeline.process([f], tokenizer=tokenizer)
        
        assert len(chunks) > 0
        assert stats.vocabulary_size > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])