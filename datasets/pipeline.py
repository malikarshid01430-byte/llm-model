from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from datasets.chunker import DocumentChunker
from preprocessing.text_processor import TextProcessor


@dataclass
class DatasetConfig:
    """Configuration for dataset processing."""

    chunk_size: int = 512
    chunk_overlap: int = 32
    min_chunk_length: int = 50
    max_chunk_length: int = 2000
    deduplicate: bool = True
    similarity_threshold: float = 0.8
    language: str = "en"


@dataclass
class DatasetStats:
    """Statistics about a dataset."""

    total_documents: int = 0
    total_chunks: int = 0
    total_tokens: int = 0
    total_characters: int = 0
    avg_chunk_length: float = 0.0
    min_chunk_length: int = 0
    max_chunk_length: int = 0
    vocabulary_size: int = 0
    duplicate_count: int = 0
    language_distribution: dict[str, int] = field(default_factory=dict)


class BaseLoader:
    """Base class for all document loaders."""

    def __init__(self) -> None:
        self.processor = TextProcessor()

    def load(self, path: str | Path) -> str:
        """Load document from path."""
        raise NotImplementedError

    def load_many(self, paths: Sequence[str | Path]) -> list[str]:
        """Load multiple documents."""
        return [self.load(path) for path in paths]


class TXTLoader(BaseLoader):
    """Load plain text files."""

    def load(self, path: str | Path) -> str:
        """Load text file."""
        text = Path(path).read_text(encoding="utf-8")
        return self.processor.clean(text)


class MarkdownLoader(BaseLoader):
    """Load Markdown files."""

    def load(self, path: str | Path) -> str:
        """Load markdown file and remove markdown syntax."""
        text = Path(path).read_text(encoding="utf-8")
        # Remove markdown headers
        text = re.sub(r"^#+\s+", "", text, flags=re.MULTILINE)
        # Remove markdown links
        text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)
        # Remove markdown images
        text = re.sub(r"!\[([^\]]*)\]\([^\)]+\)", "", text)
        # Remove bold/italic markers
        text = re.sub(r"\*+([^\*]+)\*+", r"\1", text)
        text = re.sub(r"_+([^_]+)_+", r"\1", text)
        # Remove code blocks
        text = re.sub(r"```[^`]*```", "", text, flags=re.DOTALL)
        text = re.sub(r"`[^`]+`", "", text)
        return self.processor.clean(text)


class HTMLLoader(BaseLoader):
    """Load HTML files."""

    def load(self, path: str | Path) -> str:
        """Load HTML file and extract text."""
        try:
            from bs4 import BeautifulSoup

            html = Path(path).read_text(encoding="utf-8")
            soup = BeautifulSoup(html, "html.parser")
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            text = soup.get_text(separator=" ")
            return self.processor.clean(text)
        except ImportError:
            # Fallback: basic HTML tag removal
            text = Path(path).read_text(encoding="utf-8")
            text = re.sub(r"<[^>]+>", " ", text)
            return self.processor.clean(text)


class CSVLoader(BaseLoader):
    """Load CSV files."""

    def __init__(self, text_columns: list[str] | None = None) -> None:
        super().__init__()
        self.text_columns = text_columns

    def load(self, path: str | Path) -> str:
        """Load CSV and concatenate text columns."""
        df = pd.read_csv(path)

        if self.text_columns is None:
            # Use all string columns
            self.text_columns = df.select_dtypes(include=["object"]).columns.tolist()

        texts = []
        for col in self.text_columns:
            if col in df.columns:
                texts.extend(df[col].dropna().astype(str).tolist())

        return self.processor.clean(" ".join(texts))


class JSONLoader(BaseLoader):
    """Load JSON files."""

    def __init__(self, text_key: str = "text") -> None:
        super().__init__()
        self.text_key = text_key

    def load(self, path: str | Path) -> str:
        """Load JSON and extract text fields."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))

        texts = []
        if isinstance(data, dict):
            data = [data]

        for item in data:
            if isinstance(item, dict):
                if self.text_key in item:
                    texts.append(str(item[self.text_key]))
                else:
                    # Extract all string values
                    texts.extend([str(v) for v in item.values() if isinstance(v, str)])
            elif isinstance(item, str):
                texts.append(item)

        return self.processor.clean(" ".join(texts))


class PDFLoader(BaseLoader):
    """Load PDF files."""

    def load(self, path: str | Path) -> str:
        """Load PDF and extract text."""
        try:
            import PyPDF2

            with open(path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                texts = []
                for page in reader.pages:
                    texts.append(page.extract_text() or "")
                return self.processor.clean(" ".join(texts))
        except ImportError:
            try:
                import pdfplumber

                with pdfplumber.open(path) as pdf:
                    texts = []
                    for page in pdf.pages:
                        texts.append(page.extract_text() or "")
                    return self.processor.clean(" ".join(texts))
            except ImportError:
                raise ImportError("Install PyPDF2 or pdfplumber for PDF support")


class DOCXLoader(BaseLoader):
    """Load DOCX files."""

    def load(self, path: str | Path) -> str:
        """Load DOCX and extract text."""
        try:
            from docx import Document

            doc = Document(str(path))
            texts = [paragraph.text for paragraph in doc.paragraphs]
            return self.processor.clean(" ".join(texts))
        except ImportError:
            raise ImportError("Install python-docx for DOCX support")


class WikipediaLoader(BaseLoader):
    """Load Wikipedia articles."""

    def __init__(self, language: str = "en") -> None:
        super().__init__()
        self.language = language

    def load(self, path: str | Path) -> str:
        """Load Wikipedia dump (XML format)."""
        try:
            import mwxml

            with open(path, "r", encoding="utf-8") as f:
                dump = mwxml.Dump.from_file(f)
                texts = []
                for page in dump:
                    for revision in page:
                        if revision.text is not None:
                            texts.append(revision.text)
                return self.processor.clean(" ".join(texts))
        except ImportError:
            # Fallback: treat as plain text
            return super().load(path)

    def load_from_api(self, title: str) -> str:
        """Load Wikipedia article from API."""
        try:
            import requests

            url = f"https://{self.language}.wikipedia.org/w/api.php"
            params = {
                "action": "query",
                "prop": "extracts",
                "explaintext": True,
                "titles": title,
                "format": "json",
            }
            response = requests.get(url, params=params)
            data = response.json()
            pages = data["query"]["pages"]
            text = " ".join(page.get("extract", "") for page in pages.values())
            return self.processor.clean(text)
        except ImportError:
            raise ImportError("Install requests for Wikipedia API support")


class OCRPipeline:
    """OCR pipeline for scanned documents."""

    def __init__(self, language: str = "eng") -> None:
        self.language = language

    def extract_text(self, image_path: str | Path) -> str:
        """Extract text from image using OCR."""
        try:
            import pytesseract
            from PIL import Image

            image = Image.open(image_path)
            text = pytesseract.image_to_string(image, lang=self.language)
            return text
        except ImportError:
            raise ImportError("Install pytesseract and Pillow for OCR support")

    def extract_text_from_pdf(self, pdf_path: str | Path) -> str:
        """Extract text from PDF using OCR."""
        try:
            import pdf2image
            import pytesseract

            images = pdf2image.convert_from_path(pdf_path)
            texts = [pytesseract.image_to_string(img) for img in images]
            return " ".join(texts)
        except ImportError:
            raise ImportError(
                "Install pdf2image, pytesseract, and Pillow for OCR support"
            )


class TextCleaner:
    """Advanced text cleaning utilities."""

    def __init__(self) -> None:
        self.processor = TextProcessor()

    def remove_urls(self, text: str) -> str:
        """Remove URLs from text."""
        text = re.sub(r"https?://[^\s]+", "", text)
        text = re.sub(r"www\.[^\s]+", "", text)
        return text

    def remove_emails(self, text: str) -> str:
        """Remove email addresses from text."""
        text = re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "", text)
        return text

    def remove_phone_numbers(self, text: str) -> str:
        """Remove phone numbers from text."""
        text = re.sub(
            r"\(?\+?[0-9]{1,3}\)?[-.\s]?[0-9]{1,4}[-.\s]?[0-9]{1,4}", "", text
        )
        return text

    def remove_special_chars(self, text: str) -> str:
        """Remove special characters."""
        text = re.sub(r"[^\w\s.,!?;:'\-()]", " ", text)
        return text

    def remove_excessive_whitespace(self, text: str) -> str:
        """Remove excessive whitespace."""
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def clean(self, text: str) -> str:
        """Apply all cleaning steps."""
        text = self.remove_urls(text)
        text = self.remove_emails(text)
        text = self.remove_phone_numbers(text)
        text = self.remove_special_chars(text)
        text = self.remove_excessive_whitespace(text)
        return text


class Deduplicator:
    """Remove duplicate or near-duplicate documents."""

    def __init__(self, similarity_threshold: float = 0.8) -> None:
        self.similarity_threshold = similarity_threshold

    def _get_hash(self, text: str) -> str:
        """Get MD5 hash of text."""
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    def deduplicate_exact(self, documents: list[str]) -> list[str]:
        """Remove exact duplicates."""
        seen = set()
        unique = []
        for doc in documents:
            doc_hash = self._get_hash(doc)
            if doc_hash not in seen:
                seen.add(doc_hash)
                unique.append(doc)
        return unique

    def deduplicate_fuzzy(self, documents: list[str]) -> list[str]:
        """Remove near-duplicate documents using MinHash."""
        try:
            from datasketch import MinHash, MinHashLSH

            lsh = MinHashLSH(threshold=self.similarity_threshold, num_perm=128)
            unique = []

            for i, doc in enumerate(documents):
                words = doc.split()
                if not words:
                    continue

                m = MinHash(num_perm=128)
                for word in words:
                    m.update(word.encode("utf-8"))

                if not lsh.query(m):
                    lsh.insert(i, m)
                    unique.append(doc)

            return unique
        except ImportError:
            # Fallback to exact deduplication
            return self.deduplicate_exact(documents)

    def deduplicate(self, documents: list[str], fuzzy: bool = False) -> list[str]:
        """Deduplicate documents."""
        if fuzzy:
            return self.deduplicate_fuzzy(documents)
        return self.deduplicate_exact(documents)


class DatasetStatistics:
    """Compute statistics about a dataset."""

    def __init__(self) -> None:
        self.stats = DatasetStats()

    def compute(self, documents: list[str], tokenizer=None) -> DatasetStats:
        """Compute dataset statistics."""
        self.stats.total_documents = len(documents)
        self.stats.total_characters = sum(len(doc) for doc in documents)

        chunk_lengths = []
        total_tokens = 0

        for doc in documents:
            chunk_lengths.append(len(doc))
            if tokenizer:
                tokens = tokenizer.encode(doc)
                total_tokens += len(tokens)

        if chunk_lengths:
            self.stats.total_chunks = len(documents)
            self.stats.avg_chunk_length = sum(chunk_lengths) / len(chunk_lengths)
            self.stats.min_chunk_length = min(chunk_lengths)
            self.stats.max_chunk_length = max(chunk_lengths)

        self.stats.total_tokens = total_tokens

        if tokenizer:
            self.stats.vocabulary_size = len(tokenizer.vocab)

        return self.stats

    def get_summary(self) -> dict[str, Any]:
        """Get statistics summary."""
        return {
            "total_documents": self.stats.total_documents,
            "total_chunks": self.stats.total_chunks,
            "total_tokens": self.stats.total_tokens,
            "total_characters": self.stats.total_characters,
            "avg_chunk_length": round(self.stats.avg_chunk_length, 2),
            "min_chunk_length": self.stats.min_chunk_length,
            "max_chunk_length": self.stats.max_chunk_length,
            "vocabulary_size": self.stats.vocabulary_size,
            "duplicate_count": self.stats.duplicate_count,
        }


class VocabularyBuilder:
    """Build vocabulary from dataset."""

    def __init__(self, min_frequency: int = 2, max_vocab_size: int = 50000) -> None:
        self.min_frequency = min_frequency
        self.max_vocab_size = max_vocab_size
        self.vocab: dict[str, int] = {}

    def build(self, documents: list[str], tokenizer=None) -> dict[str, int]:
        """Build vocabulary from documents."""
        word_freq: dict[str | int, int] = {}

        for doc in documents:
            if tokenizer:
                tokens = tokenizer.encode(doc)
                for token in tokens:
                    word_freq[token] = word_freq.get(token, 0) + 1
            else:
                words = doc.split()
                for word in words:
                    word = word.lower()
                    word_freq[word] = word_freq.get(word, 0) + 1

        # Filter by frequency
        filtered = {k: v for k, v in word_freq.items() if v >= self.min_frequency}

        # Sort by frequency
        sorted_vocab = sorted(filtered.items(), key=lambda x: x[1], reverse=True)

        # Limit size
        sorted_vocab = sorted_vocab[: self.max_vocab_size]

        # Create vocab dict
        self.vocab = {word: idx for idx, (word, _) in enumerate(sorted_vocab)}

        return self.vocab

    def save(self, path: str | Path) -> None:
        """Save vocabulary to file."""
        Path(path).write_text(json.dumps(self.vocab, indent=2))

    def load(self, path: str | Path) -> dict[str, int]:
        """Load vocabulary from file."""
        self.vocab = json.loads(Path(path).read_text())
        return self.vocab


class DatasetValidator:
    """Validate dataset quality."""

    def __init__(self) -> None:
        self.issues: list[str] = []

    def validate_document(self, doc: str, min_length: int = 10) -> bool:
        """Validate a single document."""
        if len(doc.strip()) < min_length:
            self.issues.append(f"Document too short: {len(doc)} chars")
            return False
        return True

    def validate_dataset(self, documents: list[str]) -> tuple[bool, list[str]]:
        """Validate entire dataset."""
        self.issues = []

        if not documents:
            self.issues.append("Dataset is empty")
            return False, self.issues

        # Check for empty documents
        empty_count = sum(1 for doc in documents if not doc.strip())
        if empty_count > 0:
            self.issues.append(f"Found {empty_count} empty documents")

        # Check for very short documents
        short_count = sum(1 for doc in documents if len(doc.strip()) < 50)
        if short_count > len(documents) * 0.1:
            self.issues.append(f"Found {short_count} very short documents (>10%)")

        # Check for duplicates
        unique = set(documents)
        duplicate_count = len(documents) - len(unique)
        if duplicate_count > 0:
            self.issues.append(f"Found {duplicate_count} duplicate documents")

        return len(self.issues) == 0, self.issues

    def get_report(self) -> str:
        """Get validation report."""
        if not self.issues:
            return "Dataset validation passed"
        return "\n".join([f"- {issue}" for issue in self.issues])


class DatasetPipeline:
    """Complete dataset processing pipeline."""

    def __init__(self, config: DatasetConfig | None = None) -> None:
        self.config = config or DatasetConfig()
        self.loader: BaseLoader | None = None
        self.cleaner = TextCleaner()
        self.deduplicator = Deduplicator(self.config.similarity_threshold)
        self.chunker = DocumentChunker(
            self.config.chunk_size, self.config.chunk_overlap
        )
        self.statistics = DatasetStatistics()
        self.vocab_builder = VocabularyBuilder()
        self.validator = DatasetValidator()

    def set_loader(self, loader: BaseLoader) -> None:
        """Set document loader."""
        self.loader = loader

    def process(
        self, paths: list[str | Path], tokenizer=None
    ) -> tuple[list[str], DatasetStats]:
        """Process documents through the pipeline."""
        if self.loader is None:
            raise ValueError("Loader not set")

        # Load documents
        documents = self.loader.load_many(paths)

        # Clean documents
        documents = [self.cleaner.clean(doc) for doc in documents]

        # Deduplicate
        if self.config.deduplicate:
            documents = self.deduplicator.deduplicate(documents)

        # Chunk documents
        chunks = []
        for doc in documents:
            doc_chunks = self.chunker.chunk(doc)
            chunks.extend(doc_chunks)

        # Filter chunks by length
        chunks = [
            chunk
            for chunk in chunks
            if self.config.min_chunk_length
            <= len(chunk)
            <= self.config.max_chunk_length
        ]

        # Compute statistics
        stats = self.statistics.compute(chunks, tokenizer)

        # Validate
        is_valid, issues = self.validator.validate_dataset(chunks)
        if not is_valid:
            print(f"Dataset validation issues:\n{self.validator.get_report()}")

        return chunks, stats

    def build_vocabulary(self, documents: list[str]) -> dict[str, int]:
        """Build vocabulary from documents."""
        return self.vocab_builder.build(documents)
