# Complete Dataset Pipeline

## Overview

This document describes the complete dataset pipeline implemented for processing text data for LLM training. The pipeline includes multiple document loaders, text cleaning, deduplication, chunking, statistics computation, vocabulary building, and dataset validation.

## Components Implemented

### 1. Document Loaders

#### TXTLoader
Loads plain text files with UTF-8 encoding.

**Usage:**
```python
from datasets.pipeline import TXTLoader

loader = TXTLoader()
text = loader.load("document.txt")
```

#### MarkdownLoader
Loads Markdown files and removes markdown syntax (headers, links, images, bold/italic markers, code blocks).

**Usage:**
```python
from datasets.pipeline import MarkdownLoader

loader = MarkdownLoader()
text = loader.load("document.md")
```

#### HTMLLoader
Loads HTML files and extracts text using BeautifulSoup (with fallback to regex-based extraction).

**Usage:**
```python
from datasets.pipeline import HTMLLoader

loader = HTMLLoader()
text = loader.load("document.html")
```

#### CSVLoader
Loads CSV files and concatenates text from specified columns (or all string columns by default).

**Usage:**
```python
from datasets.pipeline import CSVLoader

# Load specific columns
loader = CSVLoader(text_columns=["title", "content"])
text = loader.load("data.csv")

# Auto-detect text columns
loader = CSVLoader()
text = loader.load("data.csv")
```

#### JSONLoader
Loads JSON files and extracts text from specified keys or all string values.

**Usage:**
```python
from datasets.pipeline import JSONLoader

# Load from specific key
loader = JSONLoader(text_key="content")
text = loader.load("data.json")

# Auto-extract all strings
loader = JSONLoader()
text = loader.load("data.json")
```

#### PDFLoader
Loads PDF files and extracts text using PyPDF2 or pdfplumber.

**Usage:**
```python
from datasets.pipeline import PDFLoader

loader = PDFLoader()
text = loader.load("document.pdf")
```

#### DOCXLoader
Loads DOCX files and extracts text from paragraphs.

**Usage:**
```python
from datasets.pipeline import DOCXLoader

loader = DOCXLoader()
text = loader.load("document.docx")
```

#### WikipediaLoader
Loads Wikipedia articles from XML dumps or API.

**Usage:**
```python
from datasets.pipeline import WikipediaLoader

# Load from dump
loader = WikipediaLoader(language="en")
text = loader.load("wikipedia_dump.xml")

# Load from API
text = loader.load_from_api("Python (programming language)")
```

### 2. OCR Pipeline

Extracts text from images and scanned PDFs using Tesseract OCR.

**Usage:**
```python
from datasets.pipeline import OCRPipeline

ocr = OCRPipeline(language="eng")

# Extract from image
text = ocr.extract_text("scanned_page.png")

# Extract from PDF
text = ocr.extract_text_from_pdf("scanned_document.pdf")
```

### 3. Text Cleaning

Advanced text cleaning utilities for removing URLs, emails, phone numbers, special characters, and excessive whitespace.

**Usage:**
```python
from datasets.pipeline import TextCleaner

cleaner = TextCleaner()

# Individual operations
text = cleaner.remove_urls(text)
text = cleaner.remove_emails(text)
text = cleaner.remove_phone_numbers(text)
text = cleaner.remove_special_chars(text)
text = cleaner.remove_excessive_whitespace(text)

# Apply all cleaning steps
text = cleaner.clean(text)
```

### 4. Deduplication

Removes duplicate or near-duplicate documents using exact matching or MinHash-based fuzzy matching.

**Usage:**
```python
from datasets.pipeline import Deduplicator

# Exact deduplication
dedup = Deduplicator()
unique_docs = dedup.deduplicate_exact(documents)

# Fuzzy deduplication (requires datasketch)
dedup = Deduplicator(similarity_threshold=0.8)
unique_docs = dedup.deduplicate_fuzzy(documents)

# Unified interface
unique_docs = dedup.deduplicate(documents, fuzzy=False)
```

### 5. Chunking

Splits documents into overlapping chunks for training.

**Usage:**
```python
from datasets.chunker import DocumentChunker

chunker = DocumentChunker(chunk_size=512, overlap=32)
chunks = chunker.chunk(long_document)
```

### 6. Dataset Statistics

Computes comprehensive statistics about the dataset.

**Usage:**
```python
from datasets.pipeline import DatasetStatistics

stats = DatasetStatistics()
result = stats.compute(documents, tokenizer=tokenizer)

# Get summary
summary = stats.get_summary()
print(f"Total documents: {summary['total_documents']}")
print(f"Total tokens: {summary['total_tokens']}")
print(f"Avg chunk length: {summary['avg_chunk_length']}")
```

**Statistics computed:**
- Total documents
- Total chunks
- Total tokens
- Total characters
- Average chunk length
- Min/max chunk length
- Vocabulary size
- Duplicate count

### 7. Vocabulary Builder

Builds vocabulary from documents with frequency filtering.

**Usage:**
```python
from datasets.pipeline import VocabularyBuilder

builder = VocabularyBuilder(min_frequency=2, max_vocab_size=50000)
vocab = builder.build(documents, tokenizer=tokenizer)

# Save vocabulary
builder.save("vocab.json")

# Load vocabulary
vocab = builder.load("vocab.json")
```

### 8. Dataset Validation

Validates dataset quality and reports issues.

**Usage:**
```python
from datasets.pipeline import DatasetValidator

validator = DatasetValidator()

# Validate single document
is_valid = validator.validate_document(doc, min_length=10)

# Validate entire dataset
is_valid, issues = validator.validate_dataset(documents)

# Get report
report = validator.get_report()
```

**Validation checks:**
- Empty documents
- Very short documents (>10% threshold)
- Duplicate documents

### 9. Complete Pipeline

End-to-end dataset processing pipeline.

**Usage:**
```python
from datasets.pipeline import DatasetPipeline, DatasetConfig, TXTLoader

# Configure pipeline
config = DatasetConfig(
    chunk_size=512,
    chunk_overlap=32,
    min_chunk_length=50,
    max_chunk_length=2000,
    deduplicate=True,
    similarity_threshold=0.8,
)

# Create pipeline
pipeline = DatasetPipeline(config=config)
pipeline.set_loader(TXTLoader())

# Process documents
chunks, stats = pipeline.process(["doc1.txt", "doc2.txt"], tokenizer=tokenizer)

# Build vocabulary
vocab = pipeline.build_vocabulary(chunks)
```

## Configuration

### DatasetConfig

```python
@dataclass
class DatasetConfig:
    chunk_size: int = 512              # Target chunk size in characters
    chunk_overlap: int = 32            # Overlap between chunks
    min_chunk_length: int = 50         # Minimum chunk length
    max_chunk_length: int = 2000       # Maximum chunk length
    deduplicate: bool = True           # Enable deduplication
    similarity_threshold: float = 0.8  # Similarity threshold for fuzzy dedup
    language: str = "en"               # Language code
```

## Integration with Tokenizer Training

The pipeline integrates seamlessly with tokenizer training:

```python
from datasets.pipeline import DatasetPipeline, TXTLoader
from tokenizer.base import BPETokenizer

# Load and process documents
pipeline = DatasetPipeline()
pipeline.set_loader(TXTLoader())
chunks, stats = pipeline.process(["corpus.txt"])

# Train tokenizer on processed chunks
tokenizer = BPETokenizer(vocab_size=10000)
tokenizer.train(chunks)

# Use tokenizer for statistics
chunks, stats = pipeline.process(["corpus.txt"], tokenizer=tokenizer)
print(f"Vocabulary size: {stats.vocabulary_size}")
print(f"Total tokens: {stats.total_tokens}")
```

## Testing

Comprehensive test suite with 23 tests covering:
- All document loaders (TXT, Markdown, HTML, JSON, CSV)
- Text cleaning
- Deduplication (exact and fuzzy)
- Dataset statistics
- Vocabulary building
- Dataset validation
- Complete pipeline integration
- Tokenizer integration

Run tests:
```bash
python -m pytest datasets/tests/test_pipeline.py -v
```

## Test Results

```
23 passed in 4.10s
- All loaders tested
- Cleaning verified
- Deduplication working
- Statistics computed correctly
- Pipeline integration validated
```

## Key Features

1. **Multiple Format Support**: TXT, Markdown, HTML, PDF, DOCX, CSV, JSON, Wikipedia
2. **OCR Support**: Extract text from images and scanned PDFs
3. **Advanced Cleaning**: Remove URLs, emails, phone numbers, special characters
4. **Deduplication**: Exact and fuzzy (MinHash) duplicate removal
5. **Smart Chunking**: Overlapping chunks with configurable size
6. **Statistics**: Comprehensive dataset metrics
7. **Vocabulary Building**: Frequency-based vocabulary construction
8. **Validation**: Quality checks and issue reporting
9. **Pipeline Integration**: Seamless end-to-end processing
10. **Tokenizer Integration**: Works with tokenizer training

## Dependencies

**Required:**
- pandas (for CSV loading)

**Optional:**
- beautifulsoup4 (for HTML parsing)
- PyPDF2 or pdfplumber (for PDF loading)
- python-docx (for DOCX loading)
- mwxml (for Wikipedia dumps)
- requests (for Wikipedia API)
- pytesseract + Pillow + pdf2image (for OCR)
- datasketch (for fuzzy deduplication)

## Performance Considerations

- **Memory Efficient**: Processes documents one at a time
- **Streaming Support**: Can be extended for large datasets
- **Parallel Processing**: Loaders can be parallelized
- **Caching**: Intermediate results can be cached
- **Incremental Processing**: Supports batch processing

## Future Enhancements

- Parallel document loading
- Streaming dataset support
- More advanced deduplication (semantic similarity)
- Language detection
- Quality scoring
- Data augmentation
- Format conversion utilities