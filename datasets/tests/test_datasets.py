from datasets.chunker import DocumentChunker
from datasets.document_loader import DocumentLoader


def test_document_loader_reads_text(tmp_path) -> None:
    file_path = tmp_path / "sample.txt"
    file_path.write_text("hello world\nthis is a test", encoding="utf-8")
    loader = DocumentLoader()
    text = loader.load(file_path)
    assert "hello world" in text


def test_chunker_splits_text() -> None:
    chunker = DocumentChunker(chunk_size=3, overlap=1)
    chunks = chunker.chunk("one two three four five")
    assert len(chunks) > 0
