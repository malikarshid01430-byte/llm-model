# Architecture Overview

This project follows a clean architecture approach with clear separation of concerns:

- config/: environment and runtime configuration
- datasets/: dataset and batching utilities
- tokenizer/: vocabulary and tokenization logic
- model/: transformer components and GPT model
- training/: optimization and checkpointing logic
- generation/: decoding and sampling helpers
- inference/: public generation entry points
- rag/: simple retrieval abstractions
- vector_database/: basic vector search scaffolding
- api/: FastAPI application surface
- frontend/: UI scaffold

The implementation is intentionally manual and educational while being structured for future extension into production-grade systems.
