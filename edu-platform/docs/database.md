# Database Design

## PostgreSQL
Relational schema stores core entities:
- `users` for authentication and role management.
- `roles` for RBAC definitions.
- `courses` for course metadata.
- `conversations` for AI chat history and memory.

## MongoDB
Used for document storage and institutional knowledge sources such as PDFs, lecture notes, and transcripts.

## Redis
Used for cache, session support, and rate limiting.

## Vector Database
Chromadb stores semantic embeddings for documents and enables retrieval-augmented generation (RAG) search.
