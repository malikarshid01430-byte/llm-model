# EduLLM

EduLLM is a modular, from-scratch educational AI platform spanning model training, inference, backend services, frontend experience, deployment scaffolding, and advanced AI ecosystem capabilities.

## Architecture

The repository follows clean architecture and SOLID principles with clear separation between:

| Layer | Directory | Purpose |
|---|---|---|
| Configuration | `config.py`, `config/` | Dataclass-based configs for model, training, inference |
| Tokenizer | `tokenizer/` | BPE, character, word, SentencePiece tokenizers |
| Model | `model/` | GPT architecture, attention, embeddings, KV cache |
| Training | `training/` | Trainer, fine-tuning (LoRA/QLoRA), dynamic batching, DPO/PPO |
| Inference | `inference/` | Engine with greedy, sampling, beam search, streaming |
| RAG | `rag/` | FAISS/Chroma vector stores, hybrid retrieval, context ranking |
| API | `api/` | FastAPI routers for health, generation, inference, fine-tuning |
| Auth | `auth/`, `authentication/` | JWT token handling, RBAC |
| Edu Platform | `edu-platform/` | Full-stack app with Next.js frontend and FastAPI backend |
| Memory | `memory/` | Conversation and long-term memory stores |
| Datasets | `datasets/` | Document loading, chunking, pipeline processing |

## Features

- Manual tokenizer implementation (BPE, character, word, SentencePiece)
- Manual transformer and GPT-style model construction with rotary embeddings
- Training loop with checkpointing, early stopping, mixed precision
- Fine-tuning support: LoRA, QLoRA, supervised fine-tuning
- Inference with greedy, top-k/top-p sampling, beam search, streaming
- Dynamic batching and curriculum learning for efficient training
- Evaluation utilities (perplexity, accuracy)
- RAG system with FAISS and ChromaDB vector stores, hybrid retrieval
- FastAPI backend with auth, enterprise dashboards, memory, agents, MCP support
- Next.js frontend shell with role-based dashboards
- Comprehensive test suite (178+ tests, 76% coverage)
- Pre-commit hooks: black, isort, ruff

## Installation

```bash
python -m pip install -r requirements.txt
pip install -r edu-platform/backend/requirements.txt
```

## Quick Start

### Training

```bash
python train.py
```

### Inference

```python
from config import ModelConfig, InferenceConfig
from model.gpt_model import GPTModel
from inference.engine import InferenceEngine

model = GPTModel(ModelConfig())
engine = InferenceEngine(model, tokenizer, InferenceConfig())
text = engine.generate("Hello, world")
```

### Tests

```bash
# Run all tests
pytest -q

# Run with coverage
pytest --cov=. --cov-report=term

# Run specific test modules
pytest tests/ training/tests/ tokenizer/tests/
```

### Quality Checks

```bash
ruff check .
black --check .
isort --check-only .
mypy --ignore-missing-imports config.py model/ training/ inference/
```

## Backend API

```bash
cd edu-platform/backend
uvicorn app.main:app --reload
```

Key endpoints:
- `GET /health` - Health check
- `POST /generate` - Text generation
- `GET /docs` - OpenAPI documentation

## Frontend

```bash
cd edu-platform/frontend
npm install
npm run dev
```

## Environment Variables

| Variable | Description | Default |
|---|---|---|
| `JWT_SECRET` | Secret key for JWT tokens | `dev-secret` (warns) |
| `JWT_ALGORITHM` | JWT signing algorithm | `HS256` |
| `POSTGRES_URL` | Database connection URL | `sqlite+aiosqlite:///./eduai.db` |
| `REDIS_URL` | Redis connection URL | `redis://localhost:6379/0` |
| `VECTOR_STORE_PATH` | Path for vector store data | `data/vector_store` |
| `DEBUG` | Enable debug mode | Auto-detected |
| `ALLOWED_ORIGINS` | CORS allowed origins | `localhost:3000` |

## Project Structure

```
llm-model/
  config.py              # Core configuration dataclasses
  train.py               # Training entry point
  model/                 # GPT model, attention, embeddings
  training/              # Trainer, fine-tuning, dynamic batching
  inference/             # Inference engines
  tokenizer/             # Tokenizer implementations
  rag/                   # RAG system with vector stores
  api/                   # FastAPI application
  auth/                  # Authentication/authorization
  datasets/              # Data loading and processing
  memory/                # Memory systems
  edu-platform/          # Full-stack education platform
    backend/             # FastAPI backend
    frontend/            # Next.js frontend
  tests/                 # Test suite
```
