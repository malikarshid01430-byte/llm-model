# EduLLM

EduLLM is a modular, from-scratch educational AI platform spanning model training, inference, backend services, frontend experience, deployment scaffolding, and advanced AI ecosystem capabilities.

## Architecture

The repository follows clean architecture and SOLID principles with clear separation between configuration, core abstractions, preprocessing, tokenizer, model, training, inference, RAG, authentication, API, frontend, memory, agents, and MCP layers.

## Features

- Manual tokenizer implementation
- Manual transformer and GPT-style model construction
- Training loop with checkpointing
- Inference and text generation
- Evaluation utilities
- RAG and vector store scaffolding
- FastAPI backend with auth, enterprise dashboards, memory, agents, MCP support, and readiness checks
- Next.js frontend shell with role-based dashboards and resilient data loading
- Logging, configuration, documentation, and test support

## Installation

```bash
python -m pip install -r requirements.txt
pip install -r edu-platform/backend/requirements.txt
```

## Training

```bash
python train.py
```

## Tests

```bash
pytest -q
```

## Backend

```bash
cd edu-platform/backend
uvicorn app.main:app --reload
```

## Frontend

```bash
cd edu-platform/frontend
npm install
npm run dev
```
