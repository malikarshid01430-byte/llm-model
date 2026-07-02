# Deployment Guide

## Docker

```bash
docker build -t llm-from-scratch .
docker run -p 8000:8000 llm-from-scratch
```

## Docker Compose

```bash
docker compose up --build
```
