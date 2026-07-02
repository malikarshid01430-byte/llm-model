# Developer Guide

## Setup

```bash
python -m pip install -r requirements.txt
```

## Running tests

```bash
python -m pytest -q tests
```

## Training

```bash
python train.py
```

## API

```bash
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```
