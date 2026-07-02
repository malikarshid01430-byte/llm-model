# API Documentation

## Health

GET /health

Returns service health information.

## Generation

POST /generate

Body:

```json
{
  "prompt": "Explain the future of AI",
  "max_new_tokens": 40
}
```
