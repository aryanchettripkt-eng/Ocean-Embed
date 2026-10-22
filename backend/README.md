# oceanscope-backend

FastAPI service serving OceanScope AI subsurface temperature reconstruction data, metrics, and chat.

## Development Checks
```bash
ruff check .
ruff format --check .
mypy app
pytest -q
python -m app.export_openapi --check
```
