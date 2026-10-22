# oceanscope_ml

Python package for the OceanScope AI subsurface ocean temperature reconstruction pipeline.

## Structure
- `src/oceanscope_ml/`: Core package modules (data, models, training, evaluation, inference, publish, fixtures).
- `configs/`: Pipeline configuration (`experiment.yaml`).
- `tests/`: Test suite.

## Development Checks
```bash
ruff check .
ruff format --check .
mypy src
pytest -q
```
