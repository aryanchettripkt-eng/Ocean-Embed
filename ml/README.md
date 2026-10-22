# oceanscope_ml

Python package for the OceanScope AI subsurface ocean temperature reconstruction pipeline.

## Structure
- `src/oceanscope_ml/`: Core package modules:
  - `schemas/`: Manifest JSON Schema definitions.
  - `fixtures/`: Synthetic sample outputs generator.
  - `data/` *(planned M1)*: Inspection, downloading, preprocessing, and dataset loaders.
  - `models/` *(planned M2)*: Baselines (climatology, ridge) and U-Net architecture.
  - `training/` *(planned M2)*: Training loop, checkpoints, and experiment logging.
  - `evaluation/` *(planned M3/M5)*: Error metrics, regional breakdowns, and ARGO matchups.
  - `inference/` *(planned M4)*: Batch prediction generator.
  - `publish/` *(planned M4)*: Hugging Face dataset publishing utilities.
- `configs/`: Pipeline configuration (`experiment.yaml`).
- `notebooks/`: Thin execution notebooks for Google Colab and Kaggle.
- `tests/`: Automated unit tests.

## Development Checks
```bash
ruff check .
ruff format --check .
mypy src
pytest -q
```
