# Geo RAG System

An intelligent GIS with Retrieval-Augmented Generation (RAG) for spatial queries over vector/raster data and satellite imagery.

## Quickstart

1) Create venv and install deps:
```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
```

2) Run app:
```bash
streamlit run frontend/app.py
```

3) Put sample files in `data/sample_data/`.

## Notes
- On Windows, consider using conda for installing `geopandas`/`rasterio`.
- See `config/settings.py` for model and path configuration.
