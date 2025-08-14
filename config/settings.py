import os
from pathlib import Path

# Base paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
EMBEDDINGS_DIR = PROCESSED_DIR / "embeddings"
INDEXES_DIR = PROCESSED_DIR / "indexes"
CHUNKS_DIR = PROCESSED_DIR / "chunks"
SAMPLE_DATA_DIR = DATA_DIR / "sample_data"
CHROMA_DIR = PROJECT_ROOT / "chroma_db"

# Models
TEXT_EMBED_MODEL = os.getenv("TEXT_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
IMAGE_EMBED_MODEL = os.getenv("IMAGE_EMBED_MODEL", "openai/clip-vit-base-patch32")

# Streamlit
MAP_TILE_PROVIDER = os.getenv("MAP_TILE_PROVIDER", "OpenStreetMap")

# CRS
DEFAULT_CRS = "EPSG:4326"  # WGS84

# Performance
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "4"))
CHUNK_TILE_SIZE = int(os.getenv("CHUNK_TILE_SIZE", "256"))

# Ensure directories exist
for directory in [DATA_DIR, RAW_DIR, PROCESSED_DIR, EMBEDDINGS_DIR, INDEXES_DIR, CHUNKS_DIR, SAMPLE_DATA_DIR, CHROMA_DIR]:
    directory.mkdir(parents=True, exist_ok=True)
