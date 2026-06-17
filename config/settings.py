"""Music Composer RAG - Configuration"""
import os
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
INPUT_DIR = BASE_DIR / "input"
OUTPUT_DIR = BASE_DIR / "output"
MIDI_DIR = OUTPUT_DIR / "midi"
FEATURES_DIR = OUTPUT_DIR / "features"
WAV_DIR = OUTPUT_DIR / "wav"

# Qdrant connection (only used by the legacy audio pipeline: ingest/search/app).
# Override via env, e.g. QDRANT_HOST=localhost when running `docker run qdrant/qdrant`.
QDRANT_HOST = os.environ.get("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.environ.get("QDRANT_PORT", "6333"))
COLLECTION_NAME = "music_fragments"

SAMPLE_RATE = 22050
HOP_LENGTH = 512
N_MELS = 128
N_CHROMA = 12

FRAGMENT_DURATION_SEC = 8.0
FRAGMENT_OVERLAP_SEC = 2.0

# Embedding: (mean, std, min, max) per mel band = 128 * 4 = 512
EMBEDDING_DIM = 512

SUPPORTED_TAGS = [
    "composer", "era", "genre", "instrument",
    "tempo_bpm", "key", "time_signature", "mood",
]
