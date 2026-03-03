"""Generate embedding vectors for Qdrant from extracted features."""
import numpy as np
import logging

logger = logging.getLogger(__name__)

EMBEDDING_DIM = 512


def build_embedding(features):
    """
    Build a fixed-size embedding vector from audio features.
    Layout (512 dims): mel_mean[128] + mel_std[128] + mel_min[128] + mel_max[128]
    Normalized to unit length.
    """
    mel_mean = features.get("mel_mean", np.zeros(128))
    mel_std = features.get("mel_std", np.zeros(128))
    mel_min = features.get("mel_min", np.zeros(128))
    mel_max = features.get("mel_max", np.zeros(128))

    raw = np.concatenate([mel_mean, mel_std, mel_min, mel_max])

    norm = np.linalg.norm(raw)
    if norm > 0:
        raw = raw / norm

    assert len(raw) == EMBEDDING_DIM, f"Expected {EMBEDDING_DIM}, got {len(raw)}"
    return raw.tolist()


def build_metadata_payload(features, source_info):
    """Build Qdrant payload with searchable metadata."""
    chroma = features.get("chroma_mean", np.zeros(12))
    if isinstance(chroma, np.ndarray):
        chroma = chroma.tolist()

    return {
        # Source
        "source_file": source_info.get("file", "unknown"),
        "composer": source_info.get("composer", "unknown"),
        "era": source_info.get("era", ""),
        "genre": source_info.get("genre", ""),
        "instrument": source_info.get("instrument", ""),
        # Fragment position
        "fragment_start_sec": source_info.get("start_sec", 0),
        "fragment_end_sec": source_info.get("end_sec", 0),
        "fragment_index": source_info.get("fragment_index", 0),
        # Audio features (scalar, for filtering)
        "tempo_bpm": features.get("tempo_bpm", 0),
        "key": features.get("key_label", ""),
        "key_confidence": features.get("key_confidence", 0),
        "spectral_centroid": features.get("spectral_centroid_mean", 0),
        "rms": features.get("rms_mean", 0),
        "zcr": features.get("zcr_mean", 0),
        # Chroma profile
        "chroma_profile": chroma,
        # MIDI analysis (if available)
        "note_count": features.get("note_count", 0),
        "note_density": features.get("note_density_per_beat", 0),
        "pitch_class_histogram": features.get("pitch_class_histogram", []),
        "velocity_mean": features.get("velocity_mean", 0),
    }
