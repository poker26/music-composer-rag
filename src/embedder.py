"""Generate embedding vectors for Qdrant using CLAP (Contrastive Language-Audio Pretraining)."""
import numpy as np
import logging
import tempfile
import soundfile as sf
from pathlib import Path

logger = logging.getLogger(__name__)

EMBEDDING_DIM = 512

# Lazy-loaded CLAP model (heavy, load once)
_clap_model = None


def _get_clap_model():
    """Load CLAP model on first use."""
    global _clap_model
    if _clap_model is None:
        import laion_clap
        logger.info("Loading CLAP model (first time, may take a moment)...")
        _clap_model = laion_clap.CLAP_Module(enable_fusion=False)
        _clap_model.load_ckpt()
        logger.info("CLAP model loaded.")
    return _clap_model


def build_embedding(features, audio_data=None, sr=22050):
    """
    Build embedding using CLAP if audio_data is provided,
    otherwise fall back to mel-statistics.

    Args:
        features: dict of extracted audio features
        audio_data: numpy array of audio samples (mono, float32)
        sr: sample rate
    """
    if audio_data is not None:
        try:
            return _build_clap_embedding(audio_data, sr)
        except Exception as e:
            logger.warning("CLAP embedding failed, falling back to mel: %s", e)

    return _build_mel_embedding(features)


def _build_clap_embedding(audio_data, sr):
    """Build CLAP embedding from raw audio."""
    model = _get_clap_model()

    # CLAP expects 48kHz audio files, so we need to write a temp WAV
    # and let CLAP handle resampling internally
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
        sf.write(tmp_path, audio_data, sr)

    try:
        embedding = model.get_audio_embedding_from_filelist(
            x=[tmp_path], use_tensor=False
        )
        vec = embedding[0].tolist()
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    # CLAP outputs 512-dim vector
    assert len(vec) == EMBEDDING_DIM, f"CLAP returned {len(vec)} dims, expected {EMBEDDING_DIM}"
    return vec


def _build_mel_embedding(features):
    """Fallback: build embedding from mel-spectrogram statistics."""
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
