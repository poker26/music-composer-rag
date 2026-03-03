"""Extract audio features: mel-spectrogram, chroma, tempo, key, rhythm."""
import logging
import numpy as np
import librosa

logger = logging.getLogger(__name__)


def fragment_audio(y, sr, duration=8.0, overlap=2.0):
    """Split audio into overlapping fragments. Yields (fragment, start_sec, end_sec)."""
    frag_samples = int(duration * sr)
    step_samples = int((duration - overlap) * sr)
    total = len(y)

    for start in range(0, total - frag_samples + 1, step_samples):
        end = start + frag_samples
        yield y[start:end], start / sr, end / sr

    # Tail fragment if > 50% of fragment size
    remainder_start = ((total - frag_samples) // step_samples) * step_samples + step_samples
    if remainder_start < total and (total - remainder_start) > frag_samples * 0.5:
        fragment = y[remainder_start:]
        fragment = np.pad(fragment, (0, frag_samples - len(fragment)))
        yield fragment, remainder_start / sr, total / sr


def extract_mel_features(y, sr, n_mels=128, hop_length=512):
    """Extract mel-spectrogram and compute statistical summary."""
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, hop_length=hop_length)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return {
        "mel_mean": np.mean(mel_db, axis=1),
        "mel_std": np.std(mel_db, axis=1),
        "mel_min": np.min(mel_db, axis=1),
        "mel_max": np.max(mel_db, axis=1),
        "mel_spectrogram": mel_db,
    }


def extract_chroma(y, sr, n_chroma=12, hop_length=512):
    """Extract chromagram - pitch class distribution."""
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, n_chroma=n_chroma, hop_length=hop_length)
    return {
        "chroma_mean": np.mean(chroma, axis=1),
        "chroma_std": np.std(chroma, axis=1),
        "chroma_raw": chroma,
    }


def extract_rhythm(y, sr, hop_length=512):
    """Extract tempo and rhythm features."""
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop_length)
    tempo_val = float(tempo[0]) if hasattr(tempo, "__len__") else float(tempo)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    tempogram = librosa.feature.tempogram(
        onset_envelope=onset_env, sr=sr, hop_length=hop_length
    )
    return {
        "tempo_bpm": tempo_val,
        "beat_frames": beat_frames,
        "onset_strength_mean": float(np.mean(onset_env)),
        "onset_strength_std": float(np.std(onset_env)),
        "tempogram_mean": np.mean(tempogram, axis=1),
    }


def estimate_key(y, sr):
    """Estimate musical key using Krumhansl-Kessler profiles."""
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)

    major_profile = np.array(
        [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
    )
    minor_profile = np.array(
        [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
    )
    key_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

    best_corr, best_key, best_mode = -1, "C", "major"
    for shift in range(12):
        rolled = np.roll(chroma_mean, -shift)
        corr_maj = float(np.corrcoef(rolled, major_profile)[0, 1])
        corr_min = float(np.corrcoef(rolled, minor_profile)[0, 1])
        if corr_maj > best_corr:
            best_corr, best_key, best_mode = corr_maj, key_names[shift], "major"
        if corr_min > best_corr:
            best_corr, best_key, best_mode = corr_min, key_names[shift], "minor"

    return {
        "key": best_key,
        "mode": best_mode,
        "key_confidence": best_corr,
        "key_label": f"{best_key} {best_mode}",
    }


def extract_all_features(y, sr, n_mels=128, n_chroma=12, hop_length=512):
    """Extract all features from an audio fragment."""
    mel = extract_mel_features(y, sr, n_mels, hop_length)
    chroma = extract_chroma(y, sr, n_chroma, hop_length)
    rhythm = extract_rhythm(y, sr, hop_length)
    key_info = estimate_key(y, sr)

    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=hop_length)
    zcr = librosa.feature.zero_crossing_rate(y, hop_length=hop_length)

    return {
        **mel, **chroma, **rhythm, **key_info,
        "spectral_centroid_mean": float(np.mean(spectral_centroid)),
        "spectral_rolloff_mean": float(np.mean(spectral_rolloff)),
        "zcr_mean": float(np.mean(zcr)),
        "rms_mean": float(np.mean(librosa.feature.rms(y=y))),
    }
