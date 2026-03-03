"""
Audio Loader
- Scan input directory for audio files (WAV, MP3, FLAC, OGG, M4A)
- Convert to WAV (mono, target sample rate) via ffmpeg
- Load as numpy arrays via librosa
"""
import subprocess
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

SUPPORTED_EXT = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}


def scan_audio_files(input_dir: Path) -> list[Path]:
    """Find all supported audio files recursively."""
    files = []
    for ext in SUPPORTED_EXT:
        files.extend(input_dir.rglob(f"*{ext}"))
    files.sort()
    logger.info(f"Found {len(files)} audio files in {input_dir}")
    return files


def convert_to_wav(audio_path: Path, output_dir: Path, sr: int = 22050) -> Path:
    """Convert any audio file to mono WAV at target sample rate via ffmpeg."""
    output_dir.mkdir(parents=True, exist_ok=True)
    wav_path = output_dir / f"{audio_path.stem}.wav"

    if wav_path.exists():
        logger.debug(f"WAV cache hit: {wav_path.name}")
        return wav_path

    cmd = [
        "ffmpeg", "-y", "-i", str(audio_path),
        "-ar", str(sr), "-ac", "1", "-sample_fmt", "s16",
        str(wav_path),
    ]

    try:
        subprocess.run(cmd, capture_output=True, check=True, timeout=120)
        logger.info(f"Converted: {audio_path.name} -> {wav_path.name}")
    except subprocess.CalledProcessError as e:
        logger.error(f"ffmpeg failed for {audio_path.name}: {e.stderr.decode()[:200]}")
        raise
    except FileNotFoundError:
        raise RuntimeError("ffmpeg not found. Install: sudo apt install ffmpeg")

    return wav_path


def load_audio(wav_path: Path, sr: int = 22050):
    """Load WAV file as mono numpy array via librosa."""
    import librosa
    y, actual_sr = librosa.load(str(wav_path), sr=sr, mono=True)
    logger.debug(f"Loaded {wav_path.name}: {len(y)/actual_sr:.1f}s @ {actual_sr}Hz")
    return y, actual_sr
