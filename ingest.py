#!/usr/bin/env python3
"""
Music Composer RAG - Ingestion Pipeline
Usage:
  python ingest.py --input ./input/chopin --composer "Chopin" --era "romantic"
  python ingest.py --input ./input/bach --composer "Bach" --era "baroque" --skip-midi
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config.settings import (
    INPUT_DIR, MIDI_DIR, FEATURES_DIR, WAV_DIR,
    SAMPLE_RATE, HOP_LENGTH, N_MELS, N_CHROMA,
    FRAGMENT_DURATION_SEC, FRAGMENT_OVERLAP_SEC,
    EMBEDDING_DIM, COLLECTION_NAME, QDRANT_HOST, QDRANT_PORT,
)
from src.audio_loader import scan_audio_files, convert_to_wav, load_audio
from src.feature_extractor import fragment_audio, extract_all_features
from src.midi_transcriber import transcribe_to_midi
from src.midi_analyzer import analyze_midi
from src.embedder import build_embedding, build_metadata_payload
from src.qdrant_store import MusicVectorStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("ingest")


def process_file(wav_path, metadata, store, skip_midi=False):
    """Process a single WAV file: fragment -> features -> embed -> store."""
    y, sr = load_audio(wav_path, SAMPLE_RATE)
    logger.info("Processing: %s (%.1fs)", wav_path.name, len(y) / sr)

    # Optional MIDI transcription
    midi_features = {}
    if not skip_midi:
        try:
            midi_path = transcribe_to_midi(wav_path, MIDI_DIR)
            midi_features = analyze_midi(midi_path)
            logger.info("  MIDI: %d notes detected", midi_features.get("note_count", 0))
        except Exception as e:
            logger.warning("  MIDI transcription failed: %s", e)

    # Fragment and process
    fragments = list(fragment_audio(y, sr, FRAGMENT_DURATION_SEC, FRAGMENT_OVERLAP_SEC))
    logger.info("  Split into %d fragments", len(fragments))

    batch = []
    for i, (frag_audio, start_sec, end_sec) in enumerate(fragments):
        features = extract_all_features(frag_audio, sr, N_MELS, N_CHROMA, HOP_LENGTH)
        features.update(midi_features)

        embedding = build_embedding(features)

        source_info = {
            **metadata,
            "file": wav_path.name,
            "fragment_index": i,
            "start_sec": start_sec,
            "end_sec": end_sec,
        }
        payload = build_metadata_payload(features, source_info)
        batch.append({"embedding": embedding, "payload": payload})

    if batch:
        store.upsert_batch(batch)

    return len(batch)


def main():
    parser = argparse.ArgumentParser(description="Ingest audio into Music Composer RAG")
    parser.add_argument("--input", "-i", type=str, default=str(INPUT_DIR))
    parser.add_argument("--composer", "-c", type=str, default="unknown")
    parser.add_argument("--era", type=str, default="")
    parser.add_argument("--genre", type=str, default="classical")
    parser.add_argument("--instrument", type=str, default="")
    parser.add_argument("--skip-midi", action="store_true")
    parser.add_argument("--qdrant-host", type=str, default=QDRANT_HOST)
    parser.add_argument("--qdrant-port", type=int, default=QDRANT_PORT)
    args = parser.parse_args()

    input_dir = Path(args.input)
    if not input_dir.exists():
        logger.error("Input directory not found: %s", input_dir)
        sys.exit(1)

    for d in [MIDI_DIR, FEATURES_DIR, WAV_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    store = MusicVectorStore(
        host=args.qdrant_host, port=args.qdrant_port,
        collection=COLLECTION_NAME, dim=EMBEDDING_DIM,
    )
    store.ensure_collection()

    audio_files = scan_audio_files(input_dir)
    if not audio_files:
        logger.warning("No audio files found!")
        sys.exit(0)

    metadata = {
        "composer": args.composer,
        "era": args.era,
        "genre": args.genre,
        "instrument": args.instrument,
    }

    total_fragments = 0
    for audio_path in audio_files:
        try:
            wav_path = convert_to_wav(audio_path, WAV_DIR, SAMPLE_RATE)
            n = process_file(wav_path, metadata, store, args.skip_midi)
            total_fragments += n
        except Exception as e:
            logger.error("Failed to process %s: %s", audio_path.name, e)
            continue

    stats = store.get_stats()
    logger.info("=" * 50)
    logger.info("Ingestion complete!")
    logger.info("  Files processed: %d", len(audio_files))
    logger.info("  Fragments stored: %d", total_fragments)
    logger.info("  Total in Qdrant:  %d", stats["total_points"])
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
