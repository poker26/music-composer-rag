#!/usr/bin/env python3
"""
Quick search CLI for testing the database.
Usage:
  python search.py --file query.wav --limit 5
  python search.py --file query.wav --composer "Chopin" --tempo 60-100
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config.settings import (
    SAMPLE_RATE, N_MELS, N_CHROMA, HOP_LENGTH,
    EMBEDDING_DIM, COLLECTION_NAME,
)
from src.audio_loader import load_audio
from src.feature_extractor import extract_all_features
from src.embedder import build_embedding
from src.qdrant_store import MusicVectorStore


def main():
    parser = argparse.ArgumentParser(description="Search music fragments")
    parser.add_argument("--file", "-f", type=str, required=True)
    parser.add_argument("--limit", "-n", type=int, default=5)
    parser.add_argument("--composer", "-c", type=str, default=None)
    parser.add_argument("--key", "-k", type=str, default=None)
    parser.add_argument("--tempo", type=str, default=None, help="e.g. 100-140")
    args = parser.parse_args()

    y, sr = load_audio(Path(args.file), SAMPLE_RATE)
    features = extract_all_features(y, sr, N_MELS, N_CHROMA, HOP_LENGTH)
    query_vec = build_embedding(features)

    tempo_range = None
    if args.tempo:
        lo, hi = args.tempo.split("-")
        tempo_range = (float(lo), float(hi))

    store = MusicVectorStore(collection=COLLECTION_NAME, dim=EMBEDDING_DIM)
    results = store.search_similar(
        query_vec, limit=args.limit,
        composer=args.composer, key=args.key, tempo_range=tempo_range,
    )

    print(f"\nTop {len(results)} results:\n")
    for r in results:
        p = r.payload
        print(
            f"  Score: {r.score:.4f} | {p['source_file']} "
            f"[{p['fragment_start_sec']:.1f}-{p['fragment_end_sec']:.1f}s] "
            f"| {p['composer']} | {p['key']} | {p['tempo_bpm']:.0f} BPM"
        )


if __name__ == "__main__":
    main()
