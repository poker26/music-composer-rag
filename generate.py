#!/usr/bin/env python3
"""
Music Composer RAG - Generate composition in a composer's style
Usage:
  python3 generate.py --composer "Chopin" --key "C minor" --tempo 72 --duration 60 --mood "melancholic"
  python3 generate.py --composer "Bach" --key "D major" --tempo 100 --duration 45 --mood "joyful, fugue-like"
"""
import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config.settings import (
    QDRANT_HOST, QDRANT_PORT, COLLECTION_NAME, EMBEDDING_DIM, OUTPUT_DIR,
)
from src.qdrant_store import MusicVectorStore
from src.style_profiler import build_style_profile, profile_to_prompt_text
from src.composer_architect import generate_blueprint
from src.midi_builder import build_midi

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("generate")


def main():
    parser = argparse.ArgumentParser(description="Generate music in a composer's style")
    parser.add_argument("--composer", "-c", type=str, required=True, help="Composer name (must exist in DB)")
    parser.add_argument("--key", "-k", type=str, default="C minor", help="Musical key")
    parser.add_argument("--tempo", "-t", type=int, default=100, help="Tempo in BPM")
    parser.add_argument("--duration", "-d", type=int, default=60, help="Duration in seconds")
    parser.add_argument("--mood", "-m", type=str, default="expressive", help="Mood description")
    parser.add_argument("--description", type=str, default="", help="Additional description")
    parser.add_argument("--instruments", type=str, default="piano solo", help="Instrumentation")
    parser.add_argument("--output", "-o", type=str, default=None, help="Output MIDI file path")
    parser.add_argument("--save-blueprint", action="store_true", help="Save blueprint JSON")
    args = parser.parse_args()

    # 1. Connect to Qdrant
    logger.info("Connecting to Qdrant...")
    store = MusicVectorStore(
        host=QDRANT_HOST, port=QDRANT_PORT,
        collection=COLLECTION_NAME, dim=EMBEDDING_DIM,
    )

    # 2. Build style profile
    logger.info("Building style profile for %s...", args.composer)
    profile = build_style_profile(store, args.composer)
    if not profile:
        logger.error("No data found for composer '%s'. Run ingestion first.", args.composer)
        sys.exit(1)

    profile_text = profile_to_prompt_text(profile)
    logger.info("Style profile:\n%s", profile_text)

    # 3. Generate blueprint via Claude API
    logger.info("Generating composition blueprint via Claude API...")
    params = {
        "key": args.key,
        "tempo_bpm": args.tempo,
        "duration_sec": args.duration,
        "mood": args.mood,
        "description": args.description or f"A piece in the style of {args.composer}",
        "instruments": args.instruments,
        "time_signature": [4, 4],
    }

    blueprint = generate_blueprint(profile_text, params)
    logger.info("Blueprint: '%s' - %d sections, %d tracks",
                blueprint.get("title", "Untitled"),
                len(blueprint["sections"]),
                len(blueprint["tracks"]))

    # Save blueprint if requested
    if args.save_blueprint:
        bp_path = OUTPUT_DIR / "blueprints" / f"{blueprint.get('title', 'untitled')}.json"
        bp_path.parent.mkdir(parents=True, exist_ok=True)
        with open(bp_path, "w") as f:
            json.dump(blueprint, f, indent=2)
        logger.info("Blueprint saved: %s", bp_path)

    # 4. Build MIDI
    if args.output:
        midi_path = Path(args.output)
    else:
        safe_title = "".join(c if c.isalnum() or c in " -_" else "" for c in blueprint.get("title", "generated"))
        safe_title = safe_title.strip().replace(" ", "_") or "generated"
        midi_path = OUTPUT_DIR / "generated" / f"{safe_title}.mid"

    logger.info("Building MIDI...")
    result = build_midi(blueprint, midi_path, style_profile=profile)

    logger.info("=" * 50)
    logger.info("Generation complete!")
    logger.info("  Title: %s", blueprint.get("title", "Untitled"))
    logger.info("  Key: %s", blueprint.get("key"))
    logger.info("  Tempo: %s BPM", blueprint.get("tempo_bpm"))
    logger.info("  Sections: %d", len(blueprint["sections"]))
    logger.info("  Tracks: %d", len(blueprint["tracks"]))
    logger.info("  Output: %s", result)
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
