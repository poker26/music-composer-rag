#!/usr/bin/env python3
"""
Music Composer RAG - Generate

Generation is driven by a hand-authored style profile (profiles/<composer>.json)
plus a musical form template (forms/<form>.json). The only requirement to run
this is ANTHROPIC_API_KEY.

Usage:
  python3 generate.py --composer "Chopin" --key "E minor" --tempo 72 --duration 90 --mood "melancholic" --form nocturne
  python3 generate.py --composer "Bach" --key "D minor" --tempo 100 --duration 120 --form fugue
  python3 generate.py --composer "Beethoven" --key "C minor" --tempo 140 --duration 180 --form sonata
  python3 generate.py --composer "Debussy" --key "Db major" --tempo 60 --duration 90 --form prelude
  python3 generate.py --list-forms
  python3 generate.py --list-composers
"""
import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config.settings import OUTPUT_DIR
from src.style_profiler import (
    load_static_profile, static_profile_to_prompt_text, list_static_profiles,
)
from src.composer_architect import generate_blueprint, list_available_forms
from src.midi_builder import build_midi
from src.evaluator import evaluate_blueprint, format_report

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("generate")


def main():
    parser = argparse.ArgumentParser(description="Generate music in a composer's style")
    parser.add_argument("--composer", "-c", type=str, help="Composer name (must have a profile in profiles/)")
    parser.add_argument("--key", "-k", type=str, default="C minor")
    parser.add_argument("--tempo", "-t", type=int, default=100)
    parser.add_argument("--duration", "-d", type=int, default=60, help="Duration in seconds")
    parser.add_argument("--mood", "-m", type=str, default="expressive")
    parser.add_argument("--description", type=str, default="")
    parser.add_argument("--instruments", type=str, default="piano")
    parser.add_argument("--form", "-f", type=str, default=None,
                        help="Musical form: nocturne, sonata, fugue, prelude")
    parser.add_argument("--output", "-o", type=str, default=None)
    parser.add_argument("--save-blueprint", action="store_true")
    parser.add_argument("--no-eval", action="store_true", help="Skip the symbolic evaluation report")
    parser.add_argument("--list-forms", action="store_true", help="List available forms and exit")
    parser.add_argument("--list-composers", action="store_true", help="List available composer profiles and exit")
    args = parser.parse_args()

    # List forms mode
    if args.list_forms:
        forms = list_available_forms()
        if forms:
            print("\nAvailable musical forms:")
            print("-" * 50)
            for fid, info in sorted(forms.items()):
                print(f"  {fid:12s}  {info['name']}")
                print(f"               {info['description'][:80]}")
                print()
        else:
            print("No forms found. Check the forms/ directory.")
        return

    # List composers mode
    if args.list_composers:
        profiles = list_static_profiles()
        if profiles:
            print("\nAvailable composer profiles:")
            print("-" * 50)
            for name, data in profiles.items():
                print(f"  {name:14s}  {data.get('era', '')}")
        else:
            print("No profiles found. Add JSON files to the profiles/ directory.")
        return

    if not args.composer:
        parser.error("--composer is required (or use --list-forms / --list-composers)")

    # 1. Load style profile (hand-authored, no DB needed)
    logger.info("Loading style profile for %s...", args.composer)
    profile = load_static_profile(args.composer)
    if not profile:
        available = ", ".join(list_static_profiles().keys()) or "none"
        logger.error("No profile for '%s'. Available: %s", args.composer, available)
        sys.exit(1)

    profile_text = static_profile_to_prompt_text(profile)
    logger.info("Style profile:\n%s", profile_text)

    # 2. Generate blueprint via Claude API
    form_name = args.form or "free form"
    logger.info("Generating %s in %s via Claude API...", form_name, args.key)

    params = {
        "key": args.key,
        "tempo_bpm": args.tempo,
        "duration_sec": args.duration,
        "mood": args.mood,
        "description": args.description or f"A {form_name} in the style of {args.composer}",
        "instruments": args.instruments,
        "time_signature": [4, 4],
    }

    blueprint = generate_blueprint(profile_text, params, form_id=args.form)

    # Save blueprint
    if args.save_blueprint:
        safe_title = "".join(c if c.isalnum() or c in " -_" else "" for c in blueprint.get("title", "untitled"))
        safe_title = safe_title.strip().replace(" ", "_") or "untitled"
        bp_path = OUTPUT_DIR / "blueprints" / f"{safe_title}.json"
        bp_path.parent.mkdir(parents=True, exist_ok=True)
        with open(bp_path, "w") as f:
            json.dump(blueprint, f, indent=2, ensure_ascii=False)
        logger.info("Blueprint saved: %s", bp_path)

    # 3. Build MIDI
    if args.output:
        midi_path = Path(args.output)
    else:
        safe_title = "".join(c if c.isalnum() or c in " -_" else "" for c in blueprint.get("title", "generated"))
        safe_title = safe_title.strip().replace(" ", "_") or "generated"
        midi_path = OUTPUT_DIR / "generated" / f"{safe_title}.mid"

    logger.info("Building MIDI...")
    result = build_midi(blueprint, midi_path)

    logger.info("=" * 50)
    logger.info("Generation complete!")
    logger.info("  Title: %s", blueprint.get("title", "Untitled"))
    logger.info("  Form: %s", blueprint.get("form", form_name))
    logger.info("  Key: %s", blueprint.get("key"))
    logger.info("  Tempo: %s BPM", blueprint.get("tempo_bpm"))
    logger.info("  Sections: %d", len(blueprint["sections"]))
    logger.info("  Tracks: %d", len(blueprint["tracks"]))
    logger.info("  Output: %s", result)
    logger.info("=" * 50)

    # 4. Symbolic evaluation (objective signal for tuning prompts)
    if not args.no_eval:
        print("\n" + format_report(evaluate_blueprint(blueprint)))


if __name__ == "__main__":
    main()
