"""
A/B harness: does Step-4 corpus retrieval measurably change the output?

For each test case it generates the piece twice — corpus ON and corpus OFF
(via the MUSIC_DISABLE_CORPUS ablation switch) — then compares:
  - the symbolic metrics from src.evaluator (in-key, smoothness, parallels, reprise)
  - corpus_style_match: fraction of the generated harmony's degree bigrams that
    match the composer's characteristic progressions in the corpus index. This is
    the metric that should move if retrieval is actually steering the harmony.

Needs ANTHROPIC_API_KEY (or a local .env) and music21 (requirements-corpus.txt).

    python eval_ab.py                 # default small batch
    python eval_ab.py --reps 2
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
logging.getLogger("src.composer_architect").setLevel(logging.WARNING)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("eval_ab")

BASE = Path(__file__).parent


def _load_dotenv():
    env = BASE / ".env"
    if env.is_file():
        for line in env.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


_load_dotenv()

from src.style_profiler import load_static_profile, static_profile_to_prompt_text
from src.composer_architect import generate_blueprint
from src.evaluator import evaluate_blueprint
from src import retrieval
from ingest_corpus import simplify_figure

# Default cases (composers that have a corpus index; skip slow sonata).
CASES = [
    {"composer": "Chopin", "form": "nocturne", "key": "E minor", "tempo": 72, "duration": 60},
    {"composer": "Bach", "form": "fugue", "key": "D minor", "tempo": 84, "duration": 60},
    {"composer": "Beethoven", "form": "prelude", "key": "C minor", "tempo": 100, "duration": 50},
]


def corpus_bigrams(composer):
    """Set of degree bigrams that appear in the composer's characteristic progressions."""
    index = retrieval.load_index(composer)
    if not index:
        return set()
    bigrams = set()
    for prog in index.get("progressions", []):
        roman = prog["roman"]
        for i in range(len(roman) - 1):
            bigrams.add((roman[i], roman[i + 1]))
    return bigrams


def blueprint_degree_sequence(blueprint):
    """Section chord_progressions -> simplified Roman degrees, via music21 in the piece's key."""
    from music21 import harmony, key as m21key, roman

    parts = (blueprint.get("key") or "C major").split()
    tonic = parts[0]
    mode = "minor" if len(parts) > 1 and parts[1].lower().startswith("min") else "major"
    k = m21key.Key(tonic, mode)

    degrees = []
    for sec in blueprint.get("sections", []):
        for ch in sec.get("chord_progression", []) or []:
            try:
                cs = harmony.ChordSymbol(ch)
                deg = simplify_figure(roman.romanNumeralFromChord(cs, k).figure)
            except Exception:
                deg = None
            if deg and (not degrees or degrees[-1] != deg):
                degrees.append(deg)
    return degrees


def style_match(blueprint, composer):
    """Fraction of generated harmony bigrams that match the corpus's characteristic ones."""
    corpus = corpus_bigrams(composer)
    if not corpus:
        return None
    degrees = blueprint_degree_sequence(blueprint)
    gen = [(degrees[i], degrees[i + 1]) for i in range(len(degrees) - 1)]
    if not gen:
        return 0.0
    hits = sum(1 for bg in gen if bg in corpus)
    return hits / len(gen)


def run_one(case, use_corpus):
    if use_corpus:
        os.environ.pop("MUSIC_DISABLE_CORPUS", None)
    else:
        os.environ["MUSIC_DISABLE_CORPUS"] = "1"

    profile = load_static_profile(case["composer"])
    profile_text = static_profile_to_prompt_text(profile)
    params = {
        "composer": case["composer"], "key": case["key"], "tempo_bpm": case["tempo"],
        "duration_sec": case["duration"], "mood": "expressive",
        "description": f"A {case['form']} in the style of {case['composer']}",
        "instruments": "piano", "time_signature": [4, 4],
    }
    bp = generate_blueprint(profile_text, params, form_id=case["form"])
    m = evaluate_blueprint(bp)
    return {
        "in_key": m["in_key"]["ratio"],
        "smooth_stepwise": m["voice_leading"]["pct_stepwise"],
        "parallels": m["parallels"]["parallel_fifths"] + m["parallels"]["parallel_octaves"],
        "style_match": style_match(bp, case["composer"]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=1, help="repetitions per condition (averaged)")
    ap.add_argument("--out", default="eval_ab_results.json")
    args = ap.parse_args()

    results = []
    for case in CASES:
        label = f"{case['composer']}/{case['form']}"
        for cond in ("on", "off"):
            runs = []
            for r in range(args.reps):
                logger.info("%s corpus=%s rep %d/%d ...", label, cond, r + 1, args.reps)
                try:
                    runs.append(run_one(case, use_corpus=(cond == "on")))
                except Exception as e:
                    logger.error("  failed: %s", e)
            if runs:
                avg = {k: (sum(x[k] for x in runs if x[k] is not None) / len(runs)) for k in runs[0]}
                results.append({"case": label, "corpus": cond, "reps": len(runs), **avg})

    (BASE / args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")

    # Pretty table.
    print("\n" + "=" * 78)
    print(f"{'case':<20}{'corpus':<8}{'in_key':>8}{'stepwise':>10}{'parallels':>11}{'style_match':>13}")
    print("-" * 78)
    for r in results:
        sm = f"{r['style_match']*100:.0f}%" if r.get("style_match") is not None else "-"
        print(f"{r['case']:<20}{r['corpus']:<8}{r['in_key']*100:>7.1f}%{r['smooth_stepwise']:>9.0f}%"
              f"{r['parallels']:>11.1f}{sm:>13}")
    print("=" * 78)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
