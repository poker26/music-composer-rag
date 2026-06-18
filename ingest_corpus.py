"""
Step 4 — build a retrieval index from a CLEAN symbolic score corpus.

Parses real scores (Humdrum **kern or MIDI) with music21, extracts key-independent
material — harmonic progressions (Roman numerals), cadences, and melodic motifs
(interval sequences) — and aggregates them by frequency into corpus/index/<composer>.json.

That JSON is the only thing generation needs at runtime (see src/retrieval.py); music21
is NOT a runtime dependency. Raw scores live under corpus/raw/ (gitignored, cloned).

Usage:
    pip install -r requirements-corpus.txt
    python ingest_corpus.py                 # all configured composers
    python ingest_corpus.py --composer Chopin --limit 30

Corpus sources (Craig Sapp / KernScores, engraving-derived, public domain):
    corpus/raw/bach-370-chorales        -> Bach
    corpus/raw/beethoven-piano-sonatas  -> Beethoven
    corpus/raw/chopin-mazurkas          -> Chopin
    corpus/raw/chopin-preludes          -> Chopin
"""

from __future__ import annotations

import argparse
import collections
import json
import logging
import re
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ingest_corpus")

RAW_DIR = Path(__file__).parent / "corpus" / "raw"
INDEX_DIR = Path(__file__).parent / "corpus" / "index"

# Which raw repos feed which composer profile.
COMPOSER_SOURCES = {
    "Bach": ["bach-370-chorales"],
    "Beethoven": ["beethoven-piano-sonatas"],
    "Chopin": ["chopin-mazurkas", "chopin-preludes"],
}

# Bound the work per piece — full Roman-numeral analysis is expensive.
MAX_CHORDS_PER_PIECE = 120
PROGRESSION_NGRAM = (3, 4)
MOTIF_NGRAM = 4
MOTIF_MIN_MOVES = 2  # drop static fragments (e.g. [0,0,0,0] repeated notes)
TONIC_DEGREES = {"i", "I"}
TOP_PROGRESSIONS = 24
TOP_CADENCES = 12
TOP_MOTIFS = 24

_FIG_CORE = re.compile(r"^[b#]*[ivIV]+")  # leading accidentals + roman degree


def simplify_figure(figure):
    """'vi4#2' -> 'vi', 'bVI7532' -> 'bVI', 'V7' -> 'V'. Keep the functional degree only."""
    m = _FIG_CORE.match(figure or "")
    return m.group(0) if m else None


def _dedup_consecutive(seq):
    out = []
    for x in seq:
        if not out or out[-1] != x:
            out.append(x)
    return out


def analyze_piece(path):
    """Return (key_mode, roman_degrees, melodic_intervals) for one score, or None on failure."""
    from music21 import converter, roman

    score = converter.parse(str(path))
    key = score.analyze("key")

    # --- harmony: chordify -> simplified Roman degrees ---
    degrees = []
    chords = score.chordify().recurse().getElementsByClass("Chord")
    for c in chords[:MAX_CHORDS_PER_PIECE]:
        try:
            fig = roman.romanNumeralFromChord(c, key).figure
        except Exception:
            continue
        deg = simplify_figure(fig)
        if deg:
            degrees.append(deg)
    degrees = _dedup_consecutive(degrees)

    # --- melody: top-sounding part -> consecutive interval sequence ---
    intervals = []
    if score.parts:
        top = max(score.parts, key=lambda p: _avg_pitch(p))
        pitches = [n.pitch.midi for n in top.recurse().notes if n.isNote]
        intervals = [pitches[i] - pitches[i - 1] for i in range(1, len(pitches))]
        intervals = [iv for iv in intervals if abs(iv) <= 16]  # drop part-crossing jumps

    return key.mode, degrees, intervals


def _avg_pitch(part):
    ps = [n.pitch.midi for n in part.recurse().notes if n.isNote]
    return sum(ps) / len(ps) if ps else 0


def _ngrams(seq, n):
    return [tuple(seq[i:i + n]) for i in range(len(seq) - n + 1)]


def build_composer_index(composer, repos, limit):
    files = []
    for repo in repos:
        kern_dir = RAW_DIR / repo / "kern"
        root = kern_dir if kern_dir.is_dir() else RAW_DIR / repo
        files.extend(sorted(root.rglob("*.krn")) + sorted(root.rglob("*.mid")))
    if not files:
        logger.warning("%s: no source files under %s", composer, [str(RAW_DIR / r) for r in repos])
        return None

    available = len(files)
    if limit and available > limit:
        files = files[:limit]
        logger.info("%s: analyzing %d of %d available scores (--limit %d)", composer, len(files), available, limit)
    else:
        logger.info("%s: analyzing all %d scores", composer, available)

    prog_counts = collections.Counter()
    bigram_counts = collections.Counter()
    motif_counts = collections.Counter()
    mode_counts = collections.Counter()
    ok = 0
    for i, f in enumerate(files, 1):
        try:
            mode, degrees, intervals = analyze_piece(f)
        except Exception as e:
            logger.debug("  skip %s: %s", f.name, e)
            continue
        ok += 1
        mode_counts[mode] += 1
        for g in _ngrams(degrees, 2):
            bigram_counts[g] += 1
        for n in PROGRESSION_NGRAM:
            for g in _ngrams(degrees, n):
                prog_counts[g] += 1
        for g in _ngrams(intervals, MOTIF_NGRAM):
            if sum(1 for iv in g if iv != 0) >= MOTIF_MIN_MOVES:
                motif_counts[g] += 1
        if i % 20 == 0:
            logger.info("  %s: %d/%d parsed", composer, i, len(files))

    # Cadences: most common harmonic moves that resolve onto the tonic.
    cadence_counts = collections.Counter(
        {g: c for g, c in bigram_counts.items() if g[-1] in TONIC_DEGREES and g[0] not in TONIC_DEGREES}
    )

    if not ok:
        logger.warning("%s: nothing parsed", composer)
        return None

    index = {
        "composer": composer,
        "source_repos": repos,
        "scores_available": available,
        "scores_analyzed": ok,
        "modes": dict(mode_counts),
        "progressions": [{"roman": list(g), "count": c} for g, c in prog_counts.most_common(TOP_PROGRESSIONS)],
        "cadences": [{"roman": list(g), "count": c} for g, c in cadence_counts.most_common(TOP_CADENCES)],
        "melodic_motifs": [{"intervals": list(g), "count": c} for g, c in motif_counts.most_common(TOP_MOTIFS)],
    }
    return index


def main():
    ap = argparse.ArgumentParser(description="Build the Step-4 retrieval index from symbolic scores")
    ap.add_argument("--composer", "-c", help="Only this composer (default: all configured)")
    ap.add_argument("--limit", "-l", type=int, default=40, help="Max scores per composer (0 = no cap)")
    args = ap.parse_args()

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    composers = [args.composer] if args.composer else list(COMPOSER_SOURCES)
    for composer in composers:
        repos = COMPOSER_SOURCES.get(composer)
        if not repos:
            logger.warning("No corpus configured for %s (have: %s)", composer, list(COMPOSER_SOURCES))
            continue
        index = build_composer_index(composer, repos, args.limit)
        if not index:
            continue
        out = INDEX_DIR / f"{composer.lower()}.json"
        out.write_text(json.dumps(index, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("%s -> %s (%d progressions, %d motifs, from %d scores)",
                    composer, out, len(index["progressions"]), len(index["melodic_motifs"]), index["scores_analyzed"])


if __name__ == "__main__":
    main()
