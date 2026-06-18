"""
Step 4 — runtime retrieval of real reference material for a section prompt.

Loads the per-composer JSON index built by ingest_corpus.py and formats a compact
block of REAL material (harmonic progressions, cadences, melodic motifs) drawn from
that composer's actual scores. The material is key-independent (Roman numerals and
semitone intervals), so the model transposes it into the target key itself.

Stdlib only — music21 is NOT needed at runtime. If no index exists for a composer,
retrieval is a no-op and the prompt is unchanged (graceful fallback).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

INDEX_DIR = Path(__file__).parent.parent / "corpus" / "index"

_CACHE = {}


def load_index(composer):
    """Load corpus/index/<composer>.json, or None if absent. Cached."""
    key = (composer or "").lower()
    if key in _CACHE:
        return _CACHE[key]
    path = INDEX_DIR / f"{key}.json"
    index = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None
    _CACHE[key] = index
    return index


def has_corpus(composer):
    return load_index(composer) is not None


def _fmt_intervals(intervals):
    """[2, -2, -1] -> '+2 -2 -1' (semitone steps, − for descent)."""
    return " ".join(f"+{iv}" if iv > 0 else str(iv) for iv in intervals)


def _rotate(items, offset, n):
    """Deterministic slice so different sections see different material (no RNG)."""
    if not items:
        return []
    offset %= len(items)
    return [items[(offset + i) % len(items)] for i in range(min(n, len(items)))]


def reference_block(composer, section_index=0, n_progressions=6, n_cadences=4, n_motifs=5):
    """Formatted prompt text with real material for this composer, or '' if no corpus."""
    if os.environ.get("MUSIC_DISABLE_CORPUS"):
        return ""  # ablation switch for A/B measurement
    index = load_index(composer)
    if not index:
        return ""

    progs = _rotate(index.get("progressions", []), section_index * 2, n_progressions)
    cads = index.get("cadences", [])[:n_cadences]  # cadences: always the strongest few
    motifs = _rotate(index.get("melodic_motifs", []), section_index * 3, n_motifs)

    lines = [
        f"Real reference material from {composer}'s own scores "
        f"({index.get('scores_analyzed', '?')} pieces analyzed). Use these idiomatically and "
        f"transpose the Roman numerals into the target key - do NOT copy them literally:",
    ]
    if progs:
        lines.append("  - Characteristic progressions: "
                     + "; ".join(" ".join(p["roman"]) for p in progs))
    if cads:
        lines.append("  - Typical cadences: "
                     + "; ".join(" ".join(c["roman"]) for c in cads))
    if motifs:
        lines.append("  - Melodic motif shapes (semitone steps, - = descending): "
                     + "; ".join(f"[{_fmt_intervals(m['intervals'])}]" for m in motifs))
    return "\n".join(lines)


def _main(argv):
    composer = argv[0] if argv else "Chopin"
    block = reference_block(composer, section_index=0)
    print(block or f"(no corpus index for {composer} — run ingest_corpus.py)")
    return 0


if __name__ == "__main__":
    import sys
    raise SystemExit(_main(sys.argv[1:]))
