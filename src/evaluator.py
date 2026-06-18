"""
Symbolic evaluation of a generated blueprint.

These are objective, no-API checks on the note data so we get a repeatable
signal when tuning prompts. None of them "grade" musicality on their own — they
flag concrete symbolic facts (how much of the melody is in key, how smooth the
line is, how closely a reprise restates its source, how many parallel perfect
intervals appear between the outer voices).

Run standalone on a saved blueprint:

    python -m src.evaluator output/blueprints/Nocturne_in_E_minor.json
"""

from __future__ import annotations

import difflib
import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# Diatonic scale degrees (semitones from tonic).
MAJOR = (0, 2, 4, 5, 7, 9, 11)
NATURAL_MINOR = (0, 2, 3, 5, 7, 8, 10)

_PITCH_CLASS = {
    "C": 0, "C#": 1, "DB": 1, "D": 2, "D#": 3, "EB": 3, "E": 4, "FB": 4,
    "E#": 5, "F": 5, "F#": 6, "GB": 6, "G": 7, "G#": 8, "AB": 8, "A": 9,
    "A#": 10, "BB": 10, "B": 11, "CB": 11,
}


# --------------------------------------------------------------------------- #
# Parsing helpers
# --------------------------------------------------------------------------- #
def parse_key(key_str):
    """'E minor' / 'Bb major' / 'F# minor' -> (tonic_pitch_class, 'major'|'minor')."""
    parts = (key_str or "C major").strip().split()
    tonic = parts[0].upper() if parts else "C"
    tonic = tonic.replace("♯", "#").replace("♭", "B") if tonic else "C"
    pc = _PITCH_CLASS.get(tonic, _PITCH_CLASS.get(tonic[:2], 0))
    mode = "minor" if len(parts) > 1 and parts[1].lower().startswith("min") else "major"
    return pc, mode


def allowed_pitch_classes(tonic_pc, mode):
    """Diatonic scale, plus the raised leading tone for minor (harmonic minor is idiomatic)."""
    degrees = NATURAL_MINOR if mode == "minor" else MAJOR
    allowed = {(tonic_pc + d) % 12 for d in degrees}
    if mode == "minor":
        allowed.add((tonic_pc + 11) % 12)  # raised 7th (leading tone)
    return allowed


def beats_per_bar(blueprint):
    ts = blueprint.get("time_signature") or [4, 4]
    return ts[0] if ts else 4


def _iter_voice_notes(section, voice):
    """Yield (abs_beat, pitch, duration, velocity) for one voice of one section."""
    bpb = section.get("_bpb", 4)
    for bar_entry in section.get(voice, []) or []:
        bar = bar_entry.get("bar", 1)
        for n in bar_entry.get("notes", []) or []:
            start = n.get("start_beat", 1.0)
            abs_beat = (bar - 1) * bpb + (start - 1.0)
            yield abs_beat, n["pitch"], n.get("duration", 0.0), n.get("velocity", 0)


def _all_notes(blueprint, voice):
    """All notes of a voice across the whole piece, time-sorted."""
    bpb = beats_per_bar(blueprint)
    notes = []
    for sec in blueprint.get("sections", []):
        sec["_bpb"] = bpb
        notes.extend(_iter_voice_notes(sec, voice))
    notes.sort(key=lambda t: t[0])
    return notes


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def in_key_ratio(blueprint):
    """Fraction of all sounding notes whose pitch class is in the key (idiomatic minor allowed)."""
    tonic, mode = parse_key(blueprint.get("key"))
    allowed = allowed_pitch_classes(tonic, mode)
    total = in_key = 0
    out_of_key_pcs = {}
    for voice in ("melody", "accompaniment", "bass"):
        for _, pitch, _, _ in _all_notes(blueprint, voice):
            total += 1
            pc = pitch % 12
            if pc in allowed:
                in_key += 1
            else:
                out_of_key_pcs[pc] = out_of_key_pcs.get(pc, 0) + 1
    ratio = (in_key / total) if total else 0.0
    return {"ratio": ratio, "in_key": in_key, "total": total, "out_of_key_pcs": out_of_key_pcs}


def voice_leading(blueprint, voice="melody"):
    """Smoothness of a single line: mean step size, % stepwise, largest leap."""
    notes = _all_notes(blueprint, voice)
    intervals = [abs(notes[i][1] - notes[i - 1][1]) for i in range(1, len(notes))]
    if not intervals:
        return {"mean_interval": 0.0, "pct_stepwise": 0.0, "max_leap": 0, "pct_leap_gt_octave": 0.0}
    stepwise = sum(1 for d in intervals if d <= 2)
    big = sum(1 for d in intervals if d > 12)
    return {
        "mean_interval": sum(intervals) / len(intervals),
        "pct_stepwise": 100.0 * stepwise / len(intervals),
        "max_leap": max(intervals),
        "pct_leap_gt_octave": 100.0 * big / len(intervals),
    }


def _reprise_pairs(blueprint):
    """Map each reprise section to its source section, by reprise_of or X_prime->X naming."""
    by_id = {s.get("form_id"): s for s in blueprint.get("sections", [])}
    pairs = []
    for sec in blueprint.get("sections", []):
        fid = sec.get("form_id", "") or ""
        src = sec.get("reprise_of")
        if not src and fid.endswith("_prime"):
            src = fid[: -len("_prime")]
        if not src and fid.endswith("'"):
            src = fid[:-1]
        if src and src in by_id and by_id[src] is not sec:
            pairs.append((sec, by_id[src]))
    return pairs


def _downbeat_skeleton(section):
    """Pitch of the first melody note in each bar — the structural outline of the theme."""
    by_bar = {}
    for bar_entry in section.get("melody", []) or []:
        notes = bar_entry.get("notes") or []
        if notes:
            first = min(notes, key=lambda n: n.get("start_beat", 1.0))
            by_bar[bar_entry.get("bar", 0)] = first["pitch"]
    return [by_bar[b] for b in sorted(by_bar)]


def reprise_similarity(blueprint):
    """For each reprise, how closely its melody restates the source.

    Two signals: the full ornamented note stream (harsh — a heavily decorated
    reprise scores low), and a downbeat 'skeleton' (the bar-by-bar structural
    outline, which should survive ornamentation if Theme A is still recognizable).
    """
    results = []
    for reprise, source in _reprise_pairs(blueprint):
        r_pitches = [p for _, p, _, _ in _iter_voice_notes(reprise, "melody")]
        s_pitches = [p for _, p, _, _ in _iter_voice_notes(source, "melody")]
        if not r_pitches or not s_pitches:
            continue
        pc_sim = difflib.SequenceMatcher(
            None, [p % 12 for p in s_pitches], [p % 12 for p in r_pitches]
        ).ratio()
        r_skel = [p % 12 for p in _downbeat_skeleton(reprise)]
        s_skel = [p % 12 for p in _downbeat_skeleton(source)]
        skel_sim = difflib.SequenceMatcher(None, s_skel, r_skel).ratio() if r_skel and s_skel else 0.0
        results.append({
            "reprise": reprise.get("form_id"),
            "source": source.get("form_id"),
            "pitch_class_similarity": pc_sim,
            "skeleton_similarity": skel_sim,
            "reprise_notes": len(r_pitches),
            "source_notes": len(s_pitches),
        })
    return results


def _sounding_pitch(notes, t, eps=1e-6):
    """Pitch sounding at absolute beat t in a time-sorted (abs_beat, pitch, dur, vel) list."""
    found = None
    for abs_beat, pitch, dur, _ in notes:
        if abs_beat - eps <= t < abs_beat + dur - eps:
            found = pitch  # last one wins -> the most recent onset
        elif abs_beat > t + eps:
            break
    return found


def parallel_perfects(blueprint):
    """Parallel perfect fifths/octaves between melody and bass (outer voices)."""
    melody = _all_notes(blueprint, "melody")
    bass = _all_notes(blueprint, "bass")
    if not melody or not bass:
        return {"parallel_fifths": 0, "parallel_octaves": 0, "examples": []}

    onsets = sorted({round(b, 4) for b, *_ in melody} | {round(b, 4) for b, *_ in bass})
    simultaneities = []
    for t in onsets:
        m = _sounding_pitch(melody, t)
        b = _sounding_pitch(bass, t)
        if m is not None and b is not None:
            simultaneities.append((t, m, b))

    fifths = octaves = 0
    examples = []
    for i in range(1, len(simultaneities)):
        (t0, m0, b0), (t1, m1, b1) = simultaneities[i - 1], simultaneities[i]
        iv0, iv1 = (m0 - b0) % 12, (m1 - b1) % 12
        moved = (m1 != m0) and (b1 != b0)
        same_dir = (m1 - m0) * (b1 - b0) > 0
        if not (moved and same_dir):
            continue
        if iv0 == 7 and iv1 == 7:
            fifths += 1
            if len(examples) < 5:
                examples.append({"type": "P5", "from_beat": t0, "to_beat": t1})
        elif iv0 == 0 and iv1 == 0:
            octaves += 1
            if len(examples) < 5:
                examples.append({"type": "P8", "from_beat": t0, "to_beat": t1})
    return {"parallel_fifths": fifths, "parallel_octaves": octaves, "examples": examples}


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def evaluate_blueprint(blueprint):
    return {
        "in_key": in_key_ratio(blueprint),
        "voice_leading": voice_leading(blueprint, "melody"),
        "reprise": reprise_similarity(blueprint),
        "parallels": parallel_perfects(blueprint),
    }


def format_report(metrics):
    ik = metrics["in_key"]
    vl = metrics["voice_leading"]
    pp = metrics["parallels"]
    lines = []
    lines.append("Evaluation (symbolic):")
    lines.append("-" * 50)
    lines.append(f"  In-key ratio:        {ik['ratio'] * 100:5.1f}%  ({ik['in_key']}/{ik['total']} notes)")
    lines.append(f"  Melody smoothness:   mean step {vl['mean_interval']:.2f} st, "
                 f"{vl['pct_stepwise']:.0f}% stepwise, max leap {vl['max_leap']} st")
    if metrics["reprise"]:
        for r in metrics["reprise"]:
            lines.append(f"  Reprise {r['reprise']} vs {r['source']}: "
                         f"{r['skeleton_similarity'] * 100:.0f}% skeleton, "
                         f"{r['pitch_class_similarity'] * 100:.0f}% full-stream "
                         f"({r['source_notes']}->{r['reprise_notes']} notes)")
    else:
        lines.append("  Reprise similarity:  (no reprise sections)")
    lines.append(f"  Parallel perfects:   {pp['parallel_fifths']} fifths, {pp['parallel_octaves']} octaves "
                 f"(melody vs bass)")
    lines.append("-" * 50)
    return "\n".join(lines)


def _main(argv):
    if not argv:
        print(__doc__)
        return 1
    path = Path(argv[0])
    blueprint = json.loads(path.read_text(encoding="utf-8"))
    print(format_report(evaluate_blueprint(blueprint)))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
