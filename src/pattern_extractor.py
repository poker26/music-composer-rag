"""
Pattern Extractor
Extracts real musical patterns from transcribed MIDI files in the database:
- Accompaniment figures (arpeggios, block chords, alberti bass)
- Bass patterns
- Rhythmic cells
- Velocity/dynamics curves

These patterns are used by the MIDI builder for authentic accompaniment.
"""
import logging
import numpy as np
from pathlib import Path
from collections import defaultdict

logger = logging.getLogger(__name__)


def extract_patterns_from_midi(midi_path, ticks_per_beat=480):
    """
    Extract musical patterns from a MIDI file.
    Returns dict with categorized patterns.
    """
    import mido

    mid = mido.MidiFile(str(midi_path))
    tpb = mid.ticks_per_beat or ticks_per_beat

    # Parse all notes: (start_tick, pitch, velocity, duration_ticks)
    all_notes = []
    for track in mid.tracks:
        tick = 0
        note_ons = {}
        for msg in track:
            tick += msg.time
            if msg.type == "note_on" and msg.velocity > 0:
                note_ons[msg.note] = (tick, msg.velocity)
            elif msg.type in ("note_off", "note_on"):
                if msg.type == "note_off" or msg.velocity == 0:
                    if msg.note in note_ons:
                        start, vel = note_ons.pop(msg.note)
                        all_notes.append((start, msg.note, vel, tick - start))

    if not all_notes:
        return {"accompaniment": [], "bass": [], "rhythm_cells": []}

    all_notes.sort(key=lambda x: x[0])

    # Split into bars (assuming 4/4)
    bar_ticks = tpb * 4
    bars = defaultdict(list)
    for start, pitch, vel, dur in all_notes:
        bar_num = start // bar_ticks
        bar_offset = start % bar_ticks
        bars[bar_num].append((bar_offset, pitch, vel, dur))

    # Separate layers by pitch register
    accompaniment_patterns = []
    bass_patterns = []
    rhythm_cells = []

    for bar_num in sorted(bars.keys()):
        notes = bars[bar_num]
        if len(notes) < 2:
            continue

        pitches = [n[1] for n in notes]
        median_pitch = np.median(pitches)

        # Bass notes: below median - 6
        bass_notes = [(off, p, v, d) for off, p, v, d in notes if p < median_pitch - 6]
        # Mid-range: accompaniment
        mid_notes = [(off, p, v, d) for off, p, v, d in notes if median_pitch - 6 <= p <= median_pitch + 6]
        # Upper: melody (skip for pattern extraction)

        # Normalize to relative pitches (intervals from lowest note)
        if mid_notes and len(mid_notes) >= 3:
            base_pitch = min(n[1] for n in mid_notes)
            pattern = []
            for off, p, v, d in mid_notes:
                pattern.append({
                    "offset": off / tpb,  # in beats
                    "interval": p - base_pitch,
                    "velocity_ratio": v / 127.0,
                    "duration": d / tpb,  # in beats
                })
            accompaniment_patterns.append({
                "notes": pattern,
                "note_count": len(pattern),
                "span_semitones": max(n[1] for n in mid_notes) - base_pitch,
            })

        if bass_notes and len(bass_notes) >= 1:
            base_pitch = min(n[1] for n in bass_notes)
            pattern = []
            for off, p, v, d in bass_notes:
                pattern.append({
                    "offset": off / tpb,
                    "interval": p - base_pitch,
                    "velocity_ratio": v / 127.0,
                    "duration": d / tpb,
                })
            bass_patterns.append({
                "notes": pattern,
                "note_count": len(pattern),
            })

        # Rhythm cell: just the timing pattern (offsets and durations)
        if len(notes) >= 4:
            cell = []
            for off, p, v, d in notes:
                cell.append({
                    "offset": off / tpb,
                    "duration": d / tpb,
                    "velocity_ratio": v / 127.0,
                })
            rhythm_cells.append(cell)

    return {
        "accompaniment": accompaniment_patterns[:50],  # Limit to 50 patterns
        "bass": bass_patterns[:50],
        "rhythm_cells": rhythm_cells[:50],
    }


def collect_composer_patterns(midi_dir, composer_name=None, max_files=20):
    """
    Collect patterns from all MIDI files in a directory.
    Returns aggregated pattern library.
    """
    midi_dir = Path(midi_dir)
    midi_files = list(midi_dir.glob("*.mid"))

    if not midi_files:
        logger.warning("No MIDI files found in %s", midi_dir)
        return {"accompaniment": [], "bass": [], "rhythm_cells": []}

    logger.info("Extracting patterns from %d MIDI files in %s",
                min(len(midi_files), max_files), midi_dir)

    all_acc = []
    all_bass = []
    all_rhythm = []

    for midi_path in midi_files[:max_files]:
        try:
            patterns = extract_patterns_from_midi(midi_path)
            all_acc.extend(patterns["accompaniment"])
            all_bass.extend(patterns["bass"])
            all_rhythm.extend(patterns["rhythm_cells"])
        except Exception as e:
            logger.warning("Failed to extract from %s: %s", midi_path.name, e)

    logger.info("Collected %d accompaniment, %d bass, %d rhythm patterns",
                len(all_acc), len(all_bass), len(all_rhythm))

    return {
        "accompaniment": all_acc,
        "bass": all_bass,
        "rhythm_cells": all_rhythm,
    }


def apply_pattern_to_chord(pattern, root_midi, register_low, register_high, velocity_base=80):
    """
    Apply an extracted pattern to a specific chord root.
    Returns list of (beat_offset, pitch, velocity, duration_beats).
    """
    notes = []
    for n in pattern["notes"]:
        pitch = root_midi + n["interval"]
        # Transpose to fit register
        while pitch < register_low:
            pitch += 12
        while pitch > register_high:
            pitch -= 12

        vel = int(n["velocity_ratio"] * velocity_base)
        vel = max(20, min(127, vel))

        notes.append((n["offset"], pitch, vel, n["duration"]))

    return notes
