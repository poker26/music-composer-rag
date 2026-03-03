"""Analyze MIDI: chord progressions, intervals, note density, patterns."""
import logging
from pathlib import Path
from collections import Counter
import numpy as np

logger = logging.getLogger(__name__)

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def analyze_midi(midi_path):
    """Extract musical patterns from a MIDI file."""
    import mido

    mid = mido.MidiFile(str(midi_path))
    all_notes = []
    note_ons = {}

    for track in mid.tracks:
        current_tick = 0
        for msg in track:
            current_tick += msg.time
            if msg.type == "note_on" and msg.velocity > 0:
                note_ons[msg.note] = (current_tick, msg.velocity)
            elif msg.type in ("note_off", "note_on"):
                if msg.type == "note_off" or msg.velocity == 0:
                    if msg.note in note_ons:
                        start, vel = note_ons.pop(msg.note)
                        dur = current_tick - start
                        all_notes.append((start, msg.note, vel, dur))

    if not all_notes:
        return {"note_count": 0, "error": "no notes found"}

    all_notes.sort(key=lambda x: x[0])

    pitches = [n[1] for n in all_notes]
    velocities = [n[2] for n in all_notes]
    durations = [n[3] for n in all_notes]

    # Intervals between consecutive notes
    intervals = [pitches[i + 1] - pitches[i] for i in range(len(pitches) - 1)]
    interval_counts = Counter(intervals)

    # Pitch class histogram
    pitch_classes = [p % 12 for p in pitches]
    pc_hist = np.zeros(12)
    for pc in pitch_classes:
        pc_hist[pc] += 1
    if pc_hist.sum() > 0:
        pc_hist = pc_hist / pc_hist.sum()

    # Note density
    ticks_per_beat = mid.ticks_per_beat or 480
    total_ticks = max(n[0] + n[3] for n in all_notes)
    total_beats = total_ticks / ticks_per_beat
    note_density = len(all_notes) / max(total_beats, 1)

    vel_arr = np.array(velocities, dtype=float)
    dur_arr = np.array(durations, dtype=float)
    top_intervals = interval_counts.most_common(10)

    return {
        "note_count": len(all_notes),
        "pitch_range": (min(pitches), max(pitches)),
        "pitch_mean": float(np.mean(pitches)),
        "pitch_std": float(np.std(pitches)),
        "pitch_class_histogram": pc_hist.tolist(),
        "interval_histogram": dict(top_intervals),
        "interval_mean": float(np.mean(intervals)) if intervals else 0,
        "interval_std": float(np.std(intervals)) if intervals else 0,
        "velocity_mean": float(np.mean(vel_arr)),
        "velocity_std": float(np.std(vel_arr)),
        "duration_mean": float(np.mean(dur_arr)),
        "duration_std": float(np.std(dur_arr)),
        "note_density_per_beat": note_density,
        "total_beats": total_beats,
    }
