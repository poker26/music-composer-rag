"""
Level 2 - MIDI Builder v3
- Melody: explicit notes from Claude's blueprint
- Accompaniment: chord-aware arpeggiation using rhythm templates from real patterns
- Bass: proper root movement following chord progression
- All tracks synchronized to the same bar/beat grid
"""
import logging
import random
import mido
from mido import MidiFile, MidiTrack, Message, MetaMessage
from pathlib import Path

logger = logging.getLogger(__name__)

NOTE_TO_MIDI = {
    "C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3,
    "E": 4, "F": 5, "F#": 6, "Gb": 6, "G": 7, "G#": 8,
    "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11,
}

CHORD_INTERVALS = {
    "": [0, 4, 7], "m": [0, 3, 7], "7": [0, 4, 7, 10],
    "maj7": [0, 4, 7, 11], "m7": [0, 3, 7, 10],
    "dim": [0, 3, 6], "dim7": [0, 3, 6, 9], "aug": [0, 4, 8],
    "sus4": [0, 5, 7], "sus2": [0, 2, 7],
    "6": [0, 4, 7, 9], "m6": [0, 3, 7, 9],
    "9": [0, 4, 7, 10, 14], "add9": [0, 4, 7, 14],
    "7sus4": [0, 5, 7, 10],
}

DYNAMIC_VELOCITY = {"pp": 30, "p": 50, "mp": 64, "mf": 80, "f": 100, "ff": 120}


def parse_chord(chord_str):
    chord_str = chord_str.strip()
    if not chord_str:
        return 0, [0, 4, 7]
    if len(chord_str) > 1 and chord_str[1] in ("#", "b"):
        root_name, quality = chord_str[:2], chord_str[2:]
    else:
        root_name, quality = chord_str[0], chord_str[1:]
    root = NOTE_TO_MIDI.get(root_name, 0)
    intervals = CHORD_INTERVALS.get(quality, CHORD_INTERVALS[""])
    return root, intervals


def chord_tones_in_register(root, intervals, low, high):
    """Get all chord tones within a register range."""
    tones = []
    for octave in range(0, 10):
        for iv in intervals:
            pitch = root + iv + octave * 12
            if low <= pitch <= high:
                tones.append(pitch)
    return sorted(tones)


def build_melody_track(blueprint, ticks_per_beat):
    """Build melody from Claude's explicit notes."""
    events = []
    time_sig = blueprint.get("time_signature", [4, 4])
    bar_ticks = ticks_per_beat * time_sig[0]

    for section in blueprint.get("sections", []):
        for bar_data in section.get("melody", []):
            bar_num = bar_data.get("bar", 1) - 1
            bar_start_tick = bar_num * bar_ticks

            for note in bar_data.get("notes", []):
                pitch = max(21, min(108, note.get("pitch", 60)))
                start_beat = note.get("start_beat", 1.0)
                duration = note.get("duration", 1.0)
                velocity = max(20, min(127, note.get("velocity", 80)))

                note_tick = bar_start_tick + int((start_beat - 1) * ticks_per_beat)
                dur_ticks = max(1, int(duration * ticks_per_beat))

                events.append(("note_on", note_tick, pitch, velocity))
                events.append(("note_off", note_tick + dur_ticks, pitch, 0))

    return events


def extract_rhythm_template(patterns):
    """Extract just the rhythmic timing from real patterns (not the pitches)."""
    if not patterns or not patterns.get("accompaniment"):
        return None
    templates = []
    for pat in patterns["accompaniment"]:
        template = [(n["offset"], n["duration"], n["velocity_ratio"]) for n in pat["notes"]]
        if 3 <= len(template) <= 16:
            templates.append(template)
    return templates if templates else None


def build_accompaniment_track(blueprint, ticks_per_beat, patterns=None):
    """
    Chord-aware accompaniment. Uses rhythm templates from real patterns
    but fills with correct chord tones for each bar.
    """
    events = []
    time_sig = blueprint.get("time_signature", [4, 4])
    bar_ticks = ticks_per_beat * time_sig[0]
    beats_per_bar = time_sig[0]

    rhythm_templates = extract_rhythm_template(patterns)

    # Find accompaniment register from blueprint
    reg_low, reg_high = 48, 72
    for t in blueprint.get("tracks", []):
        if t.get("role") == "accompaniment":
            reg_low = t.get("register", {}).get("low", 48)
            reg_high = t.get("register", {}).get("high", 72)
            break

    for section in blueprint.get("sections", []):
        bars = section["end_bar"] - section["start_bar"] + 1
        chords = section.get("chord_progression", ["C"] * bars)
        base_velocity = DYNAMIC_VELOCITY.get(section.get("dynamic", "mf"), 80) - 10

        for bar_idx in range(bars):
            chord_str = chords[bar_idx % len(chords)]
            root, intervals = parse_chord(chord_str)
            abs_bar = section["start_bar"] - 1 + bar_idx
            bar_start = abs_bar * bar_ticks

            tones = chord_tones_in_register(root, intervals, reg_low, reg_high)
            if not tones:
                tones = chord_tones_in_register(root, intervals, reg_low - 12, reg_high + 12)
            if not tones:
                tones = [60]

            if rhythm_templates:
                # Use real rhythm template, fill with chord tones
                template = random.choice(rhythm_templates)
                for i, (offset, dur, vel_ratio) in enumerate(template):
                    # Only use notes that fit within the bar
                    if offset >= beats_per_bar:
                        continue
                    pitch = tones[i % len(tones)]
                    vel = int(vel_ratio * base_velocity)
                    vel = max(20, min(127, vel))
                    note_tick = bar_start + int(offset * ticks_per_beat)
                    dur_ticks = max(1, int(dur * ticks_per_beat * 0.9))

                    events.append(("note_on", note_tick, pitch, vel))
                    events.append(("note_off", note_tick + dur_ticks, pitch, 0))
            else:
                # Fallback: classic nocturne-style broken chord (1-5-8-10-8-5)
                if len(tones) >= 3:
                    arp_sequence = _build_arp_sequence(tones, beats_per_bar)
                else:
                    arp_sequence = tones * 4

                note_count = 8
                note_dur = bar_ticks // note_count

                for i in range(note_count):
                    pitch = arp_sequence[i % len(arp_sequence)]
                    vel = base_velocity + random.randint(-6, 6)
                    vel = max(20, min(127, vel))
                    tick = bar_start + i * note_dur

                    events.append(("note_on", tick, pitch, vel))
                    events.append(("note_off", tick + int(note_dur * 0.85), pitch, 0))

    return events


def _build_arp_sequence(tones, beats_per_bar):
    """Build a musical arpeggio sequence from chord tones."""
    if len(tones) < 3:
        return tones * 4

    # Classic patterns: up-down, up, waltz-style
    patterns = []

    # Up-down: 1-3-5-8-5-3
    up_down = []
    for t in tones:
        up_down.append(t)
    for t in reversed(tones[1:-1]):
        up_down.append(t)
    patterns.append(up_down)

    # Nocturne style: bass-mid-high-mid (like Chopin left hand)
    if len(tones) >= 4:
        nocturne = [tones[0], tones[2], tones[3], tones[2],
                    tones[1], tones[2], tones[3], tones[2]]
        patterns.append(nocturne)

    # Simple up
    patterns.append(tones[:])

    return random.choice(patterns)


def build_bass_track(blueprint, ticks_per_beat, patterns=None):
    """Bass line following chord roots with musical patterns."""
    events = []
    time_sig = blueprint.get("time_signature", [4, 4])
    bar_ticks = ticks_per_beat * time_sig[0]

    reg_low, reg_high = 36, 55
    for t in blueprint.get("tracks", []):
        if t.get("role") == "bass":
            reg_low = t.get("register", {}).get("low", 36)
            reg_high = t.get("register", {}).get("high", 55)
            break

    prev_bass_note = None

    for section in blueprint.get("sections", []):
        bars = section["end_bar"] - section["start_bar"] + 1
        chords = section.get("chord_progression", ["C"] * bars)
        base_velocity = DYNAMIC_VELOCITY.get(section.get("dynamic", "mf"), 80)

        for bar_idx in range(bars):
            chord_str = chords[bar_idx % len(chords)]
            root, intervals = parse_chord(chord_str)
            abs_bar = section["start_bar"] - 1 + bar_idx
            bar_start = abs_bar * bar_ticks

            # Find root in bass register
            bass_root = root
            while bass_root < reg_low:
                bass_root += 12
            while bass_root > reg_high:
                bass_root -= 12

            # Fifth
            fifth = bass_root + 7
            if fifth > reg_high:
                fifth -= 12

            # Third
            third = bass_root + intervals[1] if len(intervals) > 1 else bass_root + 4
            if third > reg_high:
                third -= 12

            vel = base_velocity + random.randint(-5, 5)
            vel = max(20, min(127, vel))

            # Pattern: root on beat 1, fifth on beat 3 (classic nocturne bass)
            half = bar_ticks // 2

            # Beat 1: root (long)
            events.append(("note_on", bar_start, bass_root, vel))
            events.append(("note_off", bar_start + int(half * 0.9), bass_root, 0))

            # Beat 3: fifth or third (shorter)
            alt_note = fifth if random.random() < 0.7 else third
            events.append(("note_on", bar_start + half, alt_note, vel - 8))
            events.append(("note_off", bar_start + half + int(half * 0.85), alt_note, 0))

            prev_bass_note = bass_root

    return events


def events_to_track(events, track_name, channel, program):
    """Convert event list to MIDI track with delta times."""
    track = MidiTrack()
    track.append(MetaMessage("track_name", name=track_name, time=0))
    track.append(Message("program_change", program=program, channel=channel, time=0))

    events.sort(key=lambda e: (e[1], 0 if e[0] == "note_off" else 1))

    current_tick = 0
    for evt_type, abs_tick, pitch, vel in events:
        delta = max(0, abs_tick - current_tick)
        track.append(Message(evt_type, note=pitch, velocity=vel,
                              channel=channel, time=delta))
        current_tick = abs_tick

    track.append(MetaMessage("end_of_track", time=0))
    return track


ROLE_BUILDERS = {
    "melody": lambda bp, tpb, pat: build_melody_track(bp, tpb),
    "accompaniment": build_accompaniment_track,
    "bass": build_bass_track,
}


def build_midi(blueprint, output_path, style_profile=None, patterns=None):
    """Build multi-track MIDI from blueprint."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ticks_per_beat = 480
    mid = MidiFile(ticks_per_beat=ticks_per_beat)

    tempo_bpm = blueprint.get("tempo_bpm", 120)
    time_sig = blueprint.get("time_signature", [4, 4])

    # Meta track
    meta_track = MidiTrack()
    mid.tracks.append(meta_track)
    meta_track.append(MetaMessage("track_name",
                                   name=blueprint.get("title", "Generated"), time=0))
    meta_track.append(MetaMessage("set_tempo", tempo=mido.bpm2tempo(tempo_bpm), time=0))
    meta_track.append(MetaMessage("time_signature",
                                   numerator=time_sig[0], denominator=time_sig[1], time=0))
    meta_track.append(MetaMessage("end_of_track", time=0))

    # Build each track
    for tdef in blueprint.get("tracks", []):
        role = tdef.get("role", "melody")
        channel = tdef.get("channel", 0)
        program = tdef.get("midi_program", 0)
        name = tdef.get("name", role)

        builder = ROLE_BUILDERS.get(role)
        if builder:
            if role == "melody":
                events = builder(blueprint, ticks_per_beat, patterns)
            else:
                events = builder(blueprint, ticks_per_beat, patterns)
        else:
            # Unknown role - treat as accompaniment
            events = build_accompaniment_track(blueprint, ticks_per_beat, patterns)

        if events:
            track = events_to_track(events, name, channel, program)
            mid.tracks.append(track)
            logger.info("  Track '%s' (%s): %d events", name, role, len(events))
        else:
            logger.warning("  Track '%s' (%s): no events generated", name, role)

    mid.save(str(output_path))
    logger.info("MIDI saved: %s (%d tracks)", output_path, len(mid.tracks))
    return output_path
