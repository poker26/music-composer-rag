"""
Level 2 - MIDI Builder
Converts a composition blueprint (from Claude) into a multi-track MIDI file
using musical rules and patterns from the RAG database.

Generates:
- Melody track: based on pitch class preferences and interval patterns
- Harmony track: chord voicings from the progression
- Bass track: root/fifth patterns following the harmony
- Rhythm/ornament track: arpeggiated figures and rhythmic patterns
"""
import logging
import random
import mido
from mido import MidiFile, MidiTrack, Message, MetaMessage
from pathlib import Path

logger = logging.getLogger(__name__)

# Chord definitions: chord symbol -> intervals from root (in semitones)
CHORD_INTERVALS = {
    "": [0, 4, 7],           # major
    "m": [0, 3, 7],          # minor
    "7": [0, 4, 7, 10],      # dominant 7
    "maj7": [0, 4, 7, 11],   # major 7
    "m7": [0, 3, 7, 10],     # minor 7
    "dim": [0, 3, 6],        # diminished
    "dim7": [0, 3, 6, 9],    # diminished 7
    "aug": [0, 4, 8],        # augmented
    "sus4": [0, 5, 7],       # suspended 4
    "sus2": [0, 2, 7],       # suspended 2
    "6": [0, 4, 7, 9],       # major 6
    "m6": [0, 3, 7, 9],      # minor 6
    "9": [0, 4, 7, 10, 14],  # dominant 9
    "add9": [0, 4, 7, 14],   # add 9
}

NOTE_TO_MIDI = {
    "C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3,
    "E": 4, "F": 5, "F#": 6, "Gb": 6, "G": 7, "G#": 8,
    "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11,
}

# Dynamic to velocity mapping
DYNAMIC_VELOCITY = {
    "pp": 30, "p": 50, "mp": 64, "mf": 80, "f": 100, "ff": 120,
}


def parse_chord(chord_str):
    """Parse chord symbol into (root_midi_note, intervals)."""
    chord_str = chord_str.strip()
    if not chord_str:
        return 0, [0, 4, 7]

    # Extract root note
    if len(chord_str) > 1 and chord_str[1] in ("#", "b"):
        root_name = chord_str[:2]
        quality = chord_str[2:]
    else:
        root_name = chord_str[0]
        quality = chord_str[1:]

    root = NOTE_TO_MIDI.get(root_name, 0)
    intervals = CHORD_INTERVALS.get(quality, CHORD_INTERVALS[""])

    return root, intervals


def generate_melody(section, track_def, ticks_per_beat, style_profile=None):
    """Generate melody notes for a section."""
    events = []
    bars = section["end_bar"] - section["start_bar"] + 1
    chords = section.get("chord_progression", ["C"] * bars)
    velocity = DYNAMIC_VELOCITY.get(section.get("dynamic", "mf"), 80)
    notes_per_bar = track_def.get("notes_per_bar", 8)
    low = track_def.get("register", {}).get("low", 60)
    high = track_def.get("register", {}).get("high", 84)

    # Get preferred pitch classes from style profile
    preferred_pcs = list(range(12))
    if style_profile and style_profile.get("preferred_pitch_classes"):
        preferred_pcs = [NOTE_TO_MIDI.get(pc[0], 0) for pc in style_profile["preferred_pitch_classes"]]

    current_pitch = (low + high) // 2  # Start in middle of register
    tick = 0
    note_duration = ticks_per_beat * 4 // notes_per_bar  # Duration per note

    for bar_idx in range(bars):
        chord_str = chords[bar_idx % len(chords)]
        root, intervals = parse_chord(chord_str)

        # Chord tones in the melody register
        chord_tones = []
        for octave in range(0, 10):
            for iv in intervals:
                pitch = root + iv + octave * 12
                if low <= pitch <= high:
                    chord_tones.append(pitch)

        for note_idx in range(notes_per_bar):
            # Mix of chord tones and scale movement
            if random.random() < 0.6 and chord_tones:
                # Chord tone - prefer ones close to current pitch
                chord_tones_sorted = sorted(chord_tones, key=lambda p: abs(p - current_pitch))
                target = chord_tones_sorted[0] if random.random() < 0.5 else \
                    chord_tones_sorted[min(1, len(chord_tones_sorted) - 1)]
            else:
                # Stepwise movement (intervals of 1-3 semitones)
                step = random.choice([-3, -2, -1, 1, 2, 3])
                target = current_pitch + step

            # Clamp to register
            target = max(low, min(high, target))

            # Velocity variation
            vel = velocity + random.randint(-10, 10)
            vel = max(20, min(127, vel))

            # Occasional rest
            if random.random() < 0.1:
                tick += note_duration
                continue

            events.append(("note_on", tick, target, vel))
            events.append(("note_off", tick + int(note_duration * 0.9), target, 0))

            current_pitch = target
            tick += note_duration

    return events


def generate_harmony(section, track_def, ticks_per_beat, style_profile=None):
    """Generate harmony (chord voicings) for a section."""
    events = []
    bars = section["end_bar"] - section["start_bar"] + 1
    chords = section.get("chord_progression", ["C"] * bars)
    velocity = DYNAMIC_VELOCITY.get(section.get("dynamic", "mf"), 80) - 10
    low = track_def.get("register", {}).get("low", 48)
    high = track_def.get("register", {}).get("high", 72)

    tick = 0
    bar_ticks = ticks_per_beat * 4  # Assuming 4/4

    for bar_idx in range(bars):
        chord_str = chords[bar_idx % len(chords)]
        root, intervals = parse_chord(chord_str)

        # Build voicing in register
        voicing = []
        for octave in range(0, 10):
            for iv in intervals:
                pitch = root + iv + octave * 12
                if low <= pitch <= high:
                    voicing.append(pitch)

        # Limit to 4 notes max
        if len(voicing) > 4:
            # Keep spread voicing
            voicing = [voicing[0], voicing[len(voicing)//3],
                       voicing[2*len(voicing)//3], voicing[-1]]

        vel = velocity + random.randint(-5, 5)
        vel = max(20, min(127, vel))

        notes_per_bar = track_def.get("notes_per_bar", 4)

        if notes_per_bar <= 2:
            # Whole/half note chords
            for pitch in voicing:
                events.append(("note_on", tick, pitch, vel))
            for pitch in voicing:
                events.append(("note_off", tick + int(bar_ticks * 0.95), pitch, 0))
        else:
            # Arpeggiated
            arp_duration = bar_ticks // notes_per_bar
            for i in range(notes_per_bar):
                pitch = voicing[i % len(voicing)] if voicing else 60
                events.append(("note_on", tick + i * arp_duration, pitch, vel))
                events.append(("note_off", tick + i * arp_duration + int(arp_duration * 0.8), pitch, 0))

        tick += bar_ticks

    return events


def generate_bass(section, track_def, ticks_per_beat, style_profile=None):
    """Generate bass line for a section."""
    events = []
    bars = section["end_bar"] - section["start_bar"] + 1
    chords = section.get("chord_progression", ["C"] * bars)
    velocity = DYNAMIC_VELOCITY.get(section.get("dynamic", "mf"), 80) - 5
    low = track_def.get("register", {}).get("low", 36)
    high = track_def.get("register", {}).get("high", 55)

    tick = 0
    bar_ticks = ticks_per_beat * 4

    for bar_idx in range(bars):
        chord_str = chords[bar_idx % len(chords)]
        root, intervals = parse_chord(chord_str)

        # Find root in bass register
        bass_root = root
        while bass_root < low:
            bass_root += 12
        while bass_root > high:
            bass_root -= 12

        fifth = bass_root + 7
        if fifth > high:
            fifth -= 12

        vel = velocity + random.randint(-5, 5)
        vel = max(20, min(127, vel))

        notes_per_bar = track_def.get("notes_per_bar", 2)

        if notes_per_bar <= 1:
            # Whole note bass
            events.append(("note_on", tick, bass_root, vel))
            events.append(("note_off", tick + int(bar_ticks * 0.9), bass_root, 0))
        elif notes_per_bar == 2:
            # Root-fifth pattern
            half = bar_ticks // 2
            events.append(("note_on", tick, bass_root, vel))
            events.append(("note_off", tick + int(half * 0.9), bass_root, 0))
            events.append(("note_on", tick + half, fifth, vel - 5))
            events.append(("note_off", tick + half + int(half * 0.9), fifth, 0))
        else:
            # Walking bass
            note_dur = bar_ticks // notes_per_bar
            walk_notes = [bass_root, fifth, bass_root + intervals[1] if len(intervals) > 1 else bass_root + 4, fifth]
            for i in range(notes_per_bar):
                pitch = walk_notes[i % len(walk_notes)]
                pitch = max(low, min(high, pitch))
                events.append(("note_on", tick + i * note_dur, pitch, vel))
                events.append(("note_off", tick + i * note_dur + int(note_dur * 0.8), pitch, 0))

        tick += bar_ticks

    return events


def generate_rhythm(section, track_def, ticks_per_beat, style_profile=None):
    """Generate rhythm/ornament track - arpeggiated figures."""
    events = []
    bars = section["end_bar"] - section["start_bar"] + 1
    chords = section.get("chord_progression", ["C"] * bars)
    velocity = DYNAMIC_VELOCITY.get(section.get("dynamic", "mf"), 80) - 15
    low = track_def.get("register", {}).get("low", 55)
    high = track_def.get("register", {}).get("high", 79)
    notes_per_bar = track_def.get("notes_per_bar", 16)

    tick = 0
    bar_ticks = ticks_per_beat * 4
    note_dur = bar_ticks // notes_per_bar

    for bar_idx in range(bars):
        chord_str = chords[bar_idx % len(chords)]
        root, intervals = parse_chord(chord_str)

        # Build arpeggiated figure from chord tones
        arp_notes = []
        for octave in range(0, 10):
            for iv in intervals:
                pitch = root + iv + octave * 12
                if low <= pitch <= high:
                    arp_notes.append(pitch)

        if not arp_notes:
            arp_notes = [60]  # fallback

        # Arpeggio patterns: up, down, up-down, random
        pattern_type = random.choice(["up", "down", "updown", "random"])
        if pattern_type == "up":
            pattern = sorted(arp_notes)
        elif pattern_type == "down":
            pattern = sorted(arp_notes, reverse=True)
        elif pattern_type == "updown":
            up = sorted(arp_notes)
            pattern = up + up[-2:0:-1]
        else:
            pattern = arp_notes[:]
            random.shuffle(pattern)

        vel = velocity + random.randint(-8, 8)
        vel = max(15, min(110, vel))

        for i in range(notes_per_bar):
            # Occasional ghost note or rest
            if random.random() < 0.08:
                tick += note_dur
                continue

            pitch = pattern[i % len(pattern)]
            v = vel + random.randint(-5, 5)
            v = max(15, min(127, v))

            events.append(("note_on", tick, pitch, v))
            events.append(("note_off", tick + int(note_dur * 0.7), pitch, 0))
            tick += note_dur

        # Reset tick to next bar if accumulated differently
        expected_tick = (bar_idx + 1) * bar_ticks
        tick = expected_tick

    return events


# Track role -> generator function
GENERATORS = {
    "melody": generate_melody,
    "harmony": generate_harmony,
    "bass": generate_bass,
    "rhythm": generate_rhythm,
}


def build_midi(blueprint, output_path, style_profile=None):
    """
    Build a multi-track MIDI file from a blueprint.

    Args:
        blueprint: dict from composer_architect
        output_path: Path for output MIDI file
        style_profile: optional dict from style_profiler

    Returns:
        Path to generated MIDI file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ticks_per_beat = 480
    mid = MidiFile(ticks_per_beat=ticks_per_beat)

    tempo_bpm = blueprint.get("tempo_bpm", 120)
    tempo_us = mido.bpm2tempo(tempo_bpm)
    time_sig = blueprint.get("time_signature", [4, 4])

    # Track 0: tempo and meta
    meta_track = MidiTrack()
    mid.tracks.append(meta_track)
    meta_track.append(MetaMessage("track_name", name=blueprint.get("title", "Generated"), time=0))
    meta_track.append(MetaMessage("set_tempo", tempo=tempo_us, time=0))
    meta_track.append(MetaMessage("time_signature",
                                   numerator=time_sig[0],
                                   denominator=time_sig[1],
                                   time=0))

    # Add tempo changes per section
    current_tick = 0
    for section in blueprint.get("sections", []):
        section_tempo = section.get("tempo_bpm", tempo_bpm)
        if section_tempo != tempo_bpm:
            bar_offset = (section["start_bar"] - 1) * ticks_per_beat * time_sig[0]
            delta = bar_offset - current_tick
            if delta > 0:
                meta_track.append(MetaMessage("set_tempo",
                                               tempo=mido.bpm2tempo(section_tempo),
                                               time=delta))
                current_tick = bar_offset

    meta_track.append(MetaMessage("end_of_track", time=0))

    # Generate each track
    for track_def in blueprint.get("tracks", []):
        role = track_def.get("role", "melody")
        generator = GENERATORS.get(role, generate_melody)

        track = MidiTrack()
        mid.tracks.append(track)
        track.append(MetaMessage("track_name", name=track_def.get("name", role), time=0))

        # Set instrument
        channel = track_def.get("channel", 0)
        program = track_def.get("midi_program", 0)
        track.append(Message("program_change", program=program, channel=channel, time=0))

        # Collect all events across sections
        all_events = []
        section_offset = 0

        for section in blueprint.get("sections", []):
            bar_offset = (section["start_bar"] - 1) * ticks_per_beat * time_sig[0]
            events = generator(section, track_def, ticks_per_beat, style_profile)

            # Offset events to section position
            for evt_type, tick, pitch, vel in events:
                all_events.append((evt_type, tick + bar_offset, pitch, vel, channel))

        # Sort by time and convert to MIDI messages with delta times
        all_events.sort(key=lambda e: (e[1], 0 if e[0] == "note_off" else 1))

        current_tick = 0
        for evt_type, abs_tick, pitch, vel, ch in all_events:
            delta = abs_tick - current_tick
            if delta < 0:
                delta = 0
            track.append(Message(evt_type, note=pitch, velocity=vel,
                                  channel=ch, time=delta))
            current_tick = abs_tick

        track.append(MetaMessage("end_of_track", time=0))

    mid.save(str(output_path))
    logger.info("MIDI saved: %s (%d tracks)", output_path, len(mid.tracks))
    return output_path
