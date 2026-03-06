"""
Level 2 - MIDI Builder v4
All tracks come directly from Claude's blueprint.
The builder just converts JSON note data to MIDI format.
Falls back to simple patterns only if Claude didn't generate a track.
"""
import logging
import mido
from mido import MidiFile, MidiTrack, Message, MetaMessage
from pathlib import Path

logger = logging.getLogger(__name__)


def extract_track_events(blueprint, track_key, ticks_per_beat):
    """
    Extract note events from blueprint for a given track key
    (e.g., "melody", "accompaniment", "bass").
    """
    events = []
    time_sig = blueprint.get("time_signature", [4, 4])
    bar_ticks = ticks_per_beat * time_sig[0]

    for section in blueprint.get("sections", []):
        track_data = section.get(track_key, [])

        for bar_data in track_data:
            bar_num = bar_data.get("bar", 1) - 1
            bar_start_tick = bar_num * bar_ticks

            for note in bar_data.get("notes", []):
                pitch = max(21, min(108, note.get("pitch", 60)))
                start_beat = note.get("start_beat", 1.0)
                duration = note.get("duration", 1.0)
                velocity = max(1, min(127, note.get("velocity", 80)))

                note_tick = bar_start_tick + int((start_beat - 1) * ticks_per_beat)
                dur_ticks = max(1, int(duration * ticks_per_beat))

                events.append(("note_on", note_tick, pitch, velocity))
                events.append(("note_off", note_tick + dur_ticks, pitch, 0))

    return events


def events_to_track(events, track_name, channel, program):
    """Convert event list to MIDI track with delta times."""
    track = MidiTrack()
    track.append(MetaMessage("track_name", name=track_name, time=0))
    track.append(Message("program_change", program=program, channel=channel, time=0))

    # Sort: by time, then note_off before note_on at same tick
    events.sort(key=lambda e: (e[1], 0 if e[0] == "note_off" else 1))

    current_tick = 0
    for evt_type, abs_tick, pitch, vel in events:
        delta = max(0, abs_tick - current_tick)
        track.append(Message(evt_type, note=pitch, velocity=vel,
                              channel=channel, time=delta))
        current_tick = abs_tick

    track.append(MetaMessage("end_of_track", time=0))
    return track


# Map track roles to blueprint JSON keys
ROLE_TO_KEY = {
    "melody": "melody",
    "accompaniment": "accompaniment",
    "harmony": "accompaniment",
    "bass": "bass",
    "rhythm": "accompaniment",
}


def build_midi(blueprint, output_path, style_profile=None, patterns=None):
    """Build multi-track MIDI from blueprint where all tracks have explicit notes."""
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

    # Add tempo changes per section if different from base
    bar_ticks = ticks_per_beat * time_sig[0]
    current_tick = 0
    for section in blueprint.get("sections", []):
        section_tempo = section.get("tempo_bpm", tempo_bpm)
        if section_tempo != tempo_bpm:
            bar_offset = (section["start_bar"] - 1) * bar_ticks
            delta = bar_offset - current_tick
            if delta > 0:
                meta_track.append(MetaMessage("set_tempo",
                                               tempo=mido.bpm2tempo(section_tempo),
                                               time=delta))
                current_tick = bar_offset

    meta_track.append(MetaMessage("end_of_track", time=0))

    # Build each track from blueprint data
    for tdef in blueprint.get("tracks", []):
        role = tdef.get("role", "melody")
        channel = tdef.get("channel", 0)
        program = tdef.get("midi_program", 0)
        name = tdef.get("name", role)

        # Find the correct key in blueprint sections
        track_key = ROLE_TO_KEY.get(role, role)
        events = extract_track_events(blueprint, track_key, ticks_per_beat)

        if events:
            track = events_to_track(events, name, channel, program)
            mid.tracks.append(track)
            logger.info("  Track '%s' (%s): %d events", name, role, len(events))
        else:
            logger.warning("  Track '%s' (%s): no events in blueprint", name, role)

    mid.save(str(output_path))
    logger.info("MIDI saved: %s (%d tracks, tpb=%d)", output_path, len(mid.tracks), ticks_per_beat)
    return output_path
