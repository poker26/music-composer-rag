"""
Level 1 - Composition Architect (Claude API)
Takes a style profile and user parameters, generates a composition blueprint:
- Form structure (sections with timing)
- Chord progression per section
- Track assignments (melody, harmony, bass, rhythm)
- Dynamic map
- Tempo variations

Output: JSON blueprint consumed by the MIDI builder (Level 2).
"""
import os
import json
import logging
import anthropic

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an expert music composer and arranger. You create detailed musical blueprints in JSON format.

You will receive:
1. A style profile of a specific composer (statistical analysis of their music)
2. Parameters for a new composition (key, tempo, duration, mood)

You must output ONLY valid JSON (no markdown, no commentary) with this exact structure:

{
  "title": "Generated title for the piece",
  "key": "C minor",
  "time_signature": [4, 4],
  "tempo_bpm": 120,
  "total_bars": 32,
  "sections": [
    {
      "name": "Introduction",
      "start_bar": 1,
      "end_bar": 4,
      "tempo_bpm": 116,
      "dynamic": "p",
      "chord_progression": ["Cm", "G7", "Ab", "Fm"],
      "description": "Gentle opening with arpeggiated chords"
    }
  ],
  "tracks": [
    {
      "name": "Melody",
      "channel": 0,
      "instrument": "piano",
      "midi_program": 0,
      "role": "melody",
      "register": {"low": 60, "high": 84},
      "notes_per_bar": 8,
      "style_hints": "Lyrical, singing quality, use of chromatic passing tones"
    },
    {
      "name": "Harmony",
      "channel": 1,
      "instrument": "piano",
      "midi_program": 0,
      "role": "harmony",
      "register": {"low": 48, "high": 72},
      "notes_per_bar": 4,
      "style_hints": "Block chords with occasional arpeggiation"
    },
    {
      "name": "Bass",
      "channel": 2,
      "instrument": "piano",
      "midi_program": 0,
      "role": "bass",
      "register": {"low": 36, "high": 55},
      "notes_per_bar": 2,
      "style_hints": "Root-fifth patterns, occasional walking bass"
    },
    {
      "name": "Rhythm/Ornament",
      "channel": 3,
      "instrument": "piano",
      "midi_program": 0,
      "role": "rhythm",
      "register": {"low": 55, "high": 79},
      "notes_per_bar": 16,
      "style_hints": "Arpeggiated figures, tremolo, grace notes"
    }
  ]
}

Rules:
- Chord symbols use standard notation: C, Cm, C7, Cmaj7, Cdim, Caug, Csus4, etc.
- Each section must have exactly (end_bar - start_bar + 1) chords in chord_progression (one per bar)
- Dynamic markings: pp, p, mp, mf, f, ff
- MIDI program numbers: 0=piano, 24=guitar, 32=bass, 40=violin, 73=flute
- Register is MIDI note numbers (60=middle C)
- Match the style profile closely: use preferred keys, typical tempos, characteristic note density
- Create musically coherent progressions that reflect the composer's harmonic language
- Total bars should match the requested duration at the given tempo
"""


def generate_blueprint(style_profile_text, params):
    """
    Call Claude API to generate a composition blueprint.

    Args:
        style_profile_text: Human-readable style profile string
        params: dict with keys: key, tempo_bpm, duration_sec, mood, description

    Returns:
        dict: Parsed JSON blueprint
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set. Run: export ANTHROPIC_API_KEY=sk-...")

    client = anthropic.Anthropic(api_key=api_key)

    # Calculate approximate bars
    tempo = params.get("tempo_bpm", 120)
    duration = params.get("duration_sec", 60)
    beats_per_bar = params.get("time_signature", [4, 4])[0]
    total_bars = max(4, int((tempo * duration) / (60 * beats_per_bar)))

    user_message = f"""Generate a composition blueprint based on this style profile and parameters.

STYLE PROFILE:
{style_profile_text}

PARAMETERS:
- Key: {params.get('key', 'C minor')}
- Tempo: {tempo} BPM
- Time signature: {params.get('time_signature', [4, 4])}
- Duration: approximately {duration} seconds ({total_bars} bars)
- Mood: {params.get('mood', 'expressive')}
- Description: {params.get('description', 'A piece in the style of the profiled composer')}
- Instruments: {params.get('instruments', 'piano solo (4 tracks: melody, harmony, bass, rhythm)')}

Generate the JSON blueprint now. Output ONLY valid JSON, nothing else."""

    logger.info("Calling Claude API for blueprint generation...")

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )

    raw_text = response.content[0].text.strip()

    # Clean up potential markdown wrapping
    if raw_text.startswith("```"):
        lines = raw_text.split("\n")
        raw_text = "\n".join(lines[1:-1])

    try:
        blueprint = json.loads(raw_text)
    except json.JSONDecodeError as e:
        logger.error("Failed to parse Claude response as JSON: %s", e)
        logger.error("Raw response: %s", raw_text[:500])
        raise ValueError(f"Claude returned invalid JSON: {e}")

    # Validate basic structure
    required = ["sections", "tracks", "key", "tempo_bpm"]
    for field in required:
        if field not in blueprint:
            raise ValueError(f"Blueprint missing required field: {field}")

    logger.info("Blueprint generated: %s, %d sections, %d tracks, %d bars",
                blueprint.get("title", "Untitled"),
                len(blueprint["sections"]),
                len(blueprint["tracks"]),
                blueprint.get("total_bars", 0))

    return blueprint
