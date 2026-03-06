"""
Level 1 - Composition Architect (Claude API) v4
Claude generates ALL tracks: melody, accompaniment, AND bass.
This ensures proper voice leading, phrasing, and coordination between parts.
"""
import os
import json
import logging
from pathlib import Path
import anthropic

logger = logging.getLogger(__name__)

FORMS_DIR = Path(__file__).parent.parent / "forms"

SYSTEM_PROMPT = """You are a master composer with encyclopedic knowledge of music theory, counterpoint, harmony, and form.
You compose detailed musical scores output as JSON.

You will receive:
1. A style profile of a specific composer
2. Optionally, a musical FORM TEMPLATE with structural rules
3. Parameters for the composition

You must generate ALL tracks with explicit notes. This is critical for musical coherence.

OUTPUT FORMAT - valid JSON only, no markdown, no commentary:

{
  "title": "Title",
  "key": "E minor",
  "time_signature": [4, 4],
  "tempo_bpm": 72,
  "total_bars": 24,
  "form": "nocturne",
  "tracks": [
    {
      "name": "Melody",
      "channel": 0,
      "midi_program": 0,
      "role": "melody",
      "register": {"low": 60, "high": 88}
    },
    {
      "name": "Accompaniment",
      "channel": 1,
      "midi_program": 0,
      "role": "accompaniment",
      "register": {"low": 48, "high": 72}
    },
    {
      "name": "Bass",
      "channel": 2,
      "midi_program": 0,
      "role": "bass",
      "register": {"low": 28, "high": 55}
    }
  ],
  "sections": [
    {
      "name": "Theme A",
      "form_id": "A",
      "start_bar": 1,
      "end_bar": 8,
      "tempo_bpm": 72,
      "dynamic": "p",
      "chord_progression": ["Em", "Am", "D7", "G", "C", "Am", "B7", "Em"],
      "melody": [
        {"bar": 1, "notes": [
          {"pitch": 67, "start_beat": 1.0, "duration": 2.0, "velocity": 58},
          {"pitch": 71, "start_beat": 3.0, "duration": 1.0, "velocity": 62},
          {"pitch": 74, "start_beat": 4.0, "duration": 1.0, "velocity": 65}
        ]}
      ],
      "accompaniment": [
        {"bar": 1, "notes": [
          {"pitch": 52, "start_beat": 1.0, "duration": 0.5, "velocity": 40},
          {"pitch": 59, "start_beat": 1.5, "duration": 0.5, "velocity": 38},
          {"pitch": 64, "start_beat": 2.0, "duration": 0.5, "velocity": 42},
          {"pitch": 59, "start_beat": 2.5, "duration": 0.5, "velocity": 38},
          {"pitch": 52, "start_beat": 3.0, "duration": 0.5, "velocity": 40},
          {"pitch": 59, "start_beat": 3.5, "duration": 0.5, "velocity": 38},
          {"pitch": 64, "start_beat": 4.0, "duration": 0.5, "velocity": 42},
          {"pitch": 59, "start_beat": 4.5, "duration": 0.5, "velocity": 38}
        ]}
      ],
      "bass": [
        {"bar": 1, "notes": [
          {"pitch": 40, "start_beat": 1.0, "duration": 4.0, "velocity": 55}
        ]}
      ]
    }
  ]
}

COMPOSITION PRINCIPLES:

1. VOICE LEADING:
   - Accompaniment voices should move by step (1-2 semitones) or common tone when chords change
   - Avoid parallel fifths and octaves between any pair of voices
   - The bass should move by step, fourth, or fifth (not leap randomly)
   - When a chord changes, keep common tones and move other voices to the nearest chord tone

2. PHRASING:
   - Melodies have 4-bar phrases (antecedent + consequent)
   - Each phrase has an arch shape: rise to a peak, then fall
   - Leave small gaps (rests) between phrases for "breathing"
   - Dynamics swell toward phrase peaks and diminish at phrase ends
   - Velocity should follow the phrase contour (louder at climax, softer at rest points)

3. ACCOMPANIMENT PATTERNS:
   - For nocturnes: steady arpeggiated left hand (e.g., bass note on beat 1, then broken chord pattern)
   - For sonatas: varied - block chords, alberti bass, tremolo depending on character
   - For preludes: one consistent figuration pattern throughout
   - For fugues: independent contrapuntal lines (no "accompaniment" - all voices are melodic)
   - The pattern should be rhythmically REGULAR and predictable to ground the free melody

4. MELODY-ACCOMPANIMENT COORDINATION:
   - Melody notes on strong beats should be chord tones (root, 3rd, 5th, 7th)
   - Melody notes on weak beats can be passing tones, neighbor tones, suspensions
   - Accompaniment should NOT play the same rhythm as the melody (rhythmic independence)
   - When melody has long notes, accompaniment provides motion; when melody is active, accompaniment simplifies
   - Melody register should be clearly above accompaniment (no crossing voices)

5. BASS:
   - Bass on beat 1 = chord root (occasionally 3rd for first inversion)
   - Bass movement: root position -> first inversion uses stepwise bass
   - At cadences: V-I bass movement (up a fourth or down a fifth)
   - Sustained bass notes (pedal points) at section beginnings and dominant preparation

6. DYNAMICS & EXPRESSION:
   - Velocity range: pp=30-40, p=45-55, mp=58-68, mf=70-82, f=85-100, ff=105-120
   - Within a phrase, velocity gradually increases to the peak and decreases after
   - Accompaniment velocity is always 15-25 less than melody velocity
   - Bass velocity is 5-10 less than melody velocity

NOTE FORMAT:
- pitch: MIDI number (60=C4, 64=E4, 67=G4, 72=C5)
- start_beat: position in bar (1.0=beat 1, 2.5=eighth note after beat 2)
- duration: in beats (0.5=eighth, 1.0=quarter, 2.0=half)
- velocity: 1-127

EVERY bar in every section MUST have notes for ALL three tracks (melody, accompaniment, bass).
Output ONLY valid JSON."""


def list_available_forms():
    """List all form templates available in the library."""
    forms = {}
    if FORMS_DIR.exists():
        for f in FORMS_DIR.glob("*.json"):
            try:
                data = json.loads(f.read_text())
                forms[data["id"]] = {
                    "name": data["name"],
                    "description": data["description"],
                    "file": f.name,
                }
            except Exception as e:
                logger.warning("Failed to load form %s: %s", f.name, e)
    return forms


def load_form(form_id):
    """Load a specific form template by ID."""
    form_path = FORMS_DIR / f"{form_id}.json"
    if not form_path.exists():
        for f in FORMS_DIR.glob("*.json"):
            try:
                data = json.loads(f.read_text())
                if data.get("id") == form_id:
                    return data
            except Exception:
                continue
        return None
    return json.loads(form_path.read_text())


def form_to_prompt_text(form_data):
    """Convert form template to detailed text for the LLM prompt."""
    lines = []
    lines.append(f"=== MUSICAL FORM: {form_data['name']} ===")
    lines.append(form_data["description"])
    lines.append(f"Form structure: {form_data['structure']['form']}")
    lines.append(f"Typical tempo: {form_data['typical_tempo_range'][0]}-{form_data['typical_tempo_range'][1]} BPM")
    lines.append("")

    lines.append("SECTIONS (you MUST include ALL of these):")
    for i, section in enumerate(form_data["structure"]["sections"], 1):
        lines.append(f"\n--- Section {i}: {section['name']} (ID: {section['id']}) ---")
        lines.append(f"  Description: {section['description']}")
        lines.append(f"  Length: {section['bars_range'][0]}-{section['bars_range'][1]} bars")
        lines.append(f"  Dynamic arc: {section['dynamic_arc']}")
        lines.append(f"  Key relation: {section['key_relation']}")
        lines.append(f"  Melody: {section['melody_character']}")
        lines.append(f"  Accompaniment: {section['accompaniment_character']}")
        lines.append(f"  Harmonic rhythm: {section['harmonic_rhythm']}")
        lines.append(f"  Ends with: {section['ends_with']}")

    lines.append("\nCOMPOSITION RULES:")
    for rule in form_data.get("composition_rules", []):
        lines.append(f"  - {rule}")

    return "\n".join(lines)


def generate_blueprint(style_profile_text, params, form_id=None):
    """Generate full composition blueprint with all tracks."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set")

    client = anthropic.Anthropic(api_key=api_key)

    form_text = ""
    if form_id:
        form_data = load_form(form_id)
        if form_data:
            form_text = form_to_prompt_text(form_data)
            logger.info("Using form template: %s", form_data["name"])
        else:
            logger.warning("Form '%s' not found", form_id)

    tempo = params.get("tempo_bpm", 120)
    duration = params.get("duration_sec", 60)
    beats_per_bar = params.get("time_signature", [4, 4])[0]
    total_bars = max(8, int((tempo * duration) / (60 * beats_per_bar)))
    # Cap at reasonable size to fit in token limit
    total_bars = min(total_bars, 32)

    user_parts = []
    user_parts.append("Compose a piece with explicit notes for ALL tracks (melody, accompaniment, bass) in every bar.")
    user_parts.append("")
    user_parts.append(f"STYLE PROFILE:\n{style_profile_text}")

    if form_text:
        user_parts.append(f"\nFORM TEMPLATE:\n{form_text}")
        user_parts.append("\nYou MUST follow this form exactly. Every section listed must appear.")

    user_parts.append(f"""
PARAMETERS:
- Key: {params.get('key', 'C minor')}
- Tempo: {tempo} BPM
- Time signature: {params.get('time_signature', [4, 4])}
- Target length: {total_bars} bars (~{duration} seconds)
- Mood: {params.get('mood', 'expressive')}
- Description: {params.get('description', '')}
- Instruments: {params.get('instruments', 'piano')}

IMPORTANT:
- Generate notes for ALL three tracks in every bar
- Accompaniment should use a consistent rhythmic pattern (e.g., 8 eighth notes per bar for arpeggiation)
- Bass: one long note per bar on beat 1 (the chord root), optionally a second note on beat 3
- Melody: singable line with clear phrases, ornaments on weak beats
- Keep accompaniment velocity 15-25 below melody velocity
- Use proper voice leading between bars (smooth transitions)

Output ONLY valid JSON.""")

    user_message = "\n".join(user_parts)

    logger.info("Calling Claude API (form=%s, ~%d bars, all tracks)...", form_id or "free", total_bars)

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=16000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )

    raw_text = response.content[0].text.strip()

    if raw_text.startswith("```"):
        lines = raw_text.split("\n")
        start = 1
        end = len(lines) - 1
        while end > start and lines[end].strip() in ("```", ""):
            end -= 1
        raw_text = "\n".join(lines[start:end + 1])

    try:
        blueprint = json.loads(raw_text)
    except json.JSONDecodeError as e:
        logger.error("JSON parse failed: %s", e)
        logger.error("Raw (first 1500): %s", raw_text[:1500])
        raise ValueError(f"Claude returned invalid JSON: {e}")

    for field in ["sections", "tracks", "key", "tempo_bpm"]:
        if field not in blueprint:
            raise ValueError(f"Missing field: {field}")

    # Stats
    melody_notes = sum(len(b.get("notes", []))
                       for s in blueprint["sections"]
                       for b in s.get("melody", []))
    acc_notes = sum(len(b.get("notes", []))
                    for s in blueprint["sections"]
                    for b in s.get("accompaniment", []))
    bass_notes = sum(len(b.get("notes", []))
                     for s in blueprint["sections"]
                     for b in s.get("bass", []))

    logger.info("Blueprint: '%s' | %d sections | melody=%d, acc=%d, bass=%d notes",
                blueprint.get("title", "Untitled"),
                len(blueprint["sections"]),
                melody_notes, acc_notes, bass_notes)

    return blueprint
