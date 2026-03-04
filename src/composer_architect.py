"""
Level 1 - Composition Architect (Claude API) v3
Loads musical form templates from the forms/ library and uses them
to guide Claude in generating structurally correct compositions.
"""
import os
import json
import logging
from pathlib import Path
import anthropic

logger = logging.getLogger(__name__)

FORMS_DIR = Path(__file__).parent.parent / "forms"

SYSTEM_PROMPT_BASE = """You are an expert music composer with deep knowledge of music theory, form, and style.
You create detailed musical scores in JSON format.

You will receive:
1. A style profile of a specific composer
2. A musical FORM TEMPLATE with strict structural rules
3. Parameters for the composition

CRITICAL FORM RULES:
- You MUST follow the form template exactly
- Each section described in the form MUST appear in your output
- Thematic relationships described in the form (e.g. "reprise must reuse Theme A melody") are MANDATORY
- Dynamic arcs, key relationships, and cadences must match the form description

OUTPUT FORMAT - valid JSON only, no markdown, no commentary:

{
  "title": "Title",
  "key": "E minor",
  "time_signature": [4, 4],
  "tempo_bpm": 72,
  "total_bars": 32,
  "form": "nocturne",
  "sections": [
    {
      "name": "Theme A",
      "form_id": "A",
      "start_bar": 1,
      "end_bar": 8,
      "tempo_bpm": 72,
      "dynamic": "p",
      "chord_progression": ["Em", "Am", "D", "G", "C", "Am", "B7", "Em"],
      "melody": [
        {"bar": 1, "notes": [
          {"pitch": 64, "start_beat": 1.0, "duration": 2.0, "velocity": 70},
          {"pitch": 67, "start_beat": 3.0, "duration": 1.0, "velocity": 75}
        ]},
        {"bar": 2, "notes": [...]}
      ],
      "description": "Opening lyrical theme"
    }
  ],
  "tracks": [
    {"name": "Melody", "channel": 0, "midi_program": 0, "role": "melody",
     "register": {"low": 60, "high": 88}},
    {"name": "Accompaniment", "channel": 1, "midi_program": 0, "role": "accompaniment",
     "register": {"low": 48, "high": 72}},
    {"name": "Bass", "channel": 2, "midi_program": 0, "role": "bass",
     "register": {"low": 28, "high": 55}}
  ]
}

MELODY RULES:
- pitch: MIDI note number (60=C4, 64=E4, 67=G4, 72=C5)
- start_beat: position in bar (1.0 = beat 1, 1.5 = eighth after beat 1)
- duration: in beats (0.5=eighth, 1.0=quarter, 2.0=half, 4.0=whole)
- velocity: 1-127
- EVERY bar must have melody notes
- Melody on strong beats should be chord tones
- Use ornamental notes (passing tones, neighbor tones, turns) on weak beats
- Create clear phrase structure (typically 4-bar phrases with arch contour)
- When the form says "reuse theme" or "reprise", you MUST use the SAME pitch sequence with variations

Only output melody for the Melody track. Accompaniment and Bass are generated separately.
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
        # Try to find by scanning
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

    lines.append("SECTIONS (you must include ALL of these):")
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
    """
    Generate composition blueprint using Claude API with form template.

    Args:
        style_profile_text: composer style profile
        params: dict with key, tempo_bpm, duration_sec, mood, etc.
        form_id: optional form template ID (nocturne, sonata, fugue, prelude)
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set")

    client = anthropic.Anthropic(api_key=api_key)

    # Load form template
    form_text = ""
    if form_id:
        form_data = load_form(form_id)
        if form_data:
            form_text = form_to_prompt_text(form_data)
            logger.info("Using form template: %s", form_data["name"])
        else:
            logger.warning("Form '%s' not found, proceeding without form template", form_id)
            available = list_available_forms()
            if available:
                logger.info("Available forms: %s", ", ".join(available.keys()))

    tempo = params.get("tempo_bpm", 120)
    duration = params.get("duration_sec", 60)
    beats_per_bar = params.get("time_signature", [4, 4])[0]
    total_bars = max(8, int((tempo * duration) / (60 * beats_per_bar)))

    user_parts = []
    user_parts.append("Generate a composition with explicit melody notes for every bar.")
    user_parts.append("")
    user_parts.append(f"STYLE PROFILE:\n{style_profile_text}")
    user_parts.append("")

    if form_text:
        user_parts.append(f"FORM TEMPLATE:\n{form_text}")
        user_parts.append("")
        user_parts.append("You MUST follow this form template exactly. Every section listed must appear.")
        user_parts.append("")

    user_parts.append(f"""PARAMETERS:
- Key: {params.get('key', 'C minor')}
- Tempo: {tempo} BPM
- Time signature: {params.get('time_signature', [4, 4])}
- Approximate length: {total_bars} bars ({duration} seconds)
- Mood: {params.get('mood', 'expressive')}
- Description: {params.get('description', '')}
- Instruments: {params.get('instruments', 'piano')}""")

    user_parts.append("")
    user_parts.append("Write a musically coherent composition following the form and style.")
    user_parts.append("Output ONLY valid JSON.")

    user_message = "\n".join(user_parts)

    logger.info("Calling Claude API (form=%s, ~%d bars)...", form_id or "free", total_bars)

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=8192,
        system=SYSTEM_PROMPT_BASE,
        messages=[{"role": "user", "content": user_message}],
    )

    raw_text = response.content[0].text.strip()

    # Clean markdown wrapping
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
        logger.error("Raw (first 1000): %s", raw_text[:1000])
        raise ValueError(f"Claude returned invalid JSON: {e}")

    # Validate
    for field in ["sections", "tracks", "key", "tempo_bpm"]:
        if field not in blueprint:
            raise ValueError(f"Blueprint missing field: {field}")

    # Count melody notes
    total_notes = 0
    for section in blueprint["sections"]:
        for bar in section.get("melody", []):
            total_notes += len(bar.get("notes", []))

    logger.info("Blueprint: '%s', %d sections, %d tracks, %d melody notes",
                blueprint.get("title", "Untitled"),
                len(blueprint["sections"]),
                len(blueprint["tracks"]),
                total_notes)

    return blueprint
