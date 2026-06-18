"""
Composition Architect (Claude API) - section-by-section generation.

Instead of asking the model for the whole piece in one call (which capped pieces
at ~32 bars and made reprises drift), we compose ONE section per API call:

  - plan_sections()      decides how many bars each form section gets
  - _generate_section()  composes a single section, given the style profile, the
                         section spec, and "memory" of what came before
  - assemble_blueprint() stitches sections together, renumbering bars to absolute
                         positions, into the blueprint that midi_builder consumes

Passing earlier material forward lets reprise sections (marked reprise_of in the
form template) restate and ornament the original theme instead of inventing a new
one, and keeps contrasting sections motivically connected.
"""
import os
import json
import logging
from pathlib import Path
import anthropic

from src import retrieval

logger = logging.getLogger(__name__)

FORMS_DIR = Path(__file__).parent.parent / "forms"

MODEL = "claude-sonnet-4-6"
MAX_TOKENS_PER_SECTION = 14000

# Instrument tracks are fixed for the whole piece (piano: melody / accomp / bass).
DEFAULT_TRACKS = [
    {"name": "Melody", "channel": 0, "midi_program": 0, "role": "melody",
     "register": {"low": 60, "high": 88}},
    {"name": "Accompaniment", "channel": 1, "midi_program": 0, "role": "accompaniment",
     "register": {"low": 48, "high": 72}},
    {"name": "Bass", "channel": 2, "midi_program": 0, "role": "bass",
     "register": {"low": 28, "high": 55}},
]

# Generic plan used when no form template is given (small ternary with frame).
DEFAULT_FORM = {
    "name": "Free form",
    "structure": {
        "form": "Intro-A-B-A'-Coda",
        "sections": [
            {"id": "Intro", "name": "Introduction", "bars_range": [2, 4],
             "description": "Brief introduction establishing key, texture and mood.",
             "dynamic_arc": "p", "ends_with": "Half cadence"},
            {"id": "A", "name": "Theme A", "bars_range": [8, 16],
             "description": "Main theme: a clear, singable melody over a steady accompaniment.",
             "dynamic_arc": "p -> mp", "ends_with": "Imperfect authentic cadence"},
            {"id": "B", "name": "Theme B", "bars_range": [8, 16],
             "description": "Contrasting section in a related key with different character.",
             "dynamic_arc": "mp -> f -> mp", "ends_with": "Dominant preparation"},
            {"id": "A_prime", "name": "Theme A' (Reprise)", "bars_range": [8, 16],
             "description": "Return of Theme A, ornamented.", "reprise_of": "A",
             "dynamic_arc": "p -> mf -> p", "ends_with": "Perfect authentic cadence"},
            {"id": "Coda", "name": "Coda", "bars_range": [2, 4],
             "description": "Brief closing that dissolves to rest.",
             "dynamic_arc": "p -> pp", "ends_with": "Final tonic"},
        ],
    },
    "composition_rules": [],
}


SYSTEM_PROMPT = """You are a master composer with encyclopedic knowledge of music theory, counterpoint, harmony, and form.
You compose ONE SECTION of a larger piano piece at a time, output as JSON.

OUTPUT FORMAT - valid JSON only, no markdown, no commentary. One section object:

{
  "title": "Optional - only include for the FIRST section, the title of the whole piece",
  "chord_progression": ["Em", "Am", "D7", "G", "C", "Am", "B7", "Em"],
  "melody": [
    {"bar": 1, "notes": [
      {"pitch": 67, "start_beat": 1.0, "duration": 2.0, "velocity": 58},
      {"pitch": 71, "start_beat": 3.0, "duration": 1.0, "velocity": 62}
    ]}
  ],
  "accompaniment": [
    {"bar": 1, "notes": [
      {"pitch": 52, "start_beat": 1.0, "duration": 0.5, "velocity": 40}
    ]}
  ],
  "bass": [
    {"bar": 1, "notes": [
      {"pitch": 40, "start_beat": 1.0, "duration": 4.0, "velocity": 55}
    ]}
  ]
}

Number the bars of THIS section 1..N (local numbering); it will be placed into the
full piece automatically. EVERY bar from 1 to N must appear in all three tracks.

COMPOSITION PRINCIPLES:

1. VOICE LEADING:
   - Accompaniment voices move by step or common tone when chords change
   - Avoid parallel fifths and octaves between ANY pair of voices. This matters
     MOST for the OUTER voices (melody and bass): never let consecutive melody
     and bass notes move in the same direction a perfect 5th or octave apart.
     If both would land on an octave/fifth, change the bass register or octave.
   - Bass moves by step, fourth, or fifth (not random leaps)

2. PHRASING:
   - Melodies in 4-bar phrases (antecedent + consequent), arch-shaped
   - Leave small rests between phrases for breathing
   - Velocity follows the phrase contour (louder at climax, softer at rests)

3. ACCOMPANIMENT:
   - Use a rhythmically REGULAR, predictable figure to ground the free melody
   - Do NOT copy the melody's rhythm (rhythmic independence)
   - When the melody holds long notes, the accompaniment provides motion

4. MELODY-ACCOMPANIMENT:
   - Melody notes on strong beats are chord tones; weak beats may be passing/neighbor tones
   - Melody register stays clearly above the accompaniment (no voice crossing)

5. BASS:
   - Beat 1 = chord root (occasionally 3rd for first inversion)
   - V-I bass motion at cadences; pedal points at section openings
   - The bass is HARMONICALLY INDEPENDENT: it grounds the harmony with roots and
     stepwise motion. Do NOT double the melody or echo its figuration an octave
     (or fifth) below. In running-figuration textures (e.g. a Bach prelude), keep
     the broken-chord figure in ONE voice (accompaniment) while the bass holds or
     walks the chord roots underneath — the outer voices must not run in parallel.

6. DYNAMICS:
   - pp=30-40, p=45-55, mp=58-68, mf=70-82, f=85-100, ff=105-120
   - Accompaniment velocity 15-25 below melody; bass 5-10 below melody

NOTE FORMAT:
- pitch: MIDI number (60=C4, 64=E4, 67=G4, 72=C5)
- start_beat: position in bar (1.0=beat 1, 2.5=eighth after beat 2)
- duration: in beats (0.5=eighth, 1.0=quarter, 2.0=half)
- velocity: 1-127

Output ONLY valid JSON for this one section."""


# --------------------------------------------------------------------------- #
# Form loading
# --------------------------------------------------------------------------- #
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
    """Load a specific form template by ID (or by its 'id' field)."""
    form_path = FORMS_DIR / f"{form_id}.json"
    if form_path.exists():
        return json.loads(form_path.read_text())
    for f in FORMS_DIR.glob("*.json"):
        try:
            data = json.loads(f.read_text())
            if data.get("id") == form_id:
                return data
        except Exception:
            continue
    return None


# --------------------------------------------------------------------------- #
# Planning (pure)
# --------------------------------------------------------------------------- #
def plan_sections(form_data, params):
    """
    Decide how many bars each section gets, scaling the target duration across
    the form's sections while keeping each within its allowed bars_range.
    Returns a list of {id, name, spec, target_bars, reprise_of}.
    """
    beats_per_bar = params.get("time_signature", [4, 4])[0]
    tempo = params.get("tempo_bpm", 100)
    duration = params.get("duration_sec", 60)
    total_target = max(8, round((tempo * duration) / (60 * beats_per_bar)))

    sections = form_data["structure"]["sections"]
    mids = []
    for s in sections:
        lo, hi = s.get("bars_range", [8, 8])
        mids.append((lo + hi) / 2.0)
    base = sum(mids) or 1.0
    factor = total_target / base

    plan = []
    for s, mid in zip(sections, mids):
        lo, hi = s.get("bars_range", [8, 8])
        target = int(round(mid * factor))
        target = max(lo, min(hi, target))
        plan.append({
            "id": s["id"],
            "name": s.get("name", s["id"]),
            "spec": s,
            "target_bars": target,
            "reprise_of": s.get("reprise_of"),
        })
    return plan


def _section_spec_text(spec, target_bars):
    """Render a single section's spec for the prompt."""
    lines = [f"Section: {spec.get('name', spec['id'])}",
             f"Length: EXACTLY {target_bars} bars (number them 1..{target_bars})"]
    for label, key in [
        ("Character", "description"),
        ("Melody", "melody_character"),
        ("Accompaniment", "accompaniment_character"),
        ("Harmonic rhythm", "harmonic_rhythm"),
        ("Key relation", "key_relation"),
        ("Dynamic arc", "dynamic_arc"),
        ("Ends with", "ends_with"),
    ]:
        if spec.get(key):
            lines.append(f"  {label}: {spec[key]}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Assembly (pure)
# --------------------------------------------------------------------------- #
TRACK_KEYS = ("melody", "accompaniment", "bass")


def _section_bar_count(section_result, target_bars):
    """Highest bar number present across tracks (fallback: target)."""
    maxbar = 0
    for key in TRACK_KEYS:
        for bar_data in section_result.get(key, []):
            maxbar = max(maxbar, bar_data.get("bar", 0))
    return maxbar or target_bars


def assemble_blueprint(params, form_data, planned, section_results, title=None):
    """
    Stitch composed sections into a full blueprint, renumbering each section's
    local bars (1..N) to absolute positions across the piece.
    """
    cursor = 1
    out_sections = []
    for plan_item, res in zip(planned, section_results):
        n_bars = _section_bar_count(res, plan_item["target_bars"])
        start_bar = cursor
        offset = start_bar - 1

        section_obj = {
            "name": plan_item["name"],
            "form_id": plan_item["id"],
            "start_bar": start_bar,
            "end_bar": start_bar + n_bars - 1,
            "tempo_bpm": res.get("tempo_bpm", params.get("tempo_bpm", 100)),
            "dynamic": res.get("dynamic", plan_item["spec"].get("dynamic_arc", "mp")),
            "chord_progression": res.get("chord_progression", []),
        }
        for key in TRACK_KEYS:
            bars = []
            for bar_data in res.get(key, []):
                bars.append({
                    "bar": bar_data.get("bar", 1) + offset,
                    "notes": bar_data.get("notes", []),
                })
            section_obj[key] = bars
        out_sections.append(section_obj)
        cursor = start_bar + n_bars

    return {
        "title": title or params.get("description") or "Generated",
        "key": params.get("key", "C minor"),
        "time_signature": params.get("time_signature", [4, 4]),
        "tempo_bpm": params.get("tempo_bpm", 100),
        "total_bars": cursor - 1,
        "form": form_data.get("id", form_data.get("name", "free form")),
        "tracks": [dict(t) for t in DEFAULT_TRACKS],
        "sections": out_sections,
    }


# --------------------------------------------------------------------------- #
# Single-section generation (API)
# --------------------------------------------------------------------------- #
def _strip_fences(text):
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        start, end = 1, len(lines) - 1
        while end > start and lines[end].strip() in ("```", ""):
            end -= 1
        text = "\n".join(lines[start:end + 1])
    return text


def _memory_context(plan_item, composed_by_id, prev_result):
    """Build the 'what came before' context for this section."""
    parts = []

    reprise_id = plan_item.get("reprise_of")
    if reprise_id and reprise_id in composed_by_id:
        ref = composed_by_id[reprise_id]
        parts.append(
            "THIS SECTION IS A REPRISE of an earlier section. You MUST reuse the "
            "SAME melodic outline as that theme (the listener must recognize it), "
            "but ornament/vary it as the section character requires. "
            "Here is the original melody to restate:")
        parts.append("ORIGINAL MELODY (JSON): " +
                      json.dumps(ref.get("melody", []), ensure_ascii=False))
    else:
        # Keep contrasting/new sections motivically connected to the opening theme.
        theme = None
        for sid in ("A", "Exposition_P", "Exposition", "Opening", "Intro"):
            if sid in composed_by_id:
                theme = composed_by_id[sid]
                break
        if theme and theme.get("melody"):
            head = theme["melody"][:2]
            parts.append(
                "Stay motivically connected to the piece's main theme. Its opening "
                "bars were: " + json.dumps(head, ensure_ascii=False))

    if prev_result is not None:
        last_bars = {}
        for key in TRACK_KEYS:
            bars = prev_result.get(key, [])
            if bars:
                last_bars[key] = bars[-1]
        if last_bars:
            parts.append(
                "For a smooth join, the LAST bar of the previous section was: " +
                json.dumps(last_bars, ensure_ascii=False))

    return "\n\n".join(parts)


def _build_section_prompt(style_profile_text, params, form_data, plan_item,
                          memory_text, is_first, section_index=0):
    parts = ["Compose ONE section of a larger piano piece. "
             "Output notes for ALL three tracks (melody, accompaniment, bass) in every bar."]
    parts.append(f"STYLE PROFILE:\n{style_profile_text}")

    # Step 4: real reference material retrieved from the composer's own scores.
    reference = retrieval.reference_block(params.get("composer", ""), section_index=section_index)
    if reference:
        parts.append(reference)

    parts.append(
        f"OVERALL FORM: {form_data.get('name', 'free form')} "
        f"({form_data['structure'].get('form', '')}). "
        f"This is one of its sections.")

    parts.append(
        "PIECE PARAMETERS:\n"
        f"- Key: {params.get('key', 'C minor')}\n"
        f"- Tempo: {params.get('tempo_bpm', 100)} BPM\n"
        f"- Time signature: {params.get('time_signature', [4, 4])}\n"
        f"- Mood: {params.get('mood', 'expressive')}")

    parts.append("THIS SECTION TO COMPOSE NOW:\n" +
                 _section_spec_text(plan_item["spec"], plan_item["target_bars"]))

    if memory_text:
        parts.append(memory_text)

    if is_first:
        parts.append('Include a "title" field for the whole piece in your JSON.')

    parts.append("Output ONLY valid JSON for this one section.")
    return "\n\n".join(parts)


def _generate_section(client, prompt, attempts=3):
    """Compose one section, retrying if the model returns empty/invalid JSON.

    A single malformed section response used to abort the whole multi-minute
    generation; transient empties resolve on a retry.
    """
    raw = ""
    last_err = None
    for attempt in range(1, attempts + 1):
        response = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS_PER_SECTION,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.content[0].text if response.content else ""
        raw = _strip_fences(text)
        try:
            section = json.loads(raw)
            break
        except json.JSONDecodeError as e:
            last_err = e
            logger.warning("Section JSON parse failed (attempt %d/%d, stop_reason=%s, %d chars): %s",
                           attempt, attempts, getattr(response, "stop_reason", "?"), len(raw), e)
            if getattr(response, "stop_reason", None) == "max_tokens":
                logger.warning("  section hit max_tokens (%d) and was truncated", MAX_TOKENS_PER_SECTION)
    else:
        logger.error("Section parse failed after %d attempts.\nRaw (first 1200): %s", attempts, raw[:1200])
        raise ValueError(f"Claude returned invalid JSON for a section after {attempts} attempts: {last_err}")

    for key in TRACK_KEYS:
        if key not in section:
            logger.warning("Section missing '%s' track; using empty.", key)
            section[key] = []
    return section


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def generate_blueprint(style_profile_text, params, form_id=None):
    """Generate a full composition blueprint, one section per API call."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set")
    client = anthropic.Anthropic(api_key=api_key)

    form_data = None
    if form_id:
        form_data = load_form(form_id)
        if form_data:
            logger.info("Using form template: %s", form_data["name"])
        else:
            logger.warning("Form '%s' not found; using free-form plan.", form_id)
    if form_data is None:
        form_data = DEFAULT_FORM

    planned = plan_sections(form_data, params)
    logger.info("Plan: %d sections, %d bars total (%s)",
                len(planned), sum(p["target_bars"] for p in planned),
                ", ".join(f"{p['id']}:{p['target_bars']}" for p in planned))

    section_results = []
    composed_by_id = {}
    title = None
    prev_result = None

    for i, plan_item in enumerate(planned):
        memory_text = _memory_context(plan_item, composed_by_id, prev_result)
        prompt = _build_section_prompt(
            style_profile_text, params, form_data, plan_item,
            memory_text, is_first=(i == 0), section_index=i)

        logger.info("Composing section %d/%d: %s (%d bars)...",
                    i + 1, len(planned), plan_item["name"], plan_item["target_bars"])
        res = _generate_section(client, prompt)

        if i == 0 and res.get("title"):
            title = res["title"]

        section_results.append(res)
        composed_by_id[plan_item["id"]] = res
        prev_result = res

    blueprint = assemble_blueprint(params, form_data, planned, section_results, title=title)

    melody_notes = sum(len(b.get("notes", [])) for s in blueprint["sections"] for b in s["melody"])
    acc_notes = sum(len(b.get("notes", [])) for s in blueprint["sections"] for b in s["accompaniment"])
    bass_notes = sum(len(b.get("notes", [])) for s in blueprint["sections"] for b in s["bass"])
    logger.info("Blueprint: '%s' | %d sections | %d bars | melody=%d, acc=%d, bass=%d notes",
                blueprint.get("title", "Untitled"), len(blueprint["sections"]),
                blueprint["total_bars"], melody_notes, acc_notes, bass_notes)

    return blueprint
