"""
Style Profiler

Loads hand-authored qualitative style profiles from profiles/<composer>.json
and renders them as prompt text for the composition model. These profiles are
the style context for generation — no database or audio analysis required.
"""
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

PROFILES_DIR = Path(__file__).parent.parent / "profiles"


def _slug(name):
    """Filesystem-safe slug for a composer name."""
    return "".join(c.lower() if c.isalnum() else "_" for c in (name or "")).strip("_")


def load_static_profile(composer):
    """
    Load a hand-authored style profile from profiles/<composer>.json.
    Returns the profile dict, or None if no matching profile exists.
    """
    if not composer:
        return None
    path = PROFILES_DIR / f"{_slug(composer)}.json"
    if path.exists():
        return json.loads(path.read_text())
    # Fall back to a case-insensitive match on the "composer" field.
    if PROFILES_DIR.exists():
        for f in PROFILES_DIR.glob("*.json"):
            try:
                data = json.loads(f.read_text())
                if data.get("composer", "").lower() == composer.lower():
                    return data
            except Exception:
                continue
    return None


def list_static_profiles():
    """Return {composer_name: profile_dict} for every profile in profiles/."""
    profiles = {}
    if PROFILES_DIR.exists():
        for f in sorted(PROFILES_DIR.glob("*.json")):
            try:
                data = json.loads(f.read_text())
                profiles[data.get("composer", f.stem)] = data
            except Exception as e:
                logger.warning("Failed to load profile %s: %s", f.name, e)
    return profiles


def static_profile_to_prompt_text(profile):
    """Render a hand-authored style profile as text for the LLM prompt."""
    if not profile:
        return "No style profile available."

    lines = [f"Composer: {profile.get('composer', 'Unknown')}"]
    if profile.get("era"):
        lines.append(f"Era: {profile['era']}")
    if profile.get("description"):
        lines.append(profile["description"])

    tr = profile.get("tempo_range")
    if tr and len(tr) == 2:
        lines.append(f"Typical tempo range: {tr[0]}-{tr[1]} BPM")

    tk = profile.get("typical_keys")
    if tk:
        lines.append("Characteristic keys: " + ", ".join(tk))

    if profile.get("harmonic_language"):
        lines.append(f"Harmonic language: {profile['harmonic_language']}")

    sn = profile.get("style_notes")
    if sn:
        lines.append("\nStylistic traits:")
        for s in sn:
            lines.append(f"  - {s}")

    return "\n".join(lines)
