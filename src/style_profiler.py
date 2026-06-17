"""
Style Profiler
Extracts a composer's "musical DNA" from the Qdrant database:
- Characteristic intervals and their frequencies
- Pitch class distribution (harmonic preferences)
- Tempo and rhythm patterns
- Dynamic range (velocity)
- Note density
- Key preferences

This profile is passed to Claude API as generation context.
"""
import json
import logging
import numpy as np
from collections import Counter
from pathlib import Path

logger = logging.getLogger(__name__)

PROFILES_DIR = Path(__file__).parent.parent / "profiles"


def _slug(name):
    """Filesystem-safe slug for a composer name."""
    return "".join(c.lower() if c.isalnum() else "_" for c in (name or "")).strip("_")


def load_static_profile(composer):
    """
    Load a hand-authored qualitative style profile from profiles/<composer>.json.
    This is the profile source for generation (path C) — no Qdrant required.
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


def build_style_profile(store, composer, max_fragments=200):
    """
    Query Qdrant for a composer's fragments and build a statistical style profile.
    Returns a dict suitable for JSON serialization and LLM prompting.
    """
    from qdrant_client.models import Filter, FieldCondition, MatchValue

    # Fetch fragments for this composer
    filt = Filter(must=[
        FieldCondition(key="composer", match=MatchValue(value=composer))
    ])

    results = store.client.scroll(
        collection_name=store.collection,
        scroll_filter=filt,
        limit=max_fragments,
        with_payload=True,
    )
    points = results[0]

    if not points:
        logger.warning("No fragments found for composer: %s", composer)
        return None

    logger.info("Building style profile for %s from %d fragments", composer, len(points))

    # Collect statistics
    tempos = []
    keys = Counter()
    chroma_profiles = []
    note_densities = []
    velocity_means = []
    velocity_stds = []
    pitch_histograms = []
    interval_histograms = Counter()
    spectral_centroids = []
    rms_values = []

    for p in points:
        pay = p.payload

        if pay.get("tempo_bpm", 0) > 0:
            tempos.append(pay["tempo_bpm"])

        if pay.get("key"):
            keys[pay["key"]] += 1

        if pay.get("chroma_profile"):
            chroma_profiles.append(pay["chroma_profile"])

        if pay.get("note_density", 0) > 0:
            note_densities.append(pay["note_density"])

        if pay.get("velocity_mean", 0) > 0:
            velocity_means.append(pay["velocity_mean"])

        if pay.get("pitch_class_histogram"):
            pitch_histograms.append(pay["pitch_class_histogram"])

        if pay.get("spectral_centroid", 0) > 0:
            spectral_centroids.append(pay["spectral_centroid"])

        if pay.get("rms", 0) > 0:
            rms_values.append(pay["rms"])

    # Aggregate
    note_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

    # Average chroma profile
    avg_chroma = [0.0] * 12
    if chroma_profiles:
        arr = np.array(chroma_profiles)
        avg_chroma = np.mean(arr, axis=0).tolist()

    # Average pitch class histogram from MIDI
    avg_pitch_hist = [0.0] * 12
    if pitch_histograms:
        arr = np.array([h for h in pitch_histograms if len(h) == 12])
        if len(arr) > 0:
            avg_pitch_hist = np.mean(arr, axis=0).tolist()

    # Top keys
    top_keys = keys.most_common(5)

    # Tempo stats
    tempo_stats = {}
    if tempos:
        tempo_stats = {
            "mean": round(np.mean(tempos), 1),
            "std": round(np.std(tempos), 1),
            "min": round(min(tempos), 1),
            "max": round(max(tempos), 1),
            "median": round(np.median(tempos), 1),
        }

    # Preferred pitch classes (top 5)
    if avg_pitch_hist:
        ranked = sorted(enumerate(avg_pitch_hist), key=lambda x: -x[1])
        preferred_pitches = [(note_names[i], round(v, 3)) for i, v in ranked[:5]]
    else:
        preferred_pitches = []

    profile = {
        "composer": composer,
        "fragments_analyzed": len(points),
        "tempo": tempo_stats,
        "preferred_keys": [{"key": k, "count": c} for k, c in top_keys],
        "chroma_profile": {note_names[i]: round(v, 3) for i, v in enumerate(avg_chroma)},
        "preferred_pitch_classes": preferred_pitches,
        "pitch_histogram": {note_names[i]: round(v, 3) for i, v in enumerate(avg_pitch_hist)},
        "note_density": {
            "mean": round(np.mean(note_densities), 2) if note_densities else 0,
            "std": round(np.std(note_densities), 2) if note_densities else 0,
        },
        "velocity": {
            "mean": round(np.mean(velocity_means), 1) if velocity_means else 64,
            "range": [
                round(min(velocity_means), 1) if velocity_means else 40,
                round(max(velocity_means), 1) if velocity_means else 100,
            ],
        },
        "brightness": {
            "spectral_centroid_mean": round(np.mean(spectral_centroids), 1) if spectral_centroids else 0,
        },
        "loudness": {
            "rms_mean": round(np.mean(rms_values), 4) if rms_values else 0,
        },
    }

    logger.info("Style profile built: %d keys, tempo %.0f+-%.0f BPM",
                len(top_keys),
                tempo_stats.get("mean", 0),
                tempo_stats.get("std", 0))

    return profile


def profile_to_prompt_text(profile):
    """Convert style profile to human-readable text for LLM prompting."""
    if not profile:
        return "No style profile available."

    lines = []
    lines.append(f"Composer: {profile['composer']}")
    lines.append(f"Based on {profile['fragments_analyzed']} analyzed fragments.")
    lines.append("")

    t = profile.get("tempo", {})
    if t:
        lines.append(f"Tempo: {t['mean']} BPM (range {t['min']}-{t['max']}, std {t['std']})")

    keys = profile.get("preferred_keys", [])
    if keys:
        key_str = ", ".join(f"{k['key']} ({k['count']}x)" for k in keys)
        lines.append(f"Preferred keys: {key_str}")

    pp = profile.get("preferred_pitch_classes", [])
    if pp:
        pp_str = ", ".join(f"{name} ({weight})" for name, weight in pp)
        lines.append(f"Preferred pitch classes: {pp_str}")

    nd = profile.get("note_density", {})
    if nd.get("mean"):
        lines.append(f"Note density: {nd['mean']} notes/beat (std {nd['std']})")

    v = profile.get("velocity", {})
    if v.get("mean"):
        lines.append(f"Velocity: mean {v['mean']}, range {v['range'][0]}-{v['range'][1]}")

    return "\n".join(lines)
