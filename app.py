#!/usr/bin/env python3
"""
Music Composer RAG - Web UI (Gradio) v2
Tabs: Dashboard, Ingest, Search, Generate
"""
import argparse
import logging
import sys
import tempfile
import shutil
import subprocess
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import gradio as gr
import numpy as np

from config.settings import (
    SAMPLE_RATE, HOP_LENGTH, N_MELS, N_CHROMA,
    FRAGMENT_DURATION_SEC, FRAGMENT_OVERLAP_SEC,
    EMBEDDING_DIM, COLLECTION_NAME, QDRANT_HOST, QDRANT_PORT,
    INPUT_DIR, MIDI_DIR, WAV_DIR, OUTPUT_DIR,
)
from src.audio_loader import scan_audio_files, convert_to_wav, load_audio
from src.feature_extractor import fragment_audio, extract_all_features
from src.midi_transcriber import transcribe_to_midi
from src.midi_analyzer import analyze_midi
from src.embedder import build_embedding, build_metadata_payload
from src.qdrant_store import MusicVectorStore
from src.style_profiler import build_style_profile, profile_to_prompt_text
from src.composer_architect import generate_blueprint, list_available_forms
from src.midi_builder import build_midi

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("app")

# --- Global store ---
store = MusicVectorStore(
    host=QDRANT_HOST, port=QDRANT_PORT,
    collection=COLLECTION_NAME, dim=EMBEDDING_DIM,
)
store.ensure_collection()

# Soundfont path for FluidSynth
SOUNDFONT_PATH = Path("/usr/share/sounds/sf2/FluidR3_GM.sf2")
ALT_SOUNDFONT_PATHS = [
    Path("/usr/share/soundfonts/FluidR3_GM.sf2"),
    Path("/usr/share/sounds/sf2/default-GM.sf2"),
    Path("/usr/share/soundfonts/default.sf2"),
    Path.home() / "soundfonts" / "FluidR3_GM.sf2",
]


def find_soundfont():
    """Find an available soundfont file."""
    if SOUNDFONT_PATH.exists():
        return SOUNDFONT_PATH
    for p in ALT_SOUNDFONT_PATHS:
        if p.exists():
            return p
    return None


def midi_to_wav(midi_path, wav_path=None):
    """Convert MIDI to WAV using FluidSynth."""
    if wav_path is None:
        wav_path = Path(str(midi_path).replace(".mid", ".wav"))

    sf = find_soundfont()
    if not sf:
        raise RuntimeError("No soundfont found. Install: apt install fluid-soundfont-gm")

    cmd = [
        "fluidsynth", "-ni", str(sf),
        str(midi_path),
        "-F", str(wav_path),
        "-r", "44100",
        "-g", "1.0",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if result.returncode != 0:
        raise RuntimeError(f"FluidSynth failed: {result.stderr}")

    return wav_path


# ============================================================
# Dashboard
# ============================================================
def get_dashboard():
    try:
        stats = store.get_stats()
        total = stats["total_points"]

        composers = {}
        offset = None
        scrolled = 0
        while scrolled < total:
            result = store.client.scroll(
                collection_name=COLLECTION_NAME,
                limit=100, offset=offset,
                with_payload=["composer"],
            )
            points, next_offset = result
            if not points:
                break
            for p in points:
                c = p.payload.get("composer", "unknown")
                composers[c] = composers.get(c, 0) + 1
            offset = next_offset
            scrolled += len(points)
            if next_offset is None:
                break

        lines = [f"## Collection: {COLLECTION_NAME}",
                 f"**Total fragments:** {total}",
                 f"**Status:** {stats['status']}", ""]

        # Forms
        forms = list_available_forms()
        if forms:
            lines.append(f"**Available forms:** {', '.join(forms.keys())}")
            lines.append("")

        lines.append("### Composers")
        if composers:
            for name, count in sorted(composers.items(), key=lambda x: -x[1]):
                pct = count / total * 100 if total > 0 else 0
                bar = "█" * int(pct / 3)
                lines.append(f"- **{name}**: {count} fragments ({pct:.1f}%) {bar}")
        else:
            lines.append("_No data yet._")

        return "\n".join(lines)
    except Exception as e:
        return f"**Error:** {e}"


# ============================================================
# Ingestion
# ============================================================
def ingest_files(files, composer, era, genre, instrument, skip_midi):
    if not files:
        return "No files uploaded."

    WAV_DIR.mkdir(parents=True, exist_ok=True)
    MIDI_DIR.mkdir(parents=True, exist_ok=True)

    metadata = {
        "composer": composer or "unknown",
        "era": era or "", "genre": genre or "",
        "instrument": instrument or "",
    }

    results = []
    total_fragments = 0

    for file_path in files:
        file_path = Path(file_path)
        try:
            wav_path = convert_to_wav(file_path, WAV_DIR, SAMPLE_RATE)
            y, sr = load_audio(wav_path, SAMPLE_RATE)
            duration = len(y) / sr

            midi_features = {}
            if not skip_midi:
                try:
                    midi_path = transcribe_to_midi(wav_path, MIDI_DIR)
                    midi_features = analyze_midi(midi_path)
                except Exception as e:
                    logger.warning("MIDI failed for %s: %s", file_path.name, e)

            fragments = list(fragment_audio(y, sr, FRAGMENT_DURATION_SEC, FRAGMENT_OVERLAP_SEC))
            batch = []
            for i, (frag_audio, start_sec, end_sec) in enumerate(fragments):
                features = extract_all_features(frag_audio, sr, N_MELS, N_CHROMA, HOP_LENGTH)
                features.update(midi_features)
                embedding = build_embedding(features, audio_data=frag_audio, sr=sr)
                source_info = {**metadata, "file": file_path.name,
                               "fragment_index": i, "start_sec": start_sec, "end_sec": end_sec}
                payload = build_metadata_payload(features, source_info)
                batch.append({"embedding": embedding, "payload": payload})

            if batch:
                store.upsert_batch(batch)
            total_fragments += len(batch)
            results.append(f"  {file_path.name}: {duration:.1f}s → {len(batch)} fragments")
        except Exception as e:
            results.append(f"  {file_path.name}: ERROR — {e}")

    stats = store.get_stats()
    header = f"### Ingestion Complete\n**Added:** {total_fragments} | **Total:** {stats['total_points']}\n"
    return header + "\n".join(results)


# ============================================================
# Search
# ============================================================
def search_similar(audio_file, limit, composer_filter, key_filter, tempo_min, tempo_max):
    if audio_file is None:
        return "Upload an audio file to search."

    try:
        audio_path = Path(audio_file)
        wav_dir = Path(tempfile.mkdtemp())
        wav_path = convert_to_wav(audio_path, wav_dir, SAMPLE_RATE)
        y, sr = load_audio(wav_path, SAMPLE_RATE)

        frag_samples = int(FRAGMENT_DURATION_SEC * sr)
        if len(y) > frag_samples:
            y = y[:frag_samples]

        features = extract_all_features(y, sr, N_MELS, N_CHROMA, HOP_LENGTH)
        query_vec = build_embedding(features, audio_data=y, sr=sr)
        query_key = features.get("key_label", "?")
        query_tempo = features.get("tempo_bpm", 0)

        tempo_range = None
        if tempo_min > 0 and tempo_max > 0:
            tempo_range = (float(tempo_min), float(tempo_max))

        composer = composer_filter.strip() if composer_filter and composer_filter.strip() else None
        key = key_filter.strip() if key_filter and key_filter.strip() else None

        results = store.search_similar(
            query_vec, limit=int(limit),
            composer=composer, key=key, tempo_range=tempo_range,
        )
        shutil.rmtree(wav_dir, ignore_errors=True)

        lines = [f"### Query: {query_key} | {query_tempo:.0f} BPM | {len(y)/sr:.1f}s", "",
                 f"### Top {len(results)} Results", ""]
        if not results:
            lines.append("_No matches._")
        else:
            lines.append("| # | Score | File | Time | Composer | Key | BPM |")
            lines.append("|---|-------|------|------|----------|-----|-----|")
            for i, r in enumerate(results, 1):
                p = r.payload
                lines.append(
                    f"| {i} | {r.score:.4f} | {p['source_file'][:35]} | "
                    f"{p['fragment_start_sec']:.0f}-{p['fragment_end_sec']:.0f}s | "
                    f"{p['composer']} | {p.get('key', '?')} | {p.get('tempo_bpm', 0):.0f} |")
        return "\n".join(lines)
    except Exception as e:
        return f"**Error:** {e}"


# ============================================================
# Generate
# ============================================================
def get_composers_list():
    """Get list of composers in the database."""
    composers = set()
    try:
        result = store.client.scroll(
            collection_name=COLLECTION_NAME, limit=100,
            with_payload=["composer"],
        )
        for p in result[0]:
            composers.add(p.payload.get("composer", "unknown"))
    except Exception:
        pass
    return sorted(composers) if composers else ["Chopin"]


def get_forms_list():
    """Get list of available forms."""
    forms = list_available_forms()
    return ["(free form)"] + sorted(forms.keys())


def generate_composition(composer, form, key, tempo, duration, mood, description):
    """Generate a composition and return audio + info."""
    if not composer:
        return None, "Select a composer."

    try:
        # Build style profile
        profile = build_style_profile(store, composer)
        if not profile:
            return None, f"No data for '{composer}'. Upload audio first."

        profile_text = profile_to_prompt_text(profile)

        # Form
        form_id = None if form == "(free form)" else form

        # Generate blueprint
        params = {
            "key": key or "C minor",
            "tempo_bpm": int(tempo),
            "duration_sec": int(duration),
            "mood": mood or "expressive",
            "description": description or f"A piece in the style of {composer}",
            "instruments": "piano",
            "time_signature": [4, 4],
        }

        blueprint = generate_blueprint(profile_text, params, form_id=form_id)

        # Build MIDI
        safe_title = "".join(c if c.isalnum() or c in " -_" else "" for c in blueprint.get("title", "gen"))
        safe_title = safe_title.strip().replace(" ", "_") or "generated"
        midi_path = OUTPUT_DIR / "generated" / f"{safe_title}.mid"
        build_midi(blueprint, midi_path, style_profile=profile)

        # Convert to WAV for playback
        wav_path = midi_path.with_suffix(".wav")
        try:
            midi_to_wav(midi_path, wav_path)
            audio_output = str(wav_path)
        except Exception as e:
            logger.warning("MIDI->WAV failed: %s. Returning MIDI only.", e)
            audio_output = None

        # Format info
        melody_notes = sum(len(b.get("notes", []))
                           for s in blueprint["sections"]
                           for b in s.get("melody", []))
        acc_notes = sum(len(b.get("notes", []))
                        for s in blueprint["sections"]
                        for b in s.get("accompaniment", []))
        bass_notes = sum(len(b.get("notes", []))
                         for s in blueprint["sections"]
                         for b in s.get("bass", []))

        info_lines = [
            f"## {blueprint.get('title', 'Untitled')}",
            f"**Composer style:** {composer} | **Form:** {form or 'free'}",
            f"**Key:** {blueprint.get('key')} | **Tempo:** {blueprint.get('tempo_bpm')} BPM",
            f"**Sections:** {len(blueprint['sections'])} | **Total bars:** {blueprint.get('total_bars', '?')}",
            f"**Notes:** melody={melody_notes}, accompaniment={acc_notes}, bass={bass_notes}",
            "",
            "### Sections",
        ]
        for s in blueprint["sections"]:
            info_lines.append(
                f"- **{s['name']}** (bars {s['start_bar']}-{s['end_bar']}) "
                f"| {s.get('dynamic', '?')} | {s.get('chord_progression', ['?'])[0]}..."
            )

        info_lines.append(f"\n**MIDI:** `{midi_path}`")
        if audio_output:
            info_lines.append(f"**WAV:** `{wav_path}`")

        return audio_output, "\n".join(info_lines)

    except Exception as e:
        logger.error("Generation failed: %s", e, exc_info=True)
        return None, f"**Generation failed:** {e}"


# ============================================================
# Build UI
# ============================================================
def build_app():
    with gr.Blocks(title="Music Composer RAG", theme=gr.themes.Soft()) as app:
        gr.Markdown("# 🎵 Music Composer RAG\nAnalyze, search, and generate music in any composer's style.")

        with gr.Tabs():
            # --- Dashboard ---
            with gr.Tab("Dashboard"):
                dashboard_output = gr.Markdown(value=get_dashboard)
                refresh_btn = gr.Button("Refresh", variant="secondary")
                refresh_btn.click(fn=get_dashboard, outputs=dashboard_output)

            # --- Ingest ---
            with gr.Tab("Ingest"):
                gr.Markdown("### Upload Audio Files")
                with gr.Row():
                    with gr.Column(scale=2):
                        upload_files = gr.File(label="Audio Files", file_count="multiple",
                                               file_types=[".wav", ".mp3", ".flac", ".ogg", ".m4a"])
                    with gr.Column(scale=1):
                        composer_input = gr.Textbox(label="Composer", placeholder="e.g. Chopin")
                        era_input = gr.Textbox(label="Era", placeholder="e.g. romantic")
                        genre_input = gr.Textbox(label="Genre", value="classical")
                        instrument_input = gr.Textbox(label="Instrument", placeholder="e.g. piano")
                        skip_midi_input = gr.Checkbox(label="Skip MIDI (faster)", value=False)
                ingest_btn = gr.Button("Start Ingestion", variant="primary")
                ingest_output = gr.Markdown()
                ingest_btn.click(fn=ingest_files,
                                 inputs=[upload_files, composer_input, era_input, genre_input,
                                         instrument_input, skip_midi_input],
                                 outputs=ingest_output)

            # --- Search ---
            with gr.Tab("Search"):
                gr.Markdown("### Find Similar Fragments")
                with gr.Row():
                    with gr.Column(scale=2):
                        search_audio = gr.Audio(label="Query Audio", type="filepath",
                                                sources=["upload", "microphone"])
                    with gr.Column(scale=1):
                        limit_slider = gr.Slider(1, 50, value=10, step=1, label="Max results")
                        search_composer = gr.Textbox(label="Filter: composer")
                        search_key = gr.Textbox(label="Filter: key", placeholder="e.g. G minor")
                        with gr.Row():
                            tempo_min = gr.Number(label="Tempo min", value=0, precision=0)
                            tempo_max = gr.Number(label="Tempo max", value=0, precision=0)
                search_btn = gr.Button("Search", variant="primary")
                search_output = gr.Markdown()
                search_btn.click(fn=search_similar,
                                 inputs=[search_audio, limit_slider, search_composer,
                                         search_key, tempo_min, tempo_max],
                                 outputs=search_output)

            # --- Generate ---
            with gr.Tab("Generate"):
                gr.Markdown("### Generate a Composition\nCreate music in a composer's style using AI.")

                with gr.Row():
                    with gr.Column(scale=1):
                        gen_composer = gr.Dropdown(
                            choices=get_composers_list(),
                            label="Composer",
                            value=get_composers_list()[0] if get_composers_list() else None,
                            allow_custom_value=True,
                        )
                        gen_form = gr.Dropdown(
                            choices=get_forms_list(),
                            label="Musical Form",
                            value="nocturne",
                        )
                        gen_key = gr.Textbox(label="Key", value="E minor",
                                             placeholder="e.g. C minor, D major")
                        gen_tempo = gr.Slider(40, 200, value=72, step=2, label="Tempo (BPM)")
                        gen_duration = gr.Slider(30, 180, value=90, step=10,
                                                 label="Duration (seconds)")
                        gen_mood = gr.Textbox(label="Mood", value="melancholic, expressive",
                                              placeholder="e.g. joyful, dramatic, dreamy")
                        gen_desc = gr.Textbox(label="Description (optional)",
                                              placeholder="Additional instructions...")

                    with gr.Column(scale=2):
                        gen_audio = gr.Audio(label="Generated Audio", type="filepath",
                                             interactive=False)
                        gen_info = gr.Markdown(value="*Press Generate to create a composition.*")

                gen_btn = gr.Button("🎵 Generate", variant="primary", size="lg")
                gen_btn.click(
                    fn=generate_composition,
                    inputs=[gen_composer, gen_form, gen_key, gen_tempo,
                            gen_duration, gen_mood, gen_desc],
                    outputs=[gen_audio, gen_info],
                )

    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Music Composer RAG - Web UI")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    app = build_app()
    app.launch(server_name=args.host, server_port=args.port, share=args.share)
