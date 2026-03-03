#!/usr/bin/env python3
"""
Music Composer RAG - Web UI (Gradio)
Usage: python3 app.py [--port 7860] [--share]
"""
import argparse
import logging
import sys
import tempfile
import shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import gradio as gr
import numpy as np

from config.settings import (
    SAMPLE_RATE, HOP_LENGTH, N_MELS, N_CHROMA,
    FRAGMENT_DURATION_SEC, FRAGMENT_OVERLAP_SEC,
    EMBEDDING_DIM, COLLECTION_NAME, QDRANT_HOST, QDRANT_PORT,
    INPUT_DIR, MIDI_DIR, WAV_DIR,
)
from src.audio_loader import scan_audio_files, convert_to_wav, load_audio
from src.feature_extractor import fragment_audio, extract_all_features
from src.midi_transcriber import transcribe_to_midi
from src.midi_analyzer import analyze_midi
from src.embedder import build_embedding, build_metadata_payload
from src.qdrant_store import MusicVectorStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("app")

# --- Global store ---
store = MusicVectorStore(
    host=QDRANT_HOST, port=QDRANT_PORT,
    collection=COLLECTION_NAME, dim=EMBEDDING_DIM,
)
store.ensure_collection()


# ============================================================
# Dashboard
# ============================================================
def get_dashboard():
    """Return collection stats and composer breakdown."""
    try:
        stats = store.get_stats()
        total = stats["total_points"]

        # Get composer distribution by scrolling points
        composers = {}
        offset = None
        batch_size = 100
        scrolled = 0

        while scrolled < total:
            result = store.client.scroll(
                collection_name=COLLECTION_NAME,
                limit=batch_size,
                offset=offset,
                with_payload=["composer", "key", "tempo_bpm"],
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

        # Format output
        lines = []
        lines.append(f"## Collection: {COLLECTION_NAME}")
        lines.append(f"**Total fragments:** {total}")
        lines.append(f"**Status:** {stats['status']}")
        lines.append("")
        lines.append("### Composers")
        if composers:
            for name, count in sorted(composers.items(), key=lambda x: -x[1]):
                pct = count / total * 100 if total > 0 else 0
                bar = "#" * int(pct / 2)
                lines.append(f"- **{name}**: {count} fragments ({pct:.1f}%) {bar}")
        else:
            lines.append("_No data yet. Upload some audio!_")

        return "\n".join(lines)
    except Exception as e:
        return f"**Error connecting to Qdrant:** {e}\n\nMake sure Qdrant is running at {QDRANT_HOST}:{QDRANT_PORT}"


# ============================================================
# Ingestion
# ============================================================
def ingest_files(files, composer, era, genre, instrument, skip_midi):
    """Process uploaded audio files and store in Qdrant."""
    if not files:
        return "No files uploaded."

    WAV_DIR.mkdir(parents=True, exist_ok=True)
    MIDI_DIR.mkdir(parents=True, exist_ok=True)

    metadata = {
        "composer": composer or "unknown",
        "era": era or "",
        "genre": genre or "",
        "instrument": instrument or "",
    }

    results = []
    total_fragments = 0

    for file_path in files:
        file_path = Path(file_path)
        try:
            # Convert to WAV
            wav_path = convert_to_wav(file_path, WAV_DIR, SAMPLE_RATE)
            y, sr = load_audio(wav_path, SAMPLE_RATE)
            duration = len(y) / sr

            # MIDI transcription
            midi_features = {}
            if not skip_midi:
                try:
                    midi_path = transcribe_to_midi(wav_path, MIDI_DIR)
                    midi_features = analyze_midi(midi_path)
                except Exception as e:
                    logger.warning("MIDI failed for %s: %s", file_path.name, e)

            # Fragment and process
            fragments = list(fragment_audio(y, sr, FRAGMENT_DURATION_SEC, FRAGMENT_OVERLAP_SEC))
            batch = []
            for i, (frag_audio, start_sec, end_sec) in enumerate(fragments):
                features = extract_all_features(frag_audio, sr, N_MELS, N_CHROMA, HOP_LENGTH)
                features.update(midi_features)
                embedding = build_embedding(features)
                source_info = {
                    **metadata,
                    "file": file_path.name,
                    "fragment_index": i,
                    "start_sec": start_sec,
                    "end_sec": end_sec,
                }
                payload = build_metadata_payload(features, source_info)
                batch.append({"embedding": embedding, "payload": payload})

            if batch:
                store.upsert_batch(batch)

            total_fragments += len(batch)
            results.append(f"  {file_path.name}: {duration:.1f}s -> {len(batch)} fragments")

        except Exception as e:
            results.append(f"  {file_path.name}: ERROR - {e}")

    stats = store.get_stats()
    header = f"### Ingestion Complete\n**Fragments added:** {total_fragments} | **Total in DB:** {stats['total_points']}\n"
    return header + "\n".join(results)


# ============================================================
# Search
# ============================================================
def search_similar(audio_file, limit, composer_filter, key_filter, tempo_min, tempo_max):
    """Search for similar fragments given an audio file."""
    if audio_file is None:
        return "Upload an audio file to search."

    try:
        audio_path = Path(audio_file)
        wav_dir = Path(tempfile.mkdtemp())
        wav_path = convert_to_wav(audio_path, wav_dir, SAMPLE_RATE)
        y, sr = load_audio(wav_path, SAMPLE_RATE)

        # Use first fragment for search
        frag_samples = int(FRAGMENT_DURATION_SEC * sr)
        if len(y) > frag_samples:
            y = y[:frag_samples]

        features = extract_all_features(y, sr, N_MELS, N_CHROMA, HOP_LENGTH)
        query_vec = build_embedding(features)

        # Query info
        query_key = features.get("key_label", "?")
        query_tempo = features.get("tempo_bpm", 0)

        # Filters
        tempo_range = None
        if tempo_min > 0 and tempo_max > 0:
            tempo_range = (float(tempo_min), float(tempo_max))

        composer = composer_filter.strip() if composer_filter and composer_filter.strip() else None
        key = key_filter.strip() if key_filter and key_filter.strip() else None

        results = store.search_similar(
            query_vec, limit=int(limit),
            composer=composer, key=key, tempo_range=tempo_range,
        )

        # Cleanup
        shutil.rmtree(wav_dir, ignore_errors=True)

        # Format results
        lines = []
        lines.append(f"### Query Analysis")
        lines.append(f"**Detected key:** {query_key} | **Tempo:** {query_tempo:.0f} BPM | **Duration:** {len(y)/sr:.1f}s")
        lines.append("")
        lines.append(f"### Top {len(results)} Results")
        lines.append("")

        if not results:
            lines.append("_No matching fragments found. Try relaxing filters._")
        else:
            lines.append("| # | Score | File | Time | Composer | Key | BPM |")
            lines.append("|---|-------|------|------|----------|-----|-----|")
            for i, r in enumerate(results, 1):
                p = r.payload
                lines.append(
                    f"| {i} | {r.score:.4f} | {p['source_file'][:40]} | "
                    f"{p['fragment_start_sec']:.0f}-{p['fragment_end_sec']:.0f}s | "
                    f"{p['composer']} | {p.get('key', '?')} | {p.get('tempo_bpm', 0):.0f} |"
                )

        return "\n".join(lines)

    except Exception as e:
        return f"**Error:** {e}"


# ============================================================
# Build UI
# ============================================================
def build_app():
    with gr.Blocks(title="Music Composer RAG", theme=gr.themes.Soft()) as app:
        gr.Markdown("# Music Composer RAG\nAnalyze, index, and search musical compositions by style.")

        with gr.Tabs():
            # --- Dashboard Tab ---
            with gr.Tab("Dashboard"):
                dashboard_output = gr.Markdown(value=get_dashboard)
                refresh_btn = gr.Button("Refresh", variant="secondary")
                refresh_btn.click(fn=get_dashboard, outputs=dashboard_output)

            # --- Ingest Tab ---
            with gr.Tab("Ingest"):
                gr.Markdown("### Upload Audio Files\nUpload WAV or MP3 files to analyze and store in the vector database.")
                with gr.Row():
                    with gr.Column(scale=2):
                        upload_files = gr.File(
                            label="Audio Files",
                            file_count="multiple",
                            file_types=[".wav", ".mp3", ".flac", ".ogg", ".m4a"],
                        )
                    with gr.Column(scale=1):
                        composer_input = gr.Textbox(label="Composer", placeholder="e.g. Chopin")
                        era_input = gr.Textbox(label="Era", placeholder="e.g. romantic")
                        genre_input = gr.Textbox(label="Genre", placeholder="e.g. classical", value="classical")
                        instrument_input = gr.Textbox(label="Instrument", placeholder="e.g. piano")
                        skip_midi_input = gr.Checkbox(label="Skip MIDI transcription (faster)", value=False)

                ingest_btn = gr.Button("Start Ingestion", variant="primary")
                ingest_output = gr.Markdown()
                ingest_btn.click(
                    fn=ingest_files,
                    inputs=[upload_files, composer_input, era_input, genre_input, instrument_input, skip_midi_input],
                    outputs=ingest_output,
                )

            # --- Search Tab ---
            with gr.Tab("Search"):
                gr.Markdown("### Find Similar Fragments\nUpload an audio snippet to find similar fragments in the database.")
                with gr.Row():
                    with gr.Column(scale=2):
                        search_audio = gr.Audio(
                            label="Query Audio",
                            type="filepath",
                            sources=["upload", "microphone"],
                        )
                    with gr.Column(scale=1):
                        limit_slider = gr.Slider(minimum=1, maximum=50, value=10, step=1, label="Max results")
                        search_composer = gr.Textbox(label="Filter by composer", placeholder="Leave empty for all")
                        search_key = gr.Textbox(label="Filter by key", placeholder="e.g. G minor")
                        with gr.Row():
                            tempo_min = gr.Number(label="Tempo min", value=0, precision=0)
                            tempo_max = gr.Number(label="Tempo max", value=0, precision=0)

                search_btn = gr.Button("Search", variant="primary")
                search_output = gr.Markdown()
                search_btn.click(
                    fn=search_similar,
                    inputs=[search_audio, limit_slider, search_composer, search_key, tempo_min, tempo_max],
                    outputs=search_output,
                )

    return app


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Music Composer RAG - Web UI")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--share", action="store_true", help="Create public Gradio link")
    args = parser.parse_args()

    app = build_app()
    app.launch(server_name=args.host, server_port=args.port, share=args.share)
