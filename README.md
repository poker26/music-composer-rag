# Music Composer RAG

A RAG-based pipeline for analyzing and indexing musical compositions by style.
Feed it audio files of different composers, and it builds a searchable vector
database of musical features — mel-spectrograms, chroma profiles, tempo, key,
MIDI note patterns, and more.

## Architecture

```
WAV/MP3 -> ffmpeg -> librosa (features) -> embedding (512d)
                  -> basic-pitch (MIDI)  -> mido (analysis)  -> Qdrant
```

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt
sudo apt install ffmpeg

# 2. Start Qdrant
docker run -d -p 6333:6333 qdrant/qdrant

# 3. Ingest audio
python ingest.py --input ./input/chopin --composer "Chopin" --era "romantic" --instrument "piano"
python ingest.py --input ./input/bach --composer "Bach" --era "baroque" --instrument "organ"

# 4. Search
python search.py --file query.wav --limit 10
python search.py --file query.wav --composer "Chopin" --tempo 60-100
```

## Project Structure

```
config/settings.py       - Configuration (paths, Qdrant, audio params)
src/audio_loader.py      - Scan, convert (MP3->WAV), load audio
src/feature_extractor.py - Mel, chroma, tempo, key, rhythm extraction
src/midi_transcriber.py  - WAV -> MIDI via basic-pitch
src/midi_analyzer.py     - MIDI analysis: intervals, density, patterns
src/embedder.py          - Build 512d embeddings for Qdrant
src/qdrant_store.py      - Qdrant collection management and search
ingest.py                - CLI: batch ingestion pipeline
search.py                - CLI: similarity search
```

## Embedding Strategy

The 512-dimensional embedding is built from mel-spectrogram statistics:
(mean, std, min, max) across 128 mel bands. This captures timbral and
spectral characteristics of each 8-second audio fragment.

Metadata (tempo, key, chroma profile, MIDI analysis) is stored as Qdrant
payload fields with indexes for filtered search.

## Roadmap

- [ ] Web UI (FastAPI + React)
- [ ] CLAP/MusicFM embeddings for better similarity
- [ ] Style-conditioned generation (Music Transformer)
- [ ] Multi-track MIDI generation
- [ ] DAW integration (ReaScript)

## License

MIT
