# Music Composer RAG

Generate music in a composer's style. A musical **form template** (nocturne,
sonata, fugue, prelude) plus a **style profile** are handed to Claude, which
composes an explicit multi-track score (melody, accompaniment, bass) as JSON;
that score is rendered to a standard MIDI file.

## Quick Start (generation)

The generator needs only the Claude API — no database, no audio tooling.

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...

python generate.py --list-composers
python generate.py --list-forms

python generate.py --composer "Chopin" --form nocturne --key "E minor" --tempo 72 --duration 90 --mood melancholic
python generate.py --composer "Bach"   --form fugue    --key "D minor" --tempo 100 --duration 120
```

Output MIDI lands in `output/generated/`. Add `--save-blueprint` to also dump
the JSON score.

### Style profiles

Generation is driven by hand-authored profiles in `profiles/<composer>.json`
(qualitative style notes, characteristic keys, tempo range, harmonic language).
Add a new composer by dropping a JSON file there — see `profiles/chopin.json`
for the schema.

### Form templates

`forms/<form>.json` describes the structure of a musical form (sections, dynamic
arcs, harmonic rhythm, composition rules). They are the strongest part of the
prompt and are easy to extend.

## Project Structure

```
profiles/                  - Hand-authored composer style profiles (generation input)
forms/                     - Musical form templates (nocturne, sonata, fugue, prelude)
config/settings.py         - Paths and (legacy) Qdrant connection
src/style_profiler.py      - Load style profiles -> prompt text
src/composer_architect.py  - Build the prompt, call Claude, return a score blueprint
src/midi_builder.py        - Convert the JSON blueprint to a multi-track MIDI file
generate.py                - CLI: generate a composition
```

## Legacy audio pipeline (optional)

An earlier branch of this project ingested audio into a Qdrant vector store for
similarity search and statistical style profiling. It is **not required for
generation** and is kept separate:

```bash
pip install -r requirements.txt -r requirements-audio.txt   # + system ffmpeg
docker run -d -p 6333:6333 qdrant/qdrant
export QDRANT_HOST=localhost

python ingest.py --input ./input/chopin --composer "Chopin" --era romantic
python search.py --file query.wav --limit 10
python app.py    # Gradio UI: dashboard / ingest / search / generate
```

These modules (`ingest.py`, `search.py`, `app.py`, and the `src/audio_*`,
`src/embedder`, `src/qdrant_store`, `src/midi_transcriber`, `src/midi_analyzer`,
`src/pattern_extractor` helpers) use librosa, basic-pitch and CLAP embeddings.

## Roadmap

- [ ] Section-by-section generation to lift the ~32-bar / token ceiling
- [ ] Symbolic evaluation metrics (in-key ratio, voice-leading smoothness, reprise similarity)
- [ ] Optional symbolic retrieval: feed real progressions/figures from a clean MIDI corpus into the prompt

## License

MIT
