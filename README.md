# Music Composer RAG

Generate music in a composer's style. A musical **form template** (nocturne,
sonata, fugue, prelude) plus a **style profile** are handed to Claude, which
composes the piece **one section at a time** as explicit multi-track JSON
(melody, accompaniment, bass); that score is rendered to a standard MIDI file.

Composing section by section lets pieces grow well past the old single-call
~32-bar ceiling, and lets reprise sections restate and ornament the original
theme instead of inventing a new one.

## Quick Start

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
the JSON score. After each run a **symbolic evaluation** is printed (in-key
ratio, melody smoothness, reprise similarity, parallel fifths/octaves); skip it
with `--no-eval`, or run it standalone on a saved blueprint:

```bash
python -m src.evaluator output/blueprints/<name>.json
```

## How it works

```
profiles/<composer>.json   style context  ─┐
forms/<form>.json          structure      ─┤
CLI params (key, tempo...) ─────────────────┤
                                            ▼
        plan_sections()  →  one Claude call per section  →  assemble_blueprint()
                                            │
                                            ▼
                            multi-track MIDI (output/generated/)
```

- **Style profiles** (`profiles/<composer>.json`): qualitative style notes,
  characteristic keys, tempo range, harmonic language. Add a composer by dropping
  a JSON file there — see `profiles/chopin.json` for the schema.
- **Form templates** (`forms/<form>.json`): sections, dynamic arcs, harmonic
  rhythm, composition rules. A section may declare `reprise_of` to mark it as a
  return of an earlier section.
- **Reference corpus** (optional, `corpus/index/<composer>.json`): when present,
  each section prompt is enriched with *real* material — harmonic progressions,
  cadences, and melodic motifs mined from that composer's actual scores. See
  "Reference corpus" below. If no index exists for a composer, generation is
  unaffected.

## Reference corpus (Step 4 — optional)

`ingest_corpus.py` parses a clean symbolic score corpus (Humdrum `**kern` or MIDI)
with `music21` and extracts key-independent material — Roman-numeral progressions,
cadences, and melodic interval motifs — aggregated by frequency into
`corpus/index/<composer>.json`. At generation time `src/retrieval.py` injects this
into the section prompt (the model transposes it into the target key). `music21` is
needed only for ingestion; generation reads the JSON index with the stdlib alone.

```bash
pip install -r requirements-corpus.txt

# Clone clean public-domain corpora (engraving-derived **kern, not audio transcription):
git clone https://github.com/craigsapp/bach-370-chorales       corpus/raw/bach-370-chorales
git clone https://github.com/craigsapp/beethoven-piano-sonatas corpus/raw/beethoven-piano-sonatas
git clone https://github.com/craigsapp/chopin-mazurkas         corpus/raw/chopin-mazurkas
git clone https://github.com/craigsapp/chopin-preludes         corpus/raw/chopin-preludes

python ingest_corpus.py --limit 40        # build corpus/index/*.json
python -m src.retrieval Chopin            # preview the retrieved block
```

`corpus/raw/` is gitignored (each corpus has its own repo); the small
`corpus/index/*.json` files are committed.

## Project Structure

```
profiles/                  - Composer style profiles (generation input)
forms/                     - Musical form templates (nocturne, sonata, fugue, prelude)
config/settings.py         - Paths
src/style_profiler.py      - Load style profiles -> prompt text
src/composer_architect.py  - Plan sections, call Claude per section, assemble blueprint
src/midi_builder.py        - Convert the JSON blueprint to a multi-track MIDI file
src/evaluator.py           - Symbolic evaluation metrics on a blueprint
src/retrieval.py           - Inject real corpus material into section prompts (runtime)
ingest_corpus.py           - Build corpus/index/*.json from a symbolic score corpus
corpus/index/              - Per-composer retrieval indices (committed)
generate.py                - CLI: generate a composition
```

## Roadmap

- [x] Symbolic evaluation metrics (in-key ratio, voice-leading smoothness, reprise similarity)
- [x] Optional symbolic retrieval: feed real progressions/figures from a clean score corpus into the prompt
- [ ] MIDI -> audio rendering for quick listening
- [ ] Bias retrieval by section character (opening vs cadential vs development)
- [ ] Debussy corpus (no clean **kern source wired yet)

## License

MIT
