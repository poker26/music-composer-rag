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
the JSON score.

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

## Project Structure

```
profiles/                  - Composer style profiles (generation input)
forms/                     - Musical form templates (nocturne, sonata, fugue, prelude)
config/settings.py         - Paths
src/style_profiler.py      - Load style profiles -> prompt text
src/composer_architect.py  - Plan sections, call Claude per section, assemble blueprint
src/midi_builder.py        - Convert the JSON blueprint to a multi-track MIDI file
generate.py                - CLI: generate a composition
```

## Roadmap

- [ ] Symbolic evaluation metrics (in-key ratio, voice-leading smoothness, reprise similarity)
- [ ] Optional symbolic retrieval: feed real progressions/figures from a clean MIDI corpus into the prompt
- [ ] MIDI -> audio rendering for quick listening

## License

MIT
