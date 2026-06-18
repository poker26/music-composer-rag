# Session Handoff — project pivot to LLM composer (path C)

This branch (`claude/project-analysis-direction-kaogvl`) refactored the project
away from the audio/RAG concept toward a self-contained LLM composer. Picking up
on a local machine: read this file, then continue from "Open tasks".

## What changed (3 commits)

1. `b2da28d` — make the repo reproducible
   - `requirements.txt` reduced to the real core deps: `anthropic`, `mido`, `numpy`
   - removed the hardcoded public Qdrant IP from `config/settings.py`
   - cleaned `.gitignore`, removed a broken partial download under `input/`

2. `fc3dce6` — decouple generation from Qdrant
   - generation now runs on hand-authored profiles in `profiles/<composer>.json`
     (Chopin, Bach, Beethoven, Debussy), not DB statistics
   - `src/style_profiler.py`: `load_static_profile` / `list_static_profiles` /
     `static_profile_to_prompt_text`

3. `8222e1c` — section-by-section generation + remove all legacy
   - `src/composer_architect.py` now plans the form into sections
     (`plan_sections`), composes ONE section per Claude call, and stitches them
     with absolute bar renumbering (`assemble_blueprint`). This lifts the old
     ~32-bar single-call ceiling (a 240s nocturne now plans 52 bars).
   - reprise sections (marked `reprise_of` in `forms/*.json`) restate/ornament
     the original theme; contrasting sections stay tied to the opening theme
   - DELETED the whole audio/RAG stack: `ingest.py`, `search.py`, `app.py`,
     `requirements-audio.txt`, `input/`, and `src/{audio_loader,feature_extractor,
     midi_transcriber,midi_analyzer,embedder,qdrant_store,pattern_extractor}.py`

## Current state

- Repo is generation-only: `generate.py`, `config/settings.py`,
  `src/{style_profiler,composer_architect,midi_builder}.py`, `profiles/`, `forms/`.
- `git grep` confirms no remaining references to qdrant/clap/librosa/gradio/supabase.
- Verified WITHOUT an API key: `--list-composers`, `--list-forms`, section
  planning, bar renumbering, and MIDI assembly all work.
- **Migrated to a local machine 2026-06-18 and verified end-to-end** (Chopin
  nocturne + Bach prelude -> blueprint + MIDI). Fixed a dead model id
  (`claude-sonnet-4-20250514` -> `claude-sonnet-4-6`).

## How to run locally

```bash
git fetch origin
git checkout claude/project-analysis-direction-kaogvl
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...
python generate.py --composer "Chopin" --form nocturne --key "E minor" --tempo 72 --duration 90 --save-blueprint
```

## Open tasks

1. ~~**Run a real generation**~~ — DONE 2026-06-18. Chopin nocturne + Bach prelude
   generate cleanly. Still worth a *listening* pass to judge coherence/reprises by ear.
2. ~~**Qdrant cleanup**~~ — DONE 2026-06-18. `music_fragments` deleted from
   `45.12.72.157:6333` (was 8493 points). Other collections on that host belong to
   unrelated projects and were left untouched.
3. ~~**Step 3 — evaluation metrics**~~ — DONE 2026-06-18. `src/evaluator.py` prints a
   symbolic report after each generation (skip with `--no-eval`) and runs standalone:
   `python -m src.evaluator output/blueprints/<name>.json`. Covers in-key ratio,
   melody smoothness, reprise skeleton-vs-stream similarity, and parallel P5/P8.
4. **Step 4 (optional) — real retrieval**: feed real progressions/figures from a
   CLEAN symbolic MIDI corpus (not audio transcription) into the section prompts.
   NOT started.

### Tuning leads surfaced by the new metrics
- Bach prelude flagged **41 parallel octaves** (melody vs bass) — the figuration
  prompt likely lets the bass shadow the melody at the octave. Worth a prompt tweak.
- Chopin nocturne reprise: 88% skeleton / 16% full-stream (29->59 notes) — the theme
  is recognizable under heavy ornamentation, which is the intended behavior.

## Why the cloud session couldn't finish the Qdrant cleanup

This session ran in an isolated cloud container with a restrictive network
policy: outbound to `45.12.72.157` on both 6333 and 22 is blocked (verified),
while github.com is allowed. It has no access to the local laptop's disk or to
other sessions. The Qdrant deletion therefore must run from a machine (e.g. this
laptop) that can reach the host.
