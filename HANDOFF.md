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
- NOT yet verified: a real end-to-end generation (needs `ANTHROPIC_API_KEY`).

## How to run locally

```bash
git fetch origin
git checkout claude/project-analysis-direction-kaogvl
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...
python generate.py --composer "Chopin" --form nocturne --key "E minor" --tempo 72 --duration 90 --save-blueprint
```

## Open tasks

1. **Run a real generation** and sanity-check the MIDI / blueprint. Listen and
   judge whether section-by-section improved coherence and reprises.
2. **Qdrant cleanup (blocked in the cloud sandbox — do this locally).** The old
   collection lives at `45.12.72.157:6333`, collection `music_fragments`. From a
   machine with network access to that host:
   ```python
   from qdrant_client import QdrantClient
   c = QdrantClient(host="45.12.72.157", port=6333, api_key="<KEY>")
   print(c.get_collections())            # confirm it is the old collection
   c.delete_collection("music_fragments")
   ```
   (No Supabase artifacts exist — this project never used Supabase.)
3. **Step 3 — evaluation metrics** (planned, not started): symbolic checks on the
   output MIDI (in-key ratio, voice-leading smoothness, reprise similarity,
   parallel-fifth detection) printed alongside results, to give an objective
   signal when tuning prompts.
4. **Step 4 (optional) — real retrieval**: feed real progressions/figures from a
   CLEAN symbolic MIDI corpus (not audio transcription) into the section prompts.

## Why the cloud session couldn't finish the Qdrant cleanup

This session ran in an isolated cloud container with a restrictive network
policy: outbound to `45.12.72.157` on both 6333 and 22 is blocked (verified),
while github.com is allowed. It has no access to the local laptop's disk or to
other sessions. The Qdrant deletion therefore must run from a machine (e.g. this
laptop) that can reach the host.
