"""
Web UI for the LLM composer.

A thin FastAPI layer over the existing pipeline: pick a composer / form / key /
tempo, generate section-by-section via Claude, then play the resulting MIDI right
in the browser (html-midi-player, no server-side audio tooling) with the symbolic
evaluation report alongside.

Run locally:
    pip install -r requirements.txt -r requirements-web.txt
    export ANTHROPIC_API_KEY=sk-ant-...        # or a local .env file
    python webapp.py                            # http://127.0.0.1:7860

In production it listens on 127.0.0.1:7860 behind the nginx vhost for
composer.begemot26.ru.
"""

from __future__ import annotations

import base64
import logging
import os
import tempfile
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.style_profiler import (
    load_static_profile, static_profile_to_prompt_text, list_static_profiles,
)
from src.composer_architect import generate_blueprint, list_available_forms
from src.midi_builder import build_midi
from src.evaluator import evaluate_blueprint, format_report

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("webapp")

BASE_DIR = Path(__file__).parent
WEB_DIR = BASE_DIR / "web"


def _load_dotenv():
    """Make ANTHROPIC_API_KEY available locally without exporting it by hand."""
    env = BASE_DIR / ".env"
    if env.is_file():
        for line in env.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


_load_dotenv()

app = FastAPI(title="Music Composer")


class GenerateRequest(BaseModel):
    composer: str
    form: str | None = None
    key: str = "C minor"
    tempo: int = 100
    duration: int = 60
    mood: str = "expressive"
    description: str = ""


@app.get("/api/options")
def options():
    composers = [
        {"name": name, "era": data.get("era", "")}
        for name, data in sorted(list_static_profiles().items())
    ]
    forms = [
        {"id": fid, "name": info["name"]}
        for fid, info in sorted(list_available_forms().items())
    ]
    return {"composers": composers, "forms": forms}


@app.post("/api/generate")
def generate(req: GenerateRequest):
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise HTTPException(status_code=503, detail="ANTHROPIC_API_KEY is not set on the server.")

    profile = load_static_profile(req.composer)
    if not profile:
        available = ", ".join(list_static_profiles().keys()) or "none"
        raise HTTPException(status_code=400, detail=f"Unknown composer '{req.composer}'. Available: {available}")

    profile_text = static_profile_to_prompt_text(profile)
    form_name = req.form or "free form"
    params = {
        "composer": req.composer,
        "key": req.key,
        "tempo_bpm": req.tempo,
        "duration_sec": req.duration,
        "mood": req.mood,
        "description": req.description or f"A {form_name} in the style of {req.composer}",
        "instruments": "piano",
        "time_signature": [4, 4],
    }

    logger.info("Generating %s %s in %s...", req.composer, form_name, req.key)
    try:
        blueprint = generate_blueprint(profile_text, params, form_id=req.form or None)
    except Exception as e:
        logger.exception("Generation failed")
        raise HTTPException(status_code=500, detail=f"Generation failed: {e}")

    with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as tmp:
        midi_path = Path(tmp.name)
    build_midi(blueprint, midi_path)
    midi_b64 = base64.b64encode(midi_path.read_bytes()).decode("ascii")
    midi_path.unlink(missing_ok=True)

    metrics = evaluate_blueprint(blueprint)
    return JSONResponse({
        "title": blueprint.get("title", "Untitled"),
        "key": blueprint.get("key"),
        "tempo": blueprint.get("tempo_bpm"),
        "form": blueprint.get("form", form_name),
        "sections": len(blueprint.get("sections", [])),
        "bars": blueprint.get("total_bars"),
        "midi_base64": midi_b64,
        "metrics": metrics,
        "report": format_report(metrics),
    })


# Static front-end (mounted last so /api/* wins).
app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "7860"))
    uvicorn.run(app, host="127.0.0.1", port=port)
