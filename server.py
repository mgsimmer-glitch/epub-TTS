#!/usr/bin/env python3
"""
Web interface for epub-tts.

Run with:
  .venv/bin/uvicorn server:app --reload
  # or
  .venv/bin/python server.py
"""

import asyncio
import json
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import AsyncGenerator

import edge_tts
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse

from epub_tts import convert_epub

app = FastAPI()

DOWNLOADS_DIR = Path.home() / "Downloads"
DOWNLOADS_DIR.mkdir(exist_ok=True)


def _unique_downloads_path(name: str) -> Path:
    """Return a non-colliding path inside ~/Downloads for the given filename."""
    dest = DOWNLOADS_DIR / name
    if not dest.exists():
        return dest
    stem, suffix = dest.stem, dest.suffix
    counter = 1
    while True:
        dest = DOWNLOADS_DIR / f"{stem} ({counter}){suffix}"
        if not dest.exists():
            return dest
        counter += 1

# ---------------------------------------------------------------------------
# In-memory job store
# ---------------------------------------------------------------------------
# jobs[job_id] = {
#   "queue":   asyncio.Queue        — SSE event stream
#   "files":   list[{"name", "path"}]
#   "voice":   str
#   "tmpdir":  str                  — temp dir for uploads + outputs
#   "outputs": dict[int, Path]      — file_index -> merged mp3 path
#   "status":  "running" | "done"
# }
_jobs: dict[str, dict] = {}


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    html = (Path(__file__).parent / "templates" / "index.html").read_text()
    return HTMLResponse(content=html)


@app.get("/voices")
async def list_voices() -> list[dict]:
    voices = await edge_tts.list_voices()
    return [
        {"name": v["ShortName"], "locale": v["Locale"], "gender": v["Gender"]}
        for v in sorted(voices, key=lambda v: (v["Locale"], v["ShortName"]))
    ]


@app.post("/jobs")
async def create_job(
    files: list[UploadFile] = File(...),
    voice: str = Form("en-US-AriaNeural"),
) -> dict:
    if not files:
        raise HTTPException(400, "No files provided")

    job_id = str(uuid.uuid4())
    tmpdir = Path(tempfile.mkdtemp(prefix=f"epub_tts_{job_id[:8]}_"))

    # Save uploaded files to disk
    upload_dir = tmpdir / "input"
    upload_dir.mkdir()
    saved: list[dict] = []
    for f in files:
        dest = upload_dir / (f.filename or f"file_{len(saved)}.epub")
        dest.write_bytes(await f.read())
        saved.append({"name": f.filename or dest.name, "path": str(dest)})

    _jobs[job_id] = {
        "queue": asyncio.Queue(),
        "files": saved,
        "voice": voice,
        "tmpdir": str(tmpdir),
        "outputs": {},
        "status": "running",
    }

    asyncio.create_task(_run_job(job_id))
    return {"job_id": job_id}


@app.get("/jobs/{job_id}/stream")
async def stream_job(job_id: str) -> StreamingResponse:
    if job_id not in _jobs:
        raise HTTPException(404, "Job not found")

    job = _jobs[job_id]

    async def event_stream() -> AsyncGenerator[str, None]:
        queue: asyncio.Queue = job["queue"]
        while True:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=25.0)
                yield f"data: {json.dumps(event)}\n\n"
                if event["type"] == "batch_done":
                    break
            except asyncio.TimeoutError:
                yield "data: {\"type\":\"heartbeat\"}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/jobs/{job_id}/download/{file_index}")
async def download_file(job_id: str, file_index: int) -> FileResponse:
    if job_id not in _jobs:
        raise HTTPException(404, "Job not found")

    job = _jobs[job_id]
    mp3_path: Path | None = job["outputs"].get(file_index)

    if mp3_path is None or not mp3_path.exists():
        raise HTTPException(404, "Output not ready")

    original_name = job["files"][file_index]["name"]
    stem = Path(original_name).stem
    return FileResponse(mp3_path, filename=f"{stem}.mp3", media_type="audio/mpeg")


# ---------------------------------------------------------------------------
# Background job worker
# ---------------------------------------------------------------------------

def _make_progress_cb(queue: asyncio.Queue, filename: str):
    """Return an async progress callback that forwards events to the SSE queue."""
    async def cb(event: str, **kwargs):
        await queue.put({"type": event, "file": filename, **kwargs})
    return cb


async def _run_job(job_id: str) -> None:
    job = _jobs[job_id]
    queue: asyncio.Queue = job["queue"]
    files: list[dict] = job["files"]
    voice: str = job["voice"]
    tmpdir = Path(job["tmpdir"])

    for idx, file_info in enumerate(files):
        filename: str = file_info["name"]
        epub_path: str = file_info["path"]

        await queue.put({
            "type": "file_start",
            "file": filename,
            "file_index": idx,
            "total_files": len(files),
        })

        try:
            output_dir = str(tmpdir / f"out_{idx:03d}")
            on_progress = _make_progress_cb(queue, filename)

            merged = await convert_epub(epub_path, output_dir, voice, on_progress=on_progress)

            # Move finished MP3 to ~/Downloads
            final_path = _unique_downloads_path(merged.name)
            shutil.move(str(merged), final_path)
            job["outputs"][idx] = final_path

            await queue.put({
                "type": "file_done",
                "file": filename,
                "file_index": idx,
                "total_files": len(files),
                "download_url": f"/jobs/{job_id}/download/{idx}",
                "saved_to": str(final_path),
            })

        except Exception as exc:
            await queue.put({
                "type": "file_error",
                "file": filename,
                "file_index": idx,
                "error": str(exc),
            })

    job["status"] = "done"
    await queue.put({"type": "batch_done", "total_files": len(files)})

    # Schedule temp dir cleanup after a generous delay so downloads can finish
    asyncio.create_task(_deferred_cleanup(job_id, delay=3600))


async def _deferred_cleanup(job_id: str, delay: int) -> None:
    await asyncio.sleep(delay)
    job = _jobs.pop(job_id, None)
    if job:
        shutil.rmtree(job["tmpdir"], ignore_errors=True)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=False)
