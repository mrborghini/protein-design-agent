"""Run a bio-app inside its own venv as a subprocess, streaming progress.

Every tool wrapper builds an argv and calls ``run_in_env``. The runner:
  - creates the run's working dir (lazily; nothing touches the FS at import),
  - launches the tool via its venv interpreter/CLI (``<venv>/bin/<prog> …``),
  - streams stdout/stderr lines onto the SSE queue as ``bioapp`` log events,
  - enforces a hard per-tool timeout (kills the process group on overrun),
  - acquires ``gpu_exclusive`` for GPU tools so the LLM is evicted from VRAM first,
  - returns captured output, and raises ``BioAppError`` on non-zero exit / timeout.

POC: no sandboxing of the child process beyond the venv; failures are surfaced
verbatim (last lines of output) rather than swallowed.
"""
import asyncio
import os
import uuid
from pathlib import Path

from backend.gpu import gpu_exclusive
from backend.research import research_sink

from .config import TASKITY_MAX_OUTPUT_BYTES, WORKDIR, BioApp

# How many trailing output lines to include in an error so failures are debuggable.
_ERROR_TAIL_LINES = 25


class BioAppError(RuntimeError):
    """A bio-app run failed (non-zero exit, timeout, or launch error)."""


async def _emit(event: dict) -> None:
    queue = research_sink.get(None)
    if queue is not None:
        await queue.put(event)


async def emit_artifact(tool: str, kind: str, path: Path, **fields) -> None:
    """Announce a produced artifact (PDB/FASTA/score) to the UI.

    `kind` is a short type tag (e.g. "structure", "backbone", "sequence", "score");
    extra `fields` carry metrics (plddt, dG, …). `path` is made relative to WORKDIR
    so the download route can resolve it safely.
    """
    try:
        rel = str(path.relative_to(WORKDIR))
    except ValueError:
        rel = str(path)
    await _emit({"type": "artifact", "tool": tool, "kind": kind, "path": rel, **fields})


def new_run_dir(prefix: str) -> Path:
    """Create and return a fresh per-run directory under WORKDIR."""
    run_dir = WORKDIR / f"{prefix}-{uuid.uuid4().hex[:8]}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


async def _stream_lines(stream: asyncio.StreamReader, tool: str, sink: list[str], max_bytes: int) -> None:
    """Forward a child stream line-by-line to the SSE queue and the capture buffer.

    Every line is streamed live; the captured `sink` keeps only the trailing
    `max_bytes` (oldest lines dropped). The tail is kept on purpose — the taskity
    result convention puts the JSON on the LAST line.
    """
    used = 0
    while True:
        raw = await stream.readline()
        if not raw:
            return
        line = raw.decode("utf-8", errors="replace").rstrip()
        sink.append(line)
        used += len(line) + 1
        while used > max_bytes and len(sink) > 1:
            used -= len(sink.pop(0)) + 1
        if line:
            await _emit({"type": "bioapp", "stage": "log", "tool": tool, "text": line})


def _resolve_exe(app: BioApp, prog: str) -> Path:
    """Map argv[0] (e.g. 'boltz' or 'python') to the tool venv's bin/ executable."""
    return app.bindir / prog


def _venv_env(app: BioApp, extra_env: dict[str, str] | None = None) -> dict[str, str]:
    """Child environment with the tool's venv activated (no shell `activate` needed)."""
    env = {**os.environ}
    env["VIRTUAL_ENV"] = str(app.venv)
    env["PATH"] = f"{app.bindir}{os.pathsep}{env.get('PATH', '')}"
    env.pop("PYTHONHOME", None)  # would override the venv interpreter's paths
    if extra_env:
        env.update(extra_env)
    return env


async def _run(app: BioApp, argv: list[str], cwd: Path | None, label: str, max_output_bytes: int,
               extra_env: dict[str, str] | None = None) -> str:
    exe = _resolve_exe(app, argv[0])
    cmd = [str(exe), *argv[1:]]
    await _emit({"type": "bioapp", "stage": "start", "tool": app.key, "label": label,
                 "text": f"{label} — running in venv '{app.venv}'…"})

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=str(cwd) if cwd else None,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,  # merge so logs stay in order
            env=_venv_env(app, extra_env),
        )
    except FileNotFoundError as e:
        raise BioAppError(
            f"Could not launch '{exe}' for {app.key} — is the venv at '{app.venv}' set up? ({e})"
        ) from e

    lines: list[str] = []
    reader = asyncio.create_task(_stream_lines(proc.stdout, app.key, lines, max_output_bytes))
    try:
        await asyncio.wait_for(proc.wait(), timeout=app.timeout)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        reader.cancel()
        raise BioAppError(f"{label} timed out after {app.timeout}s and was killed.")
    finally:
        await reader  # drain remaining output

    if proc.returncode != 0:
        tail = "\n".join(lines[-_ERROR_TAIL_LINES:]) or "(no output)"
        raise BioAppError(
            f"{label} failed in venv '{app.venv}' (exit {proc.returncode}). Last output:\n{tail}"
        )

    await _emit({"type": "bioapp", "stage": "done", "tool": app.key, "label": label,
                 "text": f"{label} — done."})
    return "\n".join(lines)


async def run_in_env(
    app: BioApp, argv: list[str], cwd: Path | None = None, label: str = "",
    max_output_bytes: int = TASKITY_MAX_OUTPUT_BYTES, extra_env: dict[str, str] | None = None,
) -> str:
    """Run `argv` inside `app`'s venv and return its captured stdout (tail-capped).

    GPU tools (`app.gpu`) run under ``gpu_exclusive`` so resident LLMs are evicted
    from VRAM first and reload only after the tool exits. `extra_env` is merged into
    the child environment (e.g. taskities pass `BIOAPP_OUT`). Raises ``BioAppError``
    on failure.
    """
    label = label or app.key
    if app.gpu:
        async with gpu_exclusive():
            return await _run(app, argv, cwd, label, max_output_bytes, extra_env)
    return await _run(app, argv, cwd, label, max_output_bytes, extra_env)


async def spawn_wait(
    argv: list[str], timeout: int, cwd: Path | None = None, env: dict[str, str] | None = None,
) -> tuple[int, str]:
    """Run a subprocess to completion, capturing combined output. No SSE streaming.

    Used for one-off setup steps (venv create, pip install) where we want a simple
    (exit_code, output) result. Kills the process on timeout (returns code 124).
    """
    try:
        proc = await asyncio.create_subprocess_exec(
            *argv,
            cwd=str(cwd) if cwd else None,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env=env or {**os.environ},
        )
    except FileNotFoundError as e:
        return 127, f"could not launch {argv[0]!r}: {e}"
    try:
        out, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        return 124, f"timed out after {timeout}s"
    return proc.returncode or 0, (out or b"").decode("utf-8", errors="replace")
