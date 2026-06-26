"""Taskities — reusable, LLM-authored Python scripts that run in venvs.

Ported from the discord-agent "taskities" system (`src/taskity.ts`) and adapted to
this repo. A taskity is a **named, reusable** Python script the Bio-App Operator
writes once and runs repeatedly with different params. The contract (same as
discord-agent):

  - the script reads its params as a JSON string in ``sys.argv[1]``;
  - it does its work and prints **exactly one final JSON line** to stdout;
  - large output files go in the directory named by the ``BIOAPP_OUT`` env var.

Each taskity runs in one of two environments:
  - ``base_venv`` = a bio-app key (boltz/rfdiffusion/proteinmpnn/pyrosetta) → it runs
    with that venv's interpreter, so the script can ``import pyrosetta``/``torch``/
    ``boltz``. GPU bio-app venvs evict the LLM from VRAM first (via ``run_in_env`` →
    ``gpu_exclusive``).
  - otherwise → a **dedicated per-taskity venv** is created once and the declared
    ``packages`` are pip-installed (cached). An empty package list ⇒ a clean,
    isolated stdlib interpreter.

(Refinement vs the discord-agent original: we always run inside *a* venv — either a
bio-app one or a dedicated one — rather than a bare system Python, so there's no
``needs_venv`` flag and scripts never touch the backend's own venv.)

Runs are timeout-guarded and output-capped (see runner/config). Scripts are
lint-checked for stub/placeholder content before they're stored.

⚠️ SECURITY: this executes arbitrary, model-written Python locally. There is no
sandbox beyond the venv, timeout, and output cap. Local-only POC on an authorized
machine — never expose this to untrusted input. The lint guard catches *stubs*, not
*malice*.
"""
import asyncio
import hashlib
import json
import re
import shutil
import time
from pathlib import Path
from typing import Awaitable, Callable

from autogen_core.tools import FunctionTool

from .config import (
    ALL,
    TASKITIES_DIR,
    TASKITY_MAX_OUTPUT_BYTES,
    TASKITY_PYTHON,
    TASKITY_TIMEOUT,
    TASKITY_VENV_INSTALL_TIMEOUT,
    BioApp,
    resolve_base_venv,
)
from .runner import BioAppError, _emit, emit_artifact, new_run_dir, run_in_env, spawn_wait

_NAME_RE = re.compile(r"^[a-z][a-z0-9_-]{0,63}$")
_MAX_ARTIFACTS = 20  # cap emitted output files per run so a chatty script can't flood the UI

# --- stub/placeholder lint (ported from discord-agent's STUB_TOKENS) --- #
_STUB_TOKENS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\bYOUR[_-]?API[_-]?KEY\b", re.I), "contains a YOUR_API_KEY placeholder"),
    (re.compile(r"\bREPLACE[_-]?ME\b", re.I), "contains a REPLACE_ME placeholder"),
    (re.compile(r"<your[ _-]?key>", re.I), "contains a <your key> placeholder"),
    (re.compile(r"\bplaceholder\b", re.I), 'contains the word "placeholder"'),
    (re.compile(r"\bTODO\b"), "contains a TODO marker"),
    (re.compile(r"\bFIXME\b"), "contains a FIXME marker"),
    (re.compile(r"in a real implementation", re.I), 'contains "in a real implementation"'),
    (re.compile(r"\bsample data\b", re.I), 'contains "sample data"'),
    (re.compile(r"\bmock(ed)? data\b", re.I), 'contains "mock data"'),
    (re.compile(r"\bhardcod(e|ed|ing)\b", re.I), 'contains "hardcoded"'),
]
_URL_RE = re.compile(r"https?://", re.I)
_FETCH_HINTS = ("requests.", "urllib.request", "urlopen", "httpx.", "aiohttp", "urllib3", "fetch(")


def lint_taskity_script(script: str) -> list[str]:
    """Return a list of stub/placeholder issues; empty ⇒ the script looks real."""
    issues = [reason for pat, reason in _STUB_TOKENS if pat.search(script)]
    if _URL_RE.search(script) and not any(h in script for h in _FETCH_HINTS):
        issues.append("builds an http(s):// URL but never fetches it")
    return issues


# --- on-disk layout + in-process registry --- #
def _dir(name: str) -> Path:
    return TASKITIES_DIR / name


def _script_path(name: str) -> Path:
    return _dir(name) / "script.py"


def _meta_path(name: str) -> Path:
    return _dir(name) / "meta.json"


def _venv_dir(name: str) -> Path:
    return _dir(name) / ".venv"


def _marker_path(name: str) -> Path:
    return _dir(name) / ".venv-packages.json"


def _runs_log(name: str) -> Path:
    return _dir(name) / "runs.log"


def _load_registry() -> dict[str, dict]:
    reg: dict[str, dict] = {}
    if TASKITIES_DIR.is_dir():
        for d in sorted(TASKITIES_DIR.iterdir()):
            mp = d / "meta.json"
            if mp.is_file():
                try:
                    reg[d.name] = json.loads(mp.read_text())
                except Exception:  # noqa: BLE001 - skip a corrupt definition, don't abort import
                    pass
    return reg


_REGISTRY: dict[str, dict] = _load_registry()
_venv_locks: dict[str, asyncio.Lock] = {}

# Code-review gate. A reviewer is `async (name, description, script) -> (approved, feedback)`.
# The pipeline injects one (the Critic) per run via set_reviewer; when None (standalone use /
# unit tests) the gate is inactive. Enforced fail-closed in _run: no approval ⇒ no execution.
Reviewer = Callable[[str, str, str], Awaitable[tuple[bool, str]]]
_reviewer: Reviewer | None = None


def set_reviewer(fn: Reviewer | None) -> None:
    """Install (or clear) the script reviewer used by the run gate."""
    global _reviewer
    _reviewer = fn


def _script_sha(script: str) -> str:
    return hashlib.sha256(script.encode("utf-8")).hexdigest()


def _lock_for(name: str) -> asyncio.Lock:
    lock = _venv_locks.get(name)
    if lock is None:
        lock = _venv_locks[name] = asyncio.Lock()
    return lock


def _save(meta: dict) -> None:
    name = meta["name"]
    _dir(name).mkdir(parents=True, exist_ok=True)
    _script_path(name).write_text(meta["script"])
    _meta_path(name).write_text(json.dumps(meta, indent=2))
    _REGISTRY[name] = meta


def _remove(name: str) -> None:
    shutil.rmtree(_dir(name), ignore_errors=True)
    _REGISTRY.pop(name, None)
    _venv_locks.pop(name, None)


def _validate(name: str, base_venv: str) -> str | None:
    """Return an error message, or None if name + base_venv are acceptable."""
    if not _NAME_RE.match(name):
        return f"Invalid name '{name}'. Use ^[a-z][a-z0-9_-]{{0,63}}$ (snake-case)."
    if base_venv and base_venv not in ALL:
        return f"Unknown base_venv '{base_venv}'. Choose one of: {', '.join(ALL)} (or leave empty)."
    return None


# --- dedicated-venv setup (ported from ensureVenvReady) --- #
async def _ensure_venv(name: str, packages: list[str]) -> None:
    """Create the taskity's dedicated venv (once) and pip-install `packages` (cached)."""
    want = sorted(packages)
    async with _lock_for(name):
        venv, marker = _venv_dir(name), _marker_path(name)
        if venv.exists() and marker.exists():
            try:
                if json.loads(marker.read_text()) == want:
                    return  # venv already has exactly these packages
            except Exception:  # noqa: BLE001 - rebuild on a corrupt marker
                pass
        if not venv.exists():
            await _status(name, f"creating venv for taskity '{name}'…")
            code, out = await spawn_wait([TASKITY_PYTHON, "-m", "venv", str(venv)], timeout=120)
            if code != 0:
                raise BioAppError(f"venv create failed for '{name}': {out[-1000:]}")
        if want:
            await _status(name, f"pip installing {', '.join(want)}…")
            pip = venv / "bin" / "pip"
            code, out = await spawn_wait(
                [str(pip), "install", "--no-input", "--disable-pip-version-check", *want],
                timeout=TASKITY_VENV_INSTALL_TIMEOUT,
            )
            if code != 0:
                raise BioAppError(f"pip install failed for '{name}': {out[-1500:]}")
        marker.write_text(json.dumps(want))


async def _status(name: str, text: str) -> None:
    # Reuse the bioapp event channel so setup progress shows in the UI like any tool.
    await _emit({"type": "bioapp", "stage": "log", "tool": "taskity", "text": text})


def _last_json_line(text: str) -> tuple[object | None, str | None]:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return None, "no stdout"
    try:
        return json.loads(lines[-1]), None
    except Exception as e:  # noqa: BLE001
        return None, f"last stdout line is not JSON: {e}"


# --- run --- #
async def _run(name: str, params: dict) -> dict:
    meta = _REGISTRY.get(name)
    if meta is None:
        return {"ok": False, "error": f"No taskity named '{name}'. Use taskity_list to see what exists."}

    # --- Review gate: the exact script version must be approved before it runs. --- #
    sha = _script_sha(meta.get("script", ""))
    if _reviewer is not None and meta.get("approved_sha") != sha:
        await _status(name, f"Critic reviewing taskity '{name}' before it may run…")
        try:
            approved, feedback = await _reviewer(name, meta.get("description", ""), meta.get("script", ""))
        except Exception as e:  # noqa: BLE001 - fail closed: a broken review must not let code run
            return {"ok": False, "blocked": True,
                    "error": f"Script not run — review failed: {e}"}
        if not approved:
            return {"ok": False, "blocked": True,
                    "error": "Script not run — the Critic did not approve it. "
                             "Revise it (taskity_update) addressing the feedback, then run again.",
                    "review": feedback}
        meta["approved_sha"] = sha
        _save(meta)  # persist approval so unchanged re-runs skip re-review

    base = resolve_base_venv(meta.get("base_venv") or "")
    if base is not None:
        venv = base.venv
        eff_gpu = bool(meta.get("gpu")) or base.gpu
    else:
        try:
            await _ensure_venv(name, meta.get("packages") or [])
        except BioAppError as e:
            return {"ok": False, "error": str(e)}
        venv = _venv_dir(name)
        eff_gpu = bool(meta.get("gpu"))

    app = BioApp(key=f"taskity:{name}", venv=venv, gpu=eff_gpu, timeout=TASKITY_TIMEOUT)
    out_dir = new_run_dir(f"taskity-{name}")
    argv = ["python", str(_script_path(name)), json.dumps(params or {})]
    extra_env = {"BIOAPP_OUT": str(out_dir), "PYTHONUNBUFFERED": "1"}

    started = time.monotonic()
    timed_out = False
    try:
        stdout = await run_in_env(
            app, argv, cwd=out_dir, label=f"taskity '{name}'", extra_env=extra_env,
        )
        ok, err_msg = True, None
    except BioAppError as e:
        stdout, ok, err_msg = "", False, str(e)
        timed_out = "timed out" in err_msg
    duration_ms = round((time.monotonic() - started) * 1000)

    parsed, parse_error = _last_json_line(stdout) if ok else (None, None)
    outcome = "ok" if (ok and parse_error is None) else ("ok_non_json" if ok else ("timed_out" if timed_out else "failed"))
    _append_runs_log(name, outcome, duration_ms)

    # Surface the script + any produced files as downloadable artifacts.
    await emit_artifact("taskity", "script", _script_path(name), taskity=name)
    artifacts: list[str] = []
    for f in sorted(p for p in out_dir.rglob("*") if p.is_file())[:_MAX_ARTIFACTS]:
        await emit_artifact("taskity", "file", f, taskity=name)
        artifacts.append(f.name)

    if ok and meta.get("one_time"):
        _remove(name)

    result = {
        "ok": ok,
        "parsed_output": parsed,
        "parse_error": parse_error,
        "stdout_tail": "\n".join(stdout.splitlines()[-25:]),
        "duration_ms": duration_ms,
        "artifacts": artifacts,
    }
    if not ok:
        result["error"] = err_msg
        result["timed_out"] = timed_out
    return result


def _append_runs_log(name: str, outcome: str, duration_ms: int) -> None:
    try:
        with _runs_log(name).open("a") as fh:
            # time.time() avoids importing datetime; epoch seconds is fine for a local log.
            fh.write(f"{time.time():.0f} outcome={outcome} duration_ms={duration_ms}\n")
    except Exception:  # noqa: BLE001 - logging is best-effort
        pass


# --------------------------------------------------------------------------- #
# Tools exposed to the Operator
# --------------------------------------------------------------------------- #
_SCRIPT_CONTRACT = (
    "The script MUST read params via `json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}` and "
    "print exactly one final JSON line to stdout via `print(json.dumps(...))`. Write large output "
    "files into the directory given by the BIOAPP_OUT environment variable. No placeholders, TODOs, "
    "mock/sample data, or hardcoded fake values — write real, working code."
)


async def taskity_create(
    name: str,
    description: str,
    script: str,
    base_venv: str = "",
    packages: list[str] | None = None,
    gpu: bool = False,
    one_time: bool = False,
    params_schema: dict | None = None,
) -> str:
    """Create a new reusable Python taskity (script). Fails if the name already exists.

    Args:
      name: snake-case identifier (^[a-z][a-z0-9_-]{0,63}$).
      description: what the taskity does.
      script: full Python source. It must read params via json.loads(sys.argv[1]) and print exactly
        one final JSON line via print(json.dumps(...)); write large files into the BIOAPP_OUT dir;
        no placeholders/TODOs/mock data.
      base_venv: to use a bio-app environment, set one of boltz/rfdiffusion/proteinmpnn/pyrosetta
        (so the script can import pyrosetta/torch/boltz). GPU ones pause the LLM during the run.
        Leave empty for a dedicated venv.
      packages: pip packages for a dedicated venv (ignored when base_venv is set). Empty ⇒ stdlib only.
      gpu: force GPU eviction for a dedicated-venv script that uses the GPU (e.g. installs torch).
      one_time: delete the taskity after one successful run (scratch scripts).
      params_schema: optional JSON-schema object describing accepted params (documentation).
    """
    err = _validate(name, base_venv)
    if err:
        return err
    if name in _REGISTRY:
        return f"A taskity named '{name}' already exists — use taskity_update to change it."
    issues = lint_taskity_script(script)
    if issues:
        return "taskity_create rejected — the script looks like a stub:\n- " + "\n- ".join(issues) + \
            "\n\nFix it: write real, working code (no placeholders/TODOs/mock data)."
    _save({
        "name": name, "description": description, "script": script,
        "base_venv": base_venv, "packages": packages or [], "gpu": bool(gpu),
        "one_time": bool(one_time), "params_schema": params_schema or {},
    })
    where = f"bio-app venv '{base_venv}'" if base_venv else (
        f"dedicated venv ({', '.join(packages)})" if packages else "dedicated stdlib venv")
    return f"Created taskity '{name}' (runs in {where}). Run it with taskity_run."


async def taskity_update(
    name: str,
    description: str,
    script: str,
    base_venv: str = "",
    packages: list[str] | None = None,
    gpu: bool = False,
    one_time: bool = False,
    params_schema: dict | None = None,
) -> str:
    """Replace an existing taskity's script/metadata. Fails if it doesn't exist."""
    if name not in _REGISTRY:
        return f"No taskity named '{name}' to update — use taskity_create first."
    err = _validate(name, base_venv)
    if err:
        return err
    issues = lint_taskity_script(script)
    if issues:
        return "taskity_update rejected — the script looks like a stub:\n- " + "\n- ".join(issues)
    # Package set changed ⇒ drop the marker so the venv reinstalls on next run.
    if (packages or []) != (_REGISTRY[name].get("packages") or []):
        _marker_path(name).unlink(missing_ok=True)
    _save({
        "name": name, "description": description, "script": script,
        "base_venv": base_venv, "packages": packages or [], "gpu": bool(gpu),
        "one_time": bool(one_time), "params_schema": params_schema or {},
    })
    return f"Updated taskity '{name}'."


async def taskity_run(name: str, params: dict | None = None) -> dict:
    """Run a taskity by name with optional params (passed to the script as JSON argv).

    Returns the script's parsed JSON output plus run metadata (ok, parse_error,
    stdout_tail, duration_ms, artifacts). On failure, `ok` is false and `error`
    holds the reason.
    """
    return await _run(name, params or {})


async def taskity_view(name: str) -> str:
    """Show a taskity's metadata and full script source."""
    meta = _REGISTRY.get(name)
    if meta is None:
        return f"No taskity named '{name}'."
    head = {k: meta[k] for k in ("name", "description", "base_venv", "packages", "gpu", "one_time") if k in meta}
    return f"{json.dumps(head, indent=2)}\n\n--- script.py ---\n{meta.get('script', '')}"


async def taskity_list() -> str:
    """List all taskities (name · environment · description)."""
    if not _REGISTRY:
        return "No taskities defined yet. Create one with taskity_create."
    rows = []
    for n, m in sorted(_REGISTRY.items()):
        env = m.get("base_venv") or ("venv:" + ",".join(m.get("packages") or []) if m.get("packages") else "stdlib")
        rows.append(f"- {n} [{env}] — {m.get('description', '')}")
    return "Taskities:\n" + "\n".join(rows)


async def taskity_delete(name: str) -> str:
    """Delete a taskity and its venv/outputs."""
    if name not in _REGISTRY:
        return f"No taskity named '{name}'."
    _remove(name)
    return f"Deleted taskity '{name}'."


taskity_create_tool = FunctionTool(taskity_create, description="Create a new reusable Python taskity (script) that runs in a venv.")
taskity_update_tool = FunctionTool(taskity_update, description="Replace an existing taskity's script/metadata.")
taskity_run_tool = FunctionTool(taskity_run, description="Run a taskity by name with optional JSON params; returns its parsed output.")
taskity_view_tool = FunctionTool(taskity_view, description="Show a taskity's metadata and full script source.")
taskity_list_tool = FunctionTool(taskity_list, description="List all defined taskities.")
taskity_delete_tool = FunctionTool(taskity_delete, description="Delete a taskity and its venv/outputs.")

TASKITY_TOOLS = [
    taskity_create_tool, taskity_update_tool, taskity_run_tool,
    taskity_view_tool, taskity_list_tool, taskity_delete_tool,
]
