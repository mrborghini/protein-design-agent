"""GPU/VRAM orchestration so the LLM and GPU bio-apps never collide.

The 32 GB GPU cannot hold `gpt-oss:120b` (offloaded across VRAM+RAM) *and* a
structural-biology model (Boltz-2/RFdiffusion/ProteinMPNN) at the same time —
that OOMs. Before any GPU tool runs we therefore **evict every model Ollama has
resident** (`POST /api/generate {keep_alive: 0}`) and wait for VRAM to free, run
the tool exclusively, then let the LLM reload lazily on the next inference call.

A single process-wide ``asyncio.Lock`` serializes GPU consumers so two tools (or a
tool and a reload) never contend. Status is streamed via the shared SSE queue
(the same ``research_sink`` ContextVar the web-research tool uses).

POC / best-effort: eviction only controls *this* app's Ollama models. Anything
else using the GPU concurrently (another process, a desktop compositor) can still
OOM the bio-app — that surfaces as a tool error, not a hidden failure.
"""
import asyncio
import json
import urllib.request

from backend.agents import OLLAMA_HOST
from backend.research import research_sink

# One GPU consumer at a time, process-wide. Held for the whole tool run.
_gpu_lock = asyncio.Lock()

_UNLOAD_POLL_SECONDS = 1.0
_UNLOAD_TIMEOUT_SECONDS = 60.0


async def _emit(stage: str, text: str) -> None:
    """Push a `gpu` status event onto the active SSE queue, if one is set."""
    queue = research_sink.get(None)
    if queue is not None:
        await queue.put({"type": "gpu", "stage": stage, "text": text})


def _loaded_models_sync() -> list[str]:
    """Models Ollama currently has resident (via /api/ps). [] if none / on error."""
    try:
        req = urllib.request.Request(f"{OLLAMA_HOST}/api/ps")
        with urllib.request.urlopen(req, timeout=10) as resp:  # noqa: S310 - local Ollama
            data = json.loads(resp.read().decode("utf-8"))
        return [m["name"] for m in data.get("models", [])]
    except Exception:  # noqa: BLE001 - if we can't tell, assume nothing to evict
        return []


def _evict_sync(model: str) -> None:
    """Ask Ollama to unload `model` now (keep_alive: 0 frees its VRAM immediately)."""
    body = json.dumps({"model": model, "keep_alive": 0}).encode("utf-8")
    req = urllib.request.Request(
        f"{OLLAMA_HOST}/api/generate", data=body, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310 - local Ollama
            resp.read()
    except Exception:  # noqa: BLE001 - best-effort; the wait-loop below confirms the result
        pass


async def _free_vram() -> list[str]:
    """Evict all resident Ollama models and wait until VRAM is released.

    Returns the list of models that were evicted (so callers/UI can report what
    will reload). Times out after ``_UNLOAD_TIMEOUT_SECONDS`` and proceeds anyway —
    the tool itself will OOM-and-error if VRAM truly wasn't freed, which is honest.
    """
    loaded = await asyncio.to_thread(_loaded_models_sync)
    if not loaded:
        return []
    await _emit("evict", f"Freeing VRAM — unloading {', '.join(loaded)} from the GPU…")
    for model in loaded:
        await asyncio.to_thread(_evict_sync, model)

    waited = 0.0
    while waited < _UNLOAD_TIMEOUT_SECONDS:
        if not await asyncio.to_thread(_loaded_models_sync):
            break
        await asyncio.sleep(_UNLOAD_POLL_SECONDS)
        waited += _UNLOAD_POLL_SECONDS
    return loaded


class gpu_exclusive:
    """Async context manager granting exclusive GPU use for a bio-app run.

    On enter: serialize behind the GPU lock, then evict resident LLM(s) so the GPU
    tool has the full 32 GB. On exit: release the lock. The LLM is **not** force-
    reloaded — Ollama reloads it lazily on the next inference (the pipeline's next
    agent turn), which avoids a pointless load if more GPU tools follow.

    Usage:
        async with gpu_exclusive():
            await run_in_env(...)  # GPU tool owns the card here
    """

    def __init__(self) -> None:
        self._evicted: list[str] = []

    async def __aenter__(self) -> "gpu_exclusive":
        await _gpu_lock.acquire()
        try:
            self._evicted = await _free_vram()
        except BaseException:
            self._gpu_lock_release()
            raise
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        try:
            if self._evicted:
                await _emit(
                    "reload",
                    f"GPU tool finished — {', '.join(self._evicted)} will reload on the next turn.",
                )
        finally:
            self._gpu_lock_release()

    @staticmethod
    def _gpu_lock_release() -> None:
        if _gpu_lock.locked():
            _gpu_lock.release()
