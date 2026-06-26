"""Configuration for the external structural-biology tools (bio-apps).

POC note: every bio-app runs in its **own Python virtualenv** so its heavy GPU
deps never touch the backend venv. This module is the single source of truth for
where those venvs/weights live and how long each tool may run. Everything is read
from environment variables with sane local defaults — **no hardcoded secrets and
no machine-specific absolute paths baked into the code** (override via env).

A tool is run by invoking its venv's interpreter/CLI directly (e.g.
``<venv>/bin/boltz`` or ``<venv>/bin/python <repo>/script.py``) — no shell
activation needed (see runner.py).

The four tools, and what they need on disk (see SETUP_BIOAPPS.md):
  - boltz        — structure prediction / in-silico mutagenesis (pip-installed CLI in its venv)
  - rfdiffusion  — binder backbone generation (cloned repo + downloaded weights)
  - proteinmpnn  — sequence design (cloned repo)
  - pyrosetta    — ΔG scoring / relax (licensed wheel installed in its venv)
"""
import os
from dataclasses import dataclass
from pathlib import Path

# Where all run inputs/outputs live: WORKDIR/<run-id>/... Created lazily by the
# runner (not at import) so importing the backend never touches the filesystem.
WORKDIR = Path(os.environ.get("BIOAPP_WORKDIR", str(Path.cwd() / "runs"))).expanduser()

# Folder the Paper Analyst's RAG reads local papers (PDFs) from.
PAPERS_DIR = Path(os.environ.get("BIOAPP_PAPERS_DIR", str(Path.cwd() / "papers"))).expanduser()

# Default parent dir for the per-tool venvs when a tool's *_VENV var is unset.
_VENV_ROOT = Path(os.environ.get("BIOAPP_VENV_ROOT", str(Path.cwd() / "venvs"))).expanduser()


@dataclass(frozen=True)
class BioApp:
    """Resolved settings for one external tool.

    `venv` is the tool's virtualenv root; the runner invokes ``<venv>/bin/<prog>``.
    `repo_dir` points at a cloned source tree for the tools that ship as scripts
    (RFdiffusion, ProteinMPNN); it's empty for tools invoked as an installed
    CLI/module (Boltz, PyRosetta). `weights_dir` is the model-weights location
    where applicable. `gpu` marks tools that need the GPU (and therefore force the
    LLM out of VRAM first). `timeout` is in seconds.
    """

    key: str
    venv: Path
    gpu: bool
    timeout: int
    repo_dir: Path | None = None
    weights_dir: Path | None = None

    @property
    def bindir(self) -> Path:
        return self.venv / "bin"


def _path(var: str) -> Path | None:
    raw = os.environ.get(var, "").strip()
    return Path(raw).expanduser() if raw else None


def _venv(var: str, default_name: str) -> Path:
    return _path(var) or (_VENV_ROOT / default_name)


def _int(var: str, default: int) -> int:
    try:
        return int(os.environ.get(var, "") or default)
    except ValueError:
        return default


# Per-tool config. Defaults assume venvs under ./venvs/<tool>; override the
# *_VENV / *_DIR / *_WEIGHTS / *_TIMEOUT vars to match your install. Timeouts are
# generous because real runs take minutes→hours; the runner kills on overrun.
BOLTZ = BioApp(
    key="boltz",
    venv=_venv("BIOAPP_BOLTZ_VENV", "boltz"),
    gpu=True,
    timeout=_int("BIOAPP_BOLTZ_TIMEOUT", 3600),
    weights_dir=_path("BIOAPP_BOLTZ_WEIGHTS"),  # optional; boltz caches under ~/.boltz by default
)

RFDIFFUSION = BioApp(
    key="rfdiffusion",
    venv=_venv("BIOAPP_RFDIFFUSION_VENV", "rfdiffusion"),
    gpu=True,
    timeout=_int("BIOAPP_RFDIFFUSION_TIMEOUT", 3600),
    repo_dir=_path("BIOAPP_RFDIFFUSION_DIR"),  # clone of RFdiffusion (contains scripts/run_inference.py)
    weights_dir=_path("BIOAPP_RFDIFFUSION_WEIGHTS"),
)

PROTEINMPNN = BioApp(
    key="proteinmpnn",
    venv=_venv("BIOAPP_PROTEINMPNN_VENV", "proteinmpnn"),
    gpu=True,
    timeout=_int("BIOAPP_PROTEINMPNN_TIMEOUT", 1800),
    repo_dir=_path("BIOAPP_PROTEINMPNN_DIR"),  # clone of ProteinMPNN (contains protein_mpnn_run.py)
)

PYROSETTA = BioApp(
    key="pyrosetta",
    venv=_venv("BIOAPP_PYROSETTA_VENV", "pyrosetta"),
    gpu=False,  # CPU-bound; no VRAM eviction needed
    timeout=_int("BIOAPP_PYROSETTA_TIMEOUT", 1800),
)

ALL: dict[str, BioApp] = {a.key: a for a in (BOLTZ, RFDIFFUSION, PROTEINMPNN, PYROSETTA)}


def resolve_base_venv(key: str | None) -> BioApp | None:
    """Map a taskity's `base_venv` key to a bio-app, or None for a non-bio-app venv."""
    return ALL.get(key) if key else None


# --- Taskities: LLM-authored, reusable Python scripts run in venvs (see taskity.py) --- #
# Where taskity definitions/venvs/outputs live (under WORKDIR ⇒ covered by the runs/ gitignore).
TASKITIES_DIR = Path(os.environ.get("BIOAPP_TASKITIES_DIR", str(WORKDIR / "taskities"))).expanduser()
# Interpreter for stdlib-only taskities (no base_venv, no dedicated venv).
TASKITY_PYTHON = os.environ.get("BIOAPP_TASKITY_PYTHON", "python3")
# Per-run wall-clock cap (s); stdout/stderr byte cap; dedicated-venv pip-install cap (s).
TASKITY_TIMEOUT = _int("BIOAPP_TASKITY_TIMEOUT", 120)
TASKITY_MAX_OUTPUT_BYTES = _int("BIOAPP_TASKITY_MAX_OUTPUT_BYTES", 200_000)
TASKITY_VENV_INSTALL_TIMEOUT = _int("BIOAPP_TASKITY_VENV_INSTALL_TIMEOUT", 300)
