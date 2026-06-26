# Bio-app setup (Pipeline mode)

**POC / local-only.** Pipeline mode drives four external structural-biology tools. Each runs in
its **own Python virtualenv** so its heavy GPU dependencies never pollute the backend venv. None
of these are installed by this repo — set them up once with the steps below, then point the
backend at them with the environment variables in the last section.

The backend never "activates" a venv; it invokes each tool through its venv interpreter/CLI
directly (`<venv>/bin/boltz …`, `<venv>/bin/python <repo>/script.py …`).

> **Hardware reality (read this).** This machine has **32 GB VRAM** and **64 GB system RAM** —
> these are *not* a single 96 GB pool. `gpt-oss:120b` (MXFP4 ≈ 60–65 GB) only runs by offloading
> roughly half its weights to CPU RAM, so expect **minutes per turn**, not instant responses. More
> importantly, the LLM and any GPU bio-app **cannot share the 32 GB GPU**. The backend handles this
> by **evicting the Ollama model from VRAM before** each GPU tool (Boltz-2, RFdiffusion,
> ProteinMPNN) and letting it reload afterward — so a single pipeline run will visibly pause to
> swap the GPU back and forth. PyRosetta is CPU-only and does not trigger an eviction.

## 0. Prerequisites
- Python 3.11+ (`python -m venv` available).
- NVIDIA driver + CUDA toolkit matching each tool's PyTorch build.
- Ollama running with the rostered models pulled (`gpt-oss:120b`, `qwen3.5:9b`, `gemma4:12b`).
  Verify: `ollama list`.

The defaults below put venvs under `./venvs/<tool>`. Override any path with the env vars in §5.

## 1. Boltz-2 — structure prediction / in-silico mutagenesis (GPU)
```bash
python -m venv venvs/boltz
venvs/boltz/bin/pip install boltz          # installs the `boltz` CLI
# Weights download on first run to ~/.boltz (or set BIOAPP_BOLTZ_WEIGHTS).
```
Used to model the native state and to re-predict point mutants, comparing pLDDT to see whether a
mutation rigidifies a flexible loop.

## 2. RFdiffusion — binder backbone generation (GPU)
```bash
git clone https://github.com/RosettaCommons/RFdiffusion.git
python -m venv venvs/rfdiffusion
venvs/rfdiffusion/bin/pip install -r RFdiffusion/requirements.txt   # + SE3Transformer per the repo README
venvs/rfdiffusion/bin/pip install -e RFdiffusion
# Download model weights into RFdiffusion/models (see the repo's download script).
```
Generates a mini-protein backbone that caps an exposed hydrophobic aggregation patch. Point the
backend at the clone with `BIOAPP_RFDIFFUSION_DIR` (required) and weights with `_WEIGHTS`.

## 3. ProteinMPNN — sequence design (GPU)
```bash
git clone https://github.com/dauparas/ProteinMPNN.git
python -m venv venvs/proteinmpnn
venvs/proteinmpnn/bin/pip install torch    # match your CUDA build
```
Solves for the amino-acid sequence that folds the RFdiffusion backbone, maximizing core packing.
Point the backend at the clone with `BIOAPP_PROTEINMPNN_DIR` (required).

## 4. PyRosetta — ΔG scoring / relax (CPU)
> **License required.** PyRosetta is **free for academic/non-commercial use** but needs a license
> from Rosetta Commons (https://www.pyrosetta.org/ → "Licensing"). Obtain credentials yourself; do
> **not** commit them. For commercial use at Hypersolid, clear it via the Legal wiki first.
```bash
python -m venv venvs/pyrosetta
venvs/pyrosetta/bin/pip install pyrosetta-installer
venvs/pyrosetta/bin/python -c "import pyrosetta_installer; pyrosetta_installer.install_pyrosetta()"
```
Scores generated chaperone designs by Gibbs free energy (ΔG) and rejects clashing/high-energy ones.

## 5. Point the backend at your install
All optional — defaults assume venvs under `./venvs/<tool>`. Override only what differs:

| Variable | Default | Meaning |
|---|---|---|
| `BIOAPP_VENV_ROOT` | `./venvs` | parent dir for the per-tool venvs |
| `BIOAPP_WORKDIR` | `./runs` | where run inputs/outputs are written (`<workdir>/<run-id>/`) |
| `BIOAPP_PAPERS_DIR` | `./papers` | folder the Paper Analyst's RAG reads PDFs from |
| `BIOAPP_BOLTZ_VENV` | `./venvs/boltz` | Boltz venv root |
| `BIOAPP_RFDIFFUSION_VENV` / `_DIR` / `_WEIGHTS` | `./venvs/rfdiffusion` / — / — | venv root; cloned repo dir; weights dir |
| `BIOAPP_PROTEINMPNN_VENV` / `_DIR` | `./venvs/proteinmpnn` / — | venv root; cloned repo dir |
| `BIOAPP_PYROSETTA_VENV` | `./venvs/pyrosetta` | PyRosetta venv root |
| `BIOAPP_*_TIMEOUT` | per-tool (s) | hard kill after this many seconds |

The `*_DIR` vars are **required** for RFdiffusion and ProteinMPNN (the backend needs the path to
their run scripts); the others have working defaults.

## 6. Taskities — Operator-written Python scripts (advanced)

Beyond the four typed tools, the Bio-App Operator can **write and run its own Python scripts**
("taskities", modeled on the discord-agent system). A taskity is a named, reusable script that
reads params as JSON (`sys.argv[1]`) and prints one final JSON line; large outputs go in the dir
named by the `BIOAPP_OUT` env var. Each taskity runs in either a **bio-app venv** (set
`base_venv` to `boltz`/`rfdiffusion`/`proteinmpnn`/`pyrosetta` so the script can
`import pyrosetta`/`torch`/`boltz` — GPU ones evict the LLM first) or a **dedicated venv** that
pip-installs declared packages (cached). Definitions/venvs/outputs live under
`<BIOAPP_TASKITIES_DIR>` (default `<WORKDIR>/taskities`).

Every taskity script is also **reviewed by the Critic role and must be APPROVED before it runs**
(enforced in code, fail-closed). Approval is cached per script version (sha256), so an unchanged
script isn't re-reviewed; editing it forces a fresh review.

> ⚠️ **Security:** taskities execute **arbitrary, model-written Python on your machine** — there is
> no sandbox beyond the venv, a wall-clock timeout, and an output cap. The lint guard + Critic
> review reduce footguns but are LLM/heuristic judgment, **not** a security sandbox. This is a
> **local POC**: run it only on your own authorized machine and never point this endpoint at
> untrusted input.

| Variable | Default | Meaning |
|---|---|---|
| `BIOAPP_TASKITIES_DIR` | `<WORKDIR>/taskities` | where taskity scripts/venvs/outputs live |
| `BIOAPP_TASKITY_PYTHON` | `python3` | interpreter used to *create* dedicated taskity venvs |
| `BIOAPP_TASKITY_TIMEOUT` | `120` | per-run wall-clock cap (s) |
| `BIOAPP_TASKITY_MAX_OUTPUT_BYTES` | `200000` | captured stdout/stderr cap (tail kept) |
| `BIOAPP_TASKITY_VENV_INSTALL_TIMEOUT` | `300` | dedicated-venv `pip install` cap (s) |
