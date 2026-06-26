"""PyRosetta scoring / relax (CPU) — the pipeline's quality-control filter.

Scores a design's energy (Rosetta Energy Units, REU) with the ref2015 score
function, optionally after FastRelax, and — when interface chains are given —
reports the interface ΔG (binding energy) via InterfaceAnalyzer. High-energy or
clashing designs can then be rejected before wasting time predicting/synthesizing.

CPU-only, so this does **not** evict the LLM from VRAM. Runs a small generated
driver script through the PyRosetta venv (see SETUP_BIOAPPS.md; license required).

POC note: total REU and interface ΔG are comparative scores for ranking designs,
not absolute experimental free energies.
"""
import json
from pathlib import Path

from autogen_core.tools import FunctionTool

from .config import PYROSETTA
from .runner import BioAppError, emit_artifact, new_run_dir, run_in_env

_RESULT_MARKER = "RESULT_JSON:"

# Driver run inside the PyRosetta venv. argv: <pdb> <relax 0|1> [interface e.g. "A_B"].
_DRIVER = '''\
import json, sys
import pyrosetta
from pyrosetta import pose_from_pdb

pyrosetta.init("-mute all")
pdb, do_relax = sys.argv[1], sys.argv[2] == "1"
interface = sys.argv[3] if len(sys.argv) > 3 and sys.argv[3] else ""

pose = pose_from_pdb(pdb)
sf = pyrosetta.get_fa_scorefxn()
pre = sf(pose)
if do_relax:
    from pyrosetta.rosetta.protocols.relax import FastRelax
    fr = FastRelax()
    fr.set_scorefxn(sf)
    fr.apply(pose)
post = sf(pose)
result = {"total_score": round(post, 3), "total_score_pre_relax": round(pre, 3), "relaxed": do_relax}
if interface:
    from pyrosetta.rosetta.protocols.analysis import InterfaceAnalyzerMover
    iam = InterfaceAnalyzerMover(interface)
    iam.apply(pose)
    result["interface_dG"] = round(iam.get_interface_dG(), 3)
print("''' + _RESULT_MARKER + '''" + json.dumps(result))
'''


def _parse_result(stdout: str) -> dict:
    for line in reversed(stdout.splitlines()):
        if line.startswith(_RESULT_MARKER):
            return json.loads(line[len(_RESULT_MARKER):])
    raise BioAppError("PyRosetta produced no result line — check the log above for an init/license error.")


async def pyrosetta_score(pdb: str, relax: bool = True, interface_chains: str = "",
                          name: str = "score") -> dict:
    """Score a structure's energy (and optional interface ΔG) with PyRosetta.

    Args:
      pdb: path to the structure to score (PDB).
      relax: run FastRelax before scoring (recommended for raw generated backbones).
      interface_chains: for binders, the two sides separated by '_', e.g. "A_B",
        to also report interface ΔG (binding energy). Leave empty for a monomer.
      name: label for the run.

    Returns total score (REU), pre-relax score, and (if requested) interface ΔG.
    Lower total score / more-negative interface ΔG = better.
    """
    if not Path(pdb).exists():
        raise RuntimeError(f"PDB not found: {pdb}")

    run_dir = new_run_dir(f"rosetta-{name}")
    driver = run_dir / "score_driver.py"
    driver.write_text(_DRIVER)
    argv = ["python", str(driver), str(pdb), "1" if relax else "0", interface_chains]

    stdout = await run_in_env(PYROSETTA, argv, label=f"PyRosetta score ({name})")
    result = _parse_result(stdout)

    score_json = run_dir / "score.json"
    score_json.write_text(json.dumps(result, indent=2))
    await emit_artifact("pyrosetta", "score", score_json, name=name, **result)
    return {"name": name, **result}


pyrosetta_score_tool = FunctionTool(
    pyrosetta_score,
    description="Score a structure's Rosetta energy (REU) and optional interface ΔG with PyRosetta; lower is better.",
)
