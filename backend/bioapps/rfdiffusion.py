"""RFdiffusion binder-backbone generation (GPU).

Generates mini-protein backbone(s) that cap an exposed, aggregation-prone
hydrophobic patch on a target — the "chaperone" scaffold the rest of the pipeline
then sequences (ProteinMPNN) and scores (PyRosetta).

Wraps ``scripts/run_inference.py`` from a local RFdiffusion clone, invoked through
the tool's venv. The clone path is required (`BIOAPP_RFDIFFUSION_DIR`).

POC notes:
- The RFdiffusion contig map and hotspot residues encode real structural-biology
  intent, so the Operator agent supplies them explicitly (we don't guess a contig).
- Outputs are backbones only (no sequence yet); pass them to ProteinMPNN next.
"""
import glob
from pathlib import Path

from autogen_core.tools import FunctionTool

from .config import RFDIFFUSION
from .runner import emit_artifact, new_run_dir, run_in_env


def _run_script() -> Path:
    """Locate RFdiffusion's run_inference.py, or raise a clear setup error."""
    if RFDIFFUSION.repo_dir is None:
        raise RuntimeError(
            "RFdiffusion clone path is not set. Point BIOAPP_RFDIFFUSION_DIR at your "
            "RFdiffusion checkout (see SETUP_BIOAPPS.md)."
        )
    script = RFDIFFUSION.repo_dir / "scripts" / "run_inference.py"
    if not script.exists():
        raise RuntimeError(f"Could not find {script} — is BIOAPP_RFDIFFUSION_DIR a valid RFdiffusion clone?")
    return script


async def rfdiffusion_generate(
    target_pdb: str,
    contigs: str,
    hotspot_residues: str = "",
    num_designs: int = 4,
    name: str = "binder",
) -> dict:
    """Generate binder backbone(s) against a target with RFdiffusion.

    Args:
      target_pdb: path to the target structure (PDB).
      contigs: RFdiffusion contig map string, e.g. "A1-100/0 50-50" (target chain A
        residues 1-100, then a 50-residue de-novo binder). This is required domain
        input — supply the contig that frames your binder/target.
      hotspot_residues: optional comma-separated target residues the binder should
        engage, e.g. "A30,A33,A34".
      num_designs: number of backbones to sample.
      name: label/prefix for the run.

    Returns the generated backbone PDB paths (sequence-less; feed to ProteinMPNN).
    """
    script = _run_script()
    if not Path(target_pdb).exists():
        raise RuntimeError(f"Target PDB not found: {target_pdb}")

    run_dir = new_run_dir(f"rfdiff-{name}")
    out_prefix = run_dir / name
    argv = [
        "python", str(script),
        f"inference.output_prefix={out_prefix}",
        f"inference.input_pdb={target_pdb}",
        f"contigmap.contigs=[{contigs}]",
        f"inference.num_designs={int(num_designs)}",
    ]
    if hotspot_residues.strip():
        argv.append(f"ppi.hotspot_res=[{hotspot_residues}]")
    if RFDIFFUSION.weights_dir is not None:
        argv.append(f"inference.ckpt_override_path={RFDIFFUSION.weights_dir}")

    # Run from the repo dir so RFdiffusion's relative config/asset paths resolve.
    await run_in_env(RFDIFFUSION, argv, cwd=RFDIFFUSION.repo_dir, label=f"RFdiffusion ({name})")

    backbones = sorted(glob.glob(str(run_dir / f"{name}_*.pdb")))
    if not backbones:
        raise RuntimeError(f"RFdiffusion produced no backbones under {run_dir}.")
    for bb in backbones:
        await emit_artifact("rfdiffusion", "backbone", Path(bb), name=name)
    return {"name": name, "num_backbones": len(backbones), "backbone_paths": backbones}


rfdiffusion_generate_tool = FunctionTool(
    rfdiffusion_generate,
    description="Generate mini-protein binder backbone(s) against a target PDB with RFdiffusion (contig + optional hotspots).",
)
