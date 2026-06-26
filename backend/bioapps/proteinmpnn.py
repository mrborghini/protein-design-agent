"""ProteinMPNN sequence design (GPU).

Takes a backbone (e.g. from RFdiffusion) and solves for amino-acid sequences that
fold it — biasing toward well-packed hydrophobic cores for thermal stability. The
designed sequences are what you'd predict (Boltz-2) and score (PyRosetta) next.

Wraps ``protein_mpnn_run.py`` from a local ProteinMPNN clone, invoked through the
tool's venv. The clone path is required (`BIOAPP_PROTEINMPNN_DIR`).

POC note: single-PDB mode (`--pdb_path`) is used directly; multi-chain assemblies
or fixed-position constraints would need the repo's parsing helpers (future work).
"""
import glob
from pathlib import Path

from autogen_core.tools import FunctionTool

from .config import PROTEINMPNN
from .runner import emit_artifact, new_run_dir, run_in_env


def _run_script() -> Path:
    if PROTEINMPNN.repo_dir is None:
        raise RuntimeError(
            "ProteinMPNN clone path is not set. Point BIOAPP_PROTEINMPNN_DIR at your "
            "ProteinMPNN checkout (see SETUP_BIOAPPS.md)."
        )
    script = PROTEINMPNN.repo_dir / "protein_mpnn_run.py"
    if not script.exists():
        raise RuntimeError(f"Could not find {script} — is BIOAPP_PROTEINMPNN_DIR a valid ProteinMPNN clone?")
    return script


def _read_fastas(seqs_dir: Path) -> list[str]:
    """Collect designed sequences from ProteinMPNN's seqs/*.fa output."""
    sequences: list[str] = []
    for fa in sorted(glob.glob(str(seqs_dir / "*.fa"))):
        text = Path(fa).read_text()
        # FASTA: sequence lines are the non-header lines; the first record is the
        # input backbone, the rest are designs. Keep all non-empty sequence lines.
        for line in text.splitlines():
            if line and not line.startswith(">"):
                sequences.append(line.strip())
    return sequences


async def proteinmpnn_design(
    backbone_pdb: str,
    num_sequences: int = 8,
    sampling_temp: float = 0.1,
    chains: str = "A",
    name: str = "design",
) -> dict:
    """Design amino-acid sequences for a backbone with ProteinMPNN.

    Args:
      backbone_pdb: path to the input backbone (PDB).
      num_sequences: sequences to sample.
      sampling_temp: lower = more conservative/native-like (0.1 is a good default).
      chains: chain(s) to design, e.g. "A".
      name: label for the run.

    Returns the designed sequences (FASTA strings) and the output folder.
    """
    script = _run_script()
    if not Path(backbone_pdb).exists():
        raise RuntimeError(f"Backbone PDB not found: {backbone_pdb}")

    run_dir = new_run_dir(f"mpnn-{name}")
    argv = [
        "python", str(script),
        "--pdb_path", str(backbone_pdb),
        "--pdb_path_chains", chains,
        "--out_folder", str(run_dir),
        "--num_seq_per_target", str(int(num_sequences)),
        "--sampling_temp", str(sampling_temp),
    ]
    await run_in_env(PROTEINMPNN, argv, cwd=PROTEINMPNN.repo_dir, label=f"ProteinMPNN ({name})")

    seqs_dir = run_dir / "seqs"
    sequences = _read_fastas(seqs_dir)
    if not sequences:
        raise RuntimeError(f"ProteinMPNN produced no sequences under {seqs_dir}.")
    for fa in sorted(glob.glob(str(seqs_dir / "*.fa"))):
        await emit_artifact("proteinmpnn", "sequence", Path(fa), name=name)
    return {"name": name, "num_sequences": len(sequences), "sequences": sequences,
            "seqs_dir": str(seqs_dir)}


proteinmpnn_design_tool = FunctionTool(
    proteinmpnn_design,
    description="Design amino-acid sequences that fold a given backbone PDB with ProteinMPNN.",
)
