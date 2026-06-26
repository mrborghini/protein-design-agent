"""Boltz-2 structure prediction + in-silico point mutagenesis (GPU).

Models a protein's native fold and lets the agent test whether a point mutation
rigidifies a flexible region by comparing predicted-confidence (pLDDT) before and
after. Wraps the ``boltz`` CLI installed in its venv (see SETUP_BIOAPPS.md).

POC notes:
- Output layout varies slightly across boltz releases, so we glob for the produced
  mmCIF + confidence JSON rather than hardcoding paths.
- A higher mean pLDDT on the mutant is only a *heuristic* for added rigidity, not a
  thermodynamic stability measurement — PyRosetta ΔG is the real check downstream.
"""
import glob
import json
import re
from pathlib import Path

from autogen_core.tools import FunctionTool

from .config import BOLTZ
from .runner import emit_artifact, new_run_dir, run_in_env

# Boltz multi-record FASTA header for a single protein chain (MSA server fills the MSA).
_FASTA_HEADER = ">A|protein|"
_MUTATION_RE = re.compile(r"^([A-Z])(\d+)([A-Z])$")


def _write_fasta(run_dir: Path, name: str, sequence: str) -> Path:
    seq = "".join(sequence.split()).upper()
    fasta = run_dir / f"{name}.fasta"
    fasta.write_text(f"{_FASTA_HEADER}\n{seq}\n")
    return fasta


def _find_outputs(out_dir: Path) -> tuple[Path | None, float | None]:
    """Locate the produced mmCIF and mean pLDDT from boltz's output tree."""
    cifs = sorted(glob.glob(str(out_dir / "**" / "*_model_0.cif"), recursive=True)) \
        or sorted(glob.glob(str(out_dir / "**" / "*.cif"), recursive=True))
    structure = Path(cifs[0]) if cifs else None

    plddt: float | None = None
    confs = sorted(glob.glob(str(out_dir / "**" / "confidence_*.json"), recursive=True))
    if confs:
        try:
            data = json.loads(Path(confs[0]).read_text())
            # boltz reports `complex_plddt` (0–1); fall back to `confidence_score`.
            val = data.get("complex_plddt", data.get("confidence_score"))
            plddt = round(float(val), 4) if val is not None else None
        except Exception:  # noqa: BLE001 - confidence is best-effort metadata
            plddt = None
    return structure, plddt


def _apply_mutation(sequence: str, mutation: str) -> str:
    """Apply a `WT<pos><MUT>` mutation (1-indexed), validating the wild-type residue."""
    seq = "".join(sequence.split()).upper()
    m = _MUTATION_RE.match(mutation.strip().upper())
    if not m:
        raise ValueError(f"Mutation '{mutation}' must look like 'A123V' (wild-type, position, mutant).")
    wt, pos_s, mut = m.group(1), m.group(2), m.group(3)
    pos = int(pos_s)
    if pos < 1 or pos > len(seq):
        raise ValueError(f"Position {pos} is out of range for a {len(seq)}-residue sequence.")
    if seq[pos - 1] != wt:
        raise ValueError(f"Residue at position {pos} is '{seq[pos - 1]}', not '{wt}' as the mutation claims.")
    return seq[: pos - 1] + mut + seq[pos:]


async def _predict(name: str, sequence: str, use_msa_server: bool) -> dict:
    run_dir = new_run_dir(f"boltz-{name}")
    fasta = _write_fasta(run_dir, name, sequence)
    out_dir = run_dir / "out"
    argv = ["boltz", "predict", str(fasta), "--out_dir", str(out_dir)]
    if use_msa_server:
        argv.append("--use_msa_server")  # online MSA (POC convenience); omit for fully-local MSAs

    await run_in_env(BOLTZ, argv, label=f"Boltz-2 predict ({name})")

    structure, plddt = _find_outputs(out_dir)
    if structure is None:
        raise RuntimeError(f"Boltz produced no structure for '{name}' under {out_dir}.")
    await emit_artifact("boltz", "structure", structure, name=name, plddt=plddt)
    return {"name": name, "structure_path": str(structure), "mean_plddt": plddt}


async def boltz_predict(sequence: str, name: str = "target", use_msa_server: bool = True) -> dict:
    """Predict a protein's 3D structure from its amino-acid sequence with Boltz-2.

    Returns the path to the predicted structure (mmCIF) and its mean pLDDT
    confidence. Use `name` to label the run. Set `use_msa_server=False` only if you
    have local MSAs configured.
    """
    return await _predict(name, sequence, use_msa_server)


async def boltz_mutagenesis(sequence: str, mutation: str, name: str = "target",
                            use_msa_server: bool = True) -> dict:
    """Predict wild-type and point-mutant structures and compare confidence.

    `mutation` is e.g. 'A123V' (wild-type residue, 1-indexed position, mutant
    residue). Returns both structures' mean pLDDT and the delta — a positive delta
    is a heuristic (not proof) that the mutation rigidifies the fold.
    """
    mutant_seq = _apply_mutation(sequence, mutation)  # validates before any GPU work
    wt = await _predict(f"{name}-wt", sequence, use_msa_server)
    mut = await _predict(f"{name}-{mutation}", mutant_seq, use_msa_server)
    delta = None
    if wt["mean_plddt"] is not None and mut["mean_plddt"] is not None:
        delta = round(mut["mean_plddt"] - wt["mean_plddt"], 4)
    return {
        "mutation": mutation,
        "wild_type": wt,
        "mutant": mut,
        "delta_plddt": delta,
        "note": "Positive delta_plddt suggests increased rigidity; confirm with PyRosetta ΔG.",
    }


boltz_predict_tool = FunctionTool(
    boltz_predict,
    description="Predict a protein's 3D structure (mmCIF + mean pLDDT) from its sequence with Boltz-2.",
)
boltz_mutagenesis_tool = FunctionTool(
    boltz_mutagenesis,
    description="Compare predicted structure confidence (pLDDT) of a wild-type vs a point mutant (e.g. 'A123V').",
)
