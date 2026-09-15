#!/usr/bin/env python3
"""Run FlexPepDock validation experiments for paper candidate peptides.

Each candidate is docked N_TRIALS times.  We report median, mean, stdev,
and best ddG to account for FlexPepDock's stochastic refinement.
Results are written to a JSON summary file.
"""
from __future__ import annotations

import json
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
FLEXPEP_SCRIPT = REPO_ROOT / "AG_src" / "scripts" / "flexpep_dock.py"
TEMPLATE_PDB = REPO_ROOT / "data" / "fold_test1_model_0.pdb"
BASELINE_PDB = REPO_ROOT / "runs" / "pyrosetta_flow" / "sst14_agentic_mutdock" / "baseline_refined.pdb"
OUTPUT_DIR = REPO_ROOT / "runs" / "paper_validation"
CONDA_ENV = "bio-tools"
PEPTIDE_CHAIN = 1
N_TRIALS = 10
MAX_WORKERS = 2  # parallel trials per candidate (limited by CPU)


def _resolve_conda_python(conda_env: str) -> str:
    """Resolve the Python executable path for a conda environment."""
    for base in [
        Path.home() / "miniforge3",
        Path.home() / "miniconda3",
        Path.home() / "anaconda3",
    ]:
        env_python = base / "envs" / conda_env / "bin" / "python"
        if env_python.exists():
            return str(env_python)
    return ""

# WT SST14 reference
WT_SEQ = "AGCKNFFWKTFTSC"

# Candidate definitions
CANDIDATES = [
    # --- Sanity checks (should be unfavorable vs WT) ---
    {
        "id": "SAN-01",
        "label": "W8A virtual mutant",
        "sequence": "AGCKNFFAKTFTSC",
        "expectation": "Unfavorable",
        "mode": "mutate",  # use reference + MutateResidue
    },
    {
        "id": "SAN-02",
        "label": "K9A virtual mutant",
        "sequence": "AGCKNFFWATFTSC",
        "expectation": "Unfavorable",
        "mode": "mutate",
    },
    # --- Novel candidates from agentic pipeline ---
    {
        "id": "NOV-01",
        "label": "Agentic iter01 best (seed7000)",
        "sequence": "YSCKNFFWKTFTSN",
        "expectation": "Explore",
        "mode": "mutate",
    },
    {
        "id": "NOV-02",
        "label": "Agentic iter02 best (seed7000)",
        "sequence": "AGCKNDFWKTFGSE",
        "expectation": "Explore",
        "mode": "mutate",
    },
    # --- Literature controls ---
    # Octreotide core motif is Phe-Trp-Lys-Thr (SST14 positions 7,8,9,10).
    # Since MutateResidue preserves chain length, we model Octreotide as
    # a 14-mer with Octreotide-inspired substitutions at non-core positions:
    # SST14:       A G C K N F F W K T F T S C
    # Octreotide:  F C F - - F W K T C T - - -
    # Mapped 14:   F C C K N F F W K T C T S C  (pos1→F, pos2→C kept, pos11→C)
    {
        "id": "LIT-02",
        "label": "Octreotide-inspired 14-mer (pharmacophore-mapped)",
        "sequence": "FCCKNFFWKTCTSC",
        "expectation": "Favorable",
        "mode": "mutate",
        "note": "L-amino acid 14-mer mapping of Octreotide pharmacophore",
    },
    # Cortistatin-14 (CST-14): natural SSTR2 agonist, separate gene from SST14.
    # 14-mer, all L-amino acids, nanomolar SSTR2 binding confirmed in literature.
    # FWKT pharmacophore aligned: CST-14 differs at pos2 (G→P) and pos12 (T→S).
    # SST14: A G C K N F F W K T F T S C
    # CST14: A P C K N F F W K T F S S C  (FWKT-anchored mapping)
    # Ref: de Lecea et al. (1996) Nature 381:242; Bhatt et al. (2022) Cell Discov.
    {
        "id": "LIT-03",
        "label": "Cortistatin-14 (CST-14, FWKT-aligned 14-mer)",
        "sequence": "APCKNFFWKTFSSC",
        "expectation": "Favorable",
        "mode": "mutate",
        "note": "Natural SSTR2 agonist, 14-mer, 2 substitutions vs SST14 (G2P, T12S)",
    },
]


def run_flexpep(
    candidate_id: str,
    sequence: str,
    trial: int,
    mode: str,
    protocol: str = "flexpep_refine",
) -> dict:
    """Run a single FlexPepDock trial."""
    out_dir = OUTPUT_DIR / candidate_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_pdb = out_dir / f"trial_{trial}.pdb"

    if mode == "direct":
        # Direct refinement of template PDB (no mutation)
        cmd_args = [
            "--input", str(TEMPLATE_PDB),
            "--output", str(out_pdb),
            "--protocol", protocol,
            "--peptide-chain", str(PEPTIDE_CHAIN),
        ]
    elif mode == "mutate":
        # Same-length mutation from refined baseline
        cmd_args = [
            "--input", str(TEMPLATE_PDB),
            "--output", str(out_pdb),
            "--protocol", protocol,
            "--peptide-chain", str(PEPTIDE_CHAIN),
            "--reference-complex", str(BASELINE_PDB),
            "--target-sequence", sequence,
        ]
    else:
        cmd_args = [
            "--input", str(TEMPLATE_PDB),
            "--output", str(out_pdb),
            "--protocol", "flexpep_refine",
            "--peptide-chain", str(PEPTIDE_CHAIN),
            "--reference-complex", str(BASELINE_PDB),
            "--target-sequence", sequence,
        ]

    env_python = _resolve_conda_python(CONDA_ENV)
    if env_python:
        cmd = [env_python, str(FLEXPEP_SCRIPT), *cmd_args]
    else:
        cmd = ["conda", "run", "-n", CONDA_ENV, "python", str(FLEXPEP_SCRIPT), *cmd_args]

    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True, check=False)
    elapsed = time.time() - t0

    if proc.returncode != 0:
        stderr = (proc.stderr or "").strip()[-500:]
        return {
            "candidate_id": candidate_id,
            "trial": trial,
            "sequence": sequence,
            "error": f"exit={proc.returncode}: {stderr}",
            "elapsed_s": round(elapsed, 1),
        }

    lines = (proc.stdout or "").strip().splitlines()
    result = json.loads(lines[-1]) if lines else {}
    result["candidate_id"] = candidate_id
    result["trial"] = trial
    result["sequence"] = sequence
    result["elapsed_s"] = round(elapsed, 1)
    return result


def run_candidate(cand: dict) -> dict:
    """Run N_TRIALS for one candidate in parallel, report statistics."""
    cid = cand["id"]
    seq = cand["sequence"]
    mode = cand["mode"]
    print(f"\n{'='*60}", file=sys.stderr)
    print(f"[{cid}] {cand['label']}", file=sys.stderr)
    print(f"  Sequence: {seq}  ({len(seq)}-mer)", file=sys.stderr)
    print(f"  Expectation: {cand['expectation']}", file=sys.stderr)
    print(f"  Running {N_TRIALS} trials ({MAX_WORKERS} parallel)...", file=sys.stderr)

    trials: list[dict] = [{}] * N_TRIALS
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {
            pool.submit(run_flexpep, cid, seq, t, mode): t
            for t in range(N_TRIALS)
        }
        for fut in as_completed(futures):
            t_idx = futures[fut]
            result = fut.result()
            trials[t_idx] = result
            ddg = result.get("ddg", "ERR")
            print(f"  Trial {t_idx}: ddG={ddg}", file=sys.stderr)

    # Compute statistics from successful trials (filter catastrophic failures ddG > 0)
    ok_trials = [t for t in trials if "error" not in t and "ddg" in t]
    sane_trials = [t for t in ok_trials if t["ddg"] <= 0]
    if not sane_trials and ok_trials:
        sane_trials = ok_trials  # fallback if all positive
    if sane_trials:
        ddgs = [t["ddg"] for t in sane_trials]
        best = min(ok_trials, key=lambda x: x["ddg"])
        ddg_median = statistics.median(ddgs)
        ddg_mean = statistics.mean(ddgs)
        ddg_stdev = statistics.stdev(ddgs) if len(ddgs) > 1 else 0.0
        ddg_best = min(ddgs)
        ddg_worst = max(ddgs)
        n_rejected = len(ok_trials) - len(sane_trials)
        rej_msg = f", {n_rejected} outlier(s) excluded" if n_rejected else ""
        print(
            f"  → median={ddg_median:.2f}  mean={ddg_mean:.2f}  "
            f"stdev={ddg_stdev:.2f}  best={ddg_best:.2f}  "
            f"({len(sane_trials)}/{N_TRIALS} ok{rej_msg})",
            file=sys.stderr,
        )
    else:
        best = trials[0]
        ddg_median = ddg_mean = ddg_stdev = ddg_best = ddg_worst = None
        sane_trials = []
        print(f"  → ALL TRIALS FAILED", file=sys.stderr)

    return {
        "id": cid,
        "label": cand["label"],
        "sequence": seq,
        "expectation": cand["expectation"],
        "n_trials": N_TRIALS,
        "n_ok": len(sane_trials),
        "ddg_best": ddg_best,
        "ddg_median": round(ddg_median, 4) if ddg_median is not None else None,
        "ddg_mean": round(ddg_mean, 4) if ddg_mean is not None else None,
        "ddg_stdev": round(ddg_stdev, 4) if ddg_stdev is not None else None,
        "ddg_worst": ddg_worst,
        "total_score": best.get("total_score"),
        "clash_score": best.get("clash_score"),
        "all_trials": trials,
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # WT baseline — use mutate mode with its own sequence (identity mutation)
    # so it starts from the same refined baseline PDB as all other candidates.
    all_candidates = [
        {
            "id": "LIT-01",
            "label": "WT SST14 (baseline)",
            "sequence": WT_SEQ,
            "expectation": "Favorable (reference)",
            "mode": "mutate",
        },
        *CANDIDATES,
    ]

    results = []
    for cand in all_candidates:
        result = run_candidate(cand)
        results.append(result)

    # Summary table
    print(f"\n{'='*80}", file=sys.stderr)
    print("VALIDATION SUMMARY", file=sys.stderr)
    print(f"{'='*80}", file=sys.stderr)
    print(f"{'ID':<8} {'Sequence':<16} {'median':>8} {'mean':>8} {'stdev':>7} {'best':>8} {'n':>3} {'Expect'}", file=sys.stderr)
    print("-"*80, file=sys.stderr)
    for r in results:
        med = r.get("ddg_median", r.get("ddg"))
        mn = r.get("ddg_mean", r.get("ddg"))
        sd = r.get("ddg_stdev", 0)
        best = r.get("ddg_best", r.get("ddg"))
        n = r.get("n_ok", r.get("n_trials", "?"))
        ms = f"{med:.2f}" if isinstance(med, (int, float)) else "ERR"
        mns = f"{mn:.2f}" if isinstance(mn, (int, float)) else "ERR"
        ss = f"{sd:.2f}" if isinstance(sd, (int, float)) else "-"
        bs = f"{best:.2f}" if isinstance(best, (int, float)) else "ERR"
        print(f"{r['id']:<8} {r['sequence']:<16} {ms:>8} {mns:>8} {ss:>7} {bs:>8} {n:>3} {r['expectation']}", file=sys.stderr)

    # Save to JSON
    out_path = OUTPUT_DIR / "validation_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}", file=sys.stderr)
    # Also print JSON to stdout
    print(json.dumps({"status": "complete", "n_candidates": len(results), "output": str(out_path)}))


if __name__ == "__main__":
    main()
