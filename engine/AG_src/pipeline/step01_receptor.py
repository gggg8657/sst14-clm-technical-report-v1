"""
step01_receptor.py
==================
Step 01: SSTR2 구조 준비 (Receptor Structure Preparation)

수용체(SSTR2) PDB 구조를 준비하고 결합 포켓(binding pocket)을 정의한다.
구조 예측에는 OpenFold3 NIM API를 사용하며, API를 사용할 수 없을 때는
data/ 디렉토리의 기존 PDB 파일로 폴백(fallback)한다.

Prepares the SSTR2 receptor PDB and defines the binding pocket.
Uses the OpenFold3 NIM API for structure prediction; falls back to a
pre-existing PDB from the data/ directory when the API is unavailable.

Input:
    - SSTR2 protein sequence (FASTA / plain string)
    - Ligand/complex PDB (optional, for pocket extraction from known complex)
    - Or: pre-existing receptor PDB in data/

Output:
    - 01_receptor/sstr2_receptor.pdb   -- cleaned receptor-only PDB
    - 01_receptor/binding_pocket.json  -- pocket residue numbers + centroid

Public API:
    prepare_receptor(config)               -> Step01Output
    analyze_binding_pocket(pdb, chain, d)  -> dict
    extract_receptor_chain(pdb, chain)     -> str  (PDB text)
    fallback_load_existing(data_dir)       -> Step01Output
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal imports (schemas live in AG_SRC/schemas/)
# ---------------------------------------------------------------------------

from ..schemas.io_schemas import Step01Output


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_RECEPTOR_CHAIN: str = "B"
_DEFAULT_POCKET_CUTOFF: float = 5.0   # Angstroms


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def prepare_receptor(config: Dict[str, Any]) -> Step01Output:
    """SSTR2 수용체 구조를 준비하고 포켓 정의를 반환한다.

    Orchestration entry point for Step 01.  Tries the following in order:

    1. If ``config["receptor"]["existing_pdb"]`` is set and the file exists,
       load it directly (fastest path – no API call).
    2. Otherwise call the OpenFold3 NIM API to predict the structure.
    3. On any API failure, fall back to ``fallback_load_existing()``.

    Args:
        config: Pipeline configuration dict loaded from pipeline_config.yaml.
                Must contain ``receptor``, ``output_base_dir``, and ``run_id``
                keys (plus optionally ``receptor.existing_pdb``).

    Returns:
        Step01Output with paths to the receptor PDB and pocket JSON.
    """
    run_id: str = config.get("run_id", "default_run")
    output_base: Path = Path(config.get("output_base_dir", "runs")) / run_id
    out_dir: Path = output_base / "01_receptor"
    out_dir.mkdir(parents=True, exist_ok=True)

    receptor_cfg: Dict[str, Any] = config.get("receptor", {})
    data_dir: Path = Path(config.get("data_dir", "data"))

    # ------------------------------------------------------------------
    # Path 1: existing PDB supplied in config
    # ------------------------------------------------------------------
    existing_pdb: Optional[str] = receptor_cfg.get("existing_pdb")
    if existing_pdb and Path(existing_pdb).exists():
        logger.info("[Step01] Using pre-existing receptor PDB: %s", existing_pdb)
        receptor_chain = receptor_cfg.get("chain", _DEFAULT_RECEPTOR_CHAIN)
        receptor_pdb_text = Path(existing_pdb).read_text(encoding="utf-8")
        receptor_pdb_text = extract_receptor_chain(receptor_pdb_text, receptor_chain)
        return _finalize_output(
            receptor_pdb_text=receptor_pdb_text,
            pocket_residues=receptor_cfg.get("pocket_residues", []),
            chain_id=receptor_chain,
            out_dir=out_dir,
        )

    # ------------------------------------------------------------------
    # Path 2: OpenFold3 API
    # ------------------------------------------------------------------
    try:
        logger.info("[Step01] Calling OpenFold3 API to predict SSTR2 structure...")
        receptor_pdb_text = _call_openfold3(receptor_cfg, config)
        receptor_chain = receptor_cfg.get("chain", _DEFAULT_RECEPTOR_CHAIN)
        receptor_pdb_text = extract_receptor_chain(receptor_pdb_text, receptor_chain)
        return _finalize_output(
            receptor_pdb_text=receptor_pdb_text,
            pocket_residues=receptor_cfg.get("pocket_residues", []),
            chain_id=receptor_chain,
            out_dir=out_dir,
        )
    except Exception as exc:
        logger.warning(
            "[Step01] OpenFold3 API failed (%s). Falling back to pre-existing structure.",
            exc,
        )

    # ------------------------------------------------------------------
    # Path 3: fallback
    # ------------------------------------------------------------------
    return fallback_load_existing(data_dir, out_dir, receptor_cfg)


def analyze_binding_pocket(
    receptor_pdb: str,
    ligand_chain: str,
    distance_cutoff: float = _DEFAULT_POCKET_CUTOFF,
) -> Dict[str, Any]:
    """수용체 PDB와 리간드 체인을 기반으로 결합 포켓을 분석한다.

    Identifies receptor residues within *distance_cutoff* Angstroms of any
    atom in *ligand_chain*.  Returns residue numbers, a centroid, and a
    bounding-sphere radius.

    Args:
        receptor_pdb:    Full PDB text (may contain multiple chains).
        ligand_chain:    Chain ID of the ligand / peptide in *receptor_pdb*.
        distance_cutoff: Distance threshold in Angstroms.

    Returns:
        dict with keys:
            ``pocket_residues`` (List[int]),
            ``centroid``        (List[float]),
            ``radius``          (float).
    """
    try:
        from Bio.PDB import PDBParser  # type: ignore
        from io import StringIO
        import numpy as np

        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("complex", StringIO(receptor_pdb))

        ligand_coords: List[Tuple[float, float, float]] = []
        receptor_atoms: List[Any] = []

        for model in structure:
            for chain in model:
                for residue in chain:
                    for atom in residue:
                        if chain.id == ligand_chain:
                            ligand_coords.append(atom.get_vector().get_array())
                        else:
                            receptor_atoms.append((chain.id, residue.id[1], atom))

        if not ligand_coords:
            logger.warning(
                "[Step01] No atoms found in ligand chain '%s'. "
                "Returning empty pocket.",
                ligand_chain,
            )
            return {"pocket_residues": [], "centroid": [0.0, 0.0, 0.0], "radius": 0.0}

        ligand_arr = _np_array(ligand_coords)
        pocket_residue_set: set[int] = set()

        for chain_id, res_num, atom in receptor_atoms:
            atom_coord = _np_array([atom.get_vector().get_array()])
            dists = _cdist_min(atom_coord, ligand_arr)
            if dists <= distance_cutoff:
                pocket_residue_set.add(res_num)

        centroid = ligand_arr.mean(axis=0).tolist()
        radius = float(
            max(
                float(_np_norm(coord - _np_array([centroid])))
                for coord in ligand_arr
            )
        ) if len(ligand_arr) > 0 else 0.0

        logger.info(
            "[Step01] Pocket analysis: %d residues within %.1f Å of ligand chain '%s'.",
            len(pocket_residue_set),
            distance_cutoff,
            ligand_chain,
        )
        return {
            "pocket_residues": sorted(pocket_residue_set),
            "centroid": centroid,
            "radius": radius,
        }

    except ImportError:
        logger.warning(
            "[Step01] BioPython/numpy not available. "
            "Returning empty pocket from analyze_binding_pocket()."
        )
        return {"pocket_residues": [], "centroid": [0.0, 0.0, 0.0], "radius": 0.0}


def extract_receptor_chain(complex_pdb: str, receptor_chain: str = "B") -> str:
    """복합체 PDB에서 수용체 체인만 추출하여 반환한다.

    Filters the PDB text so that only ATOM/HETATM records belonging to
    *receptor_chain* are retained.  TER and END records are preserved.

    Args:
        complex_pdb:     Full PDB text potentially containing multiple chains.
        receptor_chain:  Chain identifier to keep (default ``"B"``).

    Returns:
        Filtered PDB text containing only the receptor chain records.
    """
    lines_out: List[str] = []
    for line in complex_pdb.splitlines():
        record = line[:6].strip()
        if record in ("ATOM", "HETATM"):
            # Column 22 (0-indexed 21) is the chain ID in standard PDB format
            if len(line) >= 22 and line[21] == receptor_chain:
                lines_out.append(line)
        elif record in ("TER", "END", "REMARK", "HEADER", "TITLE", "CRYST1"):
            lines_out.append(line)
    if not lines_out or lines_out[-1].strip() != "END":
        lines_out.append("END")
    logger.info(
        "[Step01] Extracted chain '%s': %d ATOM/HETATM lines.",
        receptor_chain,
        sum(1 for l in lines_out if l[:6].strip() in ("ATOM", "HETATM")),
    )
    return "\n".join(lines_out) + "\n"


def fallback_load_existing(
    data_dir: Path,
    out_dir: Optional[Path] = None,
    receptor_cfg: Optional[Dict[str, Any]] = None,
) -> Step01Output:
    """data/ ディレクトリの既存 SSTR2 PDB を読み込む（フォールバック）。

    Loads a pre-existing SSTR2 PDB from *data_dir* when API-based structure
    prediction is unavailable.  Searches for files matching ``sstr2*.pdb``
    or ``receptor*.pdb`` (case-insensitive).

    Args:
        data_dir:     Directory to search for pre-existing PDB files.
        out_dir:      Destination directory for copied output files.
                      If None, uses ``data_dir``.
        receptor_cfg: Optional receptor section from pipeline_config.yaml.

    Returns:
        Step01Output populated from the found PDB.

    Raises:
        FileNotFoundError: When no suitable PDB file is found in *data_dir*.
    """
    receptor_cfg = receptor_cfg or {}
    data_path = Path(data_dir)
    out_path = Path(out_dir) if out_dir else data_path

    # Search patterns in priority order
    candidates: List[Path] = []
    for pattern in ("sstr2*.pdb", "receptor*.pdb", "*.pdb"):
        candidates.extend(sorted(data_path.glob(pattern)))
    if not candidates:
        raise FileNotFoundError(
            f"[Step01] No receptor PDB found in {data_path}. "
            "Provide 'receptor.existing_pdb' in config or place a .pdb in the data/ dir."
        )

    chosen = candidates[0]
    logger.info("[Step01] Fallback: loading receptor PDB from %s", chosen)

    receptor_chain = receptor_cfg.get("chain", _DEFAULT_RECEPTOR_CHAIN)
    pdb_text = chosen.read_text(encoding="utf-8")
    # Only filter by chain if multi-chain content is detected
    chains_present = {l[21] for l in pdb_text.splitlines() if l[:4] == "ATOM" and len(l) >= 22}
    if len(chains_present) > 1:
        pdb_text = extract_receptor_chain(pdb_text, receptor_chain)

    pocket_residues: List[int] = receptor_cfg.get("pocket_residues", [])
    return _finalize_output(
        receptor_pdb_text=pdb_text,
        pocket_residues=pocket_residues,
        chain_id=receptor_chain,
        out_dir=out_path,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _finalize_output(
    receptor_pdb_text: str,
    pocket_residues: List[int],
    chain_id: str,
    out_dir: Path,
) -> Step01Output:
    """Write output files and return a Step01Output dataclass."""
    out_dir.mkdir(parents=True, exist_ok=True)

    receptor_pdb_path = out_dir / "sstr2_receptor.pdb"
    receptor_pdb_path.write_text(receptor_pdb_text, encoding="utf-8")
    logger.info("[Step01] Receptor PDB written -> %s", receptor_pdb_path)

    pocket_info: Dict[str, Any] = {
        "chain_id": chain_id,
        "pocket_residues": pocket_residues,
        "n_residues": len(pocket_residues),
    }
    pocket_json_path = out_dir / "binding_pocket.json"
    pocket_json_path.write_text(
        json.dumps(pocket_info, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info("[Step01] Pocket JSON written -> %s", pocket_json_path)

    return Step01Output(
        receptor_pdb_path=str(receptor_pdb_path),
        pocket_residues=pocket_residues,
        chain_id=chain_id,
        pocket_json_path=str(pocket_json_path),
    )


def _call_openfold3(receptor_cfg: Dict[str, Any], config: Dict[str, Any]) -> str:
    """OpenFold3 NIM API를 호출하여 구조 예측 결과 PDB 텍스트를 반환한다.

    Args:
        receptor_cfg: Receptor section from pipeline_config.
        config:       Full pipeline config (for API key discovery).

    Returns:
        PDB text string from OpenFold3.

    Raises:
        RuntimeError: On API error or missing sequence.
    """
    sequence: Optional[str] = receptor_cfg.get("sequence")
    if not sequence:
        raise RuntimeError(
            "[Step01] 'receptor.sequence' must be set in config to use OpenFold3."
        )

    try:
        from ..tools.api.base_tool import BaseTool  # type: ignore
    except ImportError:
        pass  # BaseTool import failure handled below

    import requests

    api_key = _resolve_api_key()
    endpoint = "https://health.api.nvidia.com/v1/biology/openfold/openfold3"
    payload = {
        "sequences": [{"protein": {"id": "A", "sequence": sequence}}]
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    timeout = int(config.get("tools", {}).get("api", {}).get("openfold3", {}).get("timeout_sec", 600))
    resp = requests.post(endpoint, json=payload, headers=headers, timeout=timeout)
    if resp.status_code != 200:
        raise RuntimeError(
            f"[Step01] OpenFold3 API returned HTTP {resp.status_code}: {resp.text[:300]}"
        )
    data = resp.json()
    pdb_text: Optional[str] = (
        data.get("output_pdb")
        or data.get("pdb")
        or data.get("result", {}).get("pdb")
    )
    if not pdb_text:
        raise RuntimeError(
            f"[Step01] OpenFold3 response missing 'output_pdb' key. Keys: {list(data.keys())}"
        )
    return pdb_text


def _resolve_api_key() -> str:
    """환경변수 / 키 파일에서 NVIDIA API 키를 탐색한다."""
    for var in ("NGC_CLI_API_KEY", "NVIDIA_API_KEY"):
        val = os.getenv(var, "").strip()
        if val:
            return val
    for name in ("ngc.key", "molmim.key"):
        for directory in (Path(__file__).parent.parent.parent, Path.cwd()):
            key_file = directory / name
            if key_file.exists():
                val = key_file.read_text(encoding="utf-8").strip()
                if val:
                    return val
    raise ValueError(
        "NVIDIA API key not found. Set NGC_CLI_API_KEY env var or create ngc.key."
    )


# ---------------------------------------------------------------------------
# numpy shims (avoid hard dependency for pure I/O paths)
# ---------------------------------------------------------------------------


def _np_array(lst: Any) -> Any:
    try:
        import numpy as np
        return np.array(lst, dtype=float)
    except ImportError:
        return lst


def _cdist_min(a: Any, b: Any) -> float:
    try:
        import numpy as np
        diffs = np.array(b) - np.array(a)
        return float(np.linalg.norm(diffs, axis=1).min())
    except Exception:
        return 0.0


def _np_norm(arr: Any) -> float:
    try:
        import numpy as np
        return float(np.linalg.norm(arr))
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# CLI / standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Step01: Receptor preparation standalone test")
    parser.add_argument("--data-dir", default="data", help="Directory with pre-existing PDB")
    parser.add_argument("--output-dir", default="runs/test_run/01_receptor")
    args = parser.parse_args()

    dummy_cfg: Dict[str, Any] = {
        "run_id": "test_run",
        "output_base_dir": "runs",
        "data_dir": args.data_dir,
        "receptor": {
            "chain": "B",
            "pocket_residues": [119, 122, 127, 184, 197, 205, 272, 294],
        },
    }
    try:
        result = prepare_receptor(dummy_cfg)
        print(f"Step01 result: {result}")
    except FileNotFoundError as e:
        print(f"[WARN] {e}")
        print("Place a SSTR2 PDB file in the data/ directory to test fallback loading.")
