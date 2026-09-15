#!/usr/bin/env python3
"""
download_alphafold.py
=====================
AlphaFold DB에서 단백질 구조를 다운로드하는 유틸리티 스크립트.

Usage:
    python AG_src/scripts/download_alphafold.py --uniprot P30872 --output ./receptors/

Called by run_pipeline_live.py Step 05b to fetch SSTR1/3/4/5 structures.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

ALPHAFOLD_API = "https://alphafold.ebi.ac.uk/api/prediction/{uniprot_id}"
ALPHAFOLD_PDB = "https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_v{version}.pdb"


def _get_latest_version(uniprot_id: str) -> int:
    """Query AlphaFold API for the latest model version."""
    url = ALPHAFOLD_API.format(uniprot_id=uniprot_id)
    try:
        with urllib.request.urlopen(url, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            if isinstance(data, list) and data:
                return int(data[0].get("latestVersion", 4))
    except Exception as e:
        print(f"API version lookup failed for {uniprot_id}: {e}", file=sys.stderr)
    return 4  # fallback


def download_alphafold_structure(uniprot_id: str, output_dir: str) -> str:
    """Download AlphaFold structure by UniProt ID. Uses cache if already exists.

    Automatically detects the latest model version via the AlphaFold API.

    Args:
        uniprot_id: UniProt accession (e.g. "P30872")
        output_dir: Directory to save the PDB file

    Returns:
        Path to the downloaded PDB file

    Raises:
        RuntimeError: If download fails after retries
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Check cache for any existing version
    for cached in sorted(out_dir.glob(f"AF-{uniprot_id}-F1-model_v*.pdb"), reverse=True):
        if cached.stat().st_size > 1000:
            return str(cached)

    version = _get_latest_version(uniprot_id)
    out_path = out_dir / f"AF-{uniprot_id}-F1-model_v{version}.pdb"
    url = ALPHAFOLD_PDB.format(uniprot_id=uniprot_id, version=version)

    for attempt in range(3):
        try:
            urllib.request.urlretrieve(url, str(out_path))
            if out_path.exists() and out_path.stat().st_size > 1000:
                return str(out_path)
        except urllib.error.HTTPError as e:
            print(f"HTTP {e.code} for {uniprot_id} v{version} (attempt {attempt + 1}/3)", file=sys.stderr)
        except urllib.error.URLError as e:
            print(f"URL error for {uniprot_id}: {e} (attempt {attempt + 1}/3)", file=sys.stderr)

    raise RuntimeError(f"Failed to download AlphaFold structure for {uniprot_id}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download AlphaFold structure by UniProt ID")
    parser.add_argument("--uniprot", required=True, help="UniProt accession ID (e.g. P30872)")
    parser.add_argument("--output", required=True, help="Output directory for PDB file")
    args = parser.parse_args()

    path = download_alphafold_structure(args.uniprot, args.output)
    print(f"Downloaded: {path}")


if __name__ == "__main__":
    main()
