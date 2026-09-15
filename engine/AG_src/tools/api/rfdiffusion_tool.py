"""
RFdiffusion Tool — Step02: De Novo Binder Backbone Design
===========================================================
Wraps the NVIDIA NIM RFdiffusion endpoint for de novo peptide/protein
backbone design conditioned on a receptor structure and hotspot residues.

Endpoint:
    https://health.api.nvidia.com/v1/biology/ipd/rfdiffusion/generate

Reference client: PRST_N_FM/bionemo/rfdiffusion_client.py
This module defines the *agentic tool interface* on top of the same API.

Typical use in the SSTR2 pipeline:
    - design_binder(): single backbone conditioned on SSTR2 binding pocket
    - design_multiple(): generate a diverse ensemble of N candidate backbones
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from .base_tool import BaseTool, ToolResult

logger = logging.getLogger(__name__)

_ENDPOINT = "https://health.api.nvidia.com/v1/biology/ipd/rfdiffusion/generate"
_TIMEOUT_S = 300


class RFdiffusionTool(BaseTool):
    """Tool wrapper for the RFdiffusion NIM API.

    Interface definition:
        design_binder(receptor_pdb, contigs, hotspot_res, ...) -> ToolResult
        design_multiple(receptor_pdb, contigs, hotspot_res, n_designs) -> ToolResult

    execute() dispatcher accepts action="design_binder" | "design_multiple".
    """

    name = "rfdiffusion"
    description = (
        "Design de novo protein/peptide binder backbones conditioned on a "
        "receptor structure using RFdiffusion diffusion-based generative model."
    )
    endpoint = _ENDPOINT
    timeout = _TIMEOUT_S

    # ------------------------------------------------------------------
    # Primary actions
    # ------------------------------------------------------------------

    def design_binder(
        self,
        receptor_pdb: str,
        contigs: str,
        hotspot_res: list[str] | None = None,
        diffusion_steps: int = 50,
        seed: int | None = None,
    ) -> ToolResult:
        """Design a single binder backbone.

        Args:
            receptor_pdb:    PDB content of the receptor (as a string).
                             Use Path("sstr2.pdb").read_text() to load from file.
            contigs:         RFdiffusion contig string specifying which receptor
                             residues to preserve and the length of the new binder.
                             Example: "B1-369/0 10-30"
                               -> keep chain B residues 1-369, design 10-30 aa binder.
            hotspot_res:     Receptor residues the binder must contact.
                             Example: ["B122", "B127", "B200"]
                             Where the letter is the chain ID and number is residue index.
            diffusion_steps: Number of denoising steps (default 50, range 15-200).
                             More steps -> higher quality, slower.
            seed:            Random seed for reproducibility. None = random.

        Returns:
            ToolResult where data contains:
                - "output_pdb":   PDB string of the designed complex (str)
                - "binder_chain": Chain identifier of the new binder (usually "A")

        Example:
            tool = RFdiffusionTool()
            result = tool.design_binder(
                receptor_pdb=Path("sstr2.pdb").read_text(),
                contigs="B1-369/0 15-25",
                hotspot_res=["B122", "B127"],
                seed=42,
            )
            pdb_text = result.data["output_pdb"]
        """
        if not receptor_pdb:
            return ToolResult.fail("receptor_pdb must not be empty")
        if not contigs:
            return ToolResult.fail("contigs must not be empty")

        payload: dict[str, Any] = {
            "input_pdb": receptor_pdb,
            "contigs": contigs,
            "diffusion_steps": diffusion_steps,
        }
        if hotspot_res:
            payload["hotspot_res"] = hotspot_res
        if seed is not None:
            payload["random_seed"] = seed

        try:
            data, elapsed_ms = self._post_timed("", payload)
            return ToolResult.ok(
                data=data,
                elapsed_ms=elapsed_ms,
                tool=self.name,
                action="design_binder",
                seed=seed,
            )
        except Exception as exc:
            logger.error("RFdiffusion design_binder failed: %s", exc)
            return ToolResult.fail(str(exc), tool=self.name, action="design_binder")

    def design_multiple(
        self,
        receptor_pdb: str,
        contigs: str,
        hotspot_res: list[str] | None = None,
        n_designs: int = 10,
        diffusion_steps: int = 50,
    ) -> ToolResult:
        """Design an ensemble of N binder backbones with varying seeds.

        Each design is called sequentially with seed=0..n_designs-1.
        Failed individual designs are recorded but do not abort the run.

        Args:
            receptor_pdb:    PDB content of the receptor (as a string).
            contigs:         RFdiffusion contig string.
            hotspot_res:     Receptor hotspot residues for binder contact.
            n_designs:       Total number of backbone designs to generate.
            diffusion_steps: Denoising steps per design.

        Returns:
            ToolResult where data contains:
                - "designs": list of per-design dicts, each containing:
                    - "design_idx": int
                    - "output_pdb": str  (present on success)
                    - "error":      str  (present on failure)
                - "n_success": number of successful designs
                - "n_failed":  number of failed designs

        Example:
            tool = RFdiffusionTool()
            result = tool.design_multiple(
                receptor_pdb=Path("sstr2.pdb").read_text(),
                contigs="B1-369/0 15-25",
                hotspot_res=["B122", "B127"],
                n_designs=20,
            )
            designs = result.data["designs"]
        """
        if not receptor_pdb:
            return ToolResult.fail("receptor_pdb must not be empty")

        designs: list[dict[str, Any]] = []
        n_success = 0
        n_failed = 0

        import time as _time
        t0 = _time.monotonic()

        for i in range(n_designs):
            logger.info("RFdiffusion design %d/%d (seed=%d)", i + 1, n_designs, i)
            single = self.design_binder(
                receptor_pdb=receptor_pdb,
                contigs=contigs,
                hotspot_res=hotspot_res,
                diffusion_steps=diffusion_steps,
                seed=i,
            )
            entry: dict[str, Any] = {"design_idx": i}
            if single.success:
                entry.update(single.data)
                n_success += 1
            else:
                entry["error"] = single.error
                n_failed += 1
                logger.warning("Design %d failed: %s", i, single.error)
            designs.append(entry)

        elapsed_ms = int((_time.monotonic() - t0) * 1000)
        return ToolResult.ok(
            data={
                "designs": designs,
                "n_success": n_success,
                "n_failed": n_failed,
            },
            elapsed_ms=elapsed_ms,
            tool=self.name,
            action="design_multiple",
        )

    # ------------------------------------------------------------------
    # execute() dispatcher
    # ------------------------------------------------------------------

    def execute(self, **kwargs: Any) -> ToolResult:
        """Dispatch to the appropriate action.

        Supported actions:
            "design_binder"   (default) — single backbone design
            "design_multiple"           — ensemble backbone design
        """
        action = kwargs.pop("action", "design_binder")
        if action == "design_binder":
            return self.design_binder(**kwargs)
        if action == "design_multiple":
            return self.design_multiple(**kwargs)
        return ToolResult.fail(f"Unknown action '{action}' for {self.name}")


def get_tool(**kwargs: Any) -> RFdiffusionTool:
    """Convenience factory used by the pipeline."""
    return RFdiffusionTool(**kwargs)
