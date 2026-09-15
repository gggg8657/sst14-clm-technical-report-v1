"""
PyRosetta MCP Server — Step06: Physics-Based Scoring & Design
=============================================================
Exposes PyRosetta computations as MCP (Model Context Protocol) tool endpoints
so the agent can call them like any other tool.

IMPORTANT: This is an *interface definition* and *implementation skeleton*.
Actual PyRosetta calls require the bio-tools conda environment:
    conda activate bio-tools
    conda install -c conda-forge pyrosetta  # or via Rosetta license

Each tool is registered with a JSON Schema for input and output validation.
The MCPServer class manages registration and dispatching.

Tools exposed:
    1. relax_structure      — fast relax with optional Cartesian restraints
    2. compute_ddg          — single-point mutation free energy change
    3. compute_binding_energy — interface ΔG via jump separation
    4. flexpep_dock         — FlexPepDock refinement / ab-initio
    5. fast_design          — FastDesign at specified positions
    6. energy_decomposition — per-residue energy breakdown
    7. interface_analysis   — InterfaceAnalyzer metrics

Usage (skeleton):
    server = PyRosettaMCPServer()
    result = server.dispatch("relax_structure", pdb_path="/tmp/complex.pdb")
"""

from __future__ import annotations

import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Try to import PyRosetta (optional — server can still be instantiated)
# ---------------------------------------------------------------------------

try:
    import pyrosetta  # type: ignore[import-untyped]
    import pyrosetta.rosetta as rosetta  # type: ignore[import-untyped]

    _HAS_PYROSETTA = True
except ImportError:
    _HAS_PYROSETTA = False
    logger.warning(
        "PyRosetta not found. Activate the bio-tools conda environment "
        "before calling any PyRosetta tools."
    )

from .base_server import MCPServer, MCPTool


# ---------------------------------------------------------------------------
# PyRosetta MCP Server
# ---------------------------------------------------------------------------


class PyRosettaMCPServer(MCPServer):
    """MCP server wrapping PyRosetta computations.

    On instantiation, all seven tools are registered. If PyRosetta is not
    available, the tools are registered but calling them will raise ImportError.
    """

    def __init__(self, init_flags: str = "-ex1 -ex2aro -ignore_unrecognized_res") -> None:
        super().__init__("pyrosetta")
        self._init_flags = init_flags
        self._initialised = False
        self._register_all()

    # ------------------------------------------------------------------
    # PyRosetta initialisation (lazy)
    # ------------------------------------------------------------------

    def _ensure_init(self) -> None:
        """Initialise PyRosetta once (lazy, thread-unsafe)."""
        if self._initialised:
            return
        if not _HAS_PYROSETTA:
            raise ImportError(
                "PyRosetta is not installed. "
                "Activate the bio-tools conda environment."
            )
        pyrosetta.init(self._init_flags, silent=True)
        self._initialised = True
        logger.info("PyRosetta initialised with flags: %s", self._init_flags)

    # ------------------------------------------------------------------
    # Tool registration
    # ------------------------------------------------------------------

    def _register_all(self) -> None:
        tools = [
            MCPTool(
                name="relax_structure",
                description=(
                    "Run FastRelax on the input structure to remove clashes "
                    "and optimise side-chain positions."
                ),
                input_schema={
                    "type": "object",
                    "required": ["pdb_path"],
                    "properties": {
                        "pdb_path": {
                            "type": "string",
                            "description": "Absolute path to the input PDB file.",
                        },
                        "cycles": {
                            "type": "integer",
                            "default": 3,
                            "description": "Number of FastRelax cycles.",
                        },
                        "cartesian": {
                            "type": "boolean",
                            "default": True,
                            "description": "Use Cartesian space relaxation.",
                        },
                    },
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "relaxed_pdb_path": {
                            "type": "string",
                            "description": "Path to the relaxed output PDB.",
                        },
                        "score_before": {"type": "number"},
                        "score_after": {"type": "number"},
                    },
                },
                handler=self._relax_structure,
            ),
            MCPTool(
                name="compute_ddg",
                description=(
                    "Estimate the free energy change (ΔΔG) of a single-point "
                    "mutation using the Cartesian ddG protocol."
                ),
                input_schema={
                    "type": "object",
                    "required": ["complex_pdb", "mutation_resid", "new_aa"],
                    "properties": {
                        "complex_pdb": {
                            "type": "string",
                            "description": "Path to the complex PDB file.",
                        },
                        "mutation_resid": {
                            "type": "string",
                            "description": "Residue identifier, e.g. 'A42' (chain + number).",
                        },
                        "new_aa": {
                            "type": "string",
                            "description": "Single-letter code of the new amino acid.",
                        },
                    },
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "ddg_value": {
                            "type": "number",
                            "description": "ΔΔG in Rosetta energy units (REU). "
                                           "Negative = stabilising.",
                        },
                        "wt_score": {"type": "number"},
                        "mut_score": {"type": "number"},
                    },
                },
                handler=self._compute_ddg,
            ),
            MCPTool(
                name="compute_binding_energy",
                description=(
                    "Compute the binding free energy (ΔG_bind) of a protein "
                    "complex by jump separation."
                ),
                input_schema={
                    "type": "object",
                    "required": ["complex_pdb"],
                    "properties": {
                        "complex_pdb": {
                            "type": "string",
                            "description": "Path to the complex PDB file.",
                        },
                        "jump_id": {
                            "type": "integer",
                            "default": 1,
                            "description": "Fold-tree jump ID that separates the chains.",
                        },
                    },
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "dg_bind": {
                            "type": "number",
                            "description": "ΔG_bind in REU.",
                        },
                        "dg_complex": {"type": "number"},
                        "dg_separated": {"type": "number"},
                    },
                },
                handler=self._compute_binding_energy,
            ),
            MCPTool(
                name="flexpep_dock",
                description=(
                    "Dock or refine a peptide in a receptor binding site using "
                    "the FlexPepDock protocol."
                ),
                input_schema={
                    "type": "object",
                    "required": ["complex_pdb"],
                    "properties": {
                        "complex_pdb": {
                            "type": "string",
                            "description": "Path to the complex PDB with receptor + peptide.",
                        },
                        "protocol": {
                            "type": "string",
                            "enum": ["refine", "ab_initio"],
                            "default": "refine",
                            "description": (
                                "'refine' = local refinement around input pose; "
                                "'ab_initio' = global search."
                            ),
                        },
                    },
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "docked_pdb": {
                            "type": "string",
                            "description": "Path to the best docked complex PDB.",
                        },
                        "score": {
                            "type": "number",
                            "description": "Rosetta total score of the docked pose.",
                        },
                        "reweighted_score": {"type": "number"},
                    },
                },
                handler=self._flexpep_dock,
            ),
            MCPTool(
                name="fast_design",
                description=(
                    "Run FastDesign at specified residue positions to optimise "
                    "the sequence of the binder."
                ),
                input_schema={
                    "type": "object",
                    "required": ["complex_pdb", "design_positions"],
                    "properties": {
                        "complex_pdb": {
                            "type": "string",
                            "description": "Path to the complex PDB.",
                        },
                        "design_positions": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": (
                                "List of residue identifiers to design "
                                "(e.g. ['A1', 'A2', 'A5'])."
                            ),
                        },
                        "scorefxn": {
                            "type": "string",
                            "default": "ref2015",
                            "description": "Rosetta score function name.",
                        },
                    },
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "designed_pdb": {
                            "type": "string",
                            "description": "Path to the designed complex PDB.",
                        },
                        "sequence": {
                            "type": "string",
                            "description": "Designed sequence of the binder chain.",
                        },
                        "score": {"type": "number"},
                    },
                },
                handler=self._fast_design,
            ),
            MCPTool(
                name="energy_decomposition",
                description=(
                    "Compute per-residue energy decomposition of a complex using "
                    "the ref2015 score function."
                ),
                input_schema={
                    "type": "object",
                    "required": ["complex_pdb"],
                    "properties": {
                        "complex_pdb": {
                            "type": "string",
                            "description": "Path to the complex PDB.",
                        },
                    },
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "per_residue_energies": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "resid": {"type": "string"},
                                    "total": {"type": "number"},
                                    "fa_atr": {"type": "number"},
                                    "fa_rep": {"type": "number"},
                                    "fa_sol": {"type": "number"},
                                    "hbond_sc": {"type": "number"},
                                    "hbond_bb_sc": {"type": "number"},
                                },
                            },
                        },
                        "total_score": {"type": "number"},
                    },
                },
                handler=self._energy_decomposition,
            ),
            MCPTool(
                name="interface_analysis",
                description=(
                    "Run InterfaceAnalyzer to compute interface energy, buried "
                    "surface area, and contact statistics."
                ),
                input_schema={
                    "type": "object",
                    "required": ["complex_pdb", "chain_A", "chain_B"],
                    "properties": {
                        "complex_pdb": {
                            "type": "string",
                            "description": "Path to the complex PDB.",
                        },
                        "chain_A": {
                            "type": "string",
                            "description": "Chain ID of the receptor (e.g. 'B').",
                        },
                        "chain_B": {
                            "type": "string",
                            "description": "Chain ID of the binder (e.g. 'A').",
                        },
                    },
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "dG_separated": {"type": "number"},
                        "dG_separated_dev": {"type": "number"},
                        "dSASA_int": {"type": "number"},
                        "dSASA_hphobic": {"type": "number"},
                        "dSASA_polar": {"type": "number"},
                        "packstat": {"type": "number"},
                        "n_hbonds_int": {"type": "integer"},
                        "sc_value": {"type": "number"},
                        "interface_report": {"type": "object"},
                    },
                },
                handler=self._interface_analysis,
            ),
        ]
        for tool in tools:
            self.register(tool)

    # ------------------------------------------------------------------
    # Tool handler implementations (skeletons)
    # ------------------------------------------------------------------

    def _relax_structure(
        self,
        pdb_path: str,
        cycles: int = 3,
        cartesian: bool = True,
    ) -> dict[str, Any]:
        """Skeleton: FastRelax the input structure."""
        # IMPLEMENTATION SKELETON
        # Actual code requires PyRosetta in the bio-tools conda environment.
        self._ensure_init()

        pose = pyrosetta.pose_from_pdb(pdb_path)
        scorefxn = pyrosetta.get_fa_scorefxn()
        score_before = scorefxn(pose)

        relax = rosetta.protocols.relax.FastRelax()
        relax.set_scorefxn(scorefxn)
        relax.max_iter(cycles * 100)
        if cartesian:
            relax.cartesian(True)
        relax.apply(pose)

        score_after = scorefxn(pose)
        output_path = Path(pdb_path).with_suffix(".relaxed.pdb")
        pose.dump_pdb(str(output_path))

        return {
            "relaxed_pdb_path": str(output_path),
            "score_before": score_before,
            "score_after": score_after,
        }

    def _compute_ddg(
        self,
        complex_pdb: str,
        mutation_resid: str,
        new_aa: str,
    ) -> dict[str, Any]:
        """Skeleton: Cartesian ΔΔG for a single-point mutation."""
        self._ensure_init()

        # Parse chain and residue number from resid (e.g. "A42")
        chain = mutation_resid[0]
        resnum = int(mutation_resid[1:])

        pose = pyrosetta.pose_from_pdb(complex_pdb)
        scorefxn = pyrosetta.get_fa_scorefxn()
        wt_score = scorefxn(pose)

        # Apply mutation using MutateResidue mover
        target_seqpos = pose.pdb_info().pdb2pose(chain, resnum)
        mutate = rosetta.protocols.simple_moves.MutateResidue(
            target_seqpos, new_aa
        )
        mut_pose = pose.clone()
        mutate.apply(mut_pose)
        mut_score = scorefxn(mut_pose)

        ddg = mut_score - wt_score
        return {
            "ddg_value": ddg,
            "wt_score": wt_score,
            "mut_score": mut_score,
        }

    def _compute_binding_energy(
        self,
        complex_pdb: str,
        jump_id: int = 1,
    ) -> dict[str, Any]:
        """Skeleton: ΔG_bind via jump separation."""
        self._ensure_init()

        pose = pyrosetta.pose_from_pdb(complex_pdb)
        scorefxn = pyrosetta.get_fa_scorefxn()
        dg_complex = scorefxn(pose)

        # Separate chains along the specified jump
        separated = pose.clone()
        rosetta.protocols.rigid.RigidBodyTransMover(
            separated, jump_id
        ).apply(separated)
        dg_separated = scorefxn(separated)

        dg_bind = dg_complex - dg_separated
        return {
            "dg_bind": dg_bind,
            "dg_complex": dg_complex,
            "dg_separated": dg_separated,
        }

    def _flexpep_dock(
        self,
        complex_pdb: str,
        protocol: str = "refine",
    ) -> dict[str, Any]:
        """Skeleton: FlexPepDock refinement or ab-initio."""
        self._ensure_init()

        pose = pyrosetta.pose_from_pdb(complex_pdb)
        scorefxn = pyrosetta.get_fa_scorefxn()

        # Build FlexPepDock mover via RosettaScripts-equivalent
        fpd = rosetta.protocols.flexpep_docking.FlexPepDockingProtocol()
        if protocol == "ab_initio":
            fpd.set_lowres_preoptimize(True)
        fpd.apply(pose)

        output_path = Path(complex_pdb).with_suffix(".docked.pdb")
        pose.dump_pdb(str(output_path))
        score = scorefxn(pose)

        return {
            "docked_pdb": str(output_path),
            "score": score,
            "reweighted_score": score,  # placeholder: use reweighted scorefxn
        }

    def _fast_design(
        self,
        complex_pdb: str,
        design_positions: list[str],
        scorefxn: str = "ref2015",
    ) -> dict[str, Any]:
        """Skeleton: FastDesign at specified positions."""
        self._ensure_init()

        pose = pyrosetta.pose_from_pdb(complex_pdb)
        sfxn = pyrosetta.create_score_function(scorefxn)

        # Build residue selector for design positions
        selector = rosetta.core.select.residue_selector.ResidueIndexSelector()
        seqpos_list = []
        for resid in design_positions:
            chain = resid[0]
            resnum = int(resid[1:])
            seqpos = pose.pdb_info().pdb2pose(chain, resnum)
            seqpos_list.append(str(seqpos))
        selector.set_index(",".join(seqpos_list))

        task_factory = rosetta.core.pack.task.TaskFactory()
        task_factory.push_back(
            rosetta.core.pack.task.operation.OperateOnResidueSubset(
                rosetta.core.pack.task.operation.RestrictToRepackingRLT(),
                selector,
                flip_subset=True,  # repack non-design positions
            )
        )

        fast_design = rosetta.protocols.denovo_design.movers.FastDesign()
        fast_design.set_scorefxn(sfxn)
        fast_design.set_task_factory(task_factory)
        fast_design.apply(pose)

        output_path = Path(complex_pdb).with_suffix(".designed.pdb")
        pose.dump_pdb(str(output_path))
        sequence = pose.sequence()
        score = sfxn(pose)

        return {
            "designed_pdb": str(output_path),
            "sequence": sequence,
            "score": score,
        }

    def _energy_decomposition(self, complex_pdb: str) -> dict[str, Any]:
        """Skeleton: Per-residue energy decomposition."""
        self._ensure_init()

        pose = pyrosetta.pose_from_pdb(complex_pdb)
        scorefxn = pyrosetta.get_fa_scorefxn()
        scorefxn(pose)  # fills energies

        per_residue: list[dict[str, Any]] = []
        for i in range(1, pose.total_residue() + 1):
            e = pose.energies().residue_total_energies(i)
            chain = pose.pdb_info().chain(i)
            resnum = pose.pdb_info().number(i)
            per_residue.append(
                {
                    "resid": f"{chain}{resnum}",
                    "total": e[rosetta.core.scoring.total_score],
                    "fa_atr": e[rosetta.core.scoring.fa_atr],
                    "fa_rep": e[rosetta.core.scoring.fa_rep],
                    "fa_sol": e[rosetta.core.scoring.fa_sol],
                    "hbond_sc": e[rosetta.core.scoring.hbond_sc],
                    "hbond_bb_sc": e[rosetta.core.scoring.hbond_bb_sc],
                }
            )

        total_score = pose.energies().total_energy()
        return {
            "per_residue_energies": per_residue,
            "total_score": total_score,
        }

    def _interface_analysis(
        self,
        complex_pdb: str,
        chain_A: str,
        chain_B: str,
    ) -> dict[str, Any]:
        """Skeleton: InterfaceAnalyzer metrics."""
        self._ensure_init()

        pose = pyrosetta.pose_from_pdb(complex_pdb)
        interface = f"{chain_A}_{chain_B}"

        ia = rosetta.protocols.analysis.InterfaceAnalyzerMover(interface)
        ia.set_pack_input(True)
        ia.set_pack_separated(True)
        ia.apply(pose)

        report = {
            "dG_separated": ia.get_interface_dG(),
            "dG_separated_dev": ia.get_interface_dG_dev(),
            "dSASA_int": ia.get_interface_delta_sasa(),
            "dSASA_hphobic": 0.0,    # placeholder — extract from data map
            "dSASA_polar": 0.0,       # placeholder
            "packstat": ia.get_interface_packstat(),
            "n_hbonds_int": int(ia.get_interface_Hbond_num()),
            "sc_value": ia.get_interface_sc(),
        }
        report["interface_report"] = report.copy()
        return report


def get_server(**kwargs: Any) -> PyRosettaMCPServer:
    """Convenience factory used by the pipeline."""
    return PyRosettaMCPServer(**kwargs)
