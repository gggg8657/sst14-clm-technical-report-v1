"""
AG_SRC/tools/api — NVIDIA NIM API Tool Wrappers
================================================
All API tool classes and their ToolResult type are exported from this package.

Usage:
    from AG_SRC.tools.api import (
        ToolResult,
        BaseTool,
        OpenFold3Tool,
        RFdiffusionTool,
        ProteinMPNNTool,
        ESMFoldTool,
        DiffDockTool,
        Boltz2Tool,
        MolMIMTool,
        ESM2Tool,
    )

Or use the convenience factory functions:
    from AG_SRC.tools.api.rfdiffusion_tool import get_tool as get_rfdiffusion
    tool = get_rfdiffusion()
    result = tool.design_binder(receptor_pdb=..., contigs=..., hotspot_res=...)
"""

from .base_tool import BaseTool, ToolResult
from .boltz2_tool import Boltz2Tool
from .diffdock_tool import DiffDockTool
from .esm2_tool import ESM2Tool
from .esmfold_tool import ESMFoldTool
from .molmim_tool import MolMIMTool
from .openfold3_tool import OpenFold3Tool
from .proteinmpnn_tool import ProteinMPNNTool
from .rfdiffusion_tool import RFdiffusionTool

__all__ = [
    # Base / shared types
    "BaseTool",
    "ToolResult",
    # Pipeline tools (in step order)
    "OpenFold3Tool",    # Step01 — target structure prediction
    "RFdiffusionTool",  # Step02 — de novo backbone design
    "ProteinMPNNTool",  # Step03 — inverse folding / sequence design
    "ESMFoldTool",      # Step04 — sequence-to-structure validation
    "DiffDockTool",     # Step05 — protein–ligand docking
    "Boltz2Tool",       # Step05 — complex structure + affinity (complement)
    # Auxiliary tools
    "MolMIMTool",       # Small molecule generation / optimisation
    "ESM2Tool",         # Sequence embeddings / mutation analysis
]
